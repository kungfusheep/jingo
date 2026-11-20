package jingo

// structencoder.go manages StructEncoder and its responsibilities.
// The general goal of the approach is to do as much of the necessary work as possible inside
// the 'compile' stage upon instantiation. This includes any logic, type assertions, buffering
// or otherwise. Changes made should consider first their ns/op impact and then their allocation
// profile also. Allocations should essentially remain at zero - albeit with the exclusion of the
// `.String()` stringer functionality which is somewhat out of our control.

import (
	"fmt"
	"io"
	"reflect"
	"strconv"
	"strings"
	"time"
	"unsafe"
)

// instruction describes the different ways we can execute a single instruction at runtime.
// static, leapFun/offset, fun are mutually exclusive. we've used a concrete type for speed.
type instruction struct {
	static  []byte                        // provides a fast path for writing static chunks without needing an instruction function
	kind    int                           // used to switch special paths in Marshal, like string fast path
	offset  uintptr                       // used in conjunction with leapFun
	leapFun func(unsafe.Pointer, *Buffer) // provides a fast path for simple write & avoids wrapping function to capture offset
	fun     func(unsafe.Pointer, *Buffer) // full instruction function for when the approaches above fail
	run     *run                          // consecutive static, string and int instructions written with one capacity check
}

const (
	kindNormal = iota
	kindStringField
	kindStatic
	kindInt
	kindRun
)

// run is a sequence of instructions whose output has an upper bound we can work out before writing any of it:
// the static chunks, 20 bytes per int and the lengths of the strings. it reserves that once and writes at a
// cursor, so the buffer's slice header isn't written back and reloaded around every append.
type run struct {
	fixed int           // static lengths plus 20 per int
	strs  []uintptr     // offsets of the string fields
	ops   []instruction // only kindStatic, kindStringField and kindInt
}

func (r *run) write(p unsafe.Pointer, w *Buffer) {
	need := r.bound(p)
	bs := grow(w.Bytes, need)
	n := len(bs)
	w.Bytes = bs[:r.put(p, bs[:n+need], n)]
}

// bound is the most bytes the run can write for the value at p.
func (r *run) bound(p unsafe.Pointer) int {
	need := r.fixed
	for _, off := range r.strs {
		need += len(*(*string)(unsafe.Pointer(uintptr(p) + off)))
	}
	return need
}

// put writes the run at bs[n:], which has room for bound(p) bytes, and returns the new length.
func (r *run) put(p unsafe.Pointer, bs []byte, n int) int {
	for i := range r.ops {
		ins := &r.ops[i]
		switch ins.kind {
		case kindStatic:
			n += copy(bs[n:], ins.static)
		case kindStringField:
			n += copy(bs[n:], *(*string)(unsafe.Pointer(uintptr(p) + ins.offset)))
		default:
			n = len(strconv.AppendInt(bs[:n], int64(*(*int)(unsafe.Pointer(uintptr(p) + ins.offset))), 10))
		}
	}
	return n
}

// flatRun returns the run when it's the encoder's only instruction, so a slice of these can reserve once.
func (e *StructEncoder) flatRun() *run {
	if len(e.instructions) == 1 && e.instructions[0].kind == kindRun {
		return e.instructions[0].run
	}
	return nil
}

// fuseRuns replaces each sequence of two or more bounded instructions with a single run.
func (e *StructEncoder) fuseRuns() {
	bounded := func(k int) bool { return k == kindStatic || k == kindStringField || k == kindInt }

	var out []instruction
	for i := 0; i < len(e.instructions); {
		j := i
		for j < len(e.instructions) && bounded(e.instructions[j].kind) {
			j++
		}
		if j-i < 2 {
			out = append(out, e.instructions[i])
			i++
			continue
		}

		r := &run{ops: append([]instruction(nil), e.instructions[i:j]...)}
		for _, ins := range r.ops {
			switch ins.kind {
			case kindStatic:
				r.fixed += len(ins.static)
			case kindStringField:
				r.strs = append(r.strs, ins.offset)
			case kindInt:
				r.fixed += 20
			}
		}
		out = append(out, instruction{kind: kindRun, run: r})
		i = j
	}
	e.instructions = out
}

// iface describes the memory footprint of interface{}
type iface struct {
	Type, Data unsafe.Pointer
}

// StructEncoder stores a set of instructions for converting a struct to a json document. It's
// useless to create an instance of this outside of `NewStructEncoder`.
type StructEncoder struct {
	instructions []instruction       // the instructionset to be executed during Marshal
	f            reflect.StructField // current field
	t            interface{}         // type
	i            int                 // iter
	cb           Buffer              // side buffer for static data
	cpos         int                 // side buffer position
}

// Marshal executes the instructions for a given type and writes the resulting
// json document to the io.Writer provided
func (e *StructEncoder) Marshal(s interface{}, w *Buffer) {

	p := (*(*iface)(unsafe.Pointer(&s))).Data

	for i := 0; i < len(e.instructions); i++ {

		if e.instructions[i].kind == kindRun { // statics, strings and ints with one capacity check
			e.instructions[i].run.write(p, w)
			continue
		} else if e.instructions[i].kind == kindStatic { // static data fast path
			w.Write(e.instructions[i].static)
			continue
		} else if e.instructions[i].kind == kindStringField { // string fields fast path, allows inlining of whole write
			ptrStringToBuf(unsafe.Pointer(uintptr(p)+e.instructions[i].offset), w)
			continue
		} else if e.instructions[i].kind == kindInt { // int fields fast path, allows inlining of whole write
			ptrIntToBuf(unsafe.Pointer(uintptr(p)+e.instructions[i].offset), w)
			continue
		} else if e.instructions[i].leapFun != nil { // simple 'conv' function fast path
			e.instructions[i].leapFun(unsafe.Pointer(uintptr(p)+e.instructions[i].offset), w)
			continue
		}

		e.instructions[i].fun(p, w) // all other instruction types
	}
}

// NewStructEncoder compiles a set of instructions for marhsaling a struct shape to a JSON document.
func NewStructEncoder(t interface{}) *StructEncoder {
	e := &StructEncoder{}
	e.t = t
	tt := reflect.TypeOf(t)

	e.chunk("{")

	emit := 0 // track number of fields we emit
	// pass over each field in the struct to build up our instruction set for each
	for e.i = 0; e.i < tt.NumField(); e.i++ {
		e.f = tt.Field(e.i)

		tag, opts := parseTag(e.f.Tag.Get("json")) // we're using tags to nominate inclusion
		if tag == "" {
			continue
		}
		emit++

		// write the key
		if emit > 1 {
			e.chunk(",")
		}
		e.chunk(`"` + tag + `":`)

		switch {
		/// support calling .String() when the 'stringer' option is passed
		case opts.Contains("stringer") && reflect.ValueOf(e.t).Field(e.i).MethodByName("String").Kind() != reflect.Invalid:
			e.optInstrStringer()

		/// support calling .JSONEncode(*Buffer) when the 'encoder' option is passed
		case opts.Contains("encoder"):

			// requrie explicit opt-in for JSONMarshaler implementation
			t := reflect.ValueOf(e.t).Field(e.i).Type()
			if t.Kind() != reflect.Ptr {
				t = reflect.PtrTo(t)
			}

			if _, ok := t.MethodByName("EncodeJSON"); ok {
				e.optInstrEncoderWriter()
				break
			}

			// default to JSONEncoder implementation for any other encoder fields
			e.optInstrEncoder()

		/// support writing byteslice-like items using 'raw' option.
		case opts.Contains("raw"):
			e.optInstrRaw()

		/// suport escaping reserved json characters from byteslice-like items and slices
		case opts.Contains("escape"):
			e.optInstrEscape()

		/// time is a type of struct, not a kind, so somewhat of a special case here.
		case e.f.Type == timeType:
			e.chunk(`"`)
			e.val(ptrTimeToBuf)
			e.chunk(`"`)
		case e.f.Type.Kind() == reflect.Ptr && timeType == reflect.TypeOf(e.t).Field(e.i).Type.Elem():
			e.ptrstringval(ptrTimeToBuf)

		// write the value instruction depending on type
		case e.f.Type.Kind() == reflect.Ptr:
			// create an instruction which can read from a pointer field
			e.valueInst(e.f.Type.Elem().Kind(), e.ptrval)

		default:
			// create an instruction which reads from a standard field
			e.valueInst(e.f.Type.Kind(), e.val)
		}
	}

	e.chunk("}")
	e.flunk()
	e.fuseRuns()

	return e
}

func (e *StructEncoder) appendInstructionFun(fun func(unsafe.Pointer, *Buffer)) {
	e.instructions = append(e.instructions, instruction{fun: fun})
}

func (e *StructEncoder) optInstrStringer() {
	e.chunk(`"`)

	t := reflect.ValueOf(e.t).Field(e.i).Type()
	if e.f.Type.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	conv := func(v unsafe.Pointer, w *Buffer) {
		e, ok := reflect.NewAt(t, v).Interface().(fmt.Stringer)
		if !ok {
			return
		}
		w.WriteString(e.String())
	}

	if e.f.Type.Kind() == reflect.Ptr {
		e.ptrval(conv)
	} else {
		e.val(conv)
	}

	e.chunk(`"`)
}

func (e *StructEncoder) optInstrEncoder() {
	t := reflect.ValueOf(e.t).Field(e.i).Type()
	if e.f.Type.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	conv := func(v unsafe.Pointer, w *Buffer) {
		e, ok := reflect.NewAt(t, v).Interface().(JSONEncoder)
		if !ok {
			w.Write(null)
			return
		}
		e.JSONEncode(w)
	}

	if e.f.Type.Kind() == reflect.Ptr {
		e.ptrval(conv)
	} else {
		e.val(conv)
	}
}

func (e *StructEncoder) optInstrEncoderWriter() {
	t := reflect.ValueOf(e.t).Field(e.i).Type()
	if e.f.Type.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	conv := func(v unsafe.Pointer, w *Buffer) {
		e, ok := reflect.NewAt(t, v).Interface().(JSONMarshaler)
		if !ok {
			w.Write(null)
			return
		}
		e.EncodeJSON(w)
	}

	if e.f.Type.Kind() == reflect.Ptr {
		e.ptrval(conv)
	} else {
		e.val(conv)
	}
}

func (e *StructEncoder) optInstrRaw() {
	conv := func(v unsafe.Pointer, w *Buffer) {
		s := *(*string)(v)
		if len(s) == 0 {
			w.Write(null)
			return
		}
		w.WriteString(s)
	}

	if e.f.Type.Kind() == reflect.Ptr {
		e.ptrval(conv)
	} else {
		e.val(conv)
	}
}

func (e *StructEncoder) optInstrEscape() {
	if e.f.Type.Kind() == reflect.Slice {
		e.flunk()

		/// create an escape string encoder internally instead of mirroring the struct, so people only need to pass the ,escape opt instead
		enc := NewSliceEncoder([]EscapeString{})
		f := e.f
		e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {
			var em interface{} = unsafe.Pointer(uintptr(v) + f.Offset)
			enc.Marshal(em, w)
		})
		return
	}

	if e.f.Type.Kind() == reflect.Ptr {
		e.ptrstringval(ptrEscapeStringToBuf)
	} else {
		e.chunk(`"`)
		e.val(ptrEscapeStringToBuf)
		e.chunk(`"`)
	}
}

// chunk writes a chunk of body data to the chunk buffer. only for writing static
//
//	structure and not dynamic values.
func (e *StructEncoder) chunk(b string) {
	e.cb.Write([]byte(b))
}

// flunk flushes whatever chunk data we've got buffered into a single instruction
func (e *StructEncoder) flunk() {

	b := e.cb.Bytes
	bs := b[e.cpos:]
	e.cpos = len(b)

	if len(bs) == 0 {
		return
	}

	e.instructions = append(e.instructions, instruction{static: bs, kind: kindStatic})
}

// valueInst works out the conversion function we need for `k` and creates an instruction to write it to the buffer
func (e *StructEncoder) valueInst(k reflect.Kind, instr func(func(unsafe.Pointer, *Buffer))) {

	switch k {

	case reflect.Int:

		/// fast path for int fields
		if e.f.Type.Kind() == reflect.Ptr {
			instr(ptrIntToBuf)
			return
		}
		e.flunk()
		e.instructions = append(e.instructions, instruction{offset: e.f.Offset, kind: kindInt})

	case reflect.Bool,
		reflect.Int8,
		reflect.Int16,
		reflect.Int32,
		reflect.Int64,
		reflect.Uint,
		reflect.Uint8,
		reflect.Uint16,
		reflect.Uint32,
		reflect.Uint64,
		reflect.Float32,
		reflect.Float64:
		/// standard print
		conv, ok := typeconv[k]
		if !ok {
			return
		}
		instr(conv)

	case reflect.Array:
		/// support for primitives in arrays (proabbly need arrayencoder.go here if we want to take this further)
		e.chunk("[")

		conv, ok := typeconv[e.f.Type.Elem().Kind()]
		if !ok {
			return
		}

		offset := e.f.Type.Elem().Size()
		for i := 0; i < e.f.Type.Len(); i++ {
			if i > 0 {
				e.chunk(", ")
			}

			e.flunk()
			f := e.f
			i := i
			e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {
				conv(unsafe.Pointer(uintptr(v)+f.Offset+(uintptr(i)*offset)), w)
			})
		}

		e.chunk("]")

	case reflect.Slice:

		e.flunk()

		enc := NewSliceEncoder(reflect.ValueOf(e.t).Field(e.i).Interface())
		f := e.f
		e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {
			var em interface{} = unsafe.Pointer(uintptr(v) + f.Offset)
			enc.Marshal(em, w)
		})

	case reflect.String:

		/// for strings to be nullable they need a special instruction to write quotes conditionally.
		if e.f.Type.Kind() == reflect.Ptr {
			e.ptrstringval(ptrStringToBuf)
			return
		}

		// otherwise a standard quoted print instruction
		e.chunk(`"`)

		/// fast path for strings
		e.flunk() // flush any chunk data we've buffered
		e.instructions = append(e.instructions, instruction{offset: e.f.Offset, kind: kindStringField})
		e.chunk(`"`)

	case reflect.Struct:
		// create an instruction for the field name (as per val)
		e.flunk()

		if e.f.Type.Kind() == reflect.Ptr {

			/// now cater for it being a pointer to a struct
			var inf = reflect.New(reflect.TypeOf(e.t).Field(e.i).Type.Elem()).Elem().Interface()

			var enc *StructEncoder
			if e.t == inf {
				// handle recursive structs by re-using the current encoder
				enc = e
			} else {
				enc = NewStructEncoder(inf)
			}

			// now create an instruction to marshal the field
			f := e.f
			e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {
				var em interface{} = unsafe.Pointer(*(*unsafe.Pointer)(unsafe.Pointer(uintptr(v) + f.Offset)))
				if em == unsafe.Pointer(nil) {
					w.Write(null)
					return
				}
				enc.Marshal(em, w)
			})
			return
		}

		// build a new StructEncoder for the type
		enc := NewStructEncoder(reflect.ValueOf(e.t).Field(e.i).Interface())
		// now create another instruction which calls marshal on the struct, passing our writer
		f := e.f
		e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {
			var em interface{} = unsafe.Pointer(uintptr(v) + f.Offset)
			enc.Marshal(em, w)
		})
		return

	case reflect.Invalid,
		reflect.Map,
		reflect.Interface,
		reflect.Complex64,
		reflect.Complex128,
		reflect.Chan,
		reflect.Func,
		reflect.Uintptr,
		reflect.UnsafePointer:
		// no
		panic(fmt.Sprint("unsupported type ", e.f.Type.Kind(), e.f.Name))
	}
}

// val creates an instruction to read from a field we're marshaling
func (e *StructEncoder) val(conv func(unsafe.Pointer, *Buffer)) {

	e.flunk() // flush any chunk data we've buffered
	e.instructions = append(e.instructions, instruction{leapFun: conv, offset: e.f.Offset})
}

// ptrval creates an instruction to read from a pointer field we're marshaling
func (e *StructEncoder) ptrval(conv func(unsafe.Pointer, *Buffer)) {

	e.flunk() // flush any chunk data we've buffered

	// avoids allocs at runtime
	null := []byte("null")

	f := e.f
	e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {

		p := unsafe.Pointer(*(*unsafe.Pointer)(unsafe.Pointer(uintptr(v) + f.Offset)))
		if p == unsafe.Pointer(nil) {
			w.Write(null)
			return
		}
		conv(p, w)
	})
}

// ptrstringval is essentially the same as ptrval but quotes strings if not nil
func (e *StructEncoder) ptrstringval(conv func(unsafe.Pointer, *Buffer)) {
	e.flunk() // flush any chunk data we've buffered

	// avoids allocs at runtime
	null := []byte("null")

	f := e.f
	e.appendInstructionFun(func(v unsafe.Pointer, w *Buffer) {

		p := unsafe.Pointer(*(*unsafe.Pointer)(unsafe.Pointer(uintptr(v) + f.Offset)))
		if p == unsafe.Pointer(nil) {
			w.Write(null)
			return
		}

		// quotes need to be at runtime here because we don't know if we're going to have to null the field
		w.WriteByte('"')
		conv(p, w)
		w.WriteByte('"')
	})
}

// JSONEncoder works with the `.encoder` option. Fields can implement this to encode their own JSON string straight
// into the working buffer. This can be useful if you're working with interface fields at runtime.
type JSONEncoder interface {
	JSONEncode(*Buffer)
}

// JSONMarshaler works with the `.encoder` option. Fields can implement this to encode their own JSON string straight
// into the provided `io.Writer`. This is useful if you require the functionality of `JSONEncoder` but don't want the hard
// dependency on `Buffer`.
type JSONMarshaler interface {
	EncodeJSON(io.Writer)
}

// tagOptions is the string following a comma in a struct field's "json"
// tag, or the empty string. It does not include the leading comma.
//
// this is jacked from the stdlib to remain compatible with that syntax.
type tagOptions string

// parseTag splits a struct field's json tag into its name and
// comma-separated options.
func parseTag(tag string) (string, tagOptions) {
	if idx := strings.Index(tag, ","); idx != -1 {
		return tag[:idx], tagOptions(tag[idx+1:])
	}
	return tag, tagOptions("")
}

// Contains reports whether a comma-separated list of options
// contains a particular substr flag. substr must be surrounded by a
// string boundary or commas.
func (o tagOptions) Contains(optionName string) bool {
	if len(o) == 0 {
		return false
	}
	s := string(o)
	for s != "" {
		var next string
		i := strings.Index(s, ",")
		if i >= 0 {
			s, next = s[:i], s[i+1:]
		}
		if s == optionName {
			return true
		}
		s = next
	}
	return false
}

var timeType = reflect.TypeOf(time.Time{})

// EscapeString can be used to cast your string slice encoders in replacement of `[]string` when using SliceEncoder directly.
// This is only necessary if you wish for the slice elements to be escaped of control sequences.
// e.g var mySliceEncoder = NewSliceEncoder([]jingo.EscapeString{})
// You can and should just use the `,escape` option on your struct fields when using StructEncoder.
type EscapeString string

var escapeStringType = reflect.TypeOf(EscapeString(""))
