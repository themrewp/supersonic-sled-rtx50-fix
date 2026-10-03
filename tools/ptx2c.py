"""Translate the sm_10 PTX embedded in SupersonicSled.exe into plain C.

Each PTX .entry becomes a C function that runs ONE GPU thread:
    void K_<n>(const uint8_t *P, const ThreadIdx *t)
P points at the kernel's argument block (as packed by cudaSetupArgument).
Registers are stored as raw bits (uint32_t / uint64_t / bool) and each
instruction reinterprets them according to its PTX type, exactly like
the hardware does.

usage: python ptx2c.py out.c in1.ptx in2.ptx ...
"""
import re
import sys

FLOAT_T = {"f32"}
DBL_T = {"f64"}
SIGNED_T = {"s8": "int8_t", "s16": "int16_t", "s32": "int32_t", "s64": "int64_t"}
UNSIGNED_T = {"u8": "uint8_t", "u16": "uint16_t", "u32": "uint32_t", "u64": "uint64_t",
              "b8": "uint8_t", "b16": "uint16_t", "b32": "uint32_t", "b64": "uint64_t"}
SIZE = {"s8": 1, "u8": 1, "b8": 1, "s16": 2, "u16": 2, "b16": 2,
        "s32": 4, "u32": 4, "b32": 4, "f32": 4, "s64": 8, "u64": 8, "b64": 8, "f64": 8}

SPECIAL = {
    "%tid.x": "t->tid_x", "%tid.y": "t->tid_y", "%tid.z": "0u",
    "%ntid.x": "t->ntid_x", "%ntid.y": "t->ntid_y", "%ntid.z": "1u",
    "%ctaid.x": "t->ctaid_x", "%ctaid.y": "t->ctaid_y", "%ctaid.z": "0u",
    "%nctaid.x": "t->nctaid_x", "%nctaid.y": "t->nctaid_y", "%nctaid.z": "1u",
}


def ctype(t):
    if t in FLOAT_T:
        return "float"
    if t in DBL_T:
        return "double"
    return SIGNED_T.get(t) or UNSIGNED_T[t]


def regname(r):
    return r[1:]  # %r12 -> r12, %f3 -> f3, %p1 -> p1


class Kernel:
    def __init__(self, name):
        self.name = name
        self.params = []  # (name, offset, size)
        self.param_off = {}
        self.regs = []    # (ctype, prefix, count)
        self.body = []
        self.size = 0


def split_ops(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Translator:
    def __init__(self):
        self.kernels = []
        self.symbols = set()   # .const/.global variables
        self.textures = set()

    # ---- operands -------------------------------------------------------
    def val(self, op, t, k):
        """C expression for operand `op` read as PTX type `t`."""
        op = op.strip()
        neg = False
        if op.startswith("-") and op[1:2] == "%":
            neg, op = True, op[1:]
        if op in SPECIAL:
            e = SPECIAL[op]
            e = f"(({ctype(t)}){e})" if t not in FLOAT_T | DBL_T else f"(({ctype(t)}){e})"
        elif op.startswith("%"):
            r = regname(op)
            if r.startswith("fd"):
                e = f"D({r})" if t in DBL_T else f"(({ctype(t)}){r})"
            elif r.startswith("p"):
                e = f"(({ctype(t)}){r})"
            elif t in FLOAT_T:
                e = f"F({r})"
            elif t in DBL_T:
                e = f"((double)F({r}))"
            else:
                e = f"(({ctype(t)}){r})"
        elif re.fullmatch(r"0[fF][0-9a-fA-F]{8}", op):
            e = f"F(0x{op[2:]}u)"
            if t not in FLOAT_T:
                e = f"(({ctype(t)})0x{op[2:]}u)" if t not in DBL_T else f"((double){e})"
        elif re.fullmatch(r"0[dD][0-9a-fA-F]{16}", op):
            e = f"D(0x{op[2:]}ull)"
            if t in FLOAT_T:
                e = f"((float){e})"
        elif re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)U?", op):
            lit = op.rstrip("U")
            e = f"(({ctype(t)}){lit}{'ll' if not lit.startswith('-') or True else ''})"
            if t in FLOAT_T:
                e = f"((float){lit})"
        elif re.fullmatch(r"[A-Za-z_]\w*", op):
            # address of a global symbol
            self.symbols.add(op)
            e = f"(({ctype(t)})(uintptr_t)SYM_{op})"
        else:
            raise ValueError(f"operand {op!r}")
        return f"(-{e})" if neg else e

    def store(self, dst, t, expr):
        r = regname(dst.strip())
        if r.startswith("p"):
            return f"{r} = ({expr}) != 0;"
        if r.startswith("fd"):
            return f"{r} = UD((double)({expr}));"
        if t in FLOAT_T:
            return f"{r} = UF({expr});"
        if t in DBL_T:
            return f"{r} = UF((float)({expr}));"
        if r.startswith("rh"):
            return f"{r} = (uint16_t)({expr});"
        return f"{r} = (uint32_t)({expr});"

    def addr(self, op, k, space):
        """[x+off] -> C pointer expression (uint8_t*)"""
        m = re.fullmatch(r"\[\s*([^\]+]+?)\s*(?:\+\s*(-?\d+))?\s*\]", op.strip())
        if not m:
            raise ValueError(f"address {op!r}")
        base, off = m.group(1), int(m.group(2) or 0)
        if space == "param":
            return f"(P + {k.param_off[base] + off})"
        if base.startswith("%"):
            return f"((uint8_t*)(uintptr_t)({regname(base)} + {off}u))"
        self.symbols.add(base)
        return f"((uint8_t*)SYM_{base} + {off})"

    # ---- instructions ---------------------------------------------------
    def emit(self, k, line):
        pred = None
        m = re.match(r"@(!?)%(p\d+)\s+(.*)", line)
        if m:
            pred = ("!" if m.group(1) else "") + m.group(2)
            line = m.group(3)
        line = line.rstrip(";").strip()
        parts = line.split(None, 1)
        opc = parts[0]
        ops = split_ops(parts[1]) if len(parts) > 1 else []
        code = self.instr(k, opc, ops)
        if pred:
            code = f"if ({pred}) {{ {code} }}"
        k.body.append("    " + code)

    def instr(self, k, opc, ops):
        f = opc.split(".")
        op = f[0]
        if op == "bra":
            return f"goto {ops[0].replace('$', 'L_')};"
        if op in ("exit", "ret"):
            return "return;"
        if op == "mov":
            return self.store(ops[0], f[-1], self.val(ops[1], f[-1], k))
        if op == "cvt":
            # cvt[.rnd][.sat].dtype.stype
            dt, st = f[-2], f[-1]
            mods = f[1:-2]
            src = self.val(ops[1], st, k)
            rnd = next((x for x in mods if x in ("rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi")), None)
            if dt in FLOAT_T | DBL_T:
                if st in FLOAT_T | DBL_T and rnd in ("rni", "rzi", "rmi", "rpi"):
                    fn = {"rni": "rintf", "rzi": "truncf", "rmi": "floorf", "rpi": "ceilf"}[rnd]
                    e = f"{fn}((float){src})"
                else:
                    e = f"(({ctype(dt)}){src})"
                if "sat" in mods:
                    e = f"sat01({e})"
            elif st in FLOAT_T | DBL_T:
                fn = {"rni": "rintf", "rzi": "truncf", "rmi": "floorf", "rpi": "ceilf", None: "truncf"}.get(rnd, "truncf")
                e = f"cvt_f2{'s' if dt.startswith('s') else 'u'}{SIZE[dt]*8}({fn}((float){src}))"
            else:
                e = f"(({ctype(dt)}){src})"
            return self.store(ops[0], dt, e)
        if op in ("ld", "st"):
            space = f[1]
            vec = f[2] if f[2].startswith("v") else None
            t = f[-1]
            if op == "ld":
                a = self.addr(ops[1], k, space)
                dsts = split_ops(ops[0][1:-1]) if vec else [ops[0]]
                return " ".join(
                    self.store(d, t, f"*({ctype(t)}*)({a} + {i * SIZE[t]})") for i, d in enumerate(dsts) if d != "_")
            a = self.addr(ops[0], k, space)
            srcs = split_ops(ops[1][1:-1]) if vec else [ops[1]]
            return " ".join(f"*({ctype(t)}*)({a} + {i * SIZE[t]}) = {self.val(s, t, k)};"
                            for i, s in enumerate(srcs))
        if op == "tex":
            # tex.2d.v4.f32.f32 {d0,d1,d2,d3},[tex,{x,y,z,w}]
            dsts = split_ops(ops[0][1:-1])
            inner = ops[1].strip()[1:-1]
            tname, coords = inner.split(",", 1)
            tname = tname.strip()
            self.textures.add(tname)
            cs = split_ops(coords.strip()[1:-1])
            ct = f[-1]
            x = self.val(cs[0], ct, k)
            y = self.val(cs[1], ct, k) if len(cs) > 1 else "0.0f"
            dt = f[-2]
            code = f"{{ float o_[4]; tex_fetch2d(TEX_{tname}, (float){x}, (float){y}, o_); "
            code += " ".join(self.store(d, dt, f"o_[{i}]") for i, d in enumerate(dsts) if d != "_") + " }"
            return code
        if op == "setp":
            cmp, t = f[1], f[-1]
            a, b = self.val(ops[1], t, k), self.val(ops[2], t, k)
            return f"{regname(ops[0])} = {self.cmp(cmp, a, b, t)};"
        if op == "set":
            cmp, dt, st = f[1], f[-2], f[-1]
            a, b = self.val(ops[1], st, k), self.val(ops[2], st, k)
            if dt in FLOAT_T:
                return self.store(ops[0], dt, f"({self.cmp(cmp, a, b, st)} ? 1.0f : 0.0f)")
            return self.store(ops[0], dt, f"({self.cmp(cmp, a, b, st)} ? 0xFFFFFFFFu : 0u)")
        if op == "selp":
            t = f[-1]
            return self.store(ops[0], t, f"({regname(ops[3]) if ops[3].startswith('%') else ops[3]} ? "
                                         f"{self.val(ops[1], t, k)} : {self.val(ops[2], t, k)})")
        t = f[-1]
        if op in ("neg", "abs", "sqrt", "rsqrt", "lg2", "ex2", "sin", "cos", "rcp", "not"):
            a = self.val(ops[1], t, k)
            fl = t in FLOAT_T
            e = {
                "neg": f"(-{a})",
                "abs": f"fabsf({a})" if fl else f"(({a}) < 0 ? -({a}) : ({a}))",
                "sqrt": f"sqrtf({a})", "rsqrt": f"(1.0f / sqrtf({a}))",
                "lg2": f"log2f({a})", "ex2": f"exp2f({a})",
                "sin": f"sinf({a})", "cos": f"cosf({a})", "rcp": f"(1.0f / ({a}))",
                "not": f"(~{a})",
            }[op]
            if t in DBL_T:
                e = e.replace("fabsf", "fabs").replace("sqrtf", "sqrt").replace("1.0f", "1.0")
            return self.store(ops[0], t, e)
        if op == "mad":
            a, b, c = (self.val(x, t, k) for x in ops[1:4])
            if "wide" in f:
                wt = {"u16": "uint32_t", "s16": "int32_t", "u32": "uint64_t", "s32": "int64_t"}[t]
                return self.store(ops[0], t, f"(({wt}){a} * ({wt}){b} + {c})")
            if "hi" in f:
                return self.store(ops[0], t, f"((uint32_t)(((uint64_t){a} * (uint64_t){b}) >> 32) + {c})")
            return self.store(ops[0], t, f"({a} * {b} + {c})")
        if op in ("add", "sub", "mul", "div", "rem", "min", "max", "and", "or", "xor", "shl", "shr"):
            a, b = self.val(ops[1], t, k), self.val(ops[2], t, k)
            fl = t in FLOAT_T | DBL_T
            if op == "mul" and "wide" in f:
                wt = {"u16": "uint32_t", "s16": "int32_t", "u32": "uint64_t", "s32": "int64_t"}[t]
                return f"{regname(ops[0])} = (uint32_t)(({wt}){a} * ({wt}){b});"
            if op == "mul" and "hi" in f:
                wt = "int64_t" if t.startswith("s") else "uint64_t"
                return self.store(ops[0], t, f"((({wt}){a} * ({wt}){b}) >> 32)")
            if op in ("shl", "shr"):
                b = self.val(ops[2], "u32", k)
                e = f"(({b}) > {SIZE[t]*8-1} ? {'(' + a + ' < 0 ? -1 : 0)' if (op == 'shr' and t.startswith('s')) else '0'} : ({a} {'<<' if op == 'shl' else '>>'} {b}))"
                return self.store(ops[0], t, e)
            e = {
                "add": f"({a} + {b})", "sub": f"({a} - {b})", "mul": f"({a} * {b})",
                "div": f"({a} / {b})" if fl else f"(({b}) ? ({a} / {b}) : 0)",
                "rem": f"(({b}) ? ({a} % {b}) : 0)",
                "min": f"fminf({a}, {b})" if fl else f"(({a}) < ({b}) ? ({a}) : ({b}))",
                "max": f"fmaxf({a}, {b})" if fl else f"(({a}) > ({b}) ? ({a}) : ({b}))",
                "and": f"({a} & {b})", "or": f"({a} | {b})", "xor": f"({a} ^ {b})",
            }[op]
            if "sat" in f and fl:
                e = f"sat01({e})"
            return self.store(ops[0], t, e)
        raise ValueError(f"unhandled opcode {opc} {ops}")

    @staticmethod
    def cmp(c, a, b, t):
        base = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
                "lo": "<", "ls": "<=", "hi": ">", "hs": ">="}
        if c in base:
            e = f"({a} {base[c]} {b})"
            if c == "ne" and t in FLOAT_T:  # ordered not-equal
                e = f"(({a} == {a}) && ({b} == {b}) && {e})"
            return e
        if c.endswith("u") and c[:-1] in base:  # unordered: true if either NaN
            return f"(({a} != {a}) || ({b} != {b}) || ({a} {base[c[:-1]]} {b}))"
        if c == "num":
            return f"(({a} == {a}) && ({b} == {b}))"
        if c == "nan":
            return f"(({a} != {a}) || ({b} != {b}))"
        raise ValueError(c)

    # ---- file -------------------------------------------------------------
    def parse(self, text):
        k = None
        in_params = False
        for raw in text.splitlines():
            line = raw.split("//")[0].strip()
            if not line:
                continue
            m = re.match(r"\.entry\s+(\w+)", line)
            if m:
                k = Kernel(m.group(1))
                in_params = True
                continue
            if k is None:
                m = re.match(r"\.(const|global)\s.*?(\w+)\s*(\[\d+\])?;", line)
                if m and not line.startswith(".global .texref"):
                    self.symbols.add(m.group(2))
                m = re.match(r"\.tex\s+\.\w+\s+(\w+);", line)
                if m:
                    self.textures.add(m.group(1))
                continue
            if in_params:
                m = re.match(r"\.param\s+(?:\.align\s+(\d+)\s+)?\.(\w+)\s+(\w+)(?:\[(\d+)\])?", line)
                if m:
                    align = int(m.group(1) or SIZE[m.group(2)])
                    size = int(m.group(4) or 1) * SIZE[m.group(2)]
                    k.size = (k.size + align - 1) // align * align
                    k.param_off[m.group(3)] = k.size
                    k.params.append((m.group(3), k.size, size))
                    k.size += size
                if line.startswith("{"):
                    in_params = False
                continue
            if line.startswith("}"):
                self.kernels.append(k)
                k = None
                continue
            m = re.match(r"\.reg\s+\.(\w+)\s+%(\w+)<(\d+)>;", line)
            if m:
                k.regs.append((m.group(1), m.group(2), int(m.group(3))))
                continue
            if line.startswith("."):
                continue
            m = re.match(r"(\$\w+):$", line)
            if m:
                k.body.append(f"{m.group(1).replace('$', 'L_')}:;")
                continue
            try:
                self.emit(k, line)
            except Exception as e:
                raise SystemExit(f"{k.name}: {line}\n  -> {e}")

    def write(self, path):
        o = ['/* generated by tools/ptx2c.py from the PTX inside SupersonicSled.exe - do not edit */',
             '#include "kernels.h"', ""]
        for s in sorted(self.symbols):
            o.append(f"void *SYM_{s};")
        for t in sorted(self.textures):
            o.append(f"TexSlot *TEX_{t};")
        o.append("")
        for i, k in enumerate(self.kernels):
            o.append(f"/* {k.name}  params={k.size} bytes */")
            o.append(f"static void K_{i}(const uint8_t *P, const ThreadIdx *t)\n{{")
            for (t, pre, n) in k.regs:
                c = "bool" if t == "pred" else ("uint64_t" if t == "f64" else "uint32_t")
                o.append(f"    {c} " + ", ".join(f"{pre}{j} = 0" for j in range(n)) + ";")
            o.append("    (void)P; (void)t;")
            o.extend(k.body)
            o.append("}\n")
        o.append("const KernelInfo g_kernels[] = {")
        for i, k in enumerate(self.kernels):
            offs = ", ".join(str(off) for (_, off, _) in k.params)
            o.append(f'    {{"{k.name}", K_{i}, {k.size}, {len(k.params)}, {{{offs}}}}},')
        o.append("    {0, 0, 0, 0, {0}}\n};")
        o.append("const char *const g_symbol_names[] = {" + ", ".join(f'"{s}"' for s in sorted(self.symbols)) + ", 0};")
        o.append("void **const g_symbol_slots[] = {" + ", ".join(f"&SYM_{s}" for s in sorted(self.symbols)) + ", 0};")
        o.append("const char *const g_texture_names[] = {" + ", ".join(f'"{s}"' for s in sorted(self.textures)) + ", 0};")
        o.append("TexSlot **const g_texture_slots[] = {" + ", ".join(f"&TEX_{s}" for s in sorted(self.textures)) + ", 0};")
        open(path, "w").write("\n".join(o) + "\n")


if __name__ == "__main__":
    tr = Translator()
    for p in sys.argv[2:]:
        tr.parse(open(p, encoding="latin1").read())
    tr.write(sys.argv[1])
    print(f"{len(tr.kernels)} kernels, symbols={sorted(tr.symbols)}, textures={sorted(tr.textures)}")
