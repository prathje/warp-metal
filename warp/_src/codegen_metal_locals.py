# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-local arrays for the Metal backend.

``wp.zeros(shape=..., dtype=...)`` inside a kernel or ``@wp.func`` builds a
``wp::fixedarray_t<N, T>``: an ``array_t`` whose storage lives inline on the
stack. MSL has no such type, so after user functions are inlined
:func:`fold_local_arrays` lowers every fixed array to a plain thread-local C
array (``T var_S[N]``) and resolves each array-valued local that refers to it
statically:

* the fixed array itself (``var_S``), and whole-array copies of it
  (``b = a`` copies the storage, exactly like ``fixedarray_t::operator=``);
* row views ``a[i]`` of a multi-dimensional fixed array (``wp::view``);
* arrays built over its storage with pointer arithmetic,
  ``wp.array(ptr=a.ptr + k * sizeof(T), shape=...)`` (Newton's contact
  manifold code carves two buffers out of one allocation this way).

Every such local is a ``(storage, element offset, dims)`` triple known at
compile time; reads, writes, atomics, component stores, ``len``/``shape``
and element pointers become direct accesses to the storage. Element
pointers (``wp::address``) become MSL ``thread T*`` locals.

Reverse mode is supported for 1-D fixed arrays accessed directly: the dual
is a second fixed array of the same shape (``adj_<label>``) that the element
adjoints accumulate into, as natively. Reverse mode through multi-dimensional
fixed arrays or through views of them raises.

Anything else that touches one of these locals (passing it to a native
function, storing its pointer, a loop-carried reassignment to a different
storage) raises :class:`~warp._src.codegen_metal.MetalCodegenError`.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from warp._src.codegen_metal_ast import (
    Comment,
    Empty,
    MetalASTParseError,
    Node,
    Pragma,
    _RawLine,
    _split_top_level_args,
    parse,
)

_FIXEDARRAY_CTOR_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*wp::fixedarray_t\s*<\s*(?P<size>\d+)\s*,\s*(?P<elem>.+)>\s*"
    r"\((?P<args>[^()]*)\)\s*;\s*$"
)
_ARRAY_CTOR_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*wp::array_t\s*<\s*(?P<elem>.+)>\s*\((?P<args>[^()]*)\)\s*;\s*$"
)
_DATA_PTR_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*\(\s*wp::uint64\s*\*\s*\)\s*&\s*\(\s*var_(?P<arr>\w+)\s*\.\s*data\s*\)\s*;\s*$"
)
_SHAPE_PTR_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*&\s*\(\s*var_(?P<arr>\w+)\s*\.\s*shape\s*\)\s*;\s*$"
)
_ASSIGN_CALL_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*wp::(?P<fn>\w+)\s*(?P<tpl><[^()]*>)?\s*\((?P<args>.*)\)\s*;\s*$"
)
_ASSIGN_COPY_PAT = re.compile(r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*var_(?P<rhs>\w+)\s*;\s*$")
_VOID_CALL_PAT = re.compile(r"^(?P<indent>\s*)wp::(?P<fn>\w+)\s*\((?P<args>.*)\)\s*;\s*$")
# Warp >= 1.18 component lvalues of an array element:
#   var_R = &(wp::index(var_A, i..){access});
#   wp::store(&(wp::index(var_A, i..){access}), var_V);
#   wp::array_atomic_<op>_slot(var_A, var_V, [&](auto& _e) -> auto& { return _e{access}; }, i..);
_INDEX_PREFIX = r"wp::index\(\s*var_(?P<arr>\w+)\s*,\s*(?P<idx>[^()]*?)\s*\)"
_SLOT_REF_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*&\(\s*" + _INDEX_PREFIX + r"(?P<acc>[^;]*?)\)\s*;\s*$"
)
_SLOT_STORE_PAT = re.compile(
    r"^(?P<indent>\s*)wp::store\(\s*&\(\s*" + _INDEX_PREFIX + r"(?P<acc>.*?)\)\s*,\s*(?P<val>[^;]+?)\s*\)\s*;\s*$"
)
_SLOT_LAMBDA = r"\[&\]\(auto& _e\) -> auto& \{ return _e(?P<acc>.*?); \}"
_SLOT_ATOMIC_PAT = re.compile(
    r"^(?P<indent>\s*)wp::array_atomic_(?P<op>add|sub|min|max)_slot\(\s*var_(?P<arr>\w+)\s*,\s*(?P<val>[\w\[\]\.]+)\s*,\s*"
    + _SLOT_LAMBDA
    + r"\s*,\s*(?P<idx>[^;]*?)\)\s*;\s*$"
)
_SLOT_ADJ_PAT = re.compile(
    r"^(?P<indent>\s*)wp::adj_array_(?P<op>store|atomic_add|atomic_sub)_slot\(\s*var_(?P<arr>\w+)\s*,\s*(?P<adjarr>[\w\.]+)\s*,"
    r"\s*(?P<adjval>[\w\[\]\.]+)\s*,\s*" + _SLOT_LAMBDA + r"\s*,\s*(?P<idx>[^;]*?)\)\s*;\s*$"
)
_ACCESS_TOKEN_PAT = re.compile(r"\.(?P<fn>component_ref|element_ref|row_ref)\((?P<args>[^()]*)\)|\.(?P<tf>p|q)\b")
# Declarations the inliner emits for locals of an inlined body
# (``device float2* var_7__3;``) — dropped for the locals lowered here.
_DECL_PAT = re.compile(r"^\s*(?:const\s+)?[A-Za-z_][\w:<>, ]*?[\s*]+(?P<pfx>var|adj)_(?P<label>\w+)\s*(?:=[^;]*)?;\s*$")
_VAR_TOKEN_PAT = re.compile(r"\b(?:var|adj)_(\w+)\b")
_INT_LIT_PAT = re.compile(r"^-?\d+$")

_LOCAL_ARRAY_CTYPE_PREFIXES = ("wp::fixedarray_t<", "wp::array_t<")


@dataclass(frozen=True)
class _LocalArray:
    """An array-valued local resolved to thread-local storage."""

    storage: str  # label of the storage local (``var_<storage>``)
    elem_ctype: str
    elem_msl: str
    offset: str  # element offset into the storage (MSL expression)
    dims: tuple[str, ...]  # MSL expressions (ints where known)

    def strides(self) -> list[str]:
        out: list[str] = []
        for k in range(len(self.dims)):
            rest = self.dims[k + 1 :]
            out.append(_mul_exprs(list(rest)) if rest else "1")
        return out


@dataclass(frozen=True)
class _PtrValue:
    """A ``uint64`` holding ``storage + byte_offset`` (``arr.ptr`` arithmetic)."""

    storage: str
    elem_msl: str
    elem_size: int
    byte_offset: str


def _mul_exprs(parts: list[str]) -> str:
    ints = [int(p) for p in parts if _INT_LIT_PAT.match(p)]
    rest = [f"({p})" for p in parts if not _INT_LIT_PAT.match(p)]
    k = 1
    for v in ints:
        k *= v
    if not rest:
        return str(k)
    if k != 1:
        rest.insert(0, str(k))
    return " * ".join(rest)


def _add_exprs(a: str, b: str) -> str:
    if _INT_LIT_PAT.match(a) and _INT_LIT_PAT.match(b):
        return str(int(a) + int(b))
    if a == "0":
        return b
    if b == "0":
        return a
    return f"{a} + {b}"


def _iter_leaves(nodes):
    for n in nodes:
        body = getattr(n, "body", None)
        if isinstance(body, tuple):
            yield from _iter_leaves(body)
        else:
            yield n


def _map_leaves(nodes, fn) -> list[Node]:
    out: list[Node] = []
    for n in nodes:
        body = getattr(n, "body", None)
        if isinstance(body, tuple):
            out.append(dataclasses.replace(n, body=tuple(_map_leaves(body, fn))))
        else:
            out.extend(fn(n))
    return out


def _local_array_ctype(ctype: str | None) -> bool:
    return bool(ctype) and ctype.startswith(_LOCAL_ARRAY_CTYPE_PREFIXES)


def fold_local_arrays(
    nodes: list[Node],
    adj,
    var_ctypes: dict[str, str],
    const_ints: dict[str, int],
    adjoint: bool = False,
) -> tuple[list[Node], set[str]]:
    """Lower thread-local fixed-size arrays; see the module docstring.

    ``var_ctypes`` maps every local label (kernel and inlined) to its ctype,
    ``const_ints`` every constant int local to its value. Returns the
    rewritten nodes and the labels whose declarations the caller must skip
    (the replacement declarations are prepended to the nodes).
    """
    from warp._src.codegen_metal import (  # noqa: PLC0415
        MetalCodegenError,
        _msl_var_type,
        _msl_zero_init,
    )
    from warp._src.codegen_metal_ast import _nonneg_locals  # noqa: PLC0415

    fixed_labels = {lbl for lbl, ct in var_ctypes.items() if ct and ct.startswith("wp::fixedarray_t<")}
    if not fixed_labels:
        return nodes, set()

    kname = getattr(adj, "fun_name", "?")

    def fail(msg: str, line: str | None = None) -> MetalCodegenError:
        tail = f" (line: {line.strip()})" if line else ""
        return MetalCodegenError(f"Kernel {kname!r}: {msg}{tail}")

    nonneg = _nonneg_locals(nodes, const_ints)

    # Integer constants folded through casts and arithmetic (``wp.uint64(5 * 8)``
    # byte offsets), so constant ``ptr`` offsets can be checked statically.
    folded: dict[str, int] = {}
    for _v in getattr(adj, "variables", ()):
        _c = getattr(_v, "constant", None)
        if _c is not None and not isinstance(_c, (bool, float)) and str(_v.label) not in const_ints:
            try:
                folded[str(_v.label)] = int(getattr(_c, "value", _c))
            except (TypeError, ValueError):
                pass

    def const_of(expr: str) -> int | None:
        e = expr.strip()
        mm = re.fullmatch(r"int\((.*)\)", e)
        if mm:
            e = mm.group(1).strip()
        if _INT_LIT_PAT.match(e):
            return int(e)
        if e.startswith("var_"):
            if e[4:] in folded:
                return folded[e[4:]]
            v = const_ints.get(e[4:])
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    # Only single-definition locals fold: a loop-carried ``j = int(-1)`` is
    # later reassigned through ``wp::assign``.
    _ndefs: dict[str, int] = {}
    for _n in _iter_leaves(nodes):
        _r = getattr(_n, "raw", "")
        for _mm in re.finditer(r"^\s*var_(\w+)\s*=|\bwp::(?:assign|\w+_inplace)\s*\(\s*var_(\w+)", _r):
            _lbl = _mm.group(1) or _mm.group(2)
            _ndefs[_lbl] = _ndefs.get(_lbl, 0) + (1 if _mm.group(1) else 2)

    def fold_const(raw: str) -> None:
        _m0 = re.match(r"^\s*var_(\w+)\s*=", raw)
        if _m0 is None or _ndefs.get(_m0.group(1), 0) != 1:
            return
        lit = re.match(r"^\s*var_(\w+)\s*=\s*(-?\d+)(?:[uU]?[lL]{0,2}|[lL]{0,2}[uU]?)\s*;\s*$", raw)
        if lit:
            folded[lit.group(1)] = int(lit.group(2))
            return
        m = _ASSIGN_CALL_PAT.match(raw)
        if m is None:
            return
        fn = m.group("fn")
        if fn not in ("uint64", "int64", "uint32", "int32", "int", "mul", "add", "sub"):
            return
        vals = [const_of(a) for a in _split_top_level_args(m.group("args"))]
        if not vals or any(v is None for v in vals):
            return
        if fn in ("mul", "add", "sub"):
            if len(vals) == 2:
                folded[m.group("lhs")] = {"mul": vals[0] * vals[1], "add": vals[0] + vals[1], "sub": vals[0] - vals[1]}[
                    fn
                ]
        elif len(vals) == 1:
            folded[m.group("lhs")] = vals[0]

    def dim_expr(expr: str) -> str:
        c = const_of(expr)
        return str(c) if c is not None else expr.strip()

    elem_size_cache: dict[str, int] = {}

    def elem_info(elem_ctype: str, line: str) -> tuple[str, int]:
        elem_ctype = elem_ctype.strip()
        if not elem_ctype.startswith("wp::"):
            raise fail(
                f"thread-local arrays of {elem_ctype!r} are not supported on Metal "
                "(only scalar, vector, quaternion, transform and matrix elements)",
                line,
            )
        try:
            msl = _msl_var_type(elem_ctype)
        except Exception as e:
            raise fail(f"unsupported thread-local array element type {elem_ctype!r}: {e}", line) from None
        if elem_ctype not in elem_size_cache:
            elem_size_cache[elem_ctype] = _ctype_size(elem_ctype)
        return msl, elem_size_cache[elem_ctype]

    arrays: dict[str, _LocalArray] = {}
    storages: dict[str, tuple[str, int]] = {}  # storage label -> (elem msl, size)
    data_ptrs: dict[str, str] = {}  # label of ``(uint64*)&(arr.data)`` -> array label
    ptr_values: dict[str, _PtrValue] = {}
    shape_ptrs: dict[str, str] = {}  # ``&(arr.shape)`` -> array label
    shape_values: dict[str, str] = {}  # loaded shape -> array label
    elem_ptrs: dict[str, str] = {}  # element pointer label -> MSL element type
    adj_storages: set[str] = set()

    def is_local(tok: str) -> bool:
        return tok.startswith("var_") and tok[4:] in arrays

    def flat_index(arr: _LocalArray, idx: list[str], line: str) -> str:
        if len(idx) > len(arr.dims):
            raise fail(f"too many indices for a {len(arr.dims)}-D thread-local array", line)
        strides = arr.strides()
        terms: list[str] = []
        const_off = 0
        for k, raw_i in enumerate(idx):
            i = raw_i.strip()
            c = const_of(i)
            dim = arr.dims[k]
            if c is not None:
                if c < 0:
                    dc = const_of(dim)
                    if dc is None:
                        terms.append(f"(({c}) + {dim}) * {strides[k]}" if strides[k] != "1" else f"(({c}) + {dim})")
                        continue
                    c += dc
                sk = const_of(strides[k])
                if sk is not None:
                    const_off += c * sk
                    continue
                terms.append(f"{c} * ({strides[k]})")
                continue
            # ``wp_wrap_index_local`` launders the index through a volatile
            # (Apple compiler bug; see its definition in codegen_metal).
            iv = i if (i.startswith("var_") and i[4:] in nonneg) else f"wp_wrap_index_local({i}, {dim})"
            terms.append(iv if strides[k] == "1" else f"{iv} * {strides[k]}")
        expr = arr.offset
        for t in terms:
            expr = _add_exprs(expr, t)
        return _add_exprs(expr, str(const_off))

    def elem_lvalue(arr: _LocalArray, idx: list[str], line: str) -> str:
        if len(idx) != len(arr.dims):
            raise fail(
                f"element access with {len(idx)} indices on a {len(arr.dims)}-D thread-local array",
                line,
            )
        return f"var_{arr.storage}[{flat_index(arr, idx, line)}]"

    def access_suffix(acc: str, elem_ctype: str, line: str) -> str:
        """Translate an ``{access}`` chain (component/element refs) to MSL."""
        out = ""
        pos = 0
        acc = acc.strip()
        cur = elem_ctype.strip()
        while pos < len(acc):
            m = _ACCESS_TOKEN_PAT.match(acc, pos)
            if m is None:
                raise fail(f"unsupported component access {acc!r} on a thread-local array element", line)
            pos = m.end()
            fn, args, tf = m.group("fn"), m.group("args"), m.group("tf")
            if tf is not None:
                # transform_t lowers to a 7-vector: p = [0:3], q = [3:7].
                raise fail("component stores into .p/.q of a thread-local transform array are not supported", line)
            if fn == "component_ref":
                out += f"[{args.strip()}]"
                cur = "scalar"
            elif fn == "element_ref":
                r, c = (a.strip() for a in args.split(","))
                msl = _msl_var_type(cur)
                if msl.startswith("wp_mat"):
                    cols = _mat_cols(cur)
                    out += f".c[({r}) * {cols} + ({c})]"
                else:
                    # Native MSL matrices are column-major.
                    out += f"[{c}][{r}]"
                cur = "scalar"
            else:
                raise fail("row stores into a thread-local matrix array element are not supported", line)
        return out

    decls: list[Node] = []

    def declare_storage(label: str, elem_msl: str, size: int) -> None:
        storages[label] = (elem_msl, size)
        decls.append(_RawLine(raw=f"    {elem_msl} var_{label}[{size}] = {{}};"))
        if adjoint:
            # The dual is a second fixed array, zeroed once per thread (the
            # native ``adj_<label>`` is default-constructed at kernel entry).
            adj_storages.add(label)
            decls.append(_RawLine(raw=f"    {elem_msl} adj_{label}[{size}] = {{}};"))

    def zero_fill(indent: str, name: str, elem_msl: str, size: int) -> str:
        z = _msl_zero_init(elem_msl)
        if size <= 16:
            return indent + " ".join(f"{name}[{k}] = {z};" for k in range(size))
        return f"{indent}for (int wp_fa_k = 0; wp_fa_k < {size}; ++wp_fa_k) {{ {name}[wp_fa_k] = {z}; }}"

    def copy_into(indent: str, dst: _LocalArray, src: _LocalArray, line: str) -> str:
        n_dst = _mul_exprs(list(dst.dims))
        n_src = _mul_exprs(list(src.dims))
        if dst.elem_msl != src.elem_msl or (n_dst != n_src):
            raise fail("copy between thread-local arrays of different shape or dtype", line)
        return (
            f"{indent}for (int wp_fa_k = 0; wp_fa_k < {n_dst}; ++wp_fa_k) {{ "
            f"var_{dst.storage}[{_add_exprs(dst.offset, 'wp_fa_k')}] = var_{src.storage}[{_add_exprs(src.offset, 'wp_fa_k')}]; }}"
        )

    def declare_ptr(label: str, elem_msl: str) -> None:
        if label in elem_ptrs:
            if elem_ptrs[label] != elem_msl:
                raise fail(f"element pointer var_{label} changes type")
            return
        elem_ptrs[label] = elem_msl
        decls.append(_RawLine(raw=f"    thread {elem_msl}* var_{label};"))

    # Loads through element pointers / shape and length queries, substituted
    # into any statement.
    load_pat = re.compile(r"\bwp::load\s*\(\s*var_(\w+)\s*\)")
    extract_pat = re.compile(r"\bwp::extract\s*\(\s*var_(\w+)\s*,\s*([^(),]+?)\s*\)")
    len_pat = re.compile(r"\bwp::len\s*\(\s*var_(\w+)\s*\)")

    def subst_reads(line: str) -> str:
        def _load(m: re.Match[str]) -> str:
            if m.group(1) in elem_ptrs:
                return f"(*var_{m.group(1)})"
            return m.group(0)

        def _extract(m: re.Match[str]) -> str:
            arr_label = shape_values.get(m.group(1))
            if arr_label is None:
                return m.group(0)
            dims = arrays[arr_label].dims
            k = const_of(m.group(2))
            if k is None:
                # Runtime axis: select among the static extents.
                expr = "0"
                for j in reversed(range(len(dims))):
                    expr = f"(({m.group(2)}) == {j} ? ({dims[j]}) : {expr})"
                return expr
            if not 0 <= k < len(dims):
                return "0"
            return f"({dims[k]})" if not _INT_LIT_PAT.match(dims[k]) else dims[k]

        def _len(m: re.Match[str]) -> str:
            arr = arrays.get(m.group(1))
            if arr is None:
                return m.group(0)
            return arr.dims[0] if _INT_LIT_PAT.match(arr.dims[0]) else f"({arr.dims[0]})"

        line = load_pat.sub(_load, line)
        line = extract_pat.sub(_extract, line)
        return len_pat.sub(_len, line)

    def rewrite(n: Node) -> list[Node]:
        if isinstance(n, (Comment, Empty, Pragma)):
            return [n]
        raw = n.raw
        if "var_" not in raw and "adj_" not in raw:
            return [n]
        # Inliner declarations of locals lowered here.
        if isinstance(n, _RawLine):
            m = _DECL_PAT.match(raw)
            if m and m.group("label") in skip:
                return []
            if m:
                return [n]
        out = rewrite_line(raw)
        if out is None:
            return [n]
        return _to_nodes(out)

    def rewrite_line(raw: str) -> list[str] | None:
        fold_const(raw)
        m = _FIXEDARRAY_CTOR_PAT.match(raw)
        if m:
            lhs = m.group("lhs")
            elem_msl, _sz = elem_info(m.group("elem"), raw)
            size = int(m.group("size"))
            args = [a.strip() for a in _split_top_level_args(m.group("args")) if a.strip()]
            dims = tuple(dim_expr(a) for a in args) or (str(size),)
            if lhs not in storages:
                declare_storage(lhs, elem_msl, size)
            elif storages[lhs] != (elem_msl, size):
                raise fail(f"thread-local array var_{lhs} is re-created with a different type", raw)
            prev = arrays.get(lhs)
            arrays[lhs] = _LocalArray(lhs, m.group("elem").strip(), elem_msl, "0", dims)
            if prev is not None and prev.storage != lhs:
                raise fail(f"thread-local array var_{lhs} is rebound", raw)
            return [zero_fill(m.group("indent"), f"var_{lhs}", elem_msl, size)]

        m = _DATA_PTR_PAT.match(raw)
        if m and m.group("arr") in arrays:
            data_ptrs[m.group("lhs")] = m.group("arr")
            return []

        m = _SHAPE_PTR_PAT.match(raw)
        if m and m.group("arr") in arrays:
            shape_ptrs[m.group("lhs")] = m.group("arr")
            return []

        m = _ARRAY_CTOR_PAT.match(raw)
        if m:
            args = [a.strip() for a in _split_top_level_args(m.group("args")) if a.strip()]
            if args and args[0].startswith("var_") and args[0][4:] in ptr_values:
                pv = ptr_values[args[0][4:]]
                elem_msl, elem_size = elem_info(m.group("elem"), raw)
                if elem_msl != pv.elem_msl or elem_size != pv.elem_size:
                    raise fail(
                        "wp.array(ptr=...) over a thread-local array must keep its element type",
                        raw,
                    )
                dims = tuple(dim_expr(a) for a in args[1:]) or ("0",)
                off = pv.byte_offset
                c = const_of(off)
                if c is not None and c % elem_size:
                    raise fail(
                        f"wp.array(ptr=...) over a thread-local array at byte offset {c}, which is not a "
                        f"multiple of the {elem_size}-byte element size",
                        raw,
                    )
                elem_off = str(c // elem_size) if c is not None else f"int(({off}) / {elem_size})"
                bind(m.group("lhs"), _LocalArray(pv.storage, m.group("elem").strip(), elem_msl, elem_off, dims), raw)
                return []
            return None

        m = _ASSIGN_COPY_PAT.match(raw)
        if m and (m.group("rhs") in arrays or m.group("rhs") in ptr_values):
            return assign_array(m.group("indent"), m.group("lhs"), m.group("rhs"), raw)

        m = _ASSIGN_CALL_PAT.match(raw)
        if m:
            fn = m.group("fn")
            lhs = m.group("lhs")
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            indent = m.group("indent")
            first = args[0] if args else ""
            if fn == "load" and first.startswith("var_"):
                src = first[4:]
                if src in data_ptrs:
                    arr = arrays[data_ptrs[src]]
                    esz = elem_info(arr.elem_ctype, raw)[1]
                    ptr_values[lhs] = _PtrValue(arr.storage, arr.elem_msl, esz, _mul_off(arr.offset, esz))
                    return []
                if src in shape_ptrs:
                    shape_values[lhs] = shape_ptrs[src]
                    return []
            if fn in ("add", "sub") and len(args) == 2:
                a, b = args
                if a.startswith("var_") and a[4:] in ptr_values:
                    pv = ptr_values[a[4:]]
                    cb = const_of(b)
                    if cb is not None:
                        off = str(cb if fn == "add" else -cb)
                    else:
                        off = f"int({b})" if fn == "add" else f"-int({b})"
                    ptr_values[lhs] = dataclasses.replace(pv, byte_offset=_add_exprs(pv.byte_offset, off))
                    return []
                if fn == "add" and b.startswith("var_") and b[4:] in ptr_values:
                    pv = ptr_values[b[4:]]
                    ca = const_of(a)
                    off = str(ca) if ca is not None else f"int({a})"
                    ptr_values[lhs] = dataclasses.replace(pv, byte_offset=_add_exprs(pv.byte_offset, off))
                    return []
            if fn in ("uint64", "int64") and len(args) == 1 and first.startswith("var_") and first[4:] in ptr_values:
                ptr_values[lhs] = ptr_values[first[4:]]
                return []
            if fn == "copy" and len(args) == 1 and (is_local(first) or (first[4:] in ptr_values)):
                return assign_array(indent, lhs, first[4:], raw)
            if fn == "where" and len(args) == 3 and (is_local(args[1]) or is_local(args[2])):
                return select_array(indent, lhs, args, raw)
            if fn == "address" and is_local(first):
                arr = arrays[first[4:]]
                declare_ptr(lhs, arr.elem_msl)
                return [f"{indent}var_{lhs} = &{elem_lvalue(arr, args[1:], raw)};"]
            if fn == "view" and is_local(first):
                arr = arrays[first[4:]]
                idx = args[1:]
                if not idx or len(idx) >= len(arr.dims):
                    raise fail("unsupported view of a thread-local array", raw)
                off = flat_index(arr, [*idx, *(["0"] * (len(arr.dims) - len(idx)))], raw)
                bind(lhs, dataclasses.replace(arr, offset=off, dims=arr.dims[len(idx) :]), raw)
                return []
            if fn in ("atomic_add", "atomic_sub", "atomic_min", "atomic_max") and is_local(first):
                arr = arrays[first[4:]]
                el = elem_lvalue(arr, args[1:-1], raw)
                val = subst_reads(args[-1])
                return [f"{indent}var_{lhs} = {el};", atomic_update(indent, el, fn[len("atomic_") :], val)]
            if fn == "len" or fn == "extract":
                pass  # handled by subst_reads below
        m = _VOID_CALL_PAT.match(raw)
        if m:
            fn = m.group("fn")
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            first = args[0] if args else ""
            indent = m.group("indent")
            if fn == "array_store" and is_local(first):
                arr = arrays[first[4:]]
                return [f"{indent}{elem_lvalue(arr, args[1:-1], raw)} = {subst_reads(args[-1])};"]
            if fn in ("atomic_add", "atomic_sub", "atomic_min", "atomic_max") and is_local(first):
                arr = arrays[first[4:]]
                el = elem_lvalue(arr, args[1:-1], raw)
                return [atomic_update(indent, el, fn[len("atomic_") :], subst_reads(args[-1]))]
            if fn == "store" and first.startswith("var_") and first[4:] in elem_ptrs:
                return [f"{indent}(*{first}) = {subst_reads(args[1])};"]
            if fn == "assign" and len(args) == 2 and is_local(first):
                src = args[1]
                if not is_local(src):
                    raise fail("assigning a non-local array to a thread-local array is not supported", raw)
                dst_arr, src_arr = arrays[first[4:]], arrays[src[4:]]
                if dst_arr == src_arr:
                    return []
                if dst_arr.storage == first[4:] and dst_arr.offset == "0":
                    return [copy_into(indent, dst_arr, src_arr, raw)]
                raise fail("rebinding a thread-local array view inside a loop is not supported", raw)

        m = _SLOT_REF_PAT.match(raw)
        if m and m.group("arr") in arrays:
            arr = arrays[m.group("arr")]
            idx = [a.strip() for a in _split_top_level_args(m.group("idx"))]
            el = elem_lvalue(arr, idx, raw) + access_suffix(m.group("acc"), arr.elem_ctype, raw)
            if not m.group("acc").strip():
                declare_ptr(m.group("lhs"), arr.elem_msl)
                return [f"{m.group('indent')}var_{m.group('lhs')} = &{el};"]
            # Component references are only materialised for the store that
            # follows; they are never read.
            skip.add(m.group("lhs"))
            return []
        m = _SLOT_STORE_PAT.match(raw)
        if m and m.group("arr") in arrays:
            arr = arrays[m.group("arr")]
            idx = [a.strip() for a in _split_top_level_args(m.group("idx"))]
            el = elem_lvalue(arr, idx, raw) + access_suffix(m.group("acc"), arr.elem_ctype, raw)
            return [f"{m.group('indent')}{el} = {subst_reads(m.group('val'))};"]
        m = _SLOT_ATOMIC_PAT.match(raw)
        if m and m.group("arr") in arrays:
            arr = arrays[m.group("arr")]
            idx = [a.strip() for a in _split_top_level_args(m.group("idx"))]
            el = elem_lvalue(arr, idx, raw) + access_suffix(m.group("acc"), arr.elem_ctype, raw)
            return [atomic_update(m.group("indent"), el, m.group("op"), m.group("val"))]

        if adjoint:
            res = rewrite_adjoint(raw)
            if res is not None:
                return res

        new = subst_reads(raw)
        # Any remaining reference to a lowered local is an unsupported use.
        for tok in _VAR_TOKEN_PAT.finditer(new):
            label = tok.group(1)
            if tok.group(0).startswith("adj_"):
                continue
            if label in arrays or label in data_ptrs or label in ptr_values or label in shape_ptrs:
                raise fail(
                    "unsupported use of a thread-local array (wp.zeros(shape, dtype) inside a kernel) on Metal",
                    raw,
                )
            if label in shape_values:
                raise fail("unsupported use of a thread-local array's shape", raw)
        if new == raw:
            return None
        return [new]

    def atomic_update(indent: str, el: str, op: str, val: str) -> str:
        if op == "add":
            return f"{indent}{el} = {el} + {val};"
        if op == "sub":
            return f"{indent}{el} = {el} - {val};"
        return f"{indent}{el} = wp::{op}({el}, {val});"

    def bind(lhs: str, arr: _LocalArray, raw: str) -> None:
        prev = arrays.get(lhs)
        if prev is not None and prev != arr:
            raise fail(f"array local var_{lhs} is bound to different thread-local storage on different paths", raw)
        arrays[lhs] = arr
        skip.add(lhs)

    def select_array(indent: str, lhs: str, args: list[str], raw: str) -> list[str]:
        """``var_L = wp::where(cond, var_A, var_B)`` over two views of one storage.

        Warp merges an array local assigned in an ``if``/``else`` with
        ``wp::where`` (natively a copy of the selected ``array_t`` header).
        Views of the same thread-local storage with the same element type and
        rank differ only in their element offset and extents, so the result is
        that storage with a runtime-selected offset and extents, held in
        ``int`` locals.
        """
        cond, a, b = args
        if not (is_local(a) and is_local(b)):
            raise fail("selecting between a thread-local array and another array is not supported on Metal", raw)
        arr_a, arr_b = arrays[a[4:]], arrays[b[4:]]
        if (arr_a.storage, arr_a.elem_msl, len(arr_a.dims)) != (arr_b.storage, arr_b.elem_msl, len(arr_b.dims)):
            raise fail(
                "selecting between thread-local arrays with different storage, element type or rank is not "
                f"supported on Metal ({arr_a.storage}/{arr_a.elem_msl}/{arr_a.dims} vs "
                f"{arr_b.storage}/{arr_b.elem_msl}/{arr_b.dims})",
                raw,
            )
        c = subst_reads(cond)
        out: list[str] = []

        def pick(name: str, x: str, y: str) -> str:
            if x == y:
                return x
            label = f"{lhs}__{name}"
            if label not in select_locals:
                select_locals.add(label)
                decls.append(_RawLine(raw=f"    int var_{label} = 0;"))
            out.append(f"{indent}var_{label} = ({c}) ? int({x}) : int({y});")
            return f"var_{label}"

        offset = pick("fa_off", arr_a.offset, arr_b.offset)
        dims = tuple(pick(f"fa_dim{k}", x, y) for k, (x, y) in enumerate(zip(arr_a.dims, arr_b.dims, strict=True)))
        bind(lhs, dataclasses.replace(arr_a, offset=offset, dims=dims), raw)
        return out

    select_locals: set[str] = set()

    def assign_array(indent: str, lhs: str, rhs: str, raw: str) -> list[str]:
        if rhs in ptr_values:
            ptr_values[lhs] = ptr_values[rhs]
            skip.add(lhs)
            return []
        src = arrays[rhs]
        ct = var_ctypes.get(lhs, "")
        if ct.startswith("wp::fixedarray_t<"):
            # ``fixedarray_t::operator=`` copies the storage.
            size = int(re.match(r"wp::fixedarray_t<\s*(\d+)", ct).group(1))
            if lhs not in storages:
                declare_storage(lhs, src.elem_msl, size)
            dst = _LocalArray(lhs, src.elem_ctype, src.elem_msl, "0", src.dims)
            prev = arrays.get(lhs)
            if prev is not None and prev.storage != lhs:
                raise fail(f"thread-local array var_{lhs} is rebound", raw)
            arrays[lhs] = dst
            return [copy_into(indent, dst, src, raw)]
        bind(lhs, src, raw)
        return []

    # --- reverse mode ------------------------------------------------------
    adj_addr_pat = re.compile(r"^(?P<indent>\s*)wp::adj_(?P<fn>\w+)\s*\((?P<args>.*)\)\s*;\s*$")

    def adj_lvalue(arr: _LocalArray, idx: list[str], raw: str) -> str:
        return "adj_" + elem_lvalue(arr, idx, raw)[len("var_") :]

    def rewrite_adjoint(raw: str) -> list[str] | None:
        m = adj_addr_pat.match(raw)
        if m is None:
            return None
        fn = m.group("fn")
        args = [a.strip() for a in _split_top_level_args(m.group("args"))]
        first = args[0] if args else ""
        indent = m.group("indent")
        if fn in ("load", "store") and first.startswith("var_") and first[4:] in elem_ptrs:
            # Generic pointer load/store adjoints are no-ops natively; the
            # gradient of an element read flows through ``adj_address``
            # (the pointer's value-typed dual ``adj_<ptr>``).
            return []
        if not is_local(first):
            if fn in ("copy", "assign") and any(is_local(a) for a in args):
                return [f"{indent}/* adjoint of a thread-local array copy: see adj_ storage */"]
            return None
        arr = arrays[first[4:]]
        if fn in ("address", "array_store", "atomic_add", "atomic_sub") and (
            len(arr.dims) != 1 or arr.storage != first[4:] or arr.offset != "0"
        ):
            # Natively the dual of a multi-dimensional or view-accessed fixed
            # array follows ``array_t`` stride/view semantics that this flat
            # dual storage does not reproduce; refuse rather than diverge.
            raise fail(
                "reverse mode through a multi-dimensional thread-local array, or through a view of one, is "
                "not supported on Metal (only 1-D wp.zeros arrays accessed directly)",
                raw,
            )
        if fn == "address":
            # adj_address(arr, i.., adj_arr, adj_i.., adj_ret)
            nidx = (len(args) - 3) // 2
            el = adj_lvalue(arr, args[1 : 1 + nidx], raw)
            return [f"{indent}{el} = {el} + {args[-1]};"]
        if fn == "array_store":
            # adj_array_store(arr, i.., value, adj_arr, adj_i.., adj_value)
            nidx = (len(args) - 4) // 2
            el = adj_lvalue(arr, args[1 : 1 + nidx], raw)
            return [f"{indent}{args[-1]} += {el};"]
        if fn in ("atomic_add", "atomic_sub"):
            # adj_atomic_add(arr, i.., value, adj_arr, adj_i.., adj_value, adj_ret)
            nidx = (len(args) - 5) // 2
            el = adj_lvalue(arr, args[1 : 1 + nidx], raw)
            op = "+=" if fn == "atomic_add" else "-="
            return [f"{indent}{args[-2]} {op} {el};"]
        if fn in ("view", "len", "fixedarray_t", "copy", "assign"):
            return []
        raise fail(f"reverse mode through wp::{fn} on a thread-local array is not supported on Metal", raw)

    skip: set[str] = set(fixed_labels)
    new_nodes = _map_leaves(nodes, rewrite)
    skip |= set(elem_ptrs) | set(data_ptrs) | set(ptr_values) | set(shape_ptrs) | set(shape_values)
    skip |= set(arrays)
    # Drop inliner declarations of lowered locals emitted before their uses
    # were seen (``_map_leaves`` visits a declaration before its body).
    new_nodes = _map_leaves(new_nodes, lambda n: _drop_decl(n, skip))
    return [*decls, *new_nodes], skip


def lower_inlined_array_slots(nodes: list[Node], adj) -> list[Node]:
    """Lower Warp >= 1.18 array-slot IR spliced in by the inliner.

    The kernel's own statements go through ``_lower_array_slot_ir`` before
    the AST pipeline, but a ``@wp.func`` body that stores into a component of
    an element of an array it was passed (``arr[i][j, 0] = x``) only appears
    once that body is inlined, with the parameter already renamed to the
    kernel argument. Run the same lowering over those statements.
    """
    from warp._src.codegen_metal import _lower_array_slot_ir  # noqa: PLC0415

    def is_candidate(n: Node) -> bool:
        return not isinstance(n, (Comment, Empty, Pragma, _RawLine)) and ("wp::index(" in n.raw or "_slot(" in n.raw)

    cands = [n.raw for n in _iter_leaves(nodes) if is_candidate(n)]
    if not cands:
        return nodes
    leaves = [n.raw for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    # One call for all candidates (the lowering scans the whole body per
    # call), separated by a marker line it passes through unchanged.
    sep = "// @@wp_metal_slot_sep"
    batch: list[str] = []
    for ln in cands:
        batch.extend((ln, sep))
    lowered = _lower_array_slot_ir(batch, adj, leaves)
    groups: list[list[str]] = [[]]
    for ln in lowered:
        if ln == sep:
            groups.append([])
        else:
            groups[-1].append(ln)
    it = iter(groups)

    def rewrite(n: Node) -> list[Node]:
        if not is_candidate(n):
            return [n]
        out = next(it)
        if out == [n.raw]:
            return [n]
        return _to_nodes(out)

    return _map_leaves(nodes, rewrite)


# ---------------------------------------------------------------------------
# Range-slice views of kernel array arguments
# ---------------------------------------------------------------------------
# ``sub = arr[a:b]`` / ``arr[i, a:b:s]`` keep a dimension with an offset and a
# stride, which the integer-index view fold (``fold_views``) cannot express.
# Every access through such a view is rewritten to an access of the base
# argument with full base indices (``base_k = start_k + i * step_k``), which
# the rest of the translator lowers like any direct argument access.

_SLICE_BEGIN = 2**31 - 1
_SLICE_END = -(2**31)
_SLICE_CTOR_PAT = re.compile(r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*wp::slice_t\s*\((?P<args>[^()]*)\)\s*;\s*$")
_ADJ_ARRAY_OP_PAT = re.compile(
    r"^(?P<indent>\s*)wp::adj_(?P<fn>address|array_store|atomic_add|atomic_sub|atomic_min|atomic_max)\s*\((?P<args>.*)\)\s*;\s*$"
)


@dataclass(frozen=True)
class _SliceView:
    """A view of kernel array ``base``: per base dim, a fixed index or a slice."""

    base: str
    # (kind, a, b, c): ("idx", index_expr) or ("slice", start_expr, step_expr, len_expr)
    dims: tuple[tuple[str, ...], ...]

    def view_dims(self) -> list[int]:
        return [k for k, d in enumerate(self.dims) if d[0] == "slice"]


def fold_arg_slice_views(nodes: list[Node], adj, const_ints: dict[str, int]) -> tuple[list[Node], set[str], set[str]]:
    """Lower range-slice views of kernel array arguments; see above.

    Element accesses are all this fold rewrites. A view that also reaches an
    operation that needs the array itself (``tile_load`` / ``tile_store``, a
    ``@wp.func_native`` snippet) is left in place when it is a unit-step
    slice of the leading dim of an argument (``arr[a:b]``), which stays
    contiguous: ``_lower_pointer_arrays`` turns it into a pointer array.
    Any other such view raises.

    Returns ``(nodes, labels to skip declaring, labels handed to
    _lower_pointer_arrays)``.
    """
    from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

    new_nodes, skip, views = _fold_arg_slice_views_once(nodes, adj, const_ints, frozenset())
    if not views:
        return new_nodes, skip, set()
    escaped = _slice_views_left_referenced(new_nodes, set(views))
    if not escaped:
        return new_nodes, skip, set()
    kname = getattr(adj, "fun_name", "?")
    handoff = _contiguous_arg_slices(nodes, adj, const_ints)
    bad = sorted(escaped - handoff)
    if bad:
        raise MetalCodegenError(
            f"Kernel {kname!r}: the array slice var_{bad[0]} is used by an operation that needs the array itself "
            "(tile_load / tile_store, a @wp.func_native snippet, ...); on Metal that works only for a unit-step "
            "slice of the leading dim of a kernel array argument (arr[a:b]) with a 32-bit-or-narrower scalar "
            "dtype. Index the slice element-wise, or pass the slice from the host."
        )
    new_nodes, skip, views = _fold_arg_slice_views_once(nodes, adj, const_ints, frozenset(escaped))
    left = _slice_views_left_referenced(new_nodes, set(views))
    if left:
        raise MetalCodegenError(
            f"Kernel {kname!r}: the array slice var_{sorted(left)[0]} is derived from a slice that is handed to a "
            "tile or native-snippet operation; slices of such slices are not supported on Metal"
        )
    return new_nodes, skip, set(escaped)


def _slice_views_left_referenced(nodes: list[Node], labels: set[str]) -> set[str]:
    """Folded view labels still referenced after the fold (uses it cannot
    rewrite to base-array accesses)."""
    alt = "|".join(re.escape(lbl) for lbl in sorted(labels, key=len, reverse=True))
    pat = re.compile(r"\bvar_(" + alt + r")\b|(?<![\w])(" + alt + r")_shape\b")
    found: set[str] = set()
    for n in _iter_leaves(nodes):
        if isinstance(n, (Comment, Empty, Pragma)):
            continue
        for m in pat.finditer(n.raw):
            found.add(m.group(1) or m.group(2))
    return found


def _contiguous_arg_slices(nodes: list[Node], adj, const_ints: dict[str, int]) -> set[str]:
    """Labels of ``var_V = wp::view(var_A, var_S)`` where ``A`` is a scalar
    (at most 32-bit) ``wp.array`` argument and ``S`` a unit-step slice:
    the leading-dim slices ``_lower_pointer_arrays`` can lower."""
    from warp._src.codegen_metal import (  # noqa: PLC0415
        _ARRAY_CTYPE_ELEM_PAT,
        _SCALAR_CTYPE_TO_MSL,
        _SLICE_VIEW_PAT,
        _is_array_arg,
    )

    ok_args = set()
    for a in adj.args:
        if not _is_array_arg(a) or type(a.type).__name__ not in ("array", "_ArrayAnnotation"):
            continue
        msl = _SCALAR_CTYPE_TO_MSL.get(_ARRAY_CTYPE_ELEM_PAT.sub(r"\1", a.ctype()))
        if msl is not None and msl != "wp_f64":
            ok_args.add(a.label)
    leaves = [n.raw for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    unit_slices = set()
    for ln in leaves:
        m = _SLICE_CTOR_PAT.match(ln)
        if m:
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            st = args[2] if len(args) == 3 else ""
            stv = const_ints.get(st[4:]) if st.startswith("var_") else (int(st) if _INT_LIT_PAT.match(st) else None)
            if stv == 1:
                unit_slices.add(m.group("lhs"))
    out = set()
    for ln in leaves:
        m = _SLICE_VIEW_PAT.match(ln)
        if m and m.group("src") in ok_args and m.group("sl") in unit_slices:
            out.add(m.group("lbl"))
    return out


def _fold_arg_slice_views_once(
    nodes: list[Node], adj, const_ints: dict[str, int], exclude: frozenset[str]
) -> tuple[list[Node], set[str], dict[str, "_SliceView"]]:
    from warp._src.codegen_metal import MetalCodegenError, _is_array_arg, _lower_array_slot_ir  # noqa: PLC0415
    from warp._src.codegen_metal_ast import _nonneg_locals  # noqa: PLC0415

    leaves = [n.raw for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    slices: dict[str, tuple[str, str, str]] = {}
    for ln in leaves:
        m = _SLICE_CTOR_PAT.match(ln)
        if m:
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            if len(args) == 3:
                slices[m.group("lhs")] = (args[0], args[1], args[2])
    if not slices and not any("wp::index(" in ln or "_slot(" in ln for ln in leaves):
        return nodes, set(), {}
    arg_ndim = {a.label: int(getattr(a.type, "ndim", 1)) for a in adj.args if _is_array_arg(a)}
    kname = getattr(adj, "fun_name", "?")
    nonneg = _nonneg_locals(nodes, const_ints)
    # Integer-index views of arguments stay with ``fold_views``; they are
    # only tracked here so component stores through them (slot IR, which
    # ``fold_views`` does not rewrite) can be lowered against the base.
    int_views: dict[str, _SliceView] = {}

    def const_of(expr: str) -> int | None:
        e = expr.strip()
        if _INT_LIT_PAT.match(e):
            return int(e)
        if e.startswith("var_"):
            v = const_ints.get(e[4:])
            if isinstance(v, int) and not isinstance(v, bool):
                return v
        return None

    views: dict[str, _SliceView] = {}
    shape_ptrs: dict[str, str] = {}
    shape_vals: dict[str, str] = {}
    decls: list[Node] = []
    skip: set[str] = set()
    counter = [0]

    def fresh(kind: str) -> str:
        counter[0] += 1
        name = f"slv{counter[0]}{kind}"
        decls.append(_RawLine(raw=f"    int var_{name} = 0;"))
        return f"var_{name}"

    def base_info(arr: str) -> _SliceView | None:
        if arr in views:
            return views[arr]
        if arr in arg_ndim:
            n = arg_ndim[arr]
            return _SliceView(arr, tuple(("slice", "0", "1", f"{arr}_shape[{k}]") for k in range(n)))
        return None

    def compose(src: _SliceView, idx_args: list[str], indent: str, out: list[str]) -> _SliceView | None:
        vd = src.view_dims()
        if len(idx_args) > len(vd):
            return None
        dims = list(src.dims)
        for j, a in enumerate(idx_args):
            k = vd[j]
            _kind, start, step, length = dims[k]
            if a.startswith("var_") and a[4:] in slices:
                s0, s1, st = slices[a[4:]]
                stc = const_of(st)
                if stc is None or stc == 0:
                    return None
                c0, c1 = const_of(s0), const_of(s1)
                L = length
                if c0 == _SLICE_BEGIN:
                    ns = f"(({L}) - 1)" if stc < 0 else "0"
                else:
                    ns = f"(({s0}) < 0 ? max(({s0}), -({L})) + ({L}) : min(({s0}), ({L})))"
                if c1 == _SLICE_END:
                    ne = "-1" if stc < 0 else f"({L})"
                else:
                    ne = f"(({s1}) < 0 ? max(({s1}), -({L})) + ({L}) : min(({s1}), ({L})))"
                vs, vn = fresh("s"), fresh("n")
                out.append(f"{indent}{vs} = {ns};")
                if stc > 0:
                    out.append(f"{indent}{vn} = ({ne}) > {vs} ? 1 + (({ne}) - {vs} - 1) / {stc} : 0;")
                else:
                    out.append(f"{indent}{vn} = {vs} > ({ne}) ? 1 + ({vs} - ({ne}) - 1) / {-stc} : 0;")
                new_start = vs if start == "0" and step == "1" else f"({start}) + {vs} * ({step})"
                new_step = str(stc) if step == "1" else f"({step}) * {stc}"
                dims[k] = ("slice", new_start, new_step, vn)
            else:
                if a.startswith("var_") and a[4:] in nonneg:
                    ia = a
                else:
                    ia = f"wp_wrap_index({a}, {length})"
                dims[k] = ("idx", ia if start == "0" and step == "1" else f"({start}) + {ia} * ({step})")
        return _SliceView(src.base, tuple(dims))

    def base_indices(v: _SliceView, idx: list[str], indent: str, out: list[str]) -> list[str] | None:
        vd = v.view_dims()
        if len(idx) != len(vd):
            return None
        res: list[str] = []
        j = 0
        for d in v.dims:
            if d[0] == "idx":
                expr = d[1]
            else:
                a = idx[j]
                j += 1
                ia = (
                    a
                    if (a.startswith("var_") and a[4:] in nonneg) or (const_of(a) is not None and const_of(a) >= 0)
                    else f"wp_wrap_index({a}, {d[3]})"
                )
                expr = ia if d[1] == "0" and d[2] == "1" else f"({d[1]}) + {ia} * ({d[2]})"
            if _INT_LIT_PAT.match(expr) or re.match(r"^var_\w+$", expr):
                res.append(expr)
            else:
                t = fresh("i")
                out.append(f"{indent}{t} = {expr};")
                res.append(t)
        return res

    def fail(msg: str, raw: str) -> MetalCodegenError:
        return MetalCodegenError(f"Kernel {kname!r}: {msg} (line: {raw.strip()})")

    def rewrite(n: Node) -> list[Node]:
        if isinstance(n, (Comment, Empty, Pragma, _RawLine)):
            return [n]
        raw = n.raw
        out: list[str] = []
        m = _ASSIGN_CALL_PAT.match(raw)
        if m:
            fn, lhs, indent = m.group("fn"), m.group("lhs"), m.group("indent")
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            first = args[0][4:] if args and args[0].startswith("var_") else ""
            if lhs in exclude:
                return [n]
            if fn == "view" and (
                first in views or (first in arg_ndim and any(a[4:] in slices for a in args[1:] if a.startswith("var_")))
            ):
                src = base_info(first)
                v = compose(src, args[1:], indent, out)
                if v is None:
                    raise fail("unsupported slice of an array (non-constant or zero step?)", raw)
                views[lhs] = v
                skip.add(lhs)
                return _to_nodes(out)
            if fn == "view" and (first in arg_ndim or first in int_views):
                src = int_views.get(first) or base_info(first)
                v = compose(src, args[1:], indent, out)
                if v is not None and not out:
                    int_views[lhs] = v
                return [n]
            if first in views:
                v = views[first]
                if fn in (
                    "address",
                    "atomic_add",
                    "atomic_sub",
                    "atomic_min",
                    "atomic_max",
                    "atomic_and",
                    "atomic_or",
                    "atomic_xor",
                    "atomic_exch",
                    "atomic_cas",
                ):
                    nval = {"address": 0, "atomic_cas": 2}.get(fn, 1)
                    idx = args[1 : len(args) - nval]
                    bi = base_indices(v, idx, indent, out)
                    if bi is None:
                        raise fail("index count does not match the sliced array's dimensions", raw)
                    rest = args[len(args) - nval :]
                    out.append(f"{indent}var_{lhs} = wp::{fn}({', '.join([f'var_{v.base}', *bi, *rest])});")
                    return _to_nodes(out)
                if fn == "len":
                    vd = v.view_dims()
                    return _to_nodes([f"{indent}var_{lhs} = {v.dims[vd[0]][3]};"])
        m = _SHAPE_PTR_PAT.match(raw)
        if m and m.group("arr") in views:
            shape_ptrs[m.group("lhs")] = m.group("arr")
            skip.add(m.group("lhs"))
            return []
        m = _ASSIGN_CALL_PAT.match(raw)
        if m and m.group("fn") == "load":
            a = m.group("args").strip()
            if a.startswith("var_") and a[4:] in shape_ptrs:
                shape_vals[m.group("lhs")] = shape_ptrs[a[4:]]
                skip.add(m.group("lhs"))
                return []
        m = _VOID_CALL_PAT.match(raw)
        if m:
            fn, indent = m.group("fn"), m.group("indent")
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            first = args[0][4:] if args and args[0].startswith("var_") else ""
            if first in views and fn in ("array_store", "atomic_add", "atomic_sub", "atomic_min", "atomic_max"):
                v = views[first]
                bi = base_indices(v, args[1:-1], indent, out)
                if bi is None:
                    raise fail("index count does not match the sliced array's dimensions", raw)
                out.append(f"{indent}wp::{fn}({', '.join([f'var_{v.base}', *bi, args[-1]])});")
                return _to_nodes(out)
        m = _ADJ_ARRAY_OP_PAT.match(raw)
        if m:
            args = [a.strip() for a in _split_top_level_args(m.group("args"))]
            first = args[0][4:] if args and args[0].startswith("var_") else ""
            if first in views:
                v = views[first]
                fn, indent = m.group("fn"), m.group("indent")
                fixed = {"address": 3, "array_store": 4}.get(fn, 5)
                nidx = (len(args) - fixed) // 2
                bi = base_indices(v, args[1 : 1 + nidx], indent, out)
                if bi is None:
                    raise fail("index count does not match the sliced array's dimensions", raw)
                nval = 0 if fn == "address" else 1
                vals = args[1 + nidx : 1 + nidx + nval]
                adj_rest = args[1 + nidx + nval + 1 + nidx :]
                adj_idx = [args[1 + nidx + nval + 1]] * len(bi) if nidx else []
                new = [f"var_{v.base}", *bi, *vals, f"adj_{v.base}", *adj_idx, *adj_rest]
                out.append(f"{indent}wp::adj_{fn}({', '.join(new)});")
                return _to_nodes(out)
        m = _SLOT_REF_PAT.match(raw) or _SLOT_STORE_PAT.match(raw)
        if m and (m.group("arr") in views or (m.group("arr") in int_views and m.group("acc").strip())):
            v = views.get(m.group("arr")) or int_views[m.group("arr")]
            idx = [a.strip() for a in _split_top_level_args(m.group("idx"))]
            bi = base_indices(v, idx, m.group("indent"), out)
            if bi is None:
                raise fail("index count does not match the sliced array's dimensions", raw)
            new = raw.replace(
                m.group(0)[m.start("arr") - m.start(0) - len("var_") : m.end("idx") - m.start(0)],
                f"var_{v.base}, {', '.join(bi)}",
                1,
            )
            out.extend(_lower_array_slot_ir([new], adj, [*leaves, new]))
            return _to_nodes(out)
        m = _SLOT_ATOMIC_PAT.match(raw)
        if m and (m.group("arr") in views or m.group("arr") in int_views):
            v = views.get(m.group("arr")) or int_views[m.group("arr")]
            idx = [a.strip() for a in _split_top_level_args(m.group("idx"))]
            bi = base_indices(v, idx, m.group("indent"), out)
            if bi is None:
                raise fail("index count does not match the sliced array's dimensions", raw)
            new = (
                raw[: m.start("arr")]
                + v.base
                + raw[m.end("arr") : m.start("idx")]
                + ", ".join(bi)
                + raw[m.end("idx") :]
            )
            out.extend(_lower_array_slot_ir([new], adj, [*leaves, new]))
            return _to_nodes(out)
        if shape_vals:

            def _extract(mm: re.Match[str]) -> str:
                v = views.get(shape_vals.get(mm.group(1), ""))
                if v is None:
                    return mm.group(0)
                k = const_of(mm.group(2))
                vd = v.view_dims()
                if k is None or not 0 <= k < len(vd):
                    return mm.group(0)
                return f"({v.dims[vd[k]][3]})"

            new = re.sub(r"\bwp::extract\s*\(\s*var_(\w+)\s*,\s*([^(),]+?)\s*\)", _extract, raw)
            if new != raw:
                return _to_nodes([new])
        return [n]

    new_nodes = _map_leaves(nodes, rewrite)
    if not views:
        return nodes, set(), {}
    return [*decls, *new_nodes], skip, views


# ---------------------------------------------------------------------------
# Array locals reassigned at runtime
# ---------------------------------------------------------------------------
# ``tmp = a; a = b; b = tmp`` inside a dynamic loop (mujoco_warp's polygon
# clipping ping-pongs two scratch rows this way) rebinds array locals at
# runtime, which no static alias can express. Each array local in such a
# group carries an ``int`` selector naming one of the group's candidate
# arrays (kernel array arguments or integer-index views of them, with the
# view indices captured when the view is taken); every access dispatches on
# the selector with one ``if`` per candidate.

_ASSIGN_VOID_PAT = re.compile(r"^(?P<indent>\s*)wp::assign\s*\(\s*var_(?P<dst>\w+)\s*,\s*var_(?P<src>\w+)\s*\)\s*;\s*$")
_COPY_ANY_PAT = re.compile(
    r"^(?P<indent>\s*)var_(?P<lhs>\w+)\s*=\s*(?:wp::copy\s*\(\s*)?var_(?P<src>\w+)\s*\)?\s*;\s*$"
)


def fold_dynamic_array_locals(nodes: list[Node], adj, var_ctypes: dict[str, str]) -> tuple[list[Node], set[str]]:
    """Lower runtime-rebound array locals; see above."""
    from warp._src.codegen_metal import MetalCodegenError, _is_array_arg, _lower_array_slot_ir  # noqa: PLC0415
    from warp._src.codegen_metal_ast import If  # noqa: PLC0415

    array_labels = {lbl for lbl, ct in var_ctypes.items() if ct and ct.startswith("wp::array_t<")}
    arg_ndim = {a.label: int(getattr(a.type, "ndim", 1)) for a in adj.args if _is_array_arg(a)}
    leaves = [n for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    rebound = set()
    for n in leaves:
        m = _ASSIGN_VOID_PAT.match(n.raw)
        if m and m.group("dst") in array_labels and m.group("dst") not in arg_ndim:
            rebound.add(m.group("dst"))
    if not rebound:
        return nodes, set()
    kname = getattr(adj, "fun_name", "?")

    def fail(msg: str, raw: str) -> MetalCodegenError:
        return MetalCodegenError(f"Kernel {kname!r}: {msg} (line: {raw.strip()})")

    # Groups: array locals connected by copies, assigns and views.
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for n in leaves:
        raw = n.raw
        m = _ASSIGN_VOID_PAT.match(raw)
        if m and m.group("dst") in array_labels:
            parent[find(m.group("dst"))] = find(m.group("src"))
            continue
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("lhs") in array_labels and (m.group("src") in array_labels or m.group("src") in arg_ndim):
            parent[find(m.group("lhs"))] = find(m.group("src"))
    dyn_roots = {find(x) for x in rebound}
    members = {x for x in list(parent) if find(x) in dyn_roots and x in array_labels and x not in arg_ndim}

    # Candidates, numbered per group: ``(argument, number of leading view
    # indices)`` for every view of a kernel argument taken into a member and
    # every whole argument copied into one. Each member owns its selector
    # AND its own copy of the leading indices (``var_<m>__i<k>``): a view
    # site that re-executes (``prev = cur; cur = src[i]`` in a loop) must
    # not move members bound by an earlier execution.
    decls: list[Node] = []
    cands: dict[str, list[tuple[str, int]]] = {}
    cand_at: dict[int, int] = {}  # id(node) -> candidate

    def add_cand(group: str, cand: tuple[str, int]) -> int:
        lst = cands.setdefault(group, [])
        if cand in lst:
            return lst.index(cand)
        lst.append(cand)
        return len(lst) - 1

    for n in leaves:
        raw = n.raw
        m = _ASSIGN_CALL_PAT.match(raw)
        if m and m.group("fn") == "view" and m.group("lhs") in members:
            args = [x.strip() for x in _split_top_level_args(m.group("args"))]
            if not args[0].startswith("var_") or args[0][4:] not in arg_ndim:
                raise fail("views of runtime-rebound arrays are not supported on Metal", raw)
            cand_at[id(n)] = add_cand(find(m.group("lhs")), (args[0][4:], len(args) - 1))
            continue
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("lhs") in members and m.group("src") in arg_ndim:
            cand_at[id(n)] = add_cand(find(m.group("lhs")), (m.group("src"), 0))
    nlead = {g: max((k for _b, k in lst), default=0) for g, lst in cands.items()}
    for mlabel in sorted(members):
        decls.append(_RawLine(raw=f"    int var_{mlabel}__sel = -1;"))
        for k in range(nlead.get(find(mlabel), 0)):
            decls.append(_RawLine(raw=f"    int var_{mlabel}__i{k} = 0;"))

    def lead_of(member: str, n_lead: int) -> list[str]:
        return [f"var_{member}__i{k}" for k in range(n_lead)]

    def copy_member(indent: str, dst: str, src: str) -> list[str]:
        out = [f"{indent}var_{dst}__sel = var_{src}__sel;"]
        out += [f"{indent}var_{dst}__i{k} = var_{src}__i{k};" for k in range(nlead.get(find(dst), 0))]
        return out

    skip: set[str] = set(members)
    dyn_ptrs: dict[str, str] = {}  # element pointer label -> member it points into
    shape_ptrs: dict[str, str] = {}
    shape_vals: dict[str, str] = {}

    def dispatch(indent: str, sel: str, group: str, make, member: str | None = None) -> list[Node]:
        """One ``if (sel == c) { ... }`` per candidate of ``group``.

        ``make(c, base, lead)`` gets ``member``'s leading indices for the
        candidate (empty without a member).
        """
        out: list[Node] = []
        for c, (base, n_lead) in enumerate(cands.get(group, [])):
            lead = lead_of(member, n_lead) if member else ["0"] * n_lead
            cond = f"({sel} == {c})"
            out.append(
                If(
                    raw=f"{indent}if {cond} {{",
                    cond=cond,
                    body=tuple(_to_nodes([f"{indent}    {ln}" for ln in make(c, base, lead)])),
                    raw_open=f"{indent}if {cond} {{",
                    raw_close=f"{indent}}}",
                )
            )
        return out

    def is_member(tok: str) -> bool:
        return tok.startswith("var_") and tok[4:] in members

    def rewrite(n: Node) -> list[Node]:
        if isinstance(n, (Comment, Empty, Pragma)):
            return [n]
        raw = n.raw
        if isinstance(n, _RawLine):
            m = _DECL_PAT.match(raw)
            if m and m.group("label") in skip:
                return []
            return [n]
        if id(n) in cand_at:
            c = cand_at[id(n)]
            m = _ASSIGN_CALL_PAT.match(raw) or _COPY_ANY_PAT.match(raw)
            indent, lhs = m.group("indent"), m.group("lhs")
            out = []
            if "args" in m.groupdict():
                args = [x.strip() for x in _split_top_level_args(m.group("args"))]
                out = [f"{indent}var_{lhs}__i{k} = {a};" for k, a in enumerate(args[1:])]
            out.append(f"{indent}var_{lhs}__sel = {c};")
            return _to_nodes(out)
        m = _ASSIGN_VOID_PAT.match(raw)
        if m and m.group("dst") in members:
            if m.group("src") not in members:
                raise fail("unsupported source for a runtime-rebound array local", raw)
            return _to_nodes(copy_member(m.group("indent"), m.group("dst"), m.group("src")))
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("lhs") in members:
            if m.group("src") not in members:
                raise fail("unsupported source for a runtime-rebound array local", raw)
            return _to_nodes(copy_member(m.group("indent"), m.group("lhs"), m.group("src")))
        m = _ASSIGN_CALL_PAT.match(raw)
        if m:
            fn, lhs, indent = m.group("fn"), m.group("lhs"), m.group("indent")
            args = [x.strip() for x in _split_top_level_args(m.group("args"))]
            first = args[0] if args else ""
            if fn == "address" and is_member(first):
                mem = first[4:]
                dyn_ptrs[lhs] = mem
                skip.add(lhs)
                decls.append(_RawLine(raw=f"    int var_{lhs}__s = -1;"))
                out = [
                    f"{indent}var_{lhs}__c{c} = wp::address({', '.join([f'var_{base}', *lead_of(mem, k), *args[1:]])});"
                    for c, (base, k) in enumerate(cands.get(find(mem), []))
                ]
                out.append(f"{indent}var_{lhs}__s = var_{mem}__sel;")
                return _to_nodes(out)
            if fn == "load" and first.startswith("var_") and first[4:] in dyn_ptrs:
                p = first[4:]
                return dispatch(
                    indent, f"var_{p}__s", find(dyn_ptrs[p]), lambda c, b, ld: [f"var_{lhs} = wp::load(var_{p}__c{c});"]
                )
            if fn == "load" and first.startswith("var_") and first[4:] in shape_ptrs:
                shape_vals[lhs] = shape_ptrs[first[4:]]
                skip.add(lhs)
                return []
            if fn == "extract" and first.startswith("var_") and first[4:] in shape_vals:
                mem = shape_vals[first[4:]]
                k = args[1]
                return dispatch(
                    indent,
                    f"var_{mem}__sel",
                    find(mem),
                    lambda c, b, ld: [f"var_{lhs} = {b}_shape[{len(ld)} + {k}];"],
                    mem,
                )
            if fn == "len" and is_member(first):
                mem = first[4:]
                return dispatch(
                    indent, f"var_{mem}__sel", find(mem), lambda c, b, ld: [f"var_{lhs} = {b}_shape[{len(ld)}];"], mem
                )
            if fn.startswith("atomic_") and is_member(first):
                mem = first[4:]
                return dispatch(
                    indent,
                    f"var_{mem}__sel",
                    find(mem),
                    lambda c, b, ld: [f"var_{lhs} = wp::{fn}({', '.join([f'var_{b}', *ld, *args[1:]])});"],
                    mem,
                )
        m = _SHAPE_PTR_PAT.match(raw)
        if m and m.group("arr") in members:
            shape_ptrs[m.group("lhs")] = m.group("arr")
            skip.add(m.group("lhs"))
            return []
        m = _VOID_CALL_PAT.match(raw)
        if m:
            fn, indent = m.group("fn"), m.group("indent")
            args = [x.strip() for x in _split_top_level_args(m.group("args"))]
            first = args[0] if args else ""
            if (fn == "array_store" or fn.startswith("atomic_")) and is_member(first):
                mem = first[4:]
                return dispatch(
                    indent,
                    f"var_{mem}__sel",
                    find(mem),
                    lambda c, b, ld: [f"wp::{fn}({', '.join([f'var_{b}', *ld, *args[1:]])});"],
                    mem,
                )
            if fn == "store" and first.startswith("var_") and first[4:] in dyn_ptrs:
                p = first[4:]
                return dispatch(
                    indent, f"var_{p}__s", find(dyn_ptrs[p]), lambda c, b, ld: [f"wp::store(var_{p}__c{c}, {args[1]});"]
                )
        m = _SLOT_REF_PAT.match(raw)
        if m and m.group("arr") in members:
            skip.add(m.group("lhs"))
            return []
        m = _SLOT_STORE_PAT.match(raw)
        if m and m.group("arr") in members:
            mem = m.group("arr")
            lo, hi = m.start("arr") - len("var_"), m.end("idx")

            def mk(c, b, ld, _raw=raw, _m=m, _lo=lo, _hi=hi):
                new = _raw[:_lo] + ", ".join([f"var_{b}", *ld, _m.group("idx")]) + _raw[_hi:]
                return [ln.strip() for ln in _lower_array_slot_ir([new], adj, [new])]

            return dispatch(m.group("indent"), f"var_{mem}__sel", find(mem), mk, mem)
        for tok in _VAR_TOKEN_PAT.finditer(raw):
            if tok.group(0).startswith("var_") and (
                tok.group(1) in members or tok.group(1) in dyn_ptrs or tok.group(1) in shape_ptrs
            ):
                raise fail("unsupported use of an array local that is rebound at runtime", raw)
        return [n]

    new_nodes = _map_leaves(nodes, rewrite)
    new_nodes = _map_leaves(new_nodes, lambda n: _drop_decl(n, skip))
    return [*decls, *new_nodes], skip


# ---------------------------------------------------------------------------
# Array fields of default-constructed struct locals
# ---------------------------------------------------------------------------
# ``provider = SupportMapDataProvider()`` leaves its array fields as empty
# arrays (null data, shape 0); Newton's support functions guard every element
# read with ``index < provider.field.shape[0]``. With no assignment there is
# no buffer to alias, so the reads lower to the empty array's values: shape
# 0 and (unreachable) zero elements.

_STRUCT_CTOR_PAT = re.compile(r"^\s*var_(?P<lhs>\w+)\s*=\s*(?P<name>[A-Za-z_]\w*)\s*\(\s*\)\s*;\s*$")
_FIELD_PTR_PAT = re.compile(
    r"^\s*var_(?P<lhs>\w+)\s*=\s*&\s*\(\s*var_(?P<root>\w+)\s*(?:\.|->)\s*(?P<field>\w+)\s*\)\s*;\s*$"
)
_STRUCT_ASSIGN_PAT = re.compile(r"^\s*var_(?P<lhs>\w+)\s*=\s*(?P<rhs>.+?)\s*;\s*$")


def fold_empty_struct_array_fields(
    nodes: list[Node], adj, struct_locals: dict[str, object], var_ctypes: dict[str, str]
) -> tuple[list[Node], set[str]]:
    """Lower reads of never-assigned array fields of struct locals; see above."""
    from warp._src.codegen import Struct  # noqa: PLC0415
    from warp._src.codegen_metal import _is_array_arg_type, _msl_var_type, _msl_zero_init  # noqa: PLC0415

    types: dict[str, object] = {str(v.label): v.type for v in adj.variables if isinstance(v.type, Struct)}
    types.update({k: v for k, v in struct_locals.items() if isinstance(v, Struct)})
    if not types:
        return nodes, set()

    def array_fields(st) -> set[str]:
        return {f for f, v in st.vars.items() if _is_array_arg_type(v.type)}

    candidates = {lbl for lbl, st in types.items() if array_fields(st)}
    if not candidates:
        return nodes, set()
    leaves = [n for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    # Definitions of each candidate: a no-arg constructor, or a copy of
    # another struct local; anything else (a function result, a load from
    # an array, a struct argument) disqualifies it.
    sources: dict[str, set[str]] = {}
    bad: set[str] = set()
    field_ptr: dict[str, tuple[str, str]] = {}
    for n in leaves:
        raw = n.raw
        m = _FIELD_PTR_PAT.match(raw)
        if m and m.group("root") in candidates:
            field_ptr[m.group("lhs")] = (m.group("root"), m.group("field"))
            continue
        m = _STRUCT_CTOR_PAT.match(raw)
        if m and m.group("lhs") in candidates:
            continue
        m = _STRUCT_ASSIGN_PAT.match(raw)
        if m and m.group("lhs") in candidates:
            rhs = m.group("rhs").strip()
            mm = re.fullmatch(r"(?:wp::copy\s*\(\s*)?var_(\w+)\s*\)?", rhs)
            if mm and mm.group(1) in candidates:
                sources.setdefault(m.group("lhs"), set()).add(mm.group(1))
            else:
                bad.add(m.group("lhs"))
            continue
        m = _ASSIGN_VOID_PAT.match(raw)
        if m and m.group("dst") in candidates:
            if m.group("src") in candidates:
                sources.setdefault(m.group("dst"), set()).add(m.group("src"))
            else:
                bad.add(m.group("dst"))
    # Any write through a pointer to one of its array fields disqualifies it.
    for n in leaves:
        for mm in re.finditer(r"\bwp::(?:store|assign|copy_into)\s*\(\s*var_(\w+)", n.raw):
            if mm.group(1) in field_ptr:
                bad.add(field_ptr[mm.group(1)][0])
    changed = True
    while changed:
        changed = False
        for lbl, srcs in sources.items():
            if lbl not in bad and srcs & bad:
                bad.add(lbl)
                changed = True
    empty_roots = candidates - bad
    if not empty_roots:
        return nodes, set()

    eptr: set[str] = set()  # pointers to empty array fields
    earr: set[str] = set()  # loaded empty arrays
    shape_p: set[str] = set()
    shape_v: set[str] = set()
    elem_p: set[str] = set()
    skip: set[str] = set()

    def zero_of(label: str) -> str | None:
        ct = var_ctypes.get(label)
        if not ct:
            return None
        try:
            return _msl_zero_init(_msl_var_type(ct))
        except Exception:
            return None

    def rewrite(n: Node) -> list[Node]:
        if isinstance(n, (Comment, Empty, Pragma, _RawLine)):
            return [n]
        raw = n.raw
        m = _FIELD_PTR_PAT.match(raw)
        if m:
            root, field = m.group("root"), m.group("field")
            if root in empty_roots and field in array_fields(types[root]):
                eptr.add(m.group("lhs"))
                skip.add(m.group("lhs"))
                return []
            if (root in eptr or root in earr) and field == "shape":
                shape_p.add(m.group("lhs"))
                skip.add(m.group("lhs"))
                return []
        m = _ASSIGN_CALL_PAT.match(raw)
        if m:
            fn, lhs, indent = m.group("fn"), m.group("lhs"), m.group("indent")
            args = [x.strip() for x in _split_top_level_args(m.group("args"))]
            first = args[0][4:] if args and args[0].startswith("var_") else ""
            if fn == "load" and first in eptr:
                earr.add(lhs)
                skip.add(lhs)
                return []
            if fn == "load" and first in shape_p:
                shape_v.add(lhs)
                skip.add(lhs)
                return []
            if fn == "extract" and first in shape_v:
                return _to_nodes([f"{indent}var_{lhs} = 0;"])
            if fn == "len" and first in earr:
                return _to_nodes([f"{indent}var_{lhs} = 0;"])
            if fn in ("address", "view") and first in earr:
                (elem_p if fn == "address" else earr).add(lhs)
                skip.add(lhs)
                return []
            if (fn == "load" and first in elem_p) or (fn.startswith("atomic_") and first in earr):
                z = zero_of(lhs)
                if z is None:
                    return [n]
                return _to_nodes([f"{indent}var_{lhs} = {z};"])
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("src") in earr:
            earr.add(m.group("lhs"))
            skip.add(m.group("lhs"))
            return []
        m = _VOID_CALL_PAT.match(raw)
        if m:
            args = [x.strip() for x in _split_top_level_args(m.group("args"))]
            first = args[0][4:] if args and args[0].startswith("var_") else ""
            if (m.group("fn") in ("array_store", "store") or m.group("fn").startswith("atomic_")) and (
                first in earr or first in elem_p
            ):
                # A write into an empty array: out of bounds natively, so
                # only reachable behind a failed guard; nothing to store.
                return []
        if shape_v:
            new = re.sub(
                r"\bwp::extract\s*\(\s*var_(\w+)\s*,\s*[^(),]+?\s*\)",
                lambda mm: "0" if mm.group(1) in shape_v else mm.group(0),
                raw,
            )
            if new != raw:
                return _to_nodes([new])
        return [n]

    new_nodes = _map_leaves(nodes, rewrite)
    new_nodes = _map_leaves(new_nodes, lambda n: _drop_decl(n, skip))
    return new_nodes, skip


def fold_array_copy_aliases(nodes: list[Node], adj, var_ctypes: dict[str, str]) -> tuple[list[Node], set[str]]:
    """Alias array locals that are plain copies of another array.

    ``on = plane_normal`` (mujoco_warp's polygon clipping) copies an array
    *handle*: both names refer to the same storage. When the copy is the
    local's only definition, every use is renamed to the source so the view
    fold and the argument lowering see the original array. (Locals that are
    also rebound are handled by :func:`fold_dynamic_array_locals`.)
    """
    from warp._src.codegen_metal_ast import _substitute_var_refs  # noqa: PLC0415

    array_labels = {lbl for lbl, ct in var_ctypes.items() if ct and ct.startswith("wp::array_t<")}
    arg_arrays = {a.label for a in adj.args if _local_array_ctype(a.ctype())}
    leaves = [n for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma))]
    defs: dict[str, list[str]] = {}
    for n in leaves:
        m = re.match(r"^\s*var_(\w+)\s*=", n.raw)
        if m and m.group(1) in array_labels:
            defs.setdefault(m.group(1), []).append(n.raw)
        m = _ASSIGN_VOID_PAT.match(n.raw)
        if m and m.group("dst") in array_labels:
            defs.setdefault(m.group("dst"), []).extend([n.raw, n.raw])
    alias: dict[str, str] = {}
    for lbl, lines in defs.items():
        if len(lines) != 1:
            continue
        m = _COPY_ANY_PAT.match(lines[0])
        if m and (m.group("src") in array_labels or m.group("src") in arg_arrays) and m.group("src") != lbl:
            alias[lbl] = m.group("src")
    if not alias:
        return nodes, set()

    def resolve(x: str) -> str:
        seen = set()
        while x in alias and x not in seen:
            seen.add(x)
            x = alias[x]
        return x

    subs = {f"var_{k}": f"var_{resolve(k)}" for k in alias}

    def rewrite(n: Node) -> list[Node]:
        if isinstance(n, (Comment, Empty, Pragma)):
            return [n]
        raw = n.raw
        if isinstance(n, _RawLine):
            m = _DECL_PAT.match(raw)
            if m and m.group("label") in alias:
                return []
            return [n]
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("lhs") in alias:
            return []
        if "var_" not in raw:
            return [n]
        new = _substitute_var_refs(raw, subs)
        return [n] if new == raw else _to_nodes([new])

    return _map_leaves(nodes, rewrite), set(alias)


def check_struct_copy_field_stores(nodes: list[Node], adj, struct_locals: dict[str, object]) -> None:
    """Reject a struct copy that diverges in an array field.

    ``q = p; p.data = a; ... q.data`` (or the mirror, storing into ``q``):
    natively ``q`` keeps the field value from the copy, but the array-field
    backing fold resolves fields flow-insensitively along copy edges and
    would hand ``q`` the array stored into ``p`` later. Raise rather than
    return the wrong array. A side that is redefined after the copy (e.g.
    the by-value parameter copy of an inlined function whose result is
    assigned back) is not a reader of the stale value and is ignored.
    """
    from warp._src.codegen import Struct  # noqa: PLC0415
    from warp._src.codegen_metal import MetalCodegenError, _is_array_arg_type  # noqa: PLC0415

    types: dict[str, object] = {str(v.label): v.type for v in adj.variables if isinstance(v.type, Struct)}
    types.update({k: v for k, v in struct_locals.items() if isinstance(v, Struct)})
    arr_fields = {lbl: {f for f, v in st.vars.items() if _is_array_arg_type(v.type)} for lbl, st in types.items()}
    arr_fields = {k: v for k, v in arr_fields.items() if v}
    if not arr_fields:
        return
    leaves = [n.raw for n in _iter_leaves(nodes) if not isinstance(n, (Comment, Empty, Pragma, _RawLine))]
    copies: list[tuple[int, str, str]] = []  # (position, dst, src)
    defs: dict[str, list[int]] = {}
    ptr_field: dict[str, tuple[str, str]] = {}
    stores: list[tuple[int, str, str]] = []
    reads: list[tuple[int, str, str]] = []
    for pos, raw in enumerate(leaves):
        m = _COPY_ANY_PAT.match(raw)
        if m and m.group("lhs") in arr_fields and m.group("src") in arr_fields:
            copies.append((pos, m.group("lhs"), m.group("src")))
        m = re.match(r"^\s*var_(\w+)\s*=", raw)
        if m and m.group(1) in arr_fields:
            defs.setdefault(m.group(1), []).append(pos)
        m = _ASSIGN_VOID_PAT.match(raw)
        if m and m.group("dst") in arr_fields:
            defs.setdefault(m.group("dst"), []).append(pos)
        m = _FIELD_PTR_PAT.match(raw)
        if m and m.group("root") in arr_fields and m.group("field") in arr_fields[m.group("root")]:
            ptr_field[m.group("lhs")] = (m.group("root"), m.group("field"))
            continue
        for mm in re.finditer(r"\bwp::(store|load)\s*\(\s*var_(\w+)", raw):
            if mm.group(2) in ptr_field:
                root, field = ptr_field[mm.group(2)]
                (stores if mm.group(1) == "store" else reads).append((pos, root, field))
    kname = getattr(adj, "fun_name", "?")
    for cpos, dst, src in copies:
        for spos, sroot, field in stores:
            if spos <= cpos or sroot not in (dst, src):
                continue
            other = src if sroot == dst else dst
            if sroot == dst and not any(r == src and f == field for _p, r, f in stores):
                # Only the copy was reassigned and the source's field was
                # never set: the empty-field fold keeps the source empty.
                continue
            if any(cpos < d <= spos for d in defs.get(other, ())) or any(cpos < d for d in defs.get(other, ())):
                continue
            if any(rpos > spos and rroot == other and rfield == field for rpos, rroot, rfield in reads):
                raise MetalCodegenError(
                    f"Kernel {kname!r}: a copy of a struct with array fields diverges from its source when the "
                    f"array field {field!r} is reassigned after the copy; Metal resolves array fields of struct "
                    "locals by alias and cannot keep the two apart. Assign the field before copying, or use "
                    "separate struct locals"
                )


def _to_nodes(lines: list[str]) -> list[Node]:
    out: list[Node] = []
    for ln in lines:
        try:
            out.extend(parse([ln]))
        except MetalASTParseError:
            out.append(_RawLine(raw=ln))
    return out


def _drop_decl(n: Node, skip: set[str]) -> list[Node]:
    if not isinstance(n, _RawLine):
        return [n]
    m = _DECL_PAT.match(n.raw)
    if m and m.group("label") in skip and "[" not in n.raw and "thread " not in n.raw:
        return []
    return [n]


def _mul_off(offset: str, size: int) -> str:
    c = offset.strip()
    if _INT_LIT_PAT.match(c):
        return str(int(c) * size)
    return f"({c}) * {size}"


def _mat_cols(ctype: str) -> int:
    m = re.match(r"wp::mat_t<\s*(\d+)\s*,\s*(\d+)", ctype)
    return int(m.group(2)) if m else 1


def _ctype_size(ctype: str) -> int:
    """Byte size of a Warp value ctype (``wp::vec_t<2, wp::float32>`` -> 8)."""
    from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

    scalar_sizes = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "float16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "float64": 8,
    }
    ct = ctype.strip()
    if ct.startswith("wp::"):
        ct = ct[4:]
    if ct in scalar_sizes:
        return scalar_sizes[ct]
    if ct in ("int", "float"):
        return 4
    m = re.match(r"^(vec_t|quat_t|transform_t|mat_t)\s*<(.*)>$", ct)
    if m:
        kind = m.group(1)
        args = [a.strip() for a in _split_top_level_args(m.group(2))]
        scalar = _ctype_size(args[-1])
        if kind == "vec_t":
            return int(args[0]) * scalar
        if kind == "quat_t":
            return 4 * scalar
        if kind == "transform_t":
            return 7 * scalar
        return int(args[0]) * int(args[1]) * scalar
    raise MetalCodegenError(f"MSL codegen: cannot size thread-local array element type {ctype!r}")


__all__ = [
    "check_struct_copy_field_stores",
    "fold_arg_slice_views",
    "fold_array_copy_aliases",
    "fold_dynamic_array_locals",
    "fold_empty_struct_array_fields",
    "fold_local_arrays",
    "lower_inlined_array_slots",
]
