# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Outlining of inlined ``@wp.func`` bodies for the Metal backend.

The Metal translator inlines every user function into the kernel's IR before
the (kernel-wide) regex/IR translation runs: array arguments become packed
buffers and shape slots, struct locals split into per-field locals, views and
negative indices fold against kernel arguments, and so on. All of that is a
kernel-level analysis, which is why the translator inlines in the first place.
The downside is code size: Newton's mesh/convex narrow phase becomes ~10 MiB
of MSL in one function, which Apple's compiler needs tens of minutes for.

Instead of teaching every pass about function boundaries, outlining runs on
the *translated* kernel. The inliner brackets each spliced body with marker
comments (:func:`region_markers`). After translation, :func:`outline_kernel`
turns each marked region into a separate MSL function, bottom-up:

* Identifiers the region uses but declares outside it (caller locals, kernel
  parameters, prologue locals) become parameters. Caller locals the region
  may write are passed as ``thread T&``; read-only small values are passed by
  ``const`` value; everything else by ``thread const T&``. Large structs are
  therefore never passed by value (Apple miscompiles non-inlined by-value
  structs above ~144 bytes; see ``_force_inline_helpers``). A ``const`` value
  parameter turns any write the analysis missed into a compile error rather
  than a lost store.
* The callee's own locals that the translator hoisted to kernel scope (the
  per-field struct locals) move into the function when nothing outside the
  region uses them.
* The function body is canonicalised (parameters and locals renamed by
  first use) and hashed, so every call of one overload with the same
  translation shares one function.

A region stays inline when it is not self-contained: it returns, breaks out
of a loop it does not contain, has unbalanced braces, preprocessor lines or
``threadgroup`` declarations, or one of its locals is used after it.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import re

MARKER_PREFIX = "//@@wp_outline"
_BEGIN = MARKER_PREFIX + " begin "
_END = MARKER_PREFIX + " end "


def mode() -> str:
    import warp.config as _cfg  # noqa: PLC0415

    m = str(getattr(_cfg, "metal_outline_funcs", "off") or "off").strip().lower()
    return m if m in ("auto", "all", "off") else "off"


def region_markers(fn_overload, inline_id: int) -> tuple[str, str] | None:
    """Marker comments for one inlined body, or ``None`` when outlining is off."""
    if mode() == "off":
        return None
    key = re.sub(r"\s+", "_", str(getattr(fn_overload, "key", "?")))
    return (f"{_BEGIN}{inline_id} {key}", f"{_END}{inline_id}")


def strip_markers(body: str) -> str:
    if MARKER_PREFIX not in body:
        return body
    return "\n".join(ln for ln in body.split("\n") if not ln.startswith(MARKER_PREFIX))


# ---------------------------------------------------------------------------
# Heuristics ("auto")
# ---------------------------------------------------------------------------
# Only kernels whose body is large enough for the Apple compile time to
# matter are outlined; below that, inlining is the best code anyway.
_AUTO_MIN_KERNEL_LINES = int(os.environ.get("WARP_METAL_OUTLINE_MIN_KERNEL_LINES", "") or 20000)
# A region is outlined when its (already partly outlined) body has at least
# this many lines.
_AUTO_MIN_REGION_LINES = int(os.environ.get("WARP_METAL_OUTLINE_MIN_LINES", "") or 40)
_ALL_MIN_REGION_LINES = 1


def _noinline_attr() -> str:
    v = os.environ.get("WARP_METAL_OUTLINE_NOINLINE", "1").strip().lower()
    return "__attribute__((noinline)) " if v not in ("0", "false", "off", "no") else ""


# ---------------------------------------------------------------------------
# Lexing
# ---------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"""
    (?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[a-zA-Z]*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op><<=|>>=|->|\+\+|--|&&|\|\||<<|>>|<=|>=|==|!=|\+=|-=|\*=|/=|%=|&=|\|=|\^=|::|[-+*/%&|^!~<>=?:;,.(){}\[\]])
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<ws>\s+)
  | (?P<other>.)
    """,
    re.VERBOSE,
)

_ASSIGN_OPS = frozenset({"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="})

_KEYWORDS = frozenset(
    """
    if else for while do switch case default break continue return goto sizeof
    const constant device thread threadgroup volatile static inline true false
    struct class using namespace typedef template typename auto operator new delete this
    """.split()
)


def _tokens(line: str) -> list[tuple[str, str]]:
    out = []
    for m in _TOKEN_RE.finditer(line):
        kind = m.lastgroup
        if kind == "ws":
            continue
        out.append((kind, m.group(0)))
    return out


def _code_part(line: str) -> str:
    """``line`` without a trailing ``//`` comment (strings are not used in bodies)."""
    i = line.find("//")
    return line if i < 0 else line[:i]


# Declaration at the start of a line: ``[quals] TYPE[*&...] NAME (= ...|;|[N]...)``.
_QUALS = r"(?:(?:const|constant|device|thread|threadgroup|volatile|static)\s+)*"
_TYPE = r"[A-Za-z_]\w*(?:::\w+)*(?:\s*<[^;=(){}]*>)?(?:\s*(?:\*|&)\s*(?:const\b\s*)?)*"
_DECL_RE = re.compile(
    r"^\s*(?P<type>"
    + _QUALS
    + _TYPE
    + r")\s*(?<=[\s*&])(?P<name>[A-Za-z_]\w*)\s*(?P<dims>(?:\[[^\]]*\]\s*)*)(?P<rest>=|;)"
)
_FOR_DECL_RE = re.compile(r"\bfor\s*\(\s*(?P<type>" + _QUALS + _TYPE + r")\s*(?<=[\s*&])(?P<name>[A-Za-z_]\w*)\s*=")
_SIMPLE_INIT_RE = re.compile(r"^\s*=\s*(?:[\w:]+\s*\(\s*[-+\w.]*\s*\)|[-+\w.]+)\s*;\s*$|^\s*;\s*$")


class _Decl:
    __slots__ = ("array_dims", "const", "line", "movable_init", "name", "param", "type")

    def __init__(self, name, type_, line, dims="", const=False, param=False, movable_init=False):
        self.name = name
        self.type = type_
        self.line = line
        self.array_dims = dims
        self.const = const
        self.param = param
        self.movable_init = movable_init


def _match_decl(code: str) -> tuple[str, str, str, str] | None:
    m = _DECL_RE.match(code)
    if m is None:
        return None
    type_ = " ".join(m.group("type").split())
    first = type_.split()[0].split("<")[0].rstrip("*&")
    if first in ("return", "else", "case", "goto", "delete", "new", "using", "typedef", "do"):
        return None
    if type_.rstrip("*& ").split()[-1] in _KEYWORDS:
        return None
    if m.group("name") in _KEYWORDS:
        return None
    return type_, m.group("name"), " ".join(m.group("dims").split()), code[m.start("rest") :]


def _is_const_object(type_: str) -> bool:
    return "*" not in type_ and "&" not in type_ and re.match(r"^(?:\w+\s+)*const\b", type_) is not None


_SMALL_VALUE_RE = re.compile(
    r"^(?:const\s+)?(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|float|size_t|"
    r"(?:bool|char|uchar|short|ushort|int|uint|long|ulong|half|float)[234]|(?:half|float)[234]x[234])$"
)


def _strip_const(type_: str) -> str:
    return re.sub(r"^const\s+", "", type_)


# ---------------------------------------------------------------------------
# Kernel scan
# ---------------------------------------------------------------------------
class _Region:
    __slots__ = ("begin", "children", "end", "id", "key", "parent", "scopes")

    def __init__(self, rid, key, begin, parent, scopes):
        self.id = rid
        self.key = key
        self.begin = begin
        self.end = -1
        self.parent = parent
        self.children: list[_Region] = []
        self.scopes = scopes


class _Scan:
    """One pass over the kernel: scopes + declarations, identifier posting
    lists, address-taken names and the marker tree."""

    def __init__(self, lines: list[str], params: list[_Decl]):
        self.lines = lines
        self.postings: dict[str, list[int]] = {}
        self.addr_taken: set[str] = set()
        self.decl_at: dict[int, _Decl] = {}
        self.roots: list[_Region] = []
        self.ok = True
        root_scope: dict[str, _Decl] = {p.name: p for p in params}
        stack: list[dict[str, _Decl]] = [root_scope, {}]
        region_stack: list[_Region] = []
        pending_for: _Decl | None = None
        for i, raw in enumerate(lines):
            if raw.startswith(MARKER_PREFIX):
                if raw.startswith(_BEGIN):
                    parts = raw[len(_BEGIN) :].split(" ", 1)
                    parent = region_stack[-1] if region_stack else None
                    r = _Region(parts[0], parts[1] if len(parts) > 1 else "?", i, parent, list(stack))
                    (parent.children if parent else self.roots).append(r)
                    region_stack.append(r)
                elif raw.startswith(_END):
                    if not region_stack or region_stack[-1].id != raw[len(_END) :].strip():
                        self.ok = False
                        return
                    region_stack.pop().end = i
                continue
            code = _code_part(raw)
            if not code.strip():
                continue
            d = _match_decl(code)
            if d is not None:
                type_, name, dims, rest = d
                decl = _Decl(
                    name,
                    type_,
                    i,
                    dims,
                    const=_is_const_object(type_),
                    movable_init=bool(_SIMPLE_INIT_RE.match(rest)) and not dims,
                )
                stack[-1][name] = decl
                self.decl_at[i] = decl
            else:
                mf = _FOR_DECL_RE.search(code)
                if mf is not None:
                    pending_for = _Decl(mf.group("name"), " ".join(mf.group("type").split()), i)
            toks = _tokens(code)
            prev = None
            prev2 = None
            for kind, text in toks:
                if kind == "id":
                    if prev is None or prev[1] not in (".", "->", "::"):
                        self.postings.setdefault(text, []).append(i)
                        if (
                            prev is not None
                            and prev[1] == "&"
                            and (prev2 is None or (prev2[0] == "op" and prev2[1] not in (")", "]")))
                        ):
                            self.addr_taken.add(text)
                elif text == "{":
                    stack.append({})
                    if pending_for is not None:
                        stack[-1][pending_for.name] = pending_for
                        pending_for = None
                elif text == "}":
                    if len(stack) > 2:
                        stack.pop()
                prev2 = prev
                prev = (kind, text)
        if region_stack:
            self.ok = False

    def count_in(self, name: str, lo: int, hi: int) -> int:
        p = self.postings.get(name, ())
        return bisect.bisect_right(p, hi) - bisect.bisect_left(p, lo)

    def total(self, name: str) -> int:
        return len(self.postings.get(name, ()))


def _resolve(scopes: list[dict[str, _Decl]], name: str, before: int) -> _Decl | None:
    for sc in reversed(scopes):
        d = sc.get(name)
        if d is not None and (d.param or d.line < before):
            return d
    return None


# ---------------------------------------------------------------------------
# Region analysis
# ---------------------------------------------------------------------------
class _Bail(Exception):
    pass


def _analyze_tokens(text_lines: list[str], ref_funcs: set[str], helper_ref_pos: dict[str, tuple[bool, ...]]):
    """Return ``(ordered identifiers, written names, region decl names,
    directly written names)``.

    ``written`` holds every name the region may modify, including stores
    through a subscript or ``->`` (``p[i] = v``, ``&p[i]``); ``direct`` only the
    names whose own value may change (``x = v``, ``x.f = v``, ``&x``, ``x++``,
    a by-reference argument). A pointer kernel parameter that is only stored
    through can be passed to the outlined function by value.

    Raises :class:`_Bail` for constructs that cannot leave the kernel.
    """
    order: dict[str, None] = {}
    written: set[str] = set()
    direct: set[str] = set()

    def mark(name: str, through: bool) -> None:
        written.add(name)
        if not through:
            direct.add(name)

    decls: set[str] = set()
    depth = 0
    # Stack of open braces: True for a loop/switch body.
    brace_kinds: list[bool] = []
    pending_loop = False
    paren_depth_at_loop = None
    for raw in text_lines:
        if raw.startswith(MARKER_PREFIX):
            continue
        code = _code_part(raw)
        if not code.strip():
            continue
        if code.lstrip().startswith("#"):
            raise _Bail("preprocessor")
        d = _match_decl(code)
        if d is not None:
            if "threadgroup" in d[0].split():
                raise _Bail("threadgroup decl")
            decls.add(d[1])
        mf = _FOR_DECL_RE.search(code)
        if mf is not None:
            decls.add(mf.group("name"))
        toks = _tokens(code)
        n = len(toks)
        # call stack: (callee name or None, arg index)
        calls: list[list] = []
        for k in range(n):
            kind, text = toks[k]
            if kind == "id":
                prev = toks[k - 1][1] if k > 0 else None
                if text in ("return", "goto"):
                    raise _Bail(text)
                if text in ("for", "while", "switch"):
                    pending_loop = True
                    paren_depth_at_loop = len(calls)
                    continue
                if text == "do":
                    pending_loop = True
                    paren_depth_at_loop = len(calls)
                    continue
                if text in ("break", "continue"):
                    if not any(brace_kinds):
                        raise _Bail(text)
                    continue
                if prev in (".", "->", "::"):
                    continue
                order.setdefault(text, None)
                # Written?
                pprev = toks[k - 1] if k > 0 else None
                # ``p[...]`` / ``p->f``: an access through ``p``, not of ``p``.
                through = k + 1 < n and toks[k + 1][1] in ("[", "->")
                if pprev is not None and pprev[1] in ("++", "--"):
                    mark(text, through)
                elif pprev is not None and pprev[1] == "&":
                    pp2 = toks[k - 2] if k > 1 else None
                    if pp2 is None or (pp2[0] == "op" and pp2[1] not in (")", "]")):
                        mark(text, through)
                j = k + 1
                member_call = False
                while j < n:
                    t = toks[j][1]
                    if t in (".", "->") and j + 1 < n and toks[j + 1][0] == "id":
                        if j + 2 < n and toks[j + 2][1] == "(":
                            member_call = True
                        j += 2
                        continue
                    if t == "[":
                        lvl = 0
                        while j < n:
                            if toks[j][1] == "[":
                                lvl += 1
                            elif toks[j][1] == "]":
                                lvl -= 1
                                if lvl == 0:
                                    break
                            j += 1
                        j += 1
                        continue
                    break
                nxt = toks[j][1] if j < n else None
                if member_call or nxt in _ASSIGN_OPS or nxt in ("++", "--"):
                    mark(text, through and not member_call)
                elif calls and pprev is not None and pprev[1] in ("(", ",") and nxt in (",", ")"):
                    # A whole-name argument (``j == k + 1``) to a by-reference
                    # parameter may change the name itself.
                    callee, argi = calls[-1]
                    if callee is not None:
                        pos = helper_ref_pos.get(callee)
                        if pos is not None:
                            if argi < len(pos) and pos[argi]:
                                mark(text, through)
                        elif callee in ref_funcs:
                            mark(text, through)
                continue
            if text == "(":
                callee = toks[k - 1][1] if k > 0 and toks[k - 1][0] == "id" else None
                if callee is None and k > 0 and toks[k - 1][1] == ">":
                    # ``f<3>(...)``: find the template's name.
                    lvl = 0
                    m = k - 1
                    while m >= 0:
                        if toks[m][1] == ">":
                            lvl += 1
                        elif toks[m][1] == "<":
                            lvl -= 1
                            if lvl == 0:
                                break
                        m -= 1
                    if m > 0 and toks[m - 1][0] == "id":
                        callee = toks[m - 1][1]
                calls.append([callee, 0])
            elif text == ")":
                if calls:
                    calls.pop()
            elif text == "," and calls:
                calls[-1][1] += 1
            elif text == "{":
                depth += 1
                brace_kinds.append(pending_loop and len(calls) <= (paren_depth_at_loop or 0))
                pending_loop = False
            elif text == "}":
                depth -= 1
                if depth < 0:
                    raise _Bail("unbalanced")
                brace_kinds.pop()
            elif text == ";" and pending_loop and not calls:
                # ``while (0);`` closing a do-loop, or a brace-less loop body.
                pending_loop = False
    if depth != 0:
        raise _Bail("unbalanced")
    return list(order), written, decls, direct


_ARRAY_DIM_RE = re.compile(r"\[[^\]]*\]")


def _param_decl(d: _Decl, pname: str, written: bool, addr: bool) -> tuple[str, bool]:
    """MSL parameter declaration for free variable ``d``; second item says
    whether it is a mutable reference."""
    t = d.type
    if d.param:
        # Kernel parameters (buffers, constant refs, builtins) by value.
        t = re.sub(r"\s*\[\[[^\]]*\]\]", "", t).strip()
        # ``constant T&`` scalar/vector params: by value, so a call with a
        # kernel argument and one with a local share one function.
        mref = re.match(r"^(?:const\s+)?constant\s+(?:const\s+)?(?P<v>\w+)\s*&$", t)
        if mref is not None and _SMALL_VALUE_RE.match(mref.group("v")):
            return f"const {mref.group('v')} {pname}", False
        return f"{t} {pname}", False
    if d.array_dims:
        space = "threadgroup" if "threadgroup" in t.split() else "thread"
        base = " ".join(w for w in t.split() if w not in ("thread", "threadgroup", "static"))
        if not written and not addr and not base.startswith("const "):
            base = "const " + base
        return f"{space} {base} (&{pname}){d.array_dims}", written or addr
    if "&" in t:
        # A reference local: pass the reference itself.
        return f"{t} {pname}", True
    mutable = (written or addr) and not d.const
    if mutable:
        if "*" in t:
            return f"{t} thread & {pname}", True
        return f"thread {t}& {pname}", True
    if "*" in t:
        return f"{t} const {pname}", False
    if _SMALL_VALUE_RE.match(t):
        return f"const {_strip_const(t)} {pname}", False
    return f"thread const {_strip_const(t)}& {pname}", False


_IDENT_SUB_RE = re.compile(r"(?<![\w.])(?<!->)(?<!::)([A-Za-z_]\w*)")


def _rename(line: str, ren: dict[str, str]) -> str:
    if not ren:
        return line
    return _IDENT_SUB_RE.sub(lambda m: ren.get(m.group(1), m.group(1)), line)


_HEADER_FN_RE = re.compile(r"^[^\n;{}()#]*?\b([A-Za-z_]\w*)\s*\(([^()]*)\)\s*(?:const\s*)?\{", re.M)


def _ref_taking_functions(header: str) -> set[str]:
    out: set[str] = set()
    for m in _HEADER_FN_RE.finditer(header):
        params = m.group(2)
        if re.search(r"(?<!&)&(?!&)", params):
            out.add(m.group(1))
    return out


_REF_FUNCS_CACHE: dict[int, set[str]] = {}


def _parse_params(params: list[str]) -> list[_Decl]:
    out = []
    for p in params:
        s = re.sub(r"\s*\[\[[^\]]*\]\]", "", p).strip()
        m = re.match(r"^(?P<type>.*?[\s*&])(?P<name>[A-Za-z_]\w*)$", s)
        if m is None:
            continue
        out.append(_Decl(m.group("name"), " ".join(m.group("type").split()), -1, param=True))
    return out


def outline_kernel(kernel_name: str, params: list[str], prologue: str, body: str, header: str) -> tuple[str, str]:
    """Outline marked regions of a translated kernel.

    Returns ``(new_body, functions)``; ``functions`` goes between the header
    and the kernel. With no markers (or outlining off) the body comes back
    unchanged apart from marker removal.
    """
    if MARKER_PREFIX not in body:
        return body, ""
    m = mode()
    if m == "off":
        return strip_markers(body), ""
    pro_lines = prologue.split("\n") if prologue else []
    body_lines = body.split("\n")
    lines = pro_lines + body_lines
    n_pro = len(pro_lines)
    n_code = sum(1 for ln in body_lines if ln.strip() and not ln.startswith(MARKER_PREFIX))
    if m == "auto" and n_code < _AUTO_MIN_KERNEL_LINES:
        return strip_markers(body), ""
    min_lines = _AUTO_MIN_REGION_LINES if m == "auto" else _ALL_MIN_REGION_LINES

    scan = _Scan(lines, _parse_params(params))
    if not scan.ok:
        return strip_markers(body), ""
    hkey = hash(header)
    ref_funcs = _REF_FUNCS_CACHE.get(hkey)
    if ref_funcs is None:
        ref_funcs = _ref_taking_functions(header)
        if len(_REF_FUNCS_CACHE) > 64:
            _REF_FUNCS_CACHE.clear()
        _REF_FUNCS_CACHE[hkey] = ref_funcs

    cur: list[str | None] = list(lines)
    functions: dict[str, str] = {}
    helper_ref_pos: dict[str, tuple[bool, ...]] = {}
    noinline = _noinline_attr()
    stats = {"regions": 0, "outlined": 0, "bailed": 0}
    bail_reasons: list[str] = []

    def bail(r: _Region, why: str) -> None:
        stats["bailed"] += 1
        bail_reasons.append(f"{r.key[:50]} ({r.end - r.begin} lines): {why}")

    def process(r: _Region) -> None:
        for c in r.children:
            process(c)
        stats["regions"] += 1
        text = [cur[i] for i in range(r.begin + 1, r.end) if cur[i] is not None]
        cur[r.begin] = None
        cur[r.end] = None
        n_text = sum(1 for ln in text if ln.strip() and not ln.startswith(MARKER_PREFIX))
        if n_text < min_lines:
            return
        try:
            ids, written, rdecls, direct = _analyze_tokens(text, ref_funcs, helper_ref_pos)
        except _Bail as e:
            bail(r, str(e))
            return
        free: list[_Decl] = []
        moved: list[_Decl] = []
        for name in ids:
            if name in rdecls:
                # Declared inside: it must not be used after the region.
                if (
                    scan.total(name) != scan.count_in(name, r.begin, r.end)
                    and _resolve(r.scopes, name, r.begin) is None
                ):
                    bail(r, f"local {name} used after region")
                    return
                continue
            d = _resolve(r.scopes, name, r.begin)
            if d is None:
                continue  # global (helper, type, constant) or a nested helper call
            if (
                not d.param
                and d.movable_init
                and cur[d.line] is not None
                and name.startswith("var_")
                and "__" in name
                and name not in scan.addr_taken
                and scan.total(name) == scan.count_in(name, r.begin, r.end) + 1
                and scan.count_in(name, d.line, d.line) == 1
            ):
                moved.append(d)
                continue
            if d.param and (name in direct or (name in written and "*" not in d.type)):
                # Pointer parameters are passed by value: storing through
                # one is fine, changing the pointer itself is not.
                bail(r, f"writes kernel param {name}")
                return
            free.append(d)
        ren: dict[str, str] = {}
        sig: list[str] = []
        ref_pos: list[bool] = []
        for k, d in enumerate(free):
            pname = f"p{k}"
            ren[d.name] = pname
            decl, is_ref = _param_decl(d, pname, d.name in written, d.name in scan.addr_taken)
            sig.append(decl)
            ref_pos.append(is_ref)
        local_names = [nm for nm in ids if nm in rdecls] + [d.name for d in moved]
        seen = set(ren)
        k = 0
        for nm in ids:
            if nm in seen:
                continue
            if nm in rdecls or any(d.name == nm for d in moved):
                ren[nm] = f"l{k}"
                k += 1
                seen.add(nm)
        del local_names
        body_out = [_rename(cur[d.line], ren) for d in moved]
        body_out += [_rename(ln, ren) for ln in text if not ln.startswith(MARKER_PREFIX)]
        canon = "\n".join(body_out)
        params_txt = ", ".join(sig)
        h = hashlib.sha1((params_txt + "\n" + canon).encode()).hexdigest()[:16]
        fname = f"wp_outl_{h}"
        if fname not in functions:
            short = re.sub(r"[^\w]", "_", r.key)[:60]
            functions[fname] = f"// {short}\n{noinline}void {fname}({params_txt}) {{\n{canon}\n}}\n"
            helper_ref_pos[fname] = tuple(ref_pos)
        for i in range(r.begin + 1, r.end):
            cur[i] = None
        for d in moved:
            cur[d.line] = None
        cur[r.begin] = f"    {fname}({', '.join(d.name for d in free)});"
        stats["outlined"] += 1

    for r in scan.roots:
        process(r)
    if not functions:
        return strip_markers(body), ""
    new_lines = [ln for ln in cur[n_pro:] if ln is not None and not ln.startswith(MARKER_PREFIX)]
    # A moved declaration may have come from the prologue; keep the prologue
    # verbatim otherwise.
    if os.environ.get("WARP_METAL_OUTLINE_VERBOSE"):
        print(
            f"[warp-metal-outline] {kernel_name}: {stats['outlined']}/{stats['regions']} regions outlined "
            f"({stats['bailed']} kept inline), {len(functions)} functions",
            flush=True,
        )
        if os.environ.get("WARP_METAL_OUTLINE_VERBOSE") == "2":
            for b in bail_reasons:
                print(f"[warp-metal-outline]   kept inline: {b}", flush=True)
    return "\n".join(new_lines), "\n".join(functions.values())
