# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serial-model Metal lowerings for tile ops not covered by the core translator.

``codegen_metal`` runs a launch-tiled kernel as one thread per block
(``wp.block_dim() == 1``), with every tile held in a private, row-major
``wp_tile_RxC_<scalar>`` struct (``c[]``); single-element tiles collapse to a
plain scalar local. That is exactly Warp's CPU tile model, so each lowering
here reproduces the CPU (``block_dim=1``) semantics of ``warp/native/tile*.h``.

Covered here:

- ``tile_scatter_add`` / ``tile_scatter_masked`` (per-thread writes into a
  tile; with one thread they are plain guarded element updates),
- ``tile_load_indexed`` (1-D and 2-D),
- ``tile_reshape`` and ``tile_view`` (aliasing views, see below),
- ``tile_assign`` with an offset,
- ``tile_stack`` / ``tile_stack_push`` / ``pop`` / ``count`` / ``clear``,
- ``tile_mesh_query_aabb`` / ``tile_bvh_query_aabb`` / ``tile_bvh_query_ray``
  with ``tile_query_valid`` and the matching ``_next``,
- the reverse of ``tile_map`` / ``tile_binary_map`` over builtin functions,
- tiles whose shape has more than two dims with only leading unit dims
  (``shape=(1, R, C)``), which collapse to the ``(R, C)`` struct.

Views. On CPU/CUDA ``tile_view`` and ``tile_reshape`` return a view over the
parent's storage. The private-struct model materializes them as copies, so the
aliasing is restored explicitly: after a statement that mutates a view (or a
view of a view) its contents are written back to the parent chain, and before
a statement that reads a view whose parent can be mutated, the view is
refreshed from the parent. Both copies are plain element loops.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Tile shapes
# --------------------------------------------------------------------------

_SHAPE_T_PAT = re.compile(r"wp::tile_shape_t<\s*([\d\s,]*)>")


def full_tile_shape(ctype: str) -> tuple[int, ...] | None:
    """All dims of the first ``tile_shape_t<...>`` in ``ctype``."""
    m = _SHAPE_T_PAT.search(ctype)
    if m is None:
        return None
    dims = tuple(int(d) for d in m.group(1).split(",") if d.strip())
    return dims if dims else None


def collapse_shape(dims: tuple[int, ...], what: str = "tile") -> tuple[int, int]:
    """``(rows, cols)`` of the private struct holding a tile of shape ``dims``.

    1-D ``(N,)`` is ``(N, 1)``; 2-D is itself; higher ranks are accepted only
    when every leading dim is 1 (``(1, R, C)``), which is the same row-major
    storage as ``(R, C)``.
    """
    from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

    if len(dims) == 1:
        return dims[0], 1
    if len(dims) == 2:
        return dims[0], dims[1]
    lead = dims[:-2]
    if any(d != 1 for d in lead):
        raise MetalCodegenError(
            f"MSL codegen: {what} of shape {dims} is not supported on Metal "
            "(tiles with more than two dims need all leading dims to be 1)"
        )
    return dims[-2], dims[-1]


# Template shape lists in ``wp::tile_load<T, bc, al, d0, d1, d2>`` and the
# fill constructors: drop leading unit dims so the 2-D lowering sees ``R, C``.
_TILE_TPL_CALL_PAT = re.compile(r"\bwp::(tile_load|tile_zeros|tile_ones|tile_full|tile_empty)\s*<")


def _match_angle(text: str, start: int) -> int:
    """Index of the ``>`` closing the ``<`` at ``start``."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _split_top(s: str, sep: str = ",") -> list[str]:
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail or out:
        out.append(tail)
    return out


def _normalize_tpl_shapes(line: str) -> str:
    if "wp::tile_" not in line:
        return line
    out = []
    pos = 0
    for m in _TILE_TPL_CALL_PAT.finditer(line):
        lt = m.end() - 1
        gt = _match_angle(line, lt)
        if gt < 0:
            continue
        args = _split_top(line[lt + 1 : gt])
        # Trailing integer args are the shape.
        n_shape = 0
        for a in reversed(args):
            if a.isdigit():
                n_shape += 1
            else:
                break
        if n_shape <= 2:
            continue
        shape = tuple(int(a) for a in args[len(args) - n_shape :])
        rows, cols = collapse_shape(shape, m.group(1))
        new_args = [*args[: len(args) - n_shape], str(rows), str(cols)]
        out.append(line[pos : lt + 1])
        out.append(", ".join(new_args))
        pos = gt
    if not out:
        return line
    out.append(line[pos:])
    return "".join(out)


# --------------------------------------------------------------------------
# Kernel-level bookkeeping
# --------------------------------------------------------------------------


def _lbl(tok: str) -> str:
    tok = tok.strip()
    for pfx in ("var_", "adj_"):
        if tok.startswith(pfx):
            return tok[len(pfx) :]
    return tok


@dataclass
class _View:
    parent: str
    # "window": a tile_view at ``offsets`` (one per parent dim); "flat": a
    # tile_reshape (same element order as the parent).
    kind: str
    offsets: list[str] = field(default_factory=list)


_VIEW_DEF_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_(view|reshape)\s*<")
_STMT_CALL_PAT = re.compile(r"\bwp::(\w+)\s*(?:<[^()]*>)?\s*\(([^;]*)\)")

# Ops that write their first tile argument in place.
_MUTATE_FIRST = {
    "tile_add_inplace",
    "tile_sub_inplace",
    "tile_mul_inplace",
    "tile_div_inplace",
    "tile_bit_and_inplace",
    "tile_bit_or_inplace",
    "tile_bit_xor_inplace",
    "tile_assign",
    "assign",
    "tile_scatter_add",
    "tile_scatter_masked",
    "tile_sort",
}


def _mutated_labels(line: str) -> set[str]:
    """Tile labels a statement writes in place (not counting ``var_X =``)."""
    out: set[str] = set()
    for m in _STMT_CALL_PAT.finditer(line):
        op = m.group(1)
        args = _split_top(m.group(2))
        if not args:
            continue
        if op in _MUTATE_FIRST:
            out.add(_lbl(args[0]))
            if op == "tile_sort" and len(args) > 1:
                out.add(_lbl(args[1]))
        elif op in ("tile_matmul_acc", "tile_matmul") and len(args) >= 6:
            out.add(_lbl(args[5]))
        elif op.startswith("tile_cholesky_inplace") and len(args) >= 2:
            out.add(_lbl(args[1]))
        elif op == "tile_cholesky_solve_inplace" and len(args) >= 3:
            out.add(_lbl(args[2]))
    for m in re.finditer(r"\btile_(?:lower|upper)_solve_inplace\s*\(([^;]*)\)", line):
        args = _split_top(m.group(1))
        if len(args) >= 3:
            out.add(_lbl(args[2]))
    for m in re.finditer(r"\btile_i?fft\s*\(([^;]*)\)", line):
        args = _split_top(m.group(1))
        if args:
            out.add(_lbl(args[-1]))
    return out


class TileExt:
    """Per-kernel state for the extra tile lowerings (see module docstring)."""

    def __init__(
        self,
        lines: list[str],
        tile_var_dims: dict[str, tuple[int, int, str]],
        tile_var_vec_n: dict[str, int],
        var_ctypes: dict[str, str],
        view_aliases: dict | None = None,
    ):
        self.dims = tile_var_dims
        self.vec_n = tile_var_vec_n
        self.ctypes = var_ctypes
        self.view_aliases = view_aliases
        self.views: dict[str, _View] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        for label, ct in var_ctypes.items():
            if "tile_shape_t" in ct and ("tile_shared_t" in ct or "tile_register_t" in ct):
                shp = full_tile_shape(ct)
                if shp is not None:
                    self.shapes[label] = shp
        mutated: set[str] = set()
        for raw in lines:
            m = _VIEW_DEF_PAT.search(raw)
            if m is not None:
                lhs, kind = m.group(1), m.group(2)
                call = raw[m.start() :]
                lt = call.index("<")
                gt = _match_angle(call, lt)
                rest = call[gt + 1 :]
                paren = rest.find("(")
                close = rest.rfind(")")
                args = _split_top(rest[paren + 1 : close]) if paren >= 0 and close > paren else []
                if args:
                    parent = _lbl(args[0])
                    if kind == "view":
                        self.views[lhs] = _View(parent, "window", args[1:])
                    else:
                        self.views[lhs] = _View(parent, "flat")
            mutated |= _mutated_labels(raw)
        # A parent is "volatile" when it (or another view of it) can change
        # after a view of it was taken; views of volatile parents refresh
        # before every use.
        self.volatile: set[str] = set()
        for lbl in mutated:
            node = lbl
            self.volatile.add(node)
            seen = set()
            while node in self.views and node not in seen:
                seen.add(node)
                node = self.views[node].parent
                self.volatile.add(node)
        self._label_pat = re.compile(r"\b(?:var|adj)_(\w+)\b")

    def rank2_labels(self) -> set[str]:
        """Tile locals declared with two or more dims (``(R, 1)`` included)."""
        return {label for label, shp in self.shapes.items() if len(shp) >= 2}

    # ------------------------------------------------------------------
    # element addressing
    # ------------------------------------------------------------------
    def _shape(self, label: str) -> tuple[int, ...] | None:
        shp = self.shapes.get(label)
        if shp is not None:
            return shp
        d = self.dims.get(label)
        if d is None:
            return None
        return (d[0], d[1]) if d[1] > 1 else (d[0],)

    def _n(self, label: str) -> int:
        d = self.dims.get(label)
        return 1 if d is None else d[0] * d[1]

    def _elem(self, var: str, label: str, flat: str, comp: int | None = None) -> str:
        """Scalar slot ``flat`` (element index) of a tile local."""
        vn = self.vec_n.get(label, 0)
        if self._n(label) == 1:
            base = var
            return base if not vn else f"{base}[{comp}]"
        if vn:
            return f"{var}.c[({flat}) * {vn} + {comp}]"
        return f"{var}.c[{flat}]"

    def _copy_loop(self, dst_var: str, dst_label: str, src_var: str, src_label: str, n: int, map_src) -> str:
        """``dst[k] = src[map_src(k)]`` for every element ``k < n``."""
        vn = self.vec_n.get(dst_label, 0)
        comps = list(range(vn)) if vn else [None]
        body = " ".join(
            f"{self._elem(dst_var, dst_label, '_tv_k', c)} = {self._elem(src_var, src_label, map_src('_tv_k'), c)};"
            for c in comps
        )
        return f"for (int _tv_k = 0; _tv_k < {n}; ++_tv_k) {{ {body} }}"

    def _window_map(self, view: str) -> tuple[callable, callable] | None:
        """(view_flat -> parent_flat expression builder, n) for a window view."""
        v = self.views[view]
        pshape = self._shape(v.parent)
        vshape = self._shape(view)
        if pshape is None or vshape is None:
            return None
        offs = v.offsets
        pd = len(offs) if offs else len(pshape)
        # Parent offsets address the parent's full rank; drop unit leading
        # dims beyond what the materialized struct keeps.
        pshape_full = pshape
        if len(pshape_full) < pd:
            pshape_full = (1,) * (pd - len(pshape_full)) + tuple(pshape_full)
        vd = len(vshape)

        def build(k: str) -> str:
            # view coordinates from the flat index (row-major)
            coords: list[str] = []
            rem = k
            for i in range(vd):
                stride = math.prod(vshape[i + 1 :])
                coords.append(f"(({rem}) / {stride})" if stride != 1 else f"({rem})")
                rem = f"(({rem}) % {stride})" if stride != 1 else rem
            # right-align into the parent rank
            pc: list[str] = []
            for d in range(pd):
                vi = d - (pd - vd)
                off = offs[d] if d < len(offs) else "0"
                pc.append(f"(({off}) + {coords[vi]})" if vi >= 0 else f"({off})")
            terms = []
            for d in range(pd):
                stride = math.prod(pshape_full[d + 1 :])
                terms.append(f"{pc[d]} * {stride}" if stride != 1 else pc[d])
            return " + ".join(terms)

        return build, math.prod(vshape)

    def _refresh(self, view: str) -> str:
        v = self.views[view]
        n = self._n(view)
        if v.kind == "flat":
            return self._copy_loop(f"var_{view}", view, f"var_{v.parent}", v.parent, n, lambda k: k)
        wm = self._window_map(view)
        if wm is None:
            return ""
        build, n = wm
        return self._copy_loop(f"var_{view}", view, f"var_{v.parent}", v.parent, n, build)

    def _writeback(self, view: str) -> str:
        """Copy a mutated view into its parent chain."""
        out: list[str] = []
        node = view
        seen: set[str] = set()
        while node in self.views and node not in seen:
            seen.add(node)
            v = self.views[node]
            n = self._n(node)
            vn = self.vec_n.get(node, 0)
            comps = list(range(vn)) if vn else [None]
            if v.kind == "flat":

                def build(k: str) -> str:
                    return k

            else:
                wm = self._window_map(node)
                if wm is None:
                    break
                build, n = wm
            body = " ".join(
                f"{self._elem(f'var_{v.parent}', v.parent, build('_tw_k'), c)} = "
                f"{self._elem(f'var_{node}', node, '_tw_k', c)};"
                for c in comps
            )
            out.append(f"for (int _tw_k = 0; _tw_k < {n}; ++_tw_k) {{ {body} }}")
            node = v.parent
        return " ".join(out)

    def _view_chain_volatile(self, view: str) -> bool:
        node = view
        seen: set[str] = set()
        while node in self.views and node not in seen:
            seen.add(node)
            if self.views[node].parent in self.volatile:
                return True
            node = self.views[node].parent
        return False

    def _refresh_chain(self, view: str) -> str:
        chain: list[str] = []
        node = view
        seen: set[str] = set()
        while node in self.views and node not in seen:
            seen.add(node)
            chain.append(node)
            node = self.views[node].parent
        return " ".join(self._refresh(v) for v in reversed(chain))

    # ------------------------------------------------------------------
    # per-line entry point
    # ------------------------------------------------------------------
    def translate(self, line: str) -> str:
        line = _normalize_tpl_shapes(line)
        if "wp::" not in line and not self.views:
            return line
        m = _VIEW_DEF_PAT.search(line)
        if m is not None and m.group(1) in self.views:
            return self._lower_view_def(line, m)
        pre = ""
        post = ""
        if self.views:
            referenced = {mm.group(1) for mm in self._label_pat.finditer(line) if mm.group(0).startswith("var_")}
            views_here = [v for v in referenced if v in self.views]
            if views_here:
                pre = " ".join(self._refresh_chain(v) for v in views_here if self._view_chain_volatile(v))
                written = _mutated_labels(line)
                post = " ".join(self._writeback(v) for v in views_here if v in written)
        line = self._lower_ops(line)
        if pre or post:
            indent = line[: len(line) - len(line.lstrip())]
            body = line.strip()
            if body and not body.endswith((";", "{", "}")):
                body += ";"
            return f"{indent}{{ {pre} }} {body} {{ {post} }}" if post else f"{indent}{{ {pre} }} {body}"
        return line

    def _lower_view_def(self, line: str, m: re.Match[str]) -> str:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        lhs = m.group(1)
        v = self.views[lhs]
        if v.parent not in self.dims or lhs not in self.dims:
            raise MetalCodegenError(f"MSL codegen: tile_{m.group(2)} of an unknown tile (statement: {line.strip()!r})")
        if self._n(lhs) != self._n(v.parent) and v.kind == "flat":
            raise MetalCodegenError(f"MSL codegen: tile_reshape changes the element count ({line.strip()!r})")
        if self.vec_n.get(lhs, 0) != self.vec_n.get(v.parent, 0):
            raise MetalCodegenError(f"MSL codegen: tile view changes the element type ({line.strip()!r})")
        # Keep the core translator's matmul / solve writeback informed about
        # 2-D window views of 2-D parents.
        if self.view_aliases is not None and v.kind == "window" and len(v.offsets) == 2:
            prows, pcols, scalar = self.dims[v.parent]
            rows, cols, _ = self.dims[lhs]
            if not self.vec_n.get(lhs, 0) and rows * cols > 1 and prows * pcols > 1:
                self.view_aliases[lhs] = (v.parent, prows, pcols, rows, cols, scalar, v.offsets[0], v.offsets[1])
        indent = line[: len(line) - len(line.lstrip())]
        return f"{indent}{{ {self._refresh_chain(lhs)} }}"

    # ------------------------------------------------------------------
    # op lowerings
    # ------------------------------------------------------------------
    def _lower_ops(self, line: str) -> str:
        if "wp::" not in line:
            return line
        if "tile_zeros" in line or "tile_ones" in line or "tile_full" in line:
            line = _VEC_FILL_PAT.sub(self._repl_vec_fill, line)
        if "tile_scatter_" in line:
            line = _SCATTER_PAT.sub(self._repl_scatter, line)
        if "tile_load_indexed" in line:
            line = _LOAD_INDEXED_PAT.sub(self._repl_load_indexed, line)
        if "wp::tile_assign" in line:
            line = _ASSIGN_PAT.sub(self._repl_assign, line)
        if "tile_stack" in line:
            line = self._lower_stack(line)
        if "tile_mesh_query" in line or "tile_bvh_query" in line or "tile_query_valid" in line:
            line = self._lower_tile_query(line)
        if "adj_tile_binary_map" in line or "adj_tile_unary_map" in line or "adj_tile_map" in line:
            line = _ADJ_MAP_PAT.sub(self._repl_adj_map, line)
        return line

    def _repl_vec_fill(self, m: re.Match[str]) -> str:
        """``tile_zeros`` / ``tile_ones`` / ``tile_full`` of a vec-element tile.

        The core fill writes one scalar per element; a vec / quat /
        transform tile stores ``vn`` scalars per element.
        """
        lhs, op = m.group(1), m.group(2)
        vn = self.vec_n.get(lhs, 0)
        if not vn:
            return m.group(0)
        n = self._n(lhs)
        scalar = self.dims[lhs][2]
        if op == "full":
            value = m.group(4).strip()
            body = " ".join(f"{self._elem(f'var_{lhs}', lhs, '_tf_k', c)} = ({value})[{c}];" for c in range(vn))
        else:
            lit = "0" if op == "zeros" else "1"
            body = " ".join(f"{self._elem(f'var_{lhs}', lhs, '_tf_k', c)} = ({scalar}){lit};" for c in range(vn))
        return f"for (int _tf_k = 0; _tf_k < {n}; ++_tf_k) {{ {body} }}"

    def _repl_scatter(self, m: re.Match[str]) -> str:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        op = m.group(1)
        tpl = m.group(2) or ""
        args = _split_top(m.group(3))
        t = args[0]
        label = _lbl(t)
        shape = self._shape(label)
        if shape is None or label not in self.dims:
            raise MetalCodegenError(f"MSL codegen: tile_scatter_{op} on an unknown tile ({m.group(0)!r})")
        idx = args[1:-2]
        value, has = args[-2], args[-1]
        # Collapsed unit leading dims: drop the matching leading indices.
        while len(idx) > len(shape) and len(idx) > 1:
            idx = idx[1:]
        if len(idx) != len(shape):
            # 3-D shape collapsed into (R, C): use the trailing two indices.
            idx = idx[-len(shape) :]
        flat_terms = []
        bounds = []
        for d, ix in enumerate(idx):
            stride = math.prod(shape[d + 1 :])
            flat_terms.append(f"({ix}) * {stride}" if stride != 1 else f"({ix})")
            bounds.append(f"({ix}) >= 0 && ({ix}) < {shape[d]}")
        flat = " + ".join(flat_terms)
        del tpl
        vn = self.vec_n.get(label, 0)
        sym = "+=" if op == "add" else "="
        if vn:
            # Vec / quat / transform elements: component-wise through the
            # value's ``operator[]`` (native ``floatN`` and ``wp_vecN`` both
            # provide it).
            stmt = " ".join(f"{self._elem(t, label, flat, c)} {sym} ({value})[{c}];" for c in range(vn))
        else:
            stmt = f"{self._elem(t, label, flat)} {sym} {value};"
        return f"if (({has}) && {' && '.join(bounds)}) {{ {stmt} }}"

    def _repl_load_indexed(self, m: re.Match[str]) -> str:
        from warp._src.codegen_metal import MetalCodegenError, _build_flat_base_expr  # noqa: PLC0415

        lhs = m.group(1)
        args = _split_top(m.group(3))
        shape = self._shape(lhs)
        if shape is None or lhs not in self.dims:
            raise MetalCodegenError(f"MSL codegen: tile_load_indexed with unknown tile shapes ({m.group(0)!r})")
        if self.vec_n.get(lhs, 0):
            raise MetalCodegenError("MSL codegen: tile_load_indexed of vec elements is not supported on Metal")
        # ``(arr, lead_0..lead_k, indices, axis, off_0..off_{nd-1})``: one
        # offset per dim of the (viewed) array, which has the tile's rank;
        # leading slice indices come from folding ``arr[i]`` views.
        nd = len(shape)
        if nd not in (1, 2) or len(args) < 3 + nd:
            raise MetalCodegenError(
                f"MSL codegen: tile_load_indexed supports 1-D and 2-D tiles on Metal ({m.group(0)!r})"
            )
        arr = args[0]
        offs = args[len(args) - nd :]
        axis = args[len(args) - nd - 1]
        idx_t = args[len(args) - nd - 2]
        lead = args[1 : len(args) - nd - 2]
        idx_label = _lbl(idx_t)
        if idx_label not in self.dims:
            raise MetalCodegenError(f"MSL codegen: tile_load_indexed with an unknown index tile ({m.group(0)!r})")
        arr_name = _lbl(arr)
        base = _build_flat_base_expr(arr_name, lead, inner_dims=nd)
        nl = len(lead)
        n = math.prod(shape)
        dst = self._elem(f"var_{lhs}", lhs, "_li_k")
        zero = f"({self.dims[lhs][2]})0"

        def idx_at(c: str) -> str:
            return self._elem(idx_t, idx_label, c)

        if nd == 1:
            r = f"(({offs[0]}) + (({axis}) == 0 ? (int)({idx_at('_li_k')}) : _li_k))"
            return (
                f"for (int _li_k = 0; _li_k < {n}; ++_li_k) {{ const int _li_r = {r}; "
                f"{dst} = (_li_r >= 0 && _li_r < (int){arr_name}_shape[{nl}]) ? {arr}[{base} + _li_r] : {zero}; }}"
            )
        cols = shape[1]
        return (
            f"for (int _li_k = 0; _li_k < {n}; ++_li_k) {{ const int _li_i = _li_k / {cols}; "
            f"const int _li_j = _li_k % {cols}; "
            f"const int _li_r = ({offs[0]}) + (({axis}) == 0 ? (int)({idx_at('_li_i')}) : _li_i); "
            f"const int _li_c = ({offs[1]}) + (({axis}) == 1 ? (int)({idx_at('_li_j')}) : _li_j); "
            f"{dst} = (_li_r >= 0 && _li_r < (int){arr_name}_shape[{nl}] && _li_c >= 0 && "
            f"_li_c < (int){arr_name}_shape[{nl + 1}]) ? "
            f"{arr}[{base} + _li_r * (int){arr_name}_shape[{nl + 1}] + _li_c] : {zero}; }}"
        )

    def _repl_assign(self, m: re.Match[str]) -> str:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        args = _split_top(m.group(1))
        if len(args) < 2:
            return m.group(0)
        dst, src = args[0], args[1]
        offs = args[2:]
        dl, sl = _lbl(dst), _lbl(src)
        dshape, sshape = self._shape(dl), self._shape(sl)
        if dshape is None or sshape is None:
            return m.group(0)
        if self.vec_n.get(dl, 0) != self.vec_n.get(sl, 0):
            raise MetalCodegenError(f"MSL codegen: tile_assign between element types ({m.group(0)!r})")
        if not offs:
            offs = ["0"] * len(dshape)
        pd = len(offs)
        dfull = tuple(dshape)
        if len(dfull) < pd:
            dfull = (1,) * (pd - len(dfull)) + dfull
        sd = len(sshape)
        n = math.prod(sshape)
        if n == 1 and math.prod(dshape) == 1:
            return f"{dst} = {src}"

        def build(k: str) -> str:
            coords = []
            rem = k
            for i in range(sd):
                stride = math.prod(sshape[i + 1 :])
                coords.append(f"(({rem}) / {stride})" if stride != 1 else f"({rem})")
                rem = f"(({rem}) % {stride})" if stride != 1 else rem
            terms = []
            for d in range(pd):
                si = d - (pd - sd)
                pc = f"(({offs[d]}) + {coords[si]})" if si >= 0 else f"({offs[d]})"
                stride = math.prod(dfull[d + 1 :])
                terms.append(f"{pc} * {stride}" if stride != 1 else pc)
            return " + ".join(terms)

        vn = self.vec_n.get(dl, 0)
        comps = list(range(vn)) if vn else [None]
        body = " ".join(
            f"{self._elem(dst, dl, build('_ta_k'), c)} = {self._elem(src, sl, '_ta_k', c)};" for c in comps
        )
        return f"for (int _ta_k = 0; _ta_k < {n}; ++_ta_k) {{ {body} }}"

    # -- tile_stack -----------------------------------------------------
    def _lower_stack(self, line: str) -> str:
        line = _STACK_INIT_PAT.sub(r"wp_tile_stack_reset(var_\1)", line)
        line = re.sub(r"\bwp::tile_stack_(push|pop|clear|count)\b", r"wp_tile_stack_\1", line)
        return line

    # -- tile queries ---------------------------------------------------
    def _lower_tile_query(self, line: str) -> str:
        line = re.sub(r"\bwp::tile_mesh_query_aabb\s*\(", "wp_bvh_tile_mesh_query_aabb(", line)
        line = re.sub(r"\bwp::tile_bvh_query_aabb\s*\(", "wp_bvh_tile_query_aabb(", line)
        line = re.sub(r"\bwp::tile_bvh_query_ray\s*\(", "wp_bvh_tile_query_ray(", line)
        line = re.sub(r"\bwp::tile_mesh_query_aabb_next\s*\(", "wp_bvh_tile_query_next(", line)
        line = re.sub(r"\bwp::tile_bvh_query_next\s*\(", "wp_bvh_tile_query_next(", line)
        line = re.sub(r"\bwp::tile_query_valid\s*\(", "wp_bvh_tile_query_valid(", line)
        return line

    # -- reverse tile_map over builtins --------------------------------
    def _repl_adj_map(self, m: re.Match[str]) -> str:
        from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

        args = _split_top(m.group(2))
        # adj_tile_*map(fn, t0..tk, adj_fn, adj_t0..adj_tk, adj_ret)
        if len(args) < 4 or len(args) % 2 != 1:
            return m.group(0)
        k = (len(args) - 3) // 2
        fn = args[0]
        ins = args[1 : 1 + k]
        adj_fn = args[1 + k]
        adj_ins = args[2 + k : 2 + 2 * k]
        adj_ret = args[-1]
        if not (fn.startswith(("wp::", "wp_", "metal::"))):
            return m.group(0)
        ret_label = _lbl(adj_ret)
        n = self._n(ret_label)
        tiled = [_lbl(a) in self.dims for a in ins]
        for a in [*ins, adj_ret]:
            if self.vec_n.get(_lbl(a), 0):
                raise MetalCodegenError(
                    f"MSL codegen: reverse tile_map over vec-element tiles is not supported ({m.group(0)!r})"
                )
        adj_name = adj_fn
        if adj_name.startswith("wp::adj_"):
            adj_name = "wp_adj_" + adj_name[len("wp::adj_") :]

        def el(var: str, is_tile: bool) -> str:
            if not is_tile or n == 1:
                return var
            return f"{var}.c[_am_k]"

        call_args = [el(a, t) for a, t in zip(ins, tiled, strict=True)]
        call_args += [el(a, t) for a, t in zip(adj_ins, tiled, strict=True)]
        call_args.append(el(adj_ret, True))
        call = f"{adj_name}({', '.join(call_args)})"
        if n == 1:
            return call
        return f"for (int _am_k = 0; _am_k < {n}; ++_am_k) {{ {call}; }}"


_VEC_FILL_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_(zeros|ones|full)\s*(<.*?>)\s*\(([^;]*?)\)(?=\s*;)")
_SCATTER_PAT = re.compile(r"\bwp::tile_scatter_(add|masked)\s*(<[^()]*>)?\s*\(([^;]*?)\)(?=\s*;)")
_LOAD_INDEXED_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_load_indexed\s*(<[^()]*>)?\s*\(([^;]*?)\)(?=\s*;)")
_ASSIGN_PAT = re.compile(r"\bwp::tile_assign\s*\(([^;]*?)\)(?=\s*;)")
_STACK_INIT_PAT = re.compile(r"\bvar_(\w+)\s*=\s*wp::tile_stack_init\s*<[^;]*?>\s*\(\s*\)")
_ADJ_MAP_PAT = re.compile(r"\bwp::adj_tile_(binary_map|unary_map|map)\s*(?:<[^()]*>)?\s*\(([^;]*?)\)(?=\s*;)")


# --------------------------------------------------------------------------
# Types and helpers
# --------------------------------------------------------------------------

_STACK_CTYPE_PAT = re.compile(r"^wp::tile_stack_t<\s*(.+)\s*,\s*(\d+)\s*>$")


def msl_var_type(ctype: str) -> str | None:
    """MSL type for the tile-side value types this module owns, else None."""
    ct = ctype.strip()
    m = _STACK_CTYPE_PAT.match(ct)
    if m is not None:
        from warp._src.codegen_metal import _msl_var_type  # noqa: PLC0415

        return f"wp_tile_stack<{_msl_var_type(m.group(1).strip())}, {m.group(2)}>"
    if ct in ("wp::mesh_query_aabb_thread_block_t", "wp::bvh_query_thread_block_t"):
        return "wp_bvh_tile_query_t"
    return None


_STACK_HELPERS = """
template <typename T, int Cap> struct wp_tile_stack { T data[Cap]; int count; };
template <typename T, int Cap> inline void wp_tile_stack_reset(thread wp_tile_stack<T, Cap>& s) {
    s.count = 0;
    for (int i = 0; i < Cap; ++i) s.data[i] = T{};
}
template <typename T, int Cap> inline int wp_tile_stack_push(thread wp_tile_stack<T, Cap>& s, T value, bool has_value) {
    int idx = -1;
    if (has_value) { idx = s.count; s.count += 1; if (idx < Cap) s.data[idx] = value; }
    if (s.count > Cap) s.count = Cap;
    return (has_value && idx >= 0 && idx < Cap) ? idx : -1;
}
template <typename T, int Cap> inline void wp_tile_stack_pop(thread wp_tile_stack<T, Cap>& s, thread T& out_value, thread int& out_slot) {
    int slot = s.count - 1;
    s.count -= 1;
    bool ok = slot >= 0 && slot < Cap;
    out_value = T{};
    out_slot = ok ? slot : -1;
    if (ok) out_value = s.data[slot];
    if (s.count < 0) s.count = 0;
}
template <typename T, int Cap> inline void wp_tile_stack_clear(thread wp_tile_stack<T, Cap>& s) { s.count = 0; }
template <typename T, int Cap> inline int wp_tile_stack_count(thread wp_tile_stack<T, Cap>& s) { return s.count; }
"""

_QUERY_HELPERS = """
struct wp_bvh_tile_query_t { wp_bvh_query_t q; bool last_valid; };
inline wp_bvh_tile_query_t wp_bvh_tile_mesh_query_aabb(ulong id, float3 lower, float3 upper) {
    wp_bvh_tile_query_t t; t.q = wp_mesh_query_aabb(id, lower, upper); t.last_valid = true; return t;
}
inline wp_bvh_tile_query_t wp_bvh_tile_query_aabb(ulong id, float3 lower, float3 upper) {
    wp_bvh_tile_query_t t; t.q = wp_bvh_query_aabb(id, lower, upper, -1); t.last_valid = true; return t;
}
inline wp_bvh_tile_query_t wp_bvh_tile_query_ray(ulong id, float3 start, float3 dir) {
    wp_bvh_tile_query_t t; t.q = wp_bvh_query_ray(id, start, dir, -1); t.last_valid = true; return t;
}
inline int wp_bvh_tile_query_next(thread wp_bvh_tile_query_t& t) {
    int index = -1;
    if (!wp_bvh_query_next(t.q, index, FLT_MAX)) index = -1;
    t.last_valid = index >= 0;
    return index;
}
inline bool wp_bvh_tile_query_valid(thread const wp_bvh_tile_query_t& t) { return t.last_valid; }
"""


def emit_helpers(source: str) -> str:
    parts = []
    native = emit_native_funcs(source)
    if native:
        parts.append(native)
    if "wp_tile_stack" in source:
        parts.append(_STACK_HELPERS)
    if "wp_bvh_tile_" in source:
        parts.append(_QUERY_HELPERS)
    return "\n".join(parts)


def source_uses_bvh(source: str) -> bool:
    return "wp_bvh_tile_" in source


_TILE_SEMANTICS_PAT = re.compile(r"\bwp::(?:adj_)?(?:un)?tile\w*\b|\bbuiltin_block_dim\b|\bWP_TILE_")


def kernel_uses_tiles(adj, seen: set[int] | None = None) -> bool:
    """Whether a kernel (or a user function it calls) has block semantics.

    Such a kernel runs one thread per block on Metal, as on CPU with
    ``block_dim=1``, so the launcher must fold the trailing block dim that
    ``wp.launch_tiled`` appends. Deciding this from the *lowered* source is
    not enough: tiles of one element lower to plain scalars, and running the
    extra lanes would execute code CPU never runs (e.g. spinning forever on
    a ``tile_scatter_masked(..., t == 0)`` progress counter only lane 0
    advances).
    """
    if seen is None:
        seen = set()
    if id(adj) in seen:
        return False
    seen.add(id(adj))
    for block in getattr(adj, "blocks", None) or ():
        if any(_TILE_SEMANTICS_PAT.search(line) for line in block.body_forward):
            return True
    for func in getattr(adj, "called_user_functions", None) or ():
        if "WP_TILE" in (getattr(func, "native_snippet", None) or ""):
            return True
        func_adj = getattr(func, "adj", None)
        if func_adj is not None and kernel_uses_tiles(func_adj, seen):
            return True
    return False


# --------------------------------------------------------------------------
# ``@wp.func_native`` snippets
# --------------------------------------------------------------------------
# The inliner splices ``@wp.func`` bodies, but a native function has no Warp
# body to splice; inlining its (empty) IR silently dropped the call. Native
# calls are instead renamed to a per-snippet MSL helper whose body is the
# snippet itself (compiled by the Metal compiler, so ``#if
# defined(__CUDA_ARCH__)`` takes the host branch, like on CPU), with a small
# compat prelude for the C math names CPU snippets use. Snippets that need
# C++-only Warp types fail to compile loudly instead of being dropped.

_NATIVE_FUNC_DEFS: dict[str, str] = {}

_NATIVE_PRELUDE = """
#ifndef WP_METAL_NATIVE_COMPAT
#define WP_METAL_NATIVE_COMPAT
inline float sqrtf(float x) { return metal::precise::sqrt(x); }
inline float rsqrtf(float x) { return metal::precise::rsqrt(x); }
inline float fabsf(float x) { return metal::fabs(x); }
inline float floorf(float x) { return metal::floor(x); }
inline float ceilf(float x) { return metal::ceil(x); }
inline float fminf(float a, float b) { return metal::fmin(a, b); }
inline float fmaxf(float a, float b) { return metal::fmax(a, b); }
inline float expf(float x) { return metal::precise::exp(x); }
inline float logf(float x) { return metal::precise::log(x); }
inline float sinf(float x) { return metal::precise::sin(x); }
inline float cosf(float x) { return metal::precise::cos(x); }
inline float tanf(float x) { return metal::precise::tan(x); }
inline float atan2f(float y, float x) { return metal::precise::atan2(y, x); }
inline float powf(float x, float y) { return metal::precise::pow(x, y); }
#define WP_TILE_SYNC() threadgroup_barrier(metal::mem_flags::mem_threadgroup)
#define WP_TILE_THREAD_IDX 0
#define WP_TILE_BLOCK_DIM 1
#define CUDA_CALLABLE
#endif
"""


def native_func_call_name(fn_overload) -> str | None:
    """MSL helper name for a ``@wp.func_native`` overload (None if not native).

    Registers the helper's definition for :func:`emit_helpers`.
    """
    import hashlib  # noqa: PLC0415

    snippet = getattr(fn_overload, "native_snippet", None)
    if snippet is None:
        return None
    from warp._src.codegen import is_tile  # noqa: PLC0415
    from warp._src.codegen_metal import MetalCodegenError, _is_array_arg_type, _msl_var_type  # noqa: PLC0415

    adj = fn_overload.adj
    params = []
    for arg in adj.args:
        if is_tile(arg.type) or _is_array_arg_type(arg.type):
            raise MetalCodegenError(
                f"@wp.func_native {fn_overload.key!r}: native snippets taking tile or array parameters "
                "are not supported on Metal (the snippet is C++/CUDA code operating on Warp's native types)"
            )
        params.append(f"{_msl_var_type(arg.ctype())} {arg.label}")
    ret_type = "void"
    rv = getattr(adj, "return_var", None)
    if rv is not None and len(rv) == 1:
        ret_type = _msl_var_type(rv[0].ctype())
    elif getattr(fn_overload, "value_type", None) is not None:
        from warp._src.codegen import Var  # noqa: PLC0415

        ret_type = _msl_var_type(Var.type_to_ctype(fn_overload.value_type))
    digest = hashlib.sha1((snippet + "|" + ",".join(params) + ret_type).encode()).hexdigest()[:10]
    name = f"wpn_{fn_overload.native_func}_{digest}"
    if name not in _NATIVE_FUNC_DEFS:
        # MSL references need an address space: bit-cast idioms such as
        # ``reinterpret_cast<uint32_t&>(f)`` on a local become ``thread``.
        body = re.sub(
            r"\b(reinterpret_cast|static_cast)\s*<\s*(?!thread\b|device\b|constant\b|threadgroup\b)([\w:]+)\s*&\s*>",
            r"\1<thread \2&>",
            snippet,
        )
        _NATIVE_FUNC_DEFS[name] = f"inline {ret_type} {name}({', '.join(params)}) {{\n{body}\n}}\n"
    return name


def emit_native_funcs(source: str) -> str:
    used = [n for n in _NATIVE_FUNC_DEFS if n in source]
    if not used:
        return ""
    return _NATIVE_PRELUDE + "\n".join(_NATIVE_FUNC_DEFS[n] for n in sorted(used))
