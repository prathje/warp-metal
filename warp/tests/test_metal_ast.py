# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Metal codegen AST parser/emitter (Phase 1 of the AST migration).

The parser must recognise every IR shape Warp produces in the kernels we
care about. We assert this in two ways:

- A handful of synthetic kernels covering the IR constructs we know about
  (assignment shapes, control flow, builtins, user calls). These run on
  every CI machine and don't need MLX or mujoco_warp.

- A bulk round-trip test against synthetic kernels broad enough to exercise
  most of the IR shape inventory, asserting that ``emit(parse(lines)) ==
  lines``. The full mujoco_warp recon lives in the integration test that
  needs the mujoco_warp checkout — it's not part of this unittest module.
"""

from __future__ import annotations

import unittest

import warp as wp
from warp._src.codegen_metal import (
    _preprocess_drop_unsupported_locals,
    _preprocess_for_loops,
    _preprocess_indexref_writes,
    _preprocess_views,
    _preprocess_while_loops,
    _vec_dtype_info,
)
from warp._src.codegen_metal_ast import (
    AddrOf,
    Assign,
    BlockClose,
    BlockElse,
    BlockOpen,
    Break,
    Builtin,
    Comment,
    Const,
    Empty,
    For,
    ForIterCmp,
    Goto,
    If,
    Label,
    MetalASTParseError,
    Pragma,
    Return,
    Tid,
    UserCall,
    Var,
    VoidCall,
    While,
    WhileCondBreak,
    WhileCondTest,
    emit,
    fold,
    fold_drop_unsupported_locals,
    fold_indexref_writes,
    fold_views,
    parse,
    parse_line,
)


def _kernel_ir(kernel) -> list[str]:
    """Build a kernel's IR (forward only) and return the body lines."""
    kernel.adj.build(builder=None, default_builder_options={"enable_backward": False})
    return kernel.adj.blocks[0].body_forward


class TestMetalASTParseShapes(unittest.TestCase):
    """Verify each IR shape parses to the right node kind."""

    def test_empty_line(self):
        self.assertIsInstance(parse_line(""), Empty)
        self.assertIsInstance(parse_line("   "), Empty)

    def test_comment(self):
        self.assertIsInstance(parse_line("// foo bar"), Comment)
        self.assertIsInstance(parse_line("    // indented"), Comment)

    def test_pragma_directive(self):
        self.assertIsInstance(parse_line('#line 42 "x.py"'), Pragma)

    def test_return(self):
        self.assertIsInstance(parse_line("return;"), Return)
        self.assertIsInstance(parse_line("    return ;"), Return)

    def test_block_open_simple_cond(self):
        n = parse_line("if (var_29) {")
        self.assertIsInstance(n, BlockOpen)
        self.assertEqual(n.cond, "var_29")

    def test_block_open_negated_cond(self):
        n = parse_line("if (!var_12) {")
        self.assertIsInstance(n, BlockOpen)
        self.assertEqual(n.cond, "!var_12")

    def test_block_else(self):
        self.assertIsInstance(parse_line("} else {"), BlockElse)

    def test_block_close(self):
        self.assertIsInstance(parse_line("}"), BlockClose)
        self.assertIsInstance(parse_line("    }"), BlockClose)

    def test_label(self):
        for name in ("start_for_0", "end_for_42", "start_while_3", "end_while_17"):
            n = parse_line(f"{name}:;")
            self.assertIsInstance(n, Label)
            self.assertEqual(n.name, name)

    def test_goto(self):
        n = parse_line("goto end_for_2;")
        self.assertIsInstance(n, Goto)
        self.assertEqual(n.target, "end_for_2")

    def test_for_iter_cmp(self):
        n = parse_line("if (iter_cmp(var_13) == 0) goto end_for_0;")
        self.assertIsInstance(n, ForIterCmp)
        self.assertEqual(n.iter_var, "13")
        self.assertEqual(n.end_label, "end_for_0")

    def test_while_cond(self):
        n = parse_line("if ((var_27) == false) goto end_while_2;")
        self.assertIsInstance(n, WhileCondTest)
        self.assertEqual(n.cond_var, "27")
        self.assertEqual(n.end_label, "end_while_2")

    def test_tid_arity(self):
        n = parse_line("builtin_tid1d(var_0);")
        self.assertIsInstance(n, Tid)
        self.assertEqual(n.arity, 1)
        self.assertEqual(n.targets, ("0",))
        n = parse_line("builtin_tid2d(var_0, var_1);")
        self.assertEqual(n.arity, 2)
        self.assertEqual(n.targets, ("0", "1"))
        n = parse_line("builtin_tid3d(var_0, var_1, var_2);")
        self.assertEqual(n.arity, 3)
        self.assertEqual(n.targets, ("0", "1", "2"))

    def test_assign_var_alias(self):
        n = parse_line("var_5 = var_8;")
        self.assertIsInstance(n, Assign)
        self.assertEqual(n.lhs, "5")
        self.assertIsInstance(n.expr, Var)
        self.assertEqual(n.expr.label, "8")

    def test_assign_const(self):
        for raw in ("var_3 = 0;", "var_4 = 1.5f;", "var_2 = true;", "var_5 = -42;"):
            n = parse_line(raw)
            self.assertIsInstance(n, Assign)
            self.assertIsInstance(n.expr, Const)

    def test_assign_builtin_call(self):
        n = parse_line("var_4 = wp::address(var_arr, var_0, var_1);")
        self.assertIsInstance(n, Assign)
        self.assertIsInstance(n.expr, Builtin)
        self.assertEqual(n.expr.name, "address")
        self.assertEqual(n.expr.args, ("var_arr", "var_0", "var_1"))

    def test_assign_builtin_with_template(self):
        n = parse_line("var_28 = wp::vec_t<10, wp::float32>();")
        self.assertIsInstance(n, Assign)
        self.assertIsInstance(n.expr, Builtin)
        self.assertEqual(n.expr.name, "vec_t")

    def test_assign_user_call(self):
        n = parse_line("var_115 = rot_vec_quat_0(var_110, var_116);")
        self.assertIsInstance(n, Assign)
        self.assertIsInstance(n.expr, UserCall)
        self.assertEqual(n.expr.name, "rot_vec_quat_0")
        self.assertEqual(n.expr.args, ("var_110", "var_116"))

    def test_assign_addrof_shape(self):
        n = parse_line("var_2 = &(var_arr.shape);")
        self.assertIsInstance(n, Assign)
        self.assertIsInstance(n.expr, AddrOf)
        self.assertEqual(n.expr.inner_kind, "shape")
        self.assertEqual(n.expr.inner_target, "arr")

    def test_assign_addrof_field_arrow(self):
        n = parse_line("var_18 = &(var_struct->field_x);")
        self.assertIsInstance(n.expr, AddrOf)
        self.assertEqual(n.expr.inner_kind, "field_arrow")
        self.assertEqual(n.expr.inner_target, "struct")
        self.assertEqual(n.expr.inner_field, "field_x")

    def test_assign_addrof_field_dot(self):
        n = parse_line("var_18 = &(var_struct.field_y);")
        self.assertIsInstance(n.expr, AddrOf)
        self.assertEqual(n.expr.inner_kind, "field_dot")
        self.assertEqual(n.expr.inner_field, "field_y")

    def test_void_array_store(self):
        n = parse_line("wp::array_store(var_xpos_out, var_0, var_16, var_44);")
        self.assertIsInstance(n, VoidCall)
        self.assertEqual(n.op, "array_store")
        self.assertEqual(n.args, ("var_xpos_out", "var_0", "var_16", "var_44"))

    def test_void_store(self):
        n = parse_line("wp::store(var_5, var_2);")
        self.assertEqual(n.op, "store")
        self.assertEqual(n.args, ("var_5", "var_2"))

    def test_void_inplace_three_arg(self):
        n = parse_line("wp::assign_inplace(var_28, var_36, var_35);")
        self.assertEqual(n.op, "assign_inplace")
        self.assertEqual(n.args, ("var_28", "var_36", "var_35"))

    def test_void_user_call(self):
        n = parse_line("normalize_with_norm_0(var_32, var_33, var_34);")
        self.assertIsInstance(n, VoidCall)
        self.assertEqual(n.op, "user_call")
        self.assertEqual(dict(n.extra)["name"], "normalize_with_norm_0")

    def test_void_printf(self):
        n = parse_line("printf(var_15, var_12);")
        self.assertEqual(n.op, "printf")
        self.assertEqual(n.args, ("var_15", "var_12"))

    def test_unknown_line_raises(self):
        with self.assertRaises(MetalASTParseError):
            parse_line("@@@ this is not Warp IR @@@")


class TestMetalASTRoundTrip(unittest.TestCase):
    """``emit(parse(lines)) == lines`` for every kernel we test against."""

    def _roundtrip(self, kernel):
        lines = _kernel_ir(kernel)
        nodes = parse(lines)
        out = emit(nodes)
        self.assertEqual(
            out,
            lines,
            f"round-trip mismatch on kernel {kernel.key!r}; "
            f"first divergence at index "
            f"{next((i for i, (a, b) in enumerate(zip(out, lines, strict=False)) if a != b), 'len-mismatch')}",
        )

    def test_simple_array_store(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.float32), b: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            b[tid] = a[tid] * 2.0

        self._roundtrip(k)

    def test_dynamic_for_loop(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(a[tid]):
                s += i
            out[tid] = s

        self._roundtrip(k)

    def test_two_arg_range(self):
        @wp.kernel
        def k(starts: wp.array(dtype=wp.int32), stops: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(starts[tid], stops[tid]):
                s += i
            out[tid] = s

        self._roundtrip(k)

    def test_while_loop_with_break(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            i = int(0)
            n = a[tid]
            while True:
                if i >= n:
                    break
                i += 1
            out[tid] = i

        self._roundtrip(k)

    def test_if_else(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            x = a[tid]
            if x > 0:
                out[tid] = x * 2
            else:
                out[tid] = -x

        self._roundtrip(k)

    def test_vec3_construct_extract(self):
        @wp.kernel
        def k(out: wp.array(dtype=wp.vec3)):
            tid = wp.tid()
            v = wp.vec3(1.0, 2.0, 3.0)
            v[0] = float(tid)
            out[tid] = v

        self._roundtrip(k)

    def test_atomic_add(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            wp.atomic_add(out, 0, a[tid])

        self._roundtrip(k)


class TestMetalASTFold(unittest.TestCase):
    """Verify the fold pass produces structured nodes equivalent in shape and
    output to running ``_preprocess_for_loops + _preprocess_while_loops``
    from :mod:`warp._src.codegen_metal`.
    """

    def _fold_ir(self, kernel) -> tuple[list, set[str], list[str]]:
        """Build IR, parse, fold; also return the existing-pipeline output
        for comparison."""
        lines = _kernel_ir(kernel)
        nodes = parse(lines)
        folded, skip = fold(nodes)

        # Existing pipeline output (the spec we must match).
        old_for, _ = _preprocess_for_loops(lines)
        old_lines = _preprocess_while_loops(old_for)
        return folded, skip, old_lines

    def _assert_equivalent(self, kernel):
        """Folded emit must equal the existing preprocessor output."""
        lines = _kernel_ir(kernel)
        nodes = parse(lines)
        folded, skip = fold(nodes)
        new_lines = emit(folded)

        old_for, old_skip = _preprocess_for_loops(lines)
        old_lines = _preprocess_while_loops(old_for)

        self.assertEqual(
            new_lines,
            old_lines,
            f"emit(fold(parse(...))) != preprocess(...) on kernel {kernel.key!r}",
        )
        self.assertEqual(skip, old_skip, f"skip set mismatch on kernel {kernel.key!r}")

    def test_dynamic_for_loop_one_arg_range(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(a[tid]):
                s += i
            out[tid] = s

        folded, _, _ = self._fold_ir(k)
        # The body should contain exactly one For node.
        for_nodes = [n for n in folded if isinstance(n, For)]
        self.assertEqual(len(for_nodes), 1)
        self.assertEqual(for_nodes[0].start, "0")
        self._assert_equivalent(k)

    def test_dynamic_for_loop_two_arg_range(self):
        @wp.kernel
        def k(starts: wp.array(dtype=wp.int32), stops: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(starts[tid], stops[tid]):
                s += i
            out[tid] = s

        folded, _, _ = self._fold_ir(k)
        for_nodes = [n for n in folded if isinstance(n, For)]
        self.assertEqual(len(for_nodes), 1)
        self.assertNotEqual(for_nodes[0].start, "0")  # start is the explicit var
        self._assert_equivalent(k)

    def test_while_loop(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            n = a[tid]
            s = int(0)
            while n > 0:
                s += n
                n -= 1
            out[tid] = s

        folded, _, _ = self._fold_ir(k)
        while_nodes = [n for n in folded if isinstance(n, While)]
        self.assertEqual(len(while_nodes), 1)
        # The body must contain a WhileCondBreak (the cond test was rewritten)
        # and no Goto referencing this loop's labels.
        body = while_nodes[0].body
        self.assertTrue(any(isinstance(n, WhileCondBreak) for n in body))
        self.assertFalse(
            any(isinstance(n, Goto) and ("while" in n.target) for n in body),
            "structured while body should contain no goto-to-while-labels",
        )
        self._assert_equivalent(k)

    def test_while_loop_with_break(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            i = int(0)
            n = a[tid]
            while True:
                if i >= n:
                    break
                i += 1
            out[tid] = i

        folded, _, _ = self._fold_ir(k)
        while_nodes = [n for n in folded if isinstance(n, While)]
        self.assertEqual(len(while_nodes), 1)

        # The break must have been folded into a Break node, somewhere in the
        # body or its nested If.
        def _has_break(nodes):
            for nn in nodes:
                if isinstance(nn, Break):
                    return True
                if isinstance(nn, If):
                    if _has_break(list(nn.body)):
                        return True
                    if nn.else_body is not None and _has_break(list(nn.else_body)):
                        return True
            return False

        self.assertTrue(_has_break(list(while_nodes[0].body)))
        self._assert_equivalent(k)

    def test_if_else_lowers_to_two_ifs(self):
        # Warp lowers Python ``if/else`` to two separate ``if`` statements
        # (the second on the negated condition), so we expect two If nodes
        # and never a structured else-arm.
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            x = a[tid]
            if x > 0:
                out[tid] = x * 2
            else:
                out[tid] = -x

        folded, _, _ = self._fold_ir(k)
        if_nodes = [n for n in folded if isinstance(n, If)]
        self.assertGreaterEqual(len(if_nodes), 2)
        self._assert_equivalent(k)

    def test_nested_for_in_while(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            n = a[tid]
            while n > 0:
                for j in range(n):
                    s += j
                n -= 1
            out[tid] = s

        self._assert_equivalent(k)


class TestMetalASTViewsFold(unittest.TestCase):
    """Verify ``fold_views`` produces output equivalent to running
    ``_preprocess_views`` on the existing for/while-preprocessed lines.
    """

    def _assert_views_equivalent(self, kernel):
        kernel.adj.build(builder=None, default_builder_options={"enable_backward": False})
        lines = kernel.adj.blocks[0].body_forward

        # New pipeline: parse -> fold (for/while/if) -> fold_views -> emit.
        nodes = parse(lines)
        folded, fold_skip = fold(nodes)
        view_folded, view_skip = fold_views(folded, kernel.adj)
        new_lines = emit(view_folded)

        # Existing pipeline: for/while preprocess, then views preprocess.
        old_for, old_for_skip = _preprocess_for_loops(lines)
        old_after_while = _preprocess_while_loops(old_for)
        old_lines, old_view_skip = _preprocess_views(old_after_while, kernel.adj)

        self.assertEqual(
            new_lines,
            old_lines,
            f"emit(fold_views(...)) != _preprocess_views(...) on kernel {kernel.key!r}",
        )
        # Combined skip-sets should match: for/while skip union view skip on each side.
        self.assertEqual(
            fold_skip | view_skip,
            old_for_skip | old_view_skip,
            f"combined skip set mismatch on kernel {kernel.key!r}",
        )

    def test_kernel_with_no_views(self):
        # Sanity: the pass should be a no-op (modulo the for-while fold)
        # when no slice/view ops are present.
        @wp.kernel
        def k(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            out[tid] = a[tid] * 2.0

        self._assert_views_equivalent(k)

    def test_arr2d_row_read(self):
        # ``arr2d[i]`` lowers to slice_t + view, which we fold into direct
        # ``arr[i, j]`` flattened addressing.
        @wp.kernel
        def k(a: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            row = a[tid]
            s = float(0.0)
            for j in range(a.shape[1]):
                s += row[j]
            out[tid] = s

        self._assert_views_equivalent(k)

    def test_arr2d_atomic_scatter_through_view(self):
        # ``wp.atomic_add(out2d[i], j, val)`` exercises the atomic flat-index
        # rewrite path.
        @wp.kernel
        def k(
            a: wp.array2d(dtype=wp.float32),
            cols: wp.array(dtype=wp.int32),
            out: wp.array2d(dtype=wp.float32),
        ):
            tid = wp.tid()
            row = a[tid]
            j = cols[tid]
            wp.atomic_add(out[tid], j, row[j] * 2.0)

        self._assert_views_equivalent(k)

    def test_arr2d_array_store_through_view(self):
        @wp.kernel
        def k(a: wp.array2d(dtype=wp.float32), out: wp.array2d(dtype=wp.float32)):
            tid = wp.tid()
            row_in = a[tid]
            row_out = out[tid]
            for j in range(a.shape[1]):
                row_out[j] = row_in[j] * 3.0 + 1.0

        self._assert_views_equivalent(k)

    def test_view_inside_for_loop(self):
        # The view definition itself can occur inside a for body — walker
        # must recurse into folded For nodes and not just the top level.
        @wp.kernel
        def k(a: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            s = float(0.0)
            for i in range(a.shape[0]):
                row = a[i]
                s += row[tid]
            out[tid] = s

        self._assert_views_equivalent(k)


class TestMetalASTIndexrefWritesFold(unittest.TestCase):
    """Verify ``fold_indexref_writes`` produces output equivalent to running
    ``_preprocess_indexref_writes`` on the existing pipeline output.
    """

    def _vec_arr_info(self, kernel) -> dict[str, tuple[int, str]]:
        info: dict[str, tuple[int, str]] = {}
        for arg in kernel.adj.args:
            v = _vec_dtype_info(arg)
            if v is not None:
                info[arg.label] = v
        return info

    def _assert_indexref_equivalent(self, kernel):
        kernel.adj.build(builder=None, default_builder_options={"enable_backward": False})
        lines = kernel.adj.blocks[0].body_forward
        info = self._vec_arr_info(kernel)

        # New pipeline.
        nodes = parse(lines)
        folded, _ = fold(nodes)
        view_folded, _ = fold_views(folded, kernel.adj)
        ix_folded, ix_skip = fold_indexref_writes(view_folded, kernel.adj, info)
        new_lines = emit(ix_folded)

        # Existing pipeline.
        old_for, _ = _preprocess_for_loops(lines)
        old_after_while = _preprocess_while_loops(old_for)
        old_after_views, _ = _preprocess_views(old_after_while, kernel.adj)
        old_lines, old_skip = _preprocess_indexref_writes(old_after_views, kernel.adj, info)

        self.assertEqual(
            new_lines,
            old_lines,
            f"emit(fold_indexref_writes(...)) != _preprocess_indexref_writes(...) on kernel {kernel.key!r}",
        )
        self.assertEqual(ix_skip, old_skip, f"skip set mismatch on kernel {kernel.key!r}")

    def test_no_indexref_writes_is_passthrough(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            out[tid] = a[tid] * 2.0

        self._assert_indexref_equivalent(k)

    def test_vec_component_write_via_indexref(self):
        # ``out[i, 0][k] = val`` lowers to address + indexref + store on a
        # vec-typed output array.
        @wp.kernel
        def k(out: wp.array2d(dtype=wp.spatial_vector)):
            worldid, k = wp.tid()
            out[worldid, 0][k] = float(worldid * 10 + k)

        self._assert_indexref_equivalent(k)


class TestMetalASTDropUnsupportedFold(unittest.TestCase):
    """Verify ``fold_drop_unsupported_locals`` matches the existing
    ``_preprocess_drop_unsupported_locals`` output.
    """

    def _assert_drop_equivalent(self, kernel):
        kernel.adj.build(builder=None, default_builder_options={"enable_backward": False})
        lines = kernel.adj.blocks[0].body_forward

        # New pipeline.
        nodes = parse(lines)
        folded, _ = fold(nodes)
        dropped, drop_skip = fold_drop_unsupported_locals(folded, kernel.adj)
        new_lines = emit(dropped)

        # Existing pipeline. Order matters: the regex pipeline runs
        # drop-unsupported BEFORE views/indexref. We mimic that here so the
        # comparison stays apples-to-apples.
        old_for, _ = _preprocess_for_loops(lines)
        old_after_while = _preprocess_while_loops(old_for)
        old_lines, old_skip = _preprocess_drop_unsupported_locals(old_after_while, kernel.adj)

        self.assertEqual(
            new_lines,
            old_lines,
            f"emit(fold_drop_unsupported_locals(...)) != "
            f"_preprocess_drop_unsupported_locals(...) on kernel {kernel.key!r}",
        )
        self.assertEqual(drop_skip, old_skip, f"skip set mismatch on kernel {kernel.key!r}")

    def test_no_unsupported_is_passthrough(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
            tid = wp.tid()
            out[tid] = a[tid]

        self._assert_drop_equivalent(k)

    def test_printf_dropped(self):
        @wp.kernel
        def k(out: wp.array(dtype=wp.int32), flag: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            if flag[tid] == 0:
                wp.printf("warn tid=%u\n", tid)
            out[tid] = tid * 2

        self._assert_drop_equivalent(k)

    def test_dead_tuple_dropped(self):
        @wp.kernel
        def k(idx: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            # ``wp.matrix(..., shape=(...), dtype=int)`` constructs an
            # internal tuple_t for the shape arg that is never read.
            table = wp.matrix(
                10,
                11,
                20,
                21,
                30,
                31,
                40,
                41,
                50,
                51,
                60,
                61,
                shape=(6, 2),
                dtype=int,
            )
            i = idx[tid]
            out[tid] = table[i, 0] + table[i, 1] * 100

        self._assert_drop_equivalent(k)


if __name__ == "__main__":
    unittest.main(verbosity=2)
