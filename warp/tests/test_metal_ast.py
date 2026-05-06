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
    """Structural fold tests: confirm ``fold`` produces ``For``/``While``/
    ``If`` nodes in the right shape on synthetic kernels. End-to-end
    correctness (does the kernel actually compute the right answer on
    Metal?) is covered by ``test_metal_launch.py``.
    """

    def _fold(self, kernel):
        lines = _kernel_ir(kernel)
        return fold(parse(lines))

    def test_dynamic_for_loop_one_arg_range(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(a[tid]):
                s += i
            out[tid] = s

        folded, skip = self._fold(k)
        for_nodes = [n for n in folded if isinstance(n, For)]
        self.assertEqual(len(for_nodes), 1)
        self.assertEqual(for_nodes[0].start, "0")
        # The for opener's range and iter vars are flagged as decl-skipped.
        self.assertEqual(len(skip), 2)

    def test_dynamic_for_loop_two_arg_range(self):
        @wp.kernel
        def k(starts: wp.array(dtype=wp.int32), stops: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            s = int(0)
            for i in range(starts[tid], stops[tid]):
                s += i
            out[tid] = s

        folded, _ = self._fold(k)
        for_nodes = [n for n in folded if isinstance(n, For)]
        self.assertEqual(len(for_nodes), 1)
        self.assertNotEqual(for_nodes[0].start, "0")  # start is an explicit var

    def test_while_loop_no_remaining_gotos(self):
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            n = a[tid]
            s = int(0)
            while n > 0:
                s += n
                n -= 1
            out[tid] = s

        folded, _ = self._fold(k)
        while_nodes = [n for n in folded if isinstance(n, While)]
        self.assertEqual(len(while_nodes), 1)
        body = while_nodes[0].body
        self.assertTrue(any(isinstance(n, WhileCondBreak) for n in body))
        self.assertFalse(
            any(isinstance(n, Goto) and "while" in n.target for n in body),
            "structured while body should contain no goto-to-while-labels",
        )

    def test_while_loop_with_break_folds_to_break_node(self):
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

        folded, _ = self._fold(k)
        while_nodes = [n for n in folded if isinstance(n, While)]
        self.assertEqual(len(while_nodes), 1)

        def _has_break(nodes):
            for nn in nodes:
                if isinstance(nn, Break):
                    return True
                if isinstance(nn, If) and _has_break(list(nn.body)):
                    return True
            return False

        self.assertTrue(_has_break(list(while_nodes[0].body)))

    def test_if_else_lowers_to_two_ifs(self):
        # Warp lowers Python ``if/else`` to two separate ``if`` statements
        # (the second on the negated condition), so we never see a
        # structured else-arm in the AST.
        @wp.kernel
        def k(a: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.int32)):
            tid = wp.tid()
            x = a[tid]
            if x > 0:
                out[tid] = x * 2
            else:
                out[tid] = -x

        folded, _ = self._fold(k)
        if_nodes = [n for n in folded if isinstance(n, If)]
        self.assertGreaterEqual(len(if_nodes), 2)

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

        folded, _ = self._fold(k)
        # Outer While, with For nested inside it (Warp may also wrap the
        # condition test in an If — we just need the For to be inside the
        # While body, transitively).
        whiles = [n for n in folded if isinstance(n, While)]
        self.assertEqual(len(whiles), 1)

        def _find_for(nodes):
            for nn in nodes:
                if isinstance(nn, For):
                    return True
                if isinstance(nn, (If, While, For)) and _find_for(list(nn.body)):
                    return True
            return False

        self.assertTrue(_find_for(list(whiles[0].body)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
