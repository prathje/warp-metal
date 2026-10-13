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

"""MSL lowering of ``wp.Texture1D/2D/3D`` values and ``wp.texture_sample``.

A texture value is the ``texture{1,2,3}d_t`` struct Warp passes to kernels
(``tex`` handle + geometry); on Metal ``tex`` is the gpuAddress of the
descriptor built by :mod:`warp._src.metal_texture`. Sampling is a port of the
CPU software sampler in ``warp/native/texture.h``, so Metal results follow the
CPU reference (address modes, border handling, integer normalization, mip
blending) rather than any hardware filtering approximation.
"""

from __future__ import annotations

import ctypes
import re

# Warp texture classes -> (native ctype name, MSL struct name, dims).
_TEXTURE_CLASSES = {
    "Texture1D": ("texture1d_t", "wp_texture1d_t", 1),
    "Texture2D": ("texture2d_t", "wp_texture2d_t", 2),
    "Texture3D": ("texture3d_t", "wp_texture3d_t", 3),
}
_CTYPE_TO_MSL = {f"wp::{native}": msl for native, msl, _ in _TEXTURE_CLASSES.values()}
_CTYPE_TO_MSL.update({native: msl for native, msl, _ in _TEXTURE_CLASSES.values()})


def texture_class_info(t) -> tuple[str, str, int] | None:
    """``(native, msl, dims)`` when ``t`` is a Warp texture type, else None."""
    from warp._src.texture import Texture  # noqa: PLC0415

    if isinstance(t, type) and issubclass(t, Texture):
        return _TEXTURE_CLASSES.get(t.__name__)
    return None


def msl_var_type(ctype: str) -> str | None:
    return _CTYPE_TO_MSL.get(ctype.strip())


def param_msl(arg_var) -> str | None:
    info = texture_class_info(arg_var.type)
    return None if info is None else info[1]


def pack_param(arg_var, value) -> tuple[bytes, int] | None:
    """``setBytes`` payload for a texture kernel param (its ``texture*_t``)."""
    info = texture_class_info(arg_var.type)
    if info is None:
        return None
    if value is None:
        from warp._src.texture import texture1d_t, texture2d_t, texture3d_t  # noqa: PLC0415

        inst = {1: texture1d_t, 2: texture2d_t, 3: texture3d_t}[info[2]]()
    else:
        if getattr(value, "_metal_storage", None) is None and getattr(value, "_tex_handle", 0):
            raise RuntimeError(
                f"Texture passed to a Metal kernel lives on {value.device}; create it on the Metal device"
            )
        inst = value.__ctype__()
    return bytes(inst), ctypes.sizeof(inst)


def struct_field_layout(ftype) -> tuple[str, int] | None:
    """``(msl_struct, n_scalar_slots)`` for a texture field of a ``wp.struct``."""
    info = texture_class_info(ftype)
    if info is None:
        return None
    from warp._src.texture import texture1d_t, texture2d_t, texture3d_t  # noqa: PLC0415

    ct = {1: texture1d_t, 2: texture2d_t, 3: texture3d_t}[info[2]]
    return info[1], ctypes.sizeof(ct) // 4


_INT_MEMBERS = {
    "wp_texture1d_t": ("width", "num_channels", "dtype"),
    "wp_texture2d_t": ("width", "height", "num_channels", "dtype"),
    "wp_texture3d_t": ("width", "height", "depth", "num_channels", "dtype"),
}


def int_members(msl: str) -> tuple[str, ...]:
    """The ``int32`` members of a texture struct, in layout order after ``tex``."""
    return _INT_MEMBERS[msl]


def struct_field_read(msl: str, buf: str, base: str) -> str:
    """Rebuild a texture struct from flat float32 struct storage."""
    lo = f"as_type<uint>({buf}[{base}])"
    hi = f"as_type<uint>({buf}[({base}) + 1])"
    n_ints = len(_INT_MEMBERS[msl])
    ints = ", ".join(f"as_type<int>({buf}[({base}) + {2 + k}])" for k in range(n_ints))
    return f"{msl}{{((ulong({hi}) << 32) | ulong({lo})), {ints}}}"


_SAMPLE_PAT = re.compile(r"\bwp::texture_sample\s*<\s*([^;]*?)\s*>\s*\(")


def _kind(tpl: str) -> str | None:
    t = tpl.replace(" ", "")
    if t in ("float", "wp::float32", "float32"):
        return "f"
    if t in ("wp::vec_t<2,wp::float32>", "float2", "wp::vec2f"):
        return "v2"
    if t in ("wp::vec_t<4,wp::float32>", "float4", "wp::vec4f"):
        return "v4"
    return None


def translate(line: str) -> str:
    """Rewrite ``wp::texture_sample<T>(tex, coords..., lod)`` to MSL helpers."""
    if "texture_sample" not in line:
        return line
    from warp._src.codegen_metal import MetalCodegenError  # noqa: PLC0415

    if re.search(r"\bwp::adj_texture_sample\b", line):
        raise MetalCodegenError("MSL codegen: reverse mode through wp.texture_sample() is not supported on Metal")

    def repl(m: re.Match[str]) -> str:
        kind = _kind(m.group(1))
        if kind is None:
            raise MetalCodegenError(f"MSL codegen: wp.texture_sample() with dtype {m.group(1)!r} is not supported")
        return f"wp_tex_sample_{kind}("

    return _SAMPLE_PAT.sub(repl, line)


def uses_textures(source: str) -> bool:
    return "wp_tex_sample_" in source or "wp_texture" in source


_HELPERS = r"""
struct wp_texture1d_t { ulong tex; int width; int num_channels; int dtype; };
struct wp_texture2d_t { ulong tex; int width; int height; int num_channels; int dtype; };
struct wp_texture3d_t { ulong tex; int width; int height; int depth; int num_channels; int dtype; };
struct wp_tex_desc_t {
    int ndim; int num_channels; int dtype; int filter_mode;
    int mip_filter_mode; int num_mip_levels; int addr_u; int addr_v;
    int addr_w; int normalized; int width; int height;
    int depth; int pad0;
    ulong mip_data[16];
    int mip_w[16]; int mip_h[16]; int mip_d[16];
};
inline const device wp_tex_desc_t& wp_tex_desc(ulong h) { return *reinterpret_cast<const device wp_tex_desc_t*>(h); }
inline float wp_tex_addr_coord(float coord, int size, int mode) {
    float texel = coord * size - 0.5f;
    float fsize = (float)size;
    if (mode == 0) {
        texel = texel - metal::floor(texel / fsize) * fsize;
        if (texel < 0) texel += fsize;
    } else if (mode == 1) {
        if (texel < 0.0f) texel = 0.0f;
        if (texel > fsize - 1.0f) texel = fsize - 1.0f;
    } else if (mode == 2) {
        float period = 2.0f * fsize;
        texel = texel - metal::floor(texel / period) * period;
        if (texel < 0) texel += period;
        if (texel >= fsize) texel = period - texel - 1.0f;
    }
    return texel;
}
inline int wp_tex_clamp_index(int idx, int size) { return (idx < 0) ? 0 : ((idx >= size) ? size - 1 : idx); }
inline int wp_tex_addr_index(int idx, int size, int mode) {
    if (mode == 0) { int m = idx % size; if (m < 0) m += size; return m; }
    if (mode == 1) return wp_tex_clamp_index(idx, size);
    if (mode == 2) {
        int period = 2 * size; int m = idx % period; if (m < 0) m += period;
        if (m >= size) return period - m - 1;
        return m;
    }
    return idx;
}
inline float wp_tex_decode(ulong level_data, int dtype, int idx) {
    switch (dtype) {
    case 0: return reinterpret_cast<const device uchar*>(level_data)[idx] / 255.0f;
    case 1: return reinterpret_cast<const device ushort*>(level_data)[idx] / 65535.0f;
    case 6: return reinterpret_cast<const device uint*>(level_data)[idx] / 4294967295.0f;
    case 3: { float v = reinterpret_cast<const device char*>(level_data)[idx] / 127.0f; return v < -1.0f ? -1.0f : v; }
    case 4: { float v = reinterpret_cast<const device short*>(level_data)[idx] / 32767.0f; return v < -1.0f ? -1.0f : v; }
    case 7: { float v = reinterpret_cast<const device int*>(level_data)[idx] / 2147483647.0f; return v < -1.0f ? -1.0f : v; }
    case 5: return float(reinterpret_cast<const device half*>(level_data)[idx]);
    default: return reinterpret_cast<const device float*>(level_data)[idx];
    }
}
inline float wp_tex_fetch(const device wp_tex_desc_t& t, int level, int x, int y, int z, int ch) {
    const int w = t.mip_w[level], h = t.mip_h[level], d = t.mip_d[level];
    if (x < 0 || x >= w || y < 0 || y >= h || z < 0 || z >= d || ch < 0 || ch >= t.num_channels) return 0.0f;
    int idx = ((z * h + y) * w + x) * t.num_channels + ch;
    return wp_tex_decode(t.mip_data[level], t.dtype, idx);
}
inline float wp_tex_sample_level(const device wp_tex_desc_t& t, int level, float u, float v, float wc, int ch) {
    const int nd = t.ndim;
    const int w = t.mip_w[level], h = t.mip_h[level], d = t.mip_d[level];
    float cu = t.normalized ? u : (u / (float)t.width);
    float cv = t.normalized ? v : (v / (float)t.height);
    float cw = t.normalized ? wc : (wc / (float)t.depth);
    float tx = wp_tex_addr_coord(cu, w, t.addr_u);
    float ty = nd > 1 ? wp_tex_addr_coord(cv, h, t.addr_v) : 0.0f;
    float tz = nd > 2 ? wp_tex_addr_coord(cw, d, t.addr_w) : 0.0f;
    if (t.filter_mode == 0) {
        int x = (int)metal::floor(tx + 0.5f);
        int y = (int)metal::floor(ty + 0.5f);
        int z = (int)metal::floor(tz + 0.5f);
        if (t.addr_u != 3) x = wp_tex_clamp_index(x, w);
        if (nd > 1 && t.addr_v != 3) y = wp_tex_clamp_index(y, h);
        if (nd > 2 && t.addr_w != 3) z = wp_tex_clamp_index(z, d);
        return wp_tex_fetch(t, level, x, nd > 1 ? y : 0, nd > 2 ? z : 0, ch);
    }
    int x0 = (int)metal::floor(tx); int x1 = x0 + 1; float fx = tx - x0;
    x0 = wp_tex_addr_index(x0, w, t.addr_u); x1 = wp_tex_addr_index(x1, w, t.addr_u);
    if (nd == 1) {
        float v0 = wp_tex_fetch(t, level, x0, 0, 0, ch);
        float v1 = wp_tex_fetch(t, level, x1, 0, 0, ch);
        return v0 * (1.0f - fx) + v1 * fx;
    }
    int y0 = (int)metal::floor(ty); int y1 = y0 + 1; float fy = ty - y0;
    y0 = wp_tex_addr_index(y0, h, t.addr_v); y1 = wp_tex_addr_index(y1, h, t.addr_v);
    if (nd == 2) {
        float v00 = wp_tex_fetch(t, level, x0, y0, 0, ch);
        float v10 = wp_tex_fetch(t, level, x1, y0, 0, ch);
        float v01 = wp_tex_fetch(t, level, x0, y1, 0, ch);
        float v11 = wp_tex_fetch(t, level, x1, y1, 0, ch);
        float v0 = v00 * (1.0f - fx) + v10 * fx;
        float v1 = v01 * (1.0f - fx) + v11 * fx;
        return v0 * (1.0f - fy) + v1 * fy;
    }
    int z0 = (int)metal::floor(tz); int z1 = z0 + 1; float fz = tz - z0;
    z0 = wp_tex_addr_index(z0, d, t.addr_w); z1 = wp_tex_addr_index(z1, d, t.addr_w);
    float v000 = wp_tex_fetch(t, level, x0, y0, z0, ch);
    float v100 = wp_tex_fetch(t, level, x1, y0, z0, ch);
    float v010 = wp_tex_fetch(t, level, x0, y1, z0, ch);
    float v110 = wp_tex_fetch(t, level, x1, y1, z0, ch);
    float v001 = wp_tex_fetch(t, level, x0, y0, z1, ch);
    float v101 = wp_tex_fetch(t, level, x1, y0, z1, ch);
    float v011 = wp_tex_fetch(t, level, x0, y1, z1, ch);
    float v111 = wp_tex_fetch(t, level, x1, y1, z1, ch);
    float v00 = v000 * (1.0f - fx) + v100 * fx;
    float v10 = v010 * (1.0f - fx) + v110 * fx;
    float v01 = v001 * (1.0f - fx) + v101 * fx;
    float v11 = v011 * (1.0f - fx) + v111 * fx;
    float v0 = v00 * (1.0f - fy) + v10 * fy;
    float v1 = v01 * (1.0f - fy) + v11 * fy;
    return v0 * (1.0f - fz) + v1 * fz;
}
inline float wp_tex_sample_ch(ulong h, float u, float v, float w, int ch, float lod) {
    if (h == 0ul) return 0.0f;
    const device wp_tex_desc_t& t = wp_tex_desc(h);
    if (lod < 0.0f) return wp_tex_sample_level(t, 0, u, v, w, ch);
    float cl = lod < 0.0f ? 0.0f : lod;
    float max_lod = (float)(t.num_mip_levels - 1);
    cl = cl > max_lod ? max_lod : cl;
    int level0 = (int)metal::floor(cl);
    if (t.num_mip_levels <= 1 || t.mip_filter_mode == 0) {
        int nearest = (int)metal::floor(cl + 0.5f);
        if (nearest >= t.num_mip_levels) nearest = t.num_mip_levels - 1;
        return wp_tex_sample_level(t, nearest, u, v, w, ch);
    }
    int level1 = level0 + 1;
    if (level1 >= t.num_mip_levels) level1 = t.num_mip_levels - 1;
    float fl = cl - (float)level0;
    float a = wp_tex_sample_level(t, level0, u, v, w, ch);
    float b = wp_tex_sample_level(t, level1, u, v, w, ch);
    return a * (1.0f - fl) + b * fl;
}
inline float wp_tex_sample_f(wp_texture1d_t t, float u, float lod) { return wp_tex_sample_ch(t.tex, u, 0.0f, 0.0f, 0, lod); }
inline float wp_tex_sample_f(wp_texture2d_t t, float2 uv, float lod) { return wp_tex_sample_ch(t.tex, uv.x, uv.y, 0.0f, 0, lod); }
inline float wp_tex_sample_f(wp_texture2d_t t, float u, float v, float lod) { return wp_tex_sample_ch(t.tex, u, v, 0.0f, 0, lod); }
inline float wp_tex_sample_f(wp_texture3d_t t, float3 p, float lod) { return wp_tex_sample_ch(t.tex, p.x, p.y, p.z, 0, lod); }
inline float wp_tex_sample_f(wp_texture3d_t t, float u, float v, float w, float lod) { return wp_tex_sample_ch(t.tex, u, v, w, 0, lod); }
inline float2 wp_tex_sample_v2_(ulong h, float u, float v, float w, float lod) {
    return float2(wp_tex_sample_ch(h, u, v, w, 0, lod), wp_tex_sample_ch(h, u, v, w, 1, lod));
}
inline float4 wp_tex_sample_v4_(ulong h, float u, float v, float w, float lod) {
    return float4(wp_tex_sample_ch(h, u, v, w, 0, lod), wp_tex_sample_ch(h, u, v, w, 1, lod),
                  wp_tex_sample_ch(h, u, v, w, 2, lod), wp_tex_sample_ch(h, u, v, w, 3, lod));
}
inline float2 wp_tex_sample_v2(wp_texture1d_t t, float u, float lod) { return wp_tex_sample_v2_(t.tex, u, 0.0f, 0.0f, lod); }
inline float2 wp_tex_sample_v2(wp_texture2d_t t, float2 uv, float lod) { return wp_tex_sample_v2_(t.tex, uv.x, uv.y, 0.0f, lod); }
inline float2 wp_tex_sample_v2(wp_texture2d_t t, float u, float v, float lod) { return wp_tex_sample_v2_(t.tex, u, v, 0.0f, lod); }
inline float2 wp_tex_sample_v2(wp_texture3d_t t, float3 p, float lod) { return wp_tex_sample_v2_(t.tex, p.x, p.y, p.z, lod); }
inline float2 wp_tex_sample_v2(wp_texture3d_t t, float u, float v, float w, float lod) { return wp_tex_sample_v2_(t.tex, u, v, w, lod); }
inline float4 wp_tex_sample_v4(wp_texture1d_t t, float u, float lod) { return wp_tex_sample_v4_(t.tex, u, 0.0f, 0.0f, lod); }
inline float4 wp_tex_sample_v4(wp_texture2d_t t, float2 uv, float lod) { return wp_tex_sample_v4_(t.tex, uv.x, uv.y, 0.0f, lod); }
inline float4 wp_tex_sample_v4(wp_texture2d_t t, float u, float v, float lod) { return wp_tex_sample_v4_(t.tex, u, v, 0.0f, lod); }
inline float4 wp_tex_sample_v4(wp_texture3d_t t, float3 p, float lod) { return wp_tex_sample_v4_(t.tex, p.x, p.y, p.z, lod); }
inline float4 wp_tex_sample_v4(wp_texture3d_t t, float u, float v, float w, float lod) { return wp_tex_sample_v4_(t.tex, u, v, w, lod); }
"""


def emit_helpers(source: str) -> str:
    return _HELPERS if uses_textures(source) else ""
