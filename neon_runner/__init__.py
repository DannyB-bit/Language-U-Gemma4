"""
neon_runner — Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.2.0 — OpticalCarrierV1)
================================================================================
Unified runtime, stream bridge, Single-Pass Flash-Glyph GQA Attention, and
Lossless 2D Optical Context Caching engine connecting `DannyB-bit/Language-U-Gemma4`
with `DannyB-bit/Neon-Runner`.
"""

from .codebooks import (
    CODEBOOK_BYTES,
    CODEBOOK_SHAPE,
    SENTINEL_INDEX,
    CodebookRegistryV2,
    LoadedCodebookV2,
    build_golden_s7_codebook,
    build_trit_rvq_codebook,
)
from .bridge import (
    BRIDGE_COMPONENT_ID,
    BRIDGE_VERSION,
    NeonRunnerBridgeV2,
    NeonStreamAccelerator,
    StreamHeaderV2,
    encode_matrix_opt4_v2,
    encode_matrix_opt5_v2,
    fwht8_vec_numpy,
    pack_lanu_header,
    parse_lanu_header,
)
from .glyph_attention import (
    GLYPH_ATTN_COMPONENT_ID,
    GLYPH_ATTN_VERSION,
    GlyphKVCacheV2,
    GlyphKVStats,
    dense_gqa_attention_reference,
)
from .optical_carrier import (
    OPTICAL_CARRIER_COMPONENT_ID,
    OPTICAL_CARRIER_VERSION,
    CarrierHeader,
    OpticalCarrierCodec,
    OpticalGemmaBridge,
)

__all__ = [
    "BRIDGE_COMPONENT_ID",
    "BRIDGE_VERSION",
    "CODEBOOK_BYTES",
    "CODEBOOK_SHAPE",
    "GLYPH_ATTN_COMPONENT_ID",
    "GLYPH_ATTN_VERSION",
    "GlyphKVCacheV2",
    "GlyphKVStats",
    "OPTICAL_CARRIER_COMPONENT_ID",
    "OPTICAL_CARRIER_VERSION",
    "CarrierHeader",
    "OpticalCarrierCodec",
    "OpticalGemmaBridge",
    "SENTINEL_INDEX",
    "CodebookRegistryV2",
    "LoadedCodebookV2",
    "NeonRunnerBridgeV2",
    "NeonStreamAccelerator",
    "StreamHeaderV2",
    "build_golden_s7_codebook",
    "build_trit_rvq_codebook",
    "dense_gqa_attention_reference",
    "encode_matrix_opt4_v2",
    "encode_matrix_opt5_v2",
    "fwht8_vec_numpy",
    "pack_lanu_header",
    "parse_lanu_header",
]
