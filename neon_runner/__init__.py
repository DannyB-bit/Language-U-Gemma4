"""
neon_runner — Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.0.0)
=============================================================
Unified runtime and stream bridge connecting `DannyB-bit/Language-U-Gemma4`
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

__all__ = [
    "BRIDGE_COMPONENT_ID",
    "BRIDGE_VERSION",
    "CODEBOOK_BYTES",
    "CODEBOOK_SHAPE",
    "SENTINEL_INDEX",
    "CodebookRegistryV2",
    "LoadedCodebookV2",
    "NeonRunnerBridgeV2",
    "NeonStreamAccelerator",
    "StreamHeaderV2",
    "build_golden_s7_codebook",
    "build_trit_rvq_codebook",
    "encode_matrix_opt4_v2",
    "encode_matrix_opt5_v2",
    "fwht8_vec_numpy",
    "pack_lanu_header",
    "parse_lanu_header",
]
