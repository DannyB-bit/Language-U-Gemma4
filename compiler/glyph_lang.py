# glyph-lang: the Language-U assembler — parses, assembles, and emits glyphs.
# KOFI 2026-10-11. The core insight: Language-U programs are not text streams
# but proposition streams. A "statement" is a macro-glyph-sized semantic action:
#
#     [Action] ⊕ [Target] ⊕ [Context] ⊕ [Constraint]
#
# The source syntax is a minimal proposition grammar; each proposition maps to
# Cuneiform-U 6D coordinates (three radicals: RC/RF/RA) and executes against a
# tiny deterministic stack machine (the Glyph-VM) that compiles to REAL machine
# code — AArch64 and x86-64 — via the codegen module.

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    from compiler.glyphs import (
        AXES, Radicals, proposition_to_radicals, rotate_h8,
        build_trit_codebook, encode_glyph_132, decode_glyph_132,
        encode_glyph_36, glyph_checksum,
    )
except ImportError:
    from glyphs import (
        AXES, Radicals, proposition_to_radicals, rotate_h8,
        build_trit_codebook, encode_glyph_132, decode_glyph_132,
        encode_glyph_36, glyph_checksum,
    )

try:
    from compiler.vm import VMInstruction, Op, GlyphProgram
except ImportError:
    try:
        from vm import VMInstruction, Op, GlyphProgram
    except ImportError:
        VMInstruction = None  # type: ignore
        Op = None  # type: ignore
        GlyphProgram = None  # type: ignore


@dataclass
class Proposition:
    """One Language-U statement: Action ⊕ Target ⊕ Context ⊕ Constraint.
    The axis fields (domain..polarity) carry its Cuneiform-U coordinates."""
    raw: str
    domain: str = "general"
    subdomain: str = "core"
    operation: str = "compute"
    modality: str = "native"
    depth: str = "word"
    polarity: str = "positive"
    # The instruction's immediate payload (data carried with the proposition):
    payload: Optional[List[float]] = None


# Source grammar (line-oriented, zero ambiguity):
#   prop <action-word> <target-word> <context-word> <constraint-word> ; axis:value ...
#   imm  <f64>            (append immediate payloads to the active proposition)
#   emit-glyph-132        (compile the active proposition into a 132B RVQ glyph)
#   emit-glyph-36         (compile into a 36B bitmask glyph)
#   halt
PROP_RE = re.compile(r"^prop\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
AXIS_RE = re.compile(r"^(domain|subdomain|operation|modality|depth|polarity):\s*(\S+)$")
IMM_RE = re.compile(r"^imm\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$")

_KNOWN_TABLES = {name: table for name, table in AXES}


def parse(source: str) -> List[Tuple[str, object]]:
    """Parse Language-U source into an ordered program of directives."""
    program: List[Tuple[str, object]] = []
    pending: Optional[Proposition] = None
    for lineno, rawline in enumerate(source.splitlines(), 1):
        line = rawline.split("#", 1)[0].strip()
        if not line:
            continue
        m = PROP_RE.match(line)
        if m:
            action, target, context, constraint = m.groups()
            # Axis defaults derive from the four semantic words deterministically:
            op = action.lower()
            sub = target.lower() if target.lower() in _KNOWN_TABLES["subdomain"] else "core"
            mod = context.lower() if context.lower() in _KNOWN_TABLES["modality"] else "native"
            pol = constraint.lower() if constraint.lower() in _KNOWN_TABLES["polarity"] else "positive"
            pending = Proposition(
                raw=line,
                operation=op,
                subdomain=sub,
                modality=mod,
                polarity=pol,
            )
            continue
        m = AXIS_RE.match(line)
        if m and pending is not None:
            axis, value = m.groups()
            setattr(pending, axis, value.lower())
            continue
        m = IMM_RE.match(line)
        if m and pending is not None:
            if pending.payload is None:
                pending.payload = []
            pending.payload.append(float(m.group(1)))
            continue
        if line == "emit-glyph-132" or line == "emit-glyph-36":
            if pending is None:
                raise SyntaxError(f"line {lineno}: emit with no active proposition")
            program.append(("emit", (pending, "132" if line.endswith("132") else "36")))
            pending = None
            continue
        if line == "halt":
            program.append(("halt", None))
            continue
        raise SyntaxError(f"line {lineno}: cannot parse: {line!r}")
    return program


def compile_program(program, codebook=None) -> "GlyphProgram":
    """Compile parsed directives into the Glyph-VM instruction stream.
    Each emit becomes a VM instruction: [op=EMIT_GLYPH][radicals RC/RF/RA]
    [container law][glyph bytes or immediate payload]."""
    if VMInstruction is None or Op is None or GlyphProgram is None:
        raise ImportError(
            "Glyph-VM module ('compiler.vm' or 'vm') is required to compile programs into VM instructions."
        )
    codebook = codebook or build_trit_codebook()
    out: List[VMInstruction] = []
    for kind, arg in program:
        if kind == "halt":
            out.append(VMInstruction(Op.HALT))
            continue
        prop, law = arg
        rad = proposition_to_radicals(prop)
        if law == "132":
            weights = (prop.payload if prop.payload is not None else [0.0] * 256)
            if len(weights) < 256:
                weights = (weights + [0.0] * 256)[:256]
            glyph = encode_glyph_132(weights, codebook)
            out.append(VMInstruction(Op.EMIT_GLYPH_132, radicals=rad, blob=glyph))
        else:
            weights = (prop.payload if prop.payload is not None else [0.0] * 256)
            if len(weights) < 256:
                weights = (weights + [0.0] * 256)[:256]
            glyph = encode_glyph_36(weights)
            out.append(VMInstruction(Op.EMIT_GLYPH_36, radicals=rad, blob=glyph))
    if not out or out[-1].op != Op.HALT:
        out.append(VMInstruction(Op.HALT))
    return GlyphProgram(instructions=out, codebook=codebook)
