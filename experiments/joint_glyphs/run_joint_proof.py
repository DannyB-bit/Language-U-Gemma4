"""Real-token and real-QAT-weight round trips with a declared arithmetic DAG."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from joint_codec import (Glyph, QAT4, OPT5, EXACT16, EXACT32, ADD, SUB, MUL,
                         ASSERT_EQ, ASSERT_GE, STREAM, RECORD, INSTRUCTION,
                         compile_chunk, decode_chunk, decode_weights, execute,
                         exact_encode, instruction, opt5_encode)

TOKENIZER_SHA = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
SOURCE_REV = "52f3f65bc7a02d555763bc923bd1d9094898219d"
TEXT = (
    "A store starts with twelve crates, each holding eight apples. It sells twenty seven "
    "apples, then receives fifteen more apples. First multiply twelve by eight to find "
    "ninety six apples. Subtract twenty seven to get sixty nine. Add fifteen to obtain "
    "eighty four. Check the result by reversing the changes: eighty four minus fifteen "
    "plus twenty seven equals ninety six. Therefore, the store ends with eighty four "
    "apples. Each individual step uses exact integer arithmetic, keeps the same units, "
    "and checks that the final inventory cannot be negative."
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True, help="Packed-correction data directory")
    parser.add_argument("--output", type=Path, required=True, help="New proof directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tokfile = args.inputs / "activation-cache/tokenizer.json"
    tokraw = tokfile.read_bytes()
    if digest(tokraw) != TOKENIZER_SHA:
        raise ValueError("Tokenizer hash mismatch")
    tokenizer = Tokenizer.from_str(tokraw.decode())
    ids = tokenizer.encode(TEXT, add_special_tokens=False).ids
    assert len(ids) == 102 and tokenizer.decode(ids, skip_special_tokens=False) == TEXT
    counts = [20, 20, 20, 21, 21]
    chunks, start = [], 0
    for count in counts:
        chunks.append(ids[start:start+count]); start += count
    cache = args.inputs / "range-cache"
    packed = (cache / "model.language_model.layers.0.self_attn.q_proj.weight_packed.bin").read_bytes()
    scales = (cache / "model.language_model.layers.0.self_attn.q_proj.weight_scale.bin").read_bytes()
    if digest(packed) != "3dc90ad3509ba1a2f057f4a6d04635f83b2ec8ea3e07fa8edb98875436787654":
        raise ValueError("Packed source hash mismatch")
    if digest(scales) != "36439c06300c935c5ebdee90731fb715a1839983b1e0bce4f1e6493fb210f13b":
        raise ValueError("Source scale hash mismatch")
    # Independent earlier checkpoint decoder output, not this codec's output.
    reference = np.load(args.inputs / "real_activation_inputs.npz", allow_pickle=False)["w"][0, :1280].astype('<f4').reshape(5, 256)
    source_sha = hashlib.sha256(reference.tobytes()).digest()
    ops = [instruction(MUL, left=12, right=8),
           instruction(SUB, left_ref=0, right=27),
           instruction(ADD, left_ref=1, right=15),
           instruction(ASSERT_EQ, left_ref=2, right=84),
           instruction(ASSERT_GE, left_ref=2, right=0)]
    results = []
    for name, codec in (("qat_exact", QAT4), ("opt5_approximate", OPT5),
                        ("opt5_repair_fp16", EXACT16), ("opt5_repair_fp32", EXACT32)):
        glyphs = []
        for i in range(5):
            if codec == QAT4:
                lane = packed[i*128:(i+1)*128] + scales[i*16:(i+1)*16]
            elif codec == OPT5:
                lane = opt5_encode(reference[i])
            else:
                lane = exact_encode(reference[i], "fp16" if codec == EXACT16 else "fp32")
            glyphs.append(Glyph(chunks[i], codec, lane, ops[i]))
        raw = compile_chunk(glyphs, bytes.fromhex(TOKENIZER_SHA), source_sha)
        (args.output / (name + '.luj')).write_bytes(raw)
        restored, _ = decode_chunk(raw, bytes.fromhex(TOKENIZER_SHA))
        recovered_ids = [value for g in restored for value in g.tokens]
        recovered = np.stack([decode_weights(g.codec, g.weights) for g in restored])
        exact = recovered.tobytes() == reference.tobytes()
        half_exact = None
        if codec != OPT5:
            halves = np.stack([decode_weights(g.codec, g.weights, 'fp16') for g in restored])
            half_exact = halves.tobytes() == reference.astype('<f2').tobytes()
            assert exact and half_exact
        assert recovered_ids == ids
        assert tokenizer.decode(recovered_ids, skip_special_tokens=False) == TEXT
        # Actual GEMV on restored model weights versus independent reference.
        x = np.linspace(-1, 1, 256, dtype='f4')
        p, target = recovered @ x, reference @ x
        if codec != OPT5:
            np.testing.assert_array_equal(p, target)
        trace = execute(restored)
        assert trace['result'] == 84
        row = dict(name=name, glyph_count=5, tokens_per_glyph=counts,
                   mean_tokens_per_glyph=20.4, weights_per_glyph=256,
                   token_id_exact=True, english_text_exact=True, source_fp32_bit_exact=exact,
                   source_fp16_bit_exact=half_exact, source_max_abs_weight_error=float(np.max(np.abs(recovered-reference))),
                   weight_payload_bytes=[len(g.weights) for g in glyphs],
                   token_payload_bytes=[(len(g.tokens)*18+7)//8 for g in glyphs],
                   instruction_bytes_per_glyph=INSTRUCTION.size, record_header_bytes=RECORD.size,
                   shared_stream_header_bytes=STREAM.size, complete_stream_bytes=len(raw),
                   mean_complete_bytes_per_glyph=len(raw)/5, sha256=digest(raw),
                   gemv_matches_reference=bool(np.array_equal(p, target)), reasoning_trace=trace)
        results.append(row)
    exact_size = results[0]['complete_stream_bytes']
    report = dict(scope="Five combined records, actual source token IDs and QAT weight blocks; no GPT model or generation benchmark.",
                  format="LUJ1 experimental extension, NOT the existing 36/132-byte glyph ABI.",
                  source_model="google/gemma-4-31B-it-qat-w4a16-ct", source_revision=SOURCE_REV,
                  source_tensor="model.language_model.layers.0.self_attn.q_proj", source_rows=[0], source_columns=[0,1280],
                  source_fp32_sha256=source_sha.hex(), tokenizer_sha256=TOKENIZER_SHA,
                  shared_tokenizer_file_bytes=len(tokraw), source_text=TEXT, source_token_ids=ids,
                  total_tokens=102, total_weights=1280, modes=results,
                  code_vs_text="The arithmetic DAG is explicitly authored and validated; it is not inferred from prose. The stored prose is an authored explanation, not model-generated reasoning.",
                  weights_vs_reasoning="Weights are decoded and exercised in GEMV; the arithmetic DAG does not obtain its answer from those model weights.",
                  dictionary_accounting={"shared_dictionary_bytes":exact_size,
                    "five_u16_references_bytes":10, "cold_total_dictionary_plus_references":exact_size+10,
                    "note":"Replacing records with IDs requires retaining the full dictionary. These are byte accounting examples, not a separate implemented dictionary codec."})
    (args.output / 'joint_roundtrip_results.json').write_text(json.dumps(report, indent=2)+'\n')
    (args.output / 'recovered_reasoning.txt').write_text(TEXT+'\n'+results[0]['reasoning_trace']['english']+'\n')
    print(json.dumps({"total_tokens":102,"total_weights":1280,"modes":[{k:r[k] for k in ('name','complete_stream_bytes','source_fp32_bit_exact','weight_payload_bytes')} for r in results]}))


if __name__ == '__main__':
    main()
