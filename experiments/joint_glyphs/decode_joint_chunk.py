"""Decode a LUJ1 stream without access to original checkpoint weights."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from joint_codec import decode_chunk, decode_weights, execute


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stream', type=Path)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--dtype', choices=('fp16','fp32'), default='fp32')
    parser.add_argument('--weights-out', type=Path, required=True)
    args = parser.parse_args()
    tokenizer_bytes = args.tokenizer.read_bytes()
    fingerprint = hashlib.sha256(tokenizer_bytes).digest()
    glyphs, source = decode_chunk(args.stream.read_bytes(), fingerprint)
    tokens = [value for glyph in glyphs for value in glyph.tokens]
    tokenizer = Tokenizer.from_str(tokenizer_bytes.decode())
    weights = np.stack([decode_weights(g.codec, g.weights, args.dtype) for g in glyphs])
    result = execute(glyphs)
    with args.weights_out.open('xb') as out:
        np.save(out, weights, allow_pickle=False)
    print(json.dumps({'glyphs':len(glyphs),'tokens':len(tokens),'weight_count':int(weights.size),
                      'weight_dtype':str(weights.dtype),'decoded_weight_bytes':weights.nbytes,
                      'decoded_weight_sha256':hashlib.sha256(weights.tobytes()).hexdigest(),
                      'source_fp32_sha256':source.hex(),
                      'text':tokenizer.decode(tokens,skip_special_tokens=False),
                      'program':result},indent=2))


if __name__=='__main__':
    main()
