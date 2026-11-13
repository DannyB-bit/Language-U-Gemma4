#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
out="${1:?new output dir}"
mkdir -p "$out"
source_model="${2:?source safetensors}"
q8_model="${3:?quantized safetensors}"
qsha="${4:?quantized model SHA256}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
run(){
 local name="$1" prompt="$2"
 ./gemma3_native "$source_model" "$prompt" 24 --logits-out "$out/$name.source.bin" > "$out/$name.source.json"
 ./gemma3_native "$q8_model" "$prompt" 24 --expected-sha "$qsha" --logits-out "$out/$name.q8.bin" > "$out/$name.q8.json"
 ./compare_logits "$out/$name.source.bin" "$out/$name.q8.bin" > "$out/$name.metrics.json"
 echo "$name"
 cat "$out/$name.metrics.json"
}
run arithmetic '2,105,2364,107,3689,563,236743,236778,900,236743,236778,236881,106,107,105,4368,107'
run capital '2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107'
run hamlet '2,105,2364,107,15938,7971,124600,236881,106,107,105,4368,107'
run multistep '2,105,2364,107,50702,236743,236770,236778,2782,236743,236828,236764,30579,236743,236778,236832,236764,1299,1138,236743,236770,236810,236761,24181,607,1186,506,3890,236761,106,107,105,4368,107'
run minimum '2,105,2364,107,3689,563,506,7100,529,236743,236810,236812,236832,532,236743,236770,236819,236800,236881,106,107,105,4368,107'
