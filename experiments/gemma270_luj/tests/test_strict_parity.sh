#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
make compare_logits
cxx="${CXX:-g++}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
"$cxx" -std=c++17 -O2 -Wall -Wextra -Werror tests/test_logit_gate.cpp -o "$tmp/fixture"
"$tmp/fixture" "$tmp/source.bin" "$tmp/bad.bin"
./compare_logits "$tmp/source.bin" "$tmp/source.bin" --min-cosine 0.999999 --max-relative-l2 0.001 --require-argmax | grep -q '"gate_pass":true'
set +e
./compare_logits "$tmp/source.bin" "$tmp/bad.bin" --min-cosine 0.999999 --max-relative-l2 0.001 --require-argmax > "$tmp/result.json"
status=$?
set -e
test "$status" -eq 2
grep -q '"gate_pass":false' "$tmp/result.json"
echo STRICT_PARITY_NEGATIVE_CONTROL=PASS
