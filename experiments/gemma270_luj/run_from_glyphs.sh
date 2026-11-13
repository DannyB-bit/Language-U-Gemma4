#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -ne 3 ]; then
  echo 'usage: run_from_glyphs.sh <glyph-shard-dir> <comma-separated-token-ids> <max-new-tokens>' >&2
  exit 2
fi
input="$1"; tokens="$2"; steps="$3"
sha=$(tr -d '\r\n' < "$input/original.sha256")
[[ "$sha" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid model digest' >&2; exit 2; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
./luj_gemma270 decode "$input" "$tmp/model.safetensors" >&2
./gemma3_native "$tmp/model.safetensors" "$tokens" "$steps" --expected-sha "$sha"
