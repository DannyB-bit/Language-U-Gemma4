# Lossless BF16 weight glyph compiler

`compile_bf16.cpp` converts the actual uploaded Gemma safetensors weight payload
into consecutive 256-weight blocks, with an explicitly shorter final block.
It preserves the safetensors JSON header and all BF16 bits. Within each block,
it splits the BF16 low/high byte planes, then uses Zstandard level 3 only when
smaller than raw storage. This is conventional lossless block compression.

GBF1 is a new weight-only envelope, NOT the existing LUJ1 token/program ABI or
the 36-byte lossy weight codec. There are no token lanes and no implied
20.4-token payload. It cannot be passed to the existing LUJ1 loader. Records
can cross tensor boundaries; the preserved safetensors header retains shapes
and offsets. No values are rounded, retrained or regenerated.

Build with C++17 and OpenSSL/Zstandard development libraries:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror compile_bf16.cpp -lcrypto -lzstd -o compile_bf16
./compile_bf16 pack model.safetensors model.gbf
./compile_bf16 unpack model.gbf restored.safetensors
cmp model.safetensors restored.safetensors
sha256sum model.safetensors restored.safetensors
```

Header: GBF1 magic, little-endian uint64 original-file size, uint64 preserved
prefix length, 32-byte SHA-256 of original file, then verbatim safetensors
prefix (8-byte header length plus JSON). Records: uint16 raw byte length,
uint16 stored length, payload. Raw length is 512 except the last record;
equal stored/raw length means uncompressed planes. Smaller means one Zstd
frame. Decoder checks record sizes, frame sizes, total size, trailing bytes and
restored SHA-256. SHA-256 is not authentication. Input dtype scan is a narrow
BF16 guard, not a complete hostile-input safetensors schema validator.

The encoder validates a complete round trip before saving. The decoder uses
up to several full-file buffers; this implementation is not a low-RAM Pi
runtime. Native model generation, full-model glyph-backed inference, and
GPT-level quality remain unproved. Exact recovery preserves the original
checkpoint; it does not improve its reasoning capability.
