# Makefile for Language-U-Gemma4-NeonRunner-Bridge-V2 (v2.1.0 — GlyphAttentionV2)
# Automatically detects ARMv8-A NEON vs x86-64 on POSIX/Linux/macOS hosts.

CC ?= gcc
UNAME_M := $(shell uname -m 2>/dev/null || echo unknown)

ifeq ($(UNAME_M),aarch64)
  ARCH_FLAGS ?= -march=armv8-a+simd -mcpu=cortex-a72
else ifeq ($(UNAME_M),arm64)
  ARCH_FLAGS ?= -O3
else
  ARCH_FLAGS ?= -march=native
endif

CFLAGS ?= -O3 $(ARCH_FLAGS) -fopenmp -Wall -Wextra
LDFLAGS ?= -lm

TARGET_BIN = neon_stream_accelerator_v2
TARGET_LIB = libneon_stream_accelerator.so
COMPAT_LIB = libtrinity_gemv.so

all: $(TARGET_BIN) $(TARGET_LIB) $(COMPAT_LIB)

$(TARGET_BIN): neon_stream_accelerator_v2.c neon_stream_accelerator_v2.h
	$(CC) $(CFLAGS) neon_stream_accelerator_v2.c -o $(TARGET_BIN) $(LDFLAGS)

$(TARGET_LIB): neon_stream_accelerator_v2.c neon_stream_accelerator_v2.h
	$(CC) $(CFLAGS) -DNEON_SHARED_LIB_ONLY -shared -fPIC neon_stream_accelerator_v2.c -o $(TARGET_LIB) $(LDFLAGS)

$(COMPAT_LIB): $(TARGET_LIB)
	cp -f $(TARGET_LIB) $(COMPAT_LIB)

clean:
	rm -f $(TARGET_BIN) $(TARGET_LIB) $(COMPAT_LIB) *.o *.obj *.exp *.lib *.dll *.exe

.PHONY: all clean

