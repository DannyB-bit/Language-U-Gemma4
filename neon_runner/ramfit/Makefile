# RAM-fit mixed-container encoder (Gate 1) -- pure C, no Python.
CC      ?= gcc
CFLAGS  ?= -O3 -march=armv8-a+simd -mcpu=cortex-a72 -fopenmp
LDLIBS  ?= -lm

all: reencode31b embed_encode shuffle_stream

reencode31b: reencode31b.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

embed_encode: embed_encode.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

shuffle_stream: shuffle_stream.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f reencode31b embed_encode shuffle_stream

.PHONY: all clean
