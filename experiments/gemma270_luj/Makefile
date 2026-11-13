CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror
LDFLAGS ?= -lzstd -lz -lcrypto
all: luj_gemma270
luj_gemma270: luj_gemma270.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)
.PHONY: all

inference: gemma3_native
gemma3_native: gemma3_native.cpp
	$(CXX) $(CXXFLAGS) -fopenmp $< -o $@ -lcrypto

compare_logits: compare_logits.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
