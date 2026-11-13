CXX ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
LDFLAGS ?= -lcrypto
all: luj_native_planner
luj_native_planner: src/luj_native_planner.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)
test: luj_native_planner
	./luj_native_planner --bank ../planner/planner.luj --inspect
	./luj_native_planner --bank ../planner/planner.luj --question 'Start with 12; multiply by 8; subtract 27; add 15'
regression: luj_native_planner
	bash tests/run_native_tests.sh
.PHONY: all test regression
