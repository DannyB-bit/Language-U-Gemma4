# Native arithmetic interpretation

C++17 executable with no Python runtime, tokenizer, neural model or external library.
This is a closed-grammar semantic compiler, NOT a GPT model or learned language
understanding. It is a separate component; it does not yet read/write LUJ glyphs
or replace the existing neural planner. No weights or metadata are falsely
counted as providing the rules: the parser's code is part of the system size.

Build and run:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Werror semantic.cpp -o semantic
./semantic 'Start with 547; take the lower of this amount and 193; divide the running total by 1; take 449 away from the total'
g++ -std=c++17 -O0 -g -fsanitize=address,undefined test.cpp -o test
./test
```

Expected example answer: -256, correcting the previously recorded -95.
Complete anchored productions consume every clause, track reversed operand
roles, and reject unknown syntax instead of guessing from keywords. All clauses
are parsed before execution. Integer math is checked, division must be exact,
and remainder follows divisor-sign semantics. Input is bounded to ASCII, 2,000
bytes and 1–5 steps. These are deliberate restrictions, not broad intelligence.

Tests cover the known failure, operand reversal, negative remainder, overflow,
negation/conditional/trailing-text refusals and a numerical grid of addition and
reversed subtraction. These authored regression tests are not independent
language benchmarks. Neither their success nor byte-level glyph correctness
proves GPT-level reasoning.

Next integration gate: encode the typed steps using the existing native glyph
format, verify binary interoperability, then evaluate on independently authored
language tasks and report coverage alongside wrong accepted answers. Broader
reasoning requires a trained language model and independently measured quality;
this component alone cannot establish that claim. The earlier Python experiment
is retained as historical evidence; this component does not execute it.
