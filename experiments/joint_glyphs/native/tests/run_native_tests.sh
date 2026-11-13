#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bank=../planner/planner.luj
exe=./luj_native_planner
out=$($exe --bank "$bank" --inspect)
grep -q '"weights":1280' <<<"$out"
grep -q '"tokens":102' <<<"$out"
check() {
  local q="$1" want="$2" op="$3" ans
  ans=$($exe --bank "$bank" --question "$q")
  grep -q "\"answer\":$want" <<<"$ans" || { echo "FAIL: $q => $ans" >&2; return 1; }
  grep -q "\"$op\"" <<<"$ans" || { echo "FAIL op: $q => $ans" >&2; return 1; }
}
check 'Start with 12; multiply by 8; subtract 27; add 15' 84 multiply
check 'Start with 547; take the lower of this amount and 193; divide the running total by 1; take 449 away from the total' -256 minimum
check 'Start with 959; gain another 211; select the larger amount versus 969; compute this amount raised to 1; increment by 155; choose the lesser amount versus 275' 275 minimum
check 'Start with 315; divide the running total by 9; compute the integer quotient with 5; apply exponent 1; keep the remainder on division by 19; the current amount divided by 7' 1 remainder
check 'Start with -9; take the current amount modulo 5' 1 remainder
for q in 'Start with 5; do not add 3' 'Start with 9223372036854775807; add 1' 'Start with 5; divide by 0' 'Start with 2; take the lower of 5 and the higher of 6'; do
  if $exe --bank "$bank" --question "$q" >/dev/null 2>&1; then echo "unexpected acceptance: $q" >&2; exit 1; fi
done
tmp=$(mktemp -d);trap 'rm -rf "$tmp"' EXIT
$exe --bank "$bank" --question 'Start with 12; multiply by 8; subtract 27; add 15' --chunk-out "$tmp/plan.luj" >/dev/null
[ "$(stat -c %s "$tmp/plan.luj")" = 1269 ]
# The program chunk must not authenticate as the immutable model bank.
if $exe --bank "$tmp/plan.luj" --inspect >/dev/null 2>&1; then echo 'program accepted as model bank' >&2; exit 1; fi
cp "$bank" "$tmp/bad.luj";printf '\377' | dd of="$tmp/bad.luj" bs=1 seek=420 count=1 conv=notrunc status=none
if $exe --bank "$tmp/bad.luj" --inspect >/dev/null 2>&1; then echo 'tampered model accepted' >&2; exit 1; fi
echo 'NATIVE_CHECKS=PASS'
