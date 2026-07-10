#!/usr/bin/env bash
# REAL bug-fix benchmark for the repo-agent (a self-contained, reproducible "SWE-bench for Rust").
#
# Each task = a REAL algorithm + its tests, with a REAL bug class INJECTED (find the CORRECT code,
# replace with a BUGGY variant that breaks a passing test). The harness scaffolds the crate with the
# bug, CONFIRMS the baseline fails, runs the product exactly as a user does (`coding_agent --root <dir>
# query "fix the failing tests"`), re-runs cargo test, and scores RESOLVED / UNRESOLVED. Real code, real
# bugs, a real compiler+test oracle -- not synthetic fixtures.
#
#   scripts/repo_bench.sh                              # model-free (deterministic engine only)
#   NSYNTH_LOCAL_LLM_URL=... scripts/repo_bench.sh     # with the gated model lane
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="$ROOT/target/release/coding_agent"
[ -x "$AGENT" ] || { echo "build first: cargo build --release --bin coding_agent"; exit 1; }
BASE="${TMPDIR:-/tmp}/repo_bench.$$"; rm -rf "$BASE"; mkdir -p "$BASE"
export HOME="$BASE/home"; mkdir -p "$HOME"
TIMEOUT="${REPO_BENCH_TIMEOUT:-200}"
TOTAL=0; RESOLVED=0

# run_task <name> <correct-substring> <buggy-substring> <source>
# injects the bug by replacing the FIRST occurrence of the correct substring with the buggy one
# (Python string replace -> no regex/delimiter hazards).
run_task() {
  local name="$1" good="$2" bad="$3" src="$4"
  local d="$BASE/$name"; mkdir -p "$d/src"
  printf '[package]\nname="%s"\nversion="0.0.0"\nedition="2021"\n' "$name" > "$d/Cargo.toml"
  printf '%s\n' "$src" > "$d/src/lib.rs"
  GOOD="$good" BAD="$bad" python3 - "$d/src/lib.rs" <<'PY'
import os, sys
p = sys.argv[1]; g = os.environ["GOOD"]; b = os.environ["BAD"]
s = open(p).read()
if g not in s:
    print("MISSING"); sys.exit(3)
open(p, "w").write(s.replace(g, b, 1))
PY
  [ $? -eq 3 ] && { printf '%-24s SKIP (correct pattern absent)\n' "$name"; return; }
  if ( cd "$d" && cargo test --quiet >/dev/null 2>&1 ); then printf '%-24s SKIP (baseline still passes)\n' "$name"; return; fi
  TOTAL=$((TOTAL+1))
  timeout "$TIMEOUT" "$AGENT" --root "$d" query "fix the failing tests" >/dev/null 2>&1
  if ( cd "$d" && cargo test --quiet >/dev/null 2>&1 ); then
    RESOLVED=$((RESOLVED+1)); printf '%-24s RESOLVED\n' "$name"
  else
    printf '%-24s unresolved\n' "$name"
  fi
}

run_task off_by_one_sum '(i as i64) < n' '(i as i64) <= n' \
'pub fn sum_first(xs:&[i64], n:i64)->i64{ let mut s=0; let mut i=0; while i<xs.len() && (i as i64) < n { s+=xs[i]; i+=1; } s }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(sum_first(&[1,2,3,4],2),3);} #[test]fn b(){assert_eq!(sum_first(&[5,5,5,5],3),15);}}'

run_task wrong_op_area 'w * h' 'w + h' \
'pub fn area(w:i64,h:i64)->i64{ w * h }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(area(3,4),12);} #[test]fn b(){assert_eq!(area(5,2),10);}}'

run_task off_by_one_factorial 'i <= n' 'i < n' \
'pub fn factorial(n:i64)->i64{ let mut r=1; let mut i=1; while i <= n { r*=i; i+=1; } r }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(factorial(4),24);} #[test]fn b(){assert_eq!(factorial(5),120);}}'

run_task missing_guard_div 'if b == 0 { return None; } ' '' \
'pub fn safe_div(a:i64,b:i64)->Option<i64>{ if b == 0 { return None; } Some(a/b) }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(safe_div(10,2),Some(5));} #[test]fn b(){assert_eq!(safe_div(5,0),None);}}'

run_task wrong_cmp_max 'a > b' 'a < b' \
'pub fn max2(a:i64,b:i64)->i64{ if a > b { a } else { b } }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(max2(3,7),7);} #[test]fn b(){assert_eq!(max2(9,2),9);}}'

run_task off_by_one_count 'x >= t' 'x > t' \
'pub fn count_ge(xs:&[i64], t:i64)->i64{ let mut c=0; for &x in xs { if x >= t { c+=1; } } c }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(count_ge(&[1,5,5,9],5),3);} #[test]fn b(){assert_eq!(count_ge(&[2,4,6],4),2);}}'

run_task wrong_op_avg '(a + b) / 2' '(a + b) * 2' \
'pub fn avg2(a:i64,b:i64)->i64{ (a + b) / 2 }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(avg2(4,8),6);} #[test]fn b(){assert_eq!(avg2(10,20),15);}}'

run_task struct_wrong_op 'self.total += x' 'self.total -= x' \
'pub struct Acc{pub total:i64} impl Acc{ pub fn new()->Self{Acc{total:0}} pub fn add(&mut self,x:i64){ self.total += x; } pub fn get(&self)->i64{ self.total } }
#[cfg(test)] mod t{use super::*;#[test]fn a(){let mut a=Acc::new();a.add(3);a.add(4);assert_eq!(a.get(),7);}}'

run_task string_wrong_method 's.to_uppercase()' 's.to_lowercase()' \
'pub fn shout(s:String)->String{ s.to_uppercase() }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(shout("hi".to_string()),"HI");} #[test]fn b(){assert_eq!(shout("ab".to_string()),"AB");}}'

run_task fold_wrong_init 'let mut s = 0' 'let mut s = 1' \
'pub fn total(xs:&[i64])->i64{ let mut s = 0; for &x in xs { s += x; } s }
#[cfg(test)] mod t{use super::*;#[test]fn a(){assert_eq!(total(&[1,2,3]),6);} #[test]fn b(){assert_eq!(total(&[10,20]),30);}}'

echo "=================================================="
echo "REPO BENCH (real bug-fix tasks): $RESOLVED/$TOTAL resolved"
rm -rf "$BASE"
