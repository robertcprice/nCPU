# BSD vs GNU coreutils: automatic differential testing

Run the fuzzer's random probes through *both* the BSD binary (macOS
default) and the GNU binary (homebrew's `g`-prefixed variants).
Every probe where the two disagree is a real shell-script
portability hazard: code assuming one implementation's behaviour
will silently break when deployed on the other platform.

## Measured (200 probes per tool, seed=42, `LC_ALL=C`)

| tool | agreeing | divergent | rate | finding |
|---|--:|--:|--:|---|
| cut | 200 | 0 | 0.00% | ✓ byte-identical |
| fold | 200 | 0 | 0.00% | ✓ byte-identical |
| head | 199 | 0 | 0.00% | ✓ byte-identical |
| sort | 200 | 0 | 0.00% | ✓ byte-identical |
| tail | 200 | 0 | 0.00% | ✓ byte-identical |
| tr | 200 | 0 | 0.00% | ✓ byte-identical |
| uniq | 200 | 0 | 0.00% | ✓ byte-identical |
| **base64** | 104 | 96 | **48.00%** | ⚠ GNU wraps at 76 cols, BSD doesn't |
| **wc** | 0 | 200 | **100.00%** | ⚠ BSD uses 8-char field padding, GNU uses variable |

(expand, nl, paste, seq, tac all skipped — we haven't written fuzzers
for them yet; adding those is a ~10-line extension.)

## The two real compat bugs surfaced

### base64: line wrapping disagreement

```
BSD (macOS):  Yi1vZGEtU1gzOW5ZVSopRklyXS5mR3JsL2FCenlxNEpjfTd0IV9BajYlOgtcXUNKQGd6JiknNyJSb0txfSdsdy81Pz1FNnxVKmxO
GNU (linux):  Yi1vZGEtU1gzOW5ZVSopRklyXS5mR3JsL2FCenlxNEpjfTd0IV9BajYlOgtcXUNKQGd6JiknNyJS
              b0txfSdsdy81Pz1FNnxVKmx
```

Scripts that parse base64 output assuming a single unwrapped line
(common in "put this token in a header" patterns) will silently
truncate on Linux because they only read up to the first newline.
Workaround: always pipe through `tr -d '\n'` before use.

### wc: column padding disagreement

```
BSD:  '      42\n'           # 6 spaces + number
GNU:  '42\n'                  # no padding, number followed by newline
```

Scripts using `cut -c1-8` or `awk '{print $1}'` on `wc -l` output
parse different regions depending on platform. Clean solution: use
`wc -l < file` and rely on the single field.

## Why this is a novel use of our harvest pipeline

Most differential testing tools (AFL, Honggfuzz, CSmith) need
semantic-aware oracles to detect divergence. Our approach is
simpler: run the SAME aggressive fuzz probes through two
implementations of the same spec, any byte difference is a finding.
The fuzzer we built for verifying our Python reimpls turns out to
be a BSD/GNU compatibility scanner for free.

## Next steps

- Extend fuzzers to the skipped tools (nl, paste, seq, tac, expand)
  for full coverage.
- Run against **rust-coreutils** (uutils) when available — three-way
  diff (BSD vs GNU vs Rust) would be a published dataset in itself.
- Package as a standalone tool: `coreutils-diff <tool>` that any
  scriptwriter can run pre-merge to detect portability issues.

## Reproduce

```bash
brew install coreutils  # installs gsort, guniq, etc.
python3 tools/binary_harvest/diff_test.py --all --n 200 \\
    --out /tmp/bsd_vs_gnu.json
```
