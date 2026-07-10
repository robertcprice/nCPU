#!/usr/bin/env python3
# MBPP prose->code through the SYNERGISTIC Rust repo-agent (bypasses the prose-router Mog barrier).
# FINDING (2026-07-09): routing MBPP prose+examples through `coding_agent "fix the failing tests"`
# (scaffold a crate: signature inferred from examples, examples as cargo tests) gives ~58% MODEL-FREE
# (22/38 attempted) vs ~13% via the verified_nl_router prose path -- the router asks the model for the
# Mog DSL and doesn't mine the examples through the Rust ladder. The repo-agent is engine-first ->
# model-writes-Rust -> cargo-verify+repair, so with NSYNTH_LOCAL_LLM_URL set it layers the model on top,
# all cargo-gated (never-wrong). Usage:  python3 scripts/mbpp_repo_agent.py [N]   (NSYNTH_LOCAL_LLM_URL
# optional). Prep: python3 scripts/mbpp_prepare.py /tmp/mbpp_bench.jsonl

import json, os, subprocess, sys, tempfile
AGENT=os.environ.get("NSYNTH_CODING_AGENT") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "target", "release", "coding_agent")
BENCH=os.environ.get("NSYNTH_MBPP_BENCH", "/tmp/mbpp_bench.jsonl")
N=int(sys.argv[1]) if len(sys.argv)>1 else 40
TMO=int(os.environ.get("SC_TIMEOUT","200"))

def rtype(v):
    if isinstance(v,bool): return "bool"
    if isinstance(v,int): return "i64"
    if isinstance(v,str): return "String"
    if isinstance(v,list):
        if all(isinstance(x,int) and not isinstance(x,bool) for x in v): return "Vec<i64>"
        if all(isinstance(x,list) for x in v):
            inner={rtype(x) for x in v} if v else {"Vec<i64>"}
            if inner=={"Vec<i64>"}: return "Vec<Vec<i64>>"
        if all(isinstance(x,str) for x in v): return "Vec<String>"
    return None

def rlit(v):
    if isinstance(v,bool): return "true" if v else "false"
    if isinstance(v,int): return str(v)
    if isinstance(v,str): return json.dumps(v)+".to_string()"
    if isinstance(v,list):
        return "vec!["+", ".join(rlit(x) for x in v)+"]"
    return None

def default(t):
    return {"i64":"0","bool":"false","String":"String::new()"}.get(t, "Vec::new()" if t and t.startswith("Vec") else "Default::default()")

solved=refused=skip=0; n=0
base=tempfile.mkdtemp(prefix="sc_"); os.environ["HOME"]=base+"/home"; os.makedirs(os.environ["HOME"],exist_ok=True)
for line in open(BENCH):
    n+=1
    if n>N: break
    t=json.loads(line); fn=t["fn"]; exs=t["examples"]
    if not exs: skip+=1; continue
    ptypes=[rtype(a) for a in exs[0]["in"]]; rt=rtype(exs[0]["out"])
    if rt is None or any(p is None for p in ptypes): skip+=1; print(f"  {t['id']:>4} {fn:<26} SKIP(type)"); continue
    params=", ".join(f"a{i}: {p}" for i,p in enumerate(ptypes))
    stub=f"pub fn {fn}({params}) -> {rt} {{ {default(rt)} }}\n"
    tests=[]
    for e in exs:
        args=", ".join(rlit(a) for a in e["in"]); outl=rlit(e["out"])
        if args is None or outl is None: continue
        tests.append(f"    #[test] fn tt{len(tests)}() {{ assert_eq!({fn}({args}), {outl}); }}")
    if not tests: skip+=1; continue
    src=stub+"#[cfg(test)]\nmod tests {\n    use super::*;\n"+"\n".join(tests)+"\n}\n"
    d=os.path.join(base,f"t{t['id']}"); os.makedirs(d+"/src",exist_ok=True)
    open(d+"/Cargo.toml","w").write(f'[package]\nname="t{t["id"]}"\nversion="0.0.0"\nedition="2021"\n')
    open(d+"/src/lib.rs","w").write(src)
    # baseline must fail (stub) else skip
    if subprocess.run(["cargo","test","--quiet"],cwd=d,capture_output=True).returncode==0: skip+=1; continue
    try: subprocess.run([AGENT,"--root",d,"query","fix the failing tests"],capture_output=True,timeout=TMO)
    except Exception: pass
    ok = subprocess.run(["cargo","test","--quiet"],cwd=d,capture_output=True).returncode==0
    solved+=ok
    print(f"  {t['id']:>4} {fn:<26} {'RESOLVED' if ok else 'unresolved'}",flush=True)
att=solved+(n-skip-solved-(1 if n>N else 0))
print(f"\nSCAFFOLD-SOLVE (Rust repo-agent): {solved} resolved / {n-skip} attempted (skip={skip} unrepresentable)")
