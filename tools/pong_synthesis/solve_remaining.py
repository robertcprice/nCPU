import json, subprocess, time
BIN = '/Users/bobbyprice/projects/nCPU/nsynth/target/release/mog_synth'
RULES = {
 'neg': ('fn neg(v: i64) -> i64', [([3],-3),([-2],2),([5],-5),([0],0),([-11],11),([7],-7)]),
 'sub2': ('fn sub2(a: i64, b: i64) -> i64', [([7,3],4),([2,5],-3),([0,0],0),([10,-4],14),([-3,-8],5),([100,1],99)]),
 'max2': ('fn max2(a: i64, b: i64) -> i64', [([3,7],7),([9,2],9),([4,4],4),([-5,-2],-2),([0,-9],0),([12,15],15)]),
 'min2': ('fn min2(a: i64, b: i64) -> i64', [([3,7],3),([9,2],2),([4,4],4),([-5,-2],-5),([0,-9],-9),([12,15],12)]),
 'abs2': ('fn abs2(v: i64) -> i64', [([5],5),([-5],5),([0],0),([-12],12),([9],9),([-1],1)]),
 'gte': ('fn gte(a: i64, b: i64) -> i64', [([5,3],1),([3,5],0),([4,4],1),([-2,-7],1),([-7,-2],0),([0,1],0),([1,0],1),([10,10],1)]),
 'hit_top': ('fn hit_top(y: i64) -> i64', [([0],1),([-1],1),([-5],1),([1],0),([3],0),([100],0),([600],0),([-20],1)]),
 'grow': ('fn grow(v: i64) -> i64', [([5],6),([3],4),([-5],-6),([-2],-3),([1],2),([-1],-2),([10],11),([-10],-11)]),
 'crossed_right': ('fn crossed_right(prev: i64, next: i64, plane: i64) -> i64',
   [([760,770,766],1),([760,766,766],1),([766,770,766],0),([770,780,766],0),([700,710,766],0),([765,766,766],1),([766,767,766],0),([100,900,766],1)]),
 'score_if_out_right': ('fn score_if_out_right(score: i64, ball_x: i64, w: i64) -> i64',
   [([0,801,800],1),([3,805,800],4),([2,400,800],2),([5,800,800],5),([7,950,800],8),([1,0,800],1)]),
 'score_if_out_left': ('fn score_if_out_left(score: i64, ball_x: i64) -> i64',
   [([0,-1],1),([3,-10],4),([2,400],2),([5,0],5),([7,-200],8),([1,799],1)]),
}
out = {}
for name, (sig, pairs) in RULES.items():
    prob = {'name': name, 'signature': sig,
            'examples': [{'inputs': list(i), 'expected': e} for i, e in pairs]}
    t0 = time.time()
    try:
        r = subprocess.run([BIN, '--problem-json', '-'], input=json.dumps(prob),
                           capture_output=True, text=True, timeout=240)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        res = {'success': False, 'error': 'timeout'}
    except Exception as ex:
        res = {'success': False, 'error': str(ex)}
    dt = time.time() - t0
    out[name] = {'verified': bool(res.get('success')), 'method': res.get('method'),
                 'mog': res.get('code'), 'examples': prob['examples'],
                 'elapsedSec': round(dt, 1), 'error': res.get('error')}
    print(f"{name}: {'OK' if res.get('success') else 'FAIL'} {res.get('method','')} {dt:.1f}s", flush=True)
    json.dump(out, open('/tmp/pong_rules_h.json', 'w'), indent=1)
print('done:', sum(1 for v in out.values() if v['verified']), '/', len(out))
