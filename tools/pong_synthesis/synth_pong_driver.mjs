// CEGIS driver for the synthesized-Pong demo (nCPU Rung 1).
// For each game rule: synthesize Mog from I/O examples via nsynth,
// transpile to TypeScript, strip types -> JS, then verify the *shipped JS*
// exhaustively / densely over the game's actual input domain against a
// reference. Mismatches become counterexamples and we re-synthesize.
import { execFileSync } from 'node:child_process';
import { writeFileSync, readFileSync, existsSync } from 'node:fs';

const BIN = '/Users/bobbyprice/projects/nCPU/nsynth/target/release/mog_synth';
const OUT = process.env.PONG_OUT || '/tmp/pong_rules.json';

// Game constants (integer grid)
const W = 800, H = 600, PH = 120, MAXV = 11;

function synthesize(problem) {
  const json = JSON.stringify(problem);
  try {
    const out = execFileSync(BIN, ['--problem-json', '-'], {
      input: json, timeout: 240_000, maxBuffer: 64 * 1024 * 1024,
      stdio: ['pipe', 'pipe', 'pipe'],
    }).toString();
    const lines = out.trim().split('\n');
    return JSON.parse(lines[lines.length - 1]);
  } catch (e) {
    return { success: false, error: String(e.message || e).slice(0, 300) };
  }
}

// The transpiler parses block-style `if cond {\n  body\n}` but mis-parses the
// solver's single-line `if cond { body }` form. Reformat (formatting-only —
// the program itself is untouched).
function normalizeMog(mog) {
  return mog
    .split('\n')
    .flatMap((line) => {
      const m = line.match(/^(\s*)if (.+?) \{ (.+?) \}(?: else \{ (.+?) \})?\s*$/);
      if (!m) return [line];
      const [, ind, cond, thenBody, elseBody] = m;
      const stmts = (b) =>
        b.split(';').map((s) => s.trim()).filter(Boolean).map((s) => `${ind}    ${s};`);
      const out = [`${ind}if ${cond} {`, ...stmts(thenBody)];
      if (elseBody) out.push(`${ind}} else {`, ...stmts(elseBody));
      out.push(`${ind}}`);
      return out;
    })
    .join('\n');
}

function transpile(mog, target) {
  return execFileSync(BIN, ['--transpile', target], {
    input: mog, timeout: 30_000,
  }).toString().trimEnd() + '\n';
}

function tsToJs(ts) {
  return ts
    .replace(/: number(\[\])?/g, '')
    .replace(/: string/g, '');
}

function compileFn(js, name) {
  return new Function(`${js}; return ${name};`)();
}

// deterministic PRNG for reproducible sampling
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function sample(cases, n, seed) {
  const rnd = mulberry32(seed);
  const picked = [];
  const used = new Set();
  while (picked.length < Math.min(n, cases.length)) {
    const i = Math.floor(rnd() * cases.length);
    if (used.has(i)) continue;
    used.add(i);
    picked.push(cases[i]);
  }
  return picked;
}

// ----------------------------------------------------------------- rules
// Each rule: name, signature, ref (ground truth), initial examples
// (hand-picked boundary cases), domain() -> full verification case list.
const RULES = [
  {
    name: 'next_pos',
    signature: 'fn next_pos(pos: i64, vel: i64) -> i64',
    ref: (p, v) => p + v,
    examples: [
      [10, 2], [5, -3], [0, 4], [100, -7], [42, 0], [799, 11], [-5, -11], [600, 1],
    ],
    domain() {
      const cs = [];
      for (let p = -60; p <= 860; p += 1)
        for (let v = -MAXV - 1; v <= MAXV + 1; v += 1) cs.push([p, v]);
      return cs;
    },
  },
  {
    name: 'wall_bounce',
    signature: 'fn wall_bounce(y: i64, vy: i64, h: i64) -> i64',
    // y at/past top wall -> force downward (positive); at/past bottom -> force upward
    ref: (y, vy, h) => (y <= 0 ? Math.abs(vy) : y >= h ? -Math.abs(vy) : vy),
    examples: [
      [0, -3, 600], [-2, -5, 600], [0, 3, 600], [-1, 7, 600],
      [600, 4, 600], [605, 7, 600], [600, -2, 600], [610, -6, 600],
      [300, 4, 600], [300, -4, 600], [1, -9, 600], [599, 9, 600],
      [5, -3, 600], [595, 3, 600],
      [0, -3, 400], [400, 5, 400], [200, -2, 400], [399, 8, 400], [1, 6, 400],
    ],
    domain() {
      const cs = [];
      for (let y = -30; y <= 630; y += 1)
        for (let vy = -MAXV - 1; vy <= MAXV + 1; vy += 1) cs.push([y, vy, 600]);
      // generality probes at other heights
      for (let y = -10; y <= 410; y += 7)
        for (let vy = -9; vy <= 9; vy += 3) cs.push([y, vy, 400]);
      return cs;
    },
  },
  {
    name: 'paddle_hit',
    signature: 'fn paddle_hit(ball_y: i64, paddle_y: i64, ph: i64) -> i64',
    ref: (by, py, ph) => (by >= py && by <= py + ph ? 1 : 0),
    examples: [
      [100, 100, 120], [220, 100, 120], [99, 100, 120], [221, 100, 120],
      [160, 100, 120], [0, 0, 120], [120, 0, 120], [121, 0, 120],
      [500, 480, 120], [480, 480, 120], [600, 480, 120], [479, 480, 120],
      [50, 200, 120], [400, 200, 120], [250, 200, 100], [301, 200, 100],
    ],
    domain() {
      const cs = [];
      for (let by = -10; by <= 610; by += 1)
        for (let py = 0; py <= H - PH; py += 12) cs.push([by, py, PH]);
      for (let by = 0; by <= 300; by += 5)
        for (let py = 0; py <= 200; py += 25) cs.push([by, py, 100]);
      return cs;
    },
  },
  {
    name: 'select',
    signature: 'fn select(c: i64, a: i64, b: i64) -> i64',
    ref: (c, a, b) => (c !== 0 ? a : b),
    examples: [
      [1, 5, 9], [0, 5, 9], [1, -3, 7], [0, -3, 7], [2, 100, 200],
      [0, 0, 1], [1, 0, 1], [0, -50, 50], [1, 400, 300], [0, 400, 300],
    ],
    domain() {
      const cs = [];
      for (let c = 0; c <= 2; c += 1)
        for (let a = -60; a <= 860; a += 9)
          for (let b = -60; b <= 860; b += 11) cs.push([c, a, b]);
      return cs;
    },
  },
  {
    name: 'reflect_x',
    signature: 'fn reflect_x(vx: i64, hit: i64) -> i64',
    // hit is a 0/1 flag (every flag producer in the game is verified 0/1)
    ref: (vx, hit) => (hit !== 0 ? -vx : vx),
    examples: [
      [5, 1], [5, 0], [-5, 1], [-5, 0], [11, 1], [-11, 0], [-7, 1], [0, 1], [0, 0], [3, 1], [7, 0],
    ],
    domain() {
      const cs = [];
      for (let vx = -MAXV - 1; vx <= MAXV + 1; vx += 1)
        for (let hit = 0; hit <= 1; hit += 1) cs.push([vx, hit]);
      return cs;
    },
  },
  {
    name: 'flag_and',
    signature: 'fn flag_and(a: i64, b: i64) -> i64',
    ref: (a, b) => (a !== 0 && b !== 0 ? 1 : 0),
    examples: [
      [0, 0], [0, 1], [1, 0], [1, 1],
    ],
    domain() {
      const cs = [];
      for (let a = 0; a <= 1; a += 1) for (let b = 0; b <= 1; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'flag_or',
    signature: 'fn flag_or(a: i64, b: i64) -> i64',
    ref: (a, b) => (a !== 0 || b !== 0 ? 1 : 0),
    examples: [
      [0, 0], [0, 1], [1, 0], [1, 1],
    ],
    domain() {
      const cs = [];
      for (let a = 0; a <= 1; a += 1) for (let b = 0; b <= 1; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'crossed_left',
    signature: 'fn crossed_left(prev: i64, next: i64, plane: i64) -> i64',
    ref: (prev, next, plane) => (prev > plane && next <= plane ? 1 : 0),
    examples: [
      [40, 30, 34], [40, 34, 34], [40, 35, 34], [34, 30, 34], [33, 28, 34],
      [50, 45, 34], [35, 20, 34], [100, 90, 34], [30, 36, 34], [36, 36, 34],
      [60, 50, 55], [56, 55, 55], [55, 50, 55], [54, 60, 55],
    ],
    domain() {
      const cs = [];
      for (let prev = 0; prev <= 120; prev += 1)
        for (let next = prev - MAXV - 1; next <= prev + MAXV + 1; next += 1)
          cs.push([prev, next, 34]);
      for (let prev = 30; prev <= 90; prev += 3)
        for (let next = prev - 12; next <= prev + 12; next += 4) cs.push([prev, next, 55]);
      return cs;
    },
  },
  {
    name: 'crossed_right',
    signature: 'fn crossed_right(prev: i64, next: i64, plane: i64) -> i64',
    ref: (prev, next, plane) => (prev < plane && next >= plane ? 1 : 0),
    examples: [
      [760, 770, 766], [760, 766, 766], [760, 765, 766], [766, 770, 766], [767, 772, 766],
      [750, 755, 766], [765, 780, 766], [700, 710, 766], [770, 760, 766], [766, 766, 766],
      [740, 750, 745], [744, 745, 745], [745, 750, 745], [746, 740, 745],
    ],
    domain() {
      const cs = [];
      for (let prev = 680; prev <= 800; prev += 1)
        for (let next = prev - MAXV - 1; next <= prev + MAXV + 1; next += 1)
          cs.push([prev, next, 766]);
      for (let prev = 710; prev <= 770; prev += 3)
        for (let next = prev - 12; next <= prev + 12; next += 4) cs.push([prev, next, 745]);
      return cs;
    },
  },
  {
    name: 'score_if_out_right',
    signature: 'fn score_if_out_right(score: i64, ball_x: i64, w: i64) -> i64',
    ref: (s, x, w) => (x > w ? s + 1 : s),
    examples: [
      [0, 801, 800], [0, 800, 800], [0, 400, 800], [3, 805, 800], [3, 799, 800],
      [7, 810, 800], [2, 0, 800], [5, -5, 800], [1, 801, 800], [9, 500, 800],
      [4, 601, 600], [4, 600, 600], [4, 300, 600],
    ],
    domain() {
      const cs = [];
      for (let s = 0; s <= 21; s += 1)
        for (let x = -40; x <= 840; x += 1) cs.push([s, x, 800]);
      for (let s = 0; s <= 10; s += 2)
        for (let x = 560; x <= 640; x += 3) cs.push([s, x, 600]);
      return cs;
    },
  },
  {
    name: 'score_if_out_left',
    signature: 'fn score_if_out_left(score: i64, ball_x: i64) -> i64',
    ref: (s, x) => (x < 0 ? s + 1 : s),
    examples: [
      [0, -1, ], [0, 0], [0, 400], [3, -5], [3, 1], [7, -10], [2, 800], [5, 805], [1, -1], [9, 50],
    ],
    domain() {
      const cs = [];
      for (let s = 0; s <= 21; s += 1)
        for (let x = -40; x <= 840; x += 1) cs.push([s, x]);
      return cs;
    },
  },
  {
    name: 'exited_left',
    signature: 'fn exited_left(x: i64) -> i64',
    ref: (x) => (x < 0 ? 1 : 0),
    examples: [
      [-1], [0], [1], [-10], [400], [800], [-3], [12],
    ],
    domain() {
      const cs = [];
      for (let x = -40; x <= 840; x += 1) cs.push([x]);
      return cs;
    },
  },
  {
    name: 'exited_right',
    signature: 'fn exited_right(x: i64, w: i64) -> i64',
    ref: (x, w) => (x > w ? 1 : 0),
    examples: [
      [801, 800], [800, 800], [799, 800], [810, 800], [0, 800], [-5, 800], [401, 400], [400, 400], [200, 400],
    ],
    domain() {
      const cs = [];
      for (let x = -40; x <= 840; x += 1) cs.push([x, 800]);
      for (let x = 360; x <= 440; x += 1) cs.push([x, 400]);
      return cs;
    },
  },
  {
    name: 'clamp',
    signature: 'fn clamp(v: i64, lo: i64, hi: i64) -> i64',
    ref: (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v),
    examples: [
      [-5, 0, 480], [0, 0, 480], [481, 0, 480], [480, 0, 480], [240, 0, 480],
      [-1, 0, 480], [500, 0, 480], [1, 0, 480], [479, 0, 480],
      [5, 10, 90], [95, 10, 90], [50, 10, 90], [10, 10, 90], [90, 10, 90],
    ],
    domain() {
      const cs = [];
      for (let v = -40; v <= 520; v += 1) cs.push([v, 0, H - PH]);
      for (let v = -20; v <= 120; v += 1) cs.push([v, 10, 90]);
      for (let v = -15; v <= 15; v += 1) cs.push([v, -MAXV, MAXV]);
      return cs;
    },
  },
  {
    name: 'paddle_track',
    signature: 'fn paddle_track(pc: i64, ball_y: i64, speed: i64) -> i64',
    // move toward the ball, at most `speed` units per frame
    ref: (pc, by, sp) => {
      const d = by - pc;
      return d > sp ? sp : d < -sp ? -sp : d;
    },
    examples: [
      [300, 400, 5], [300, 302, 5], [300, 300, 5], [300, 200, 5], [300, 298, 5],
      [300, 305, 5], [300, 295, 5], [300, 306, 5], [300, 294, 5],
      [100, 104, 5], [500, 100, 5], [0, 600, 5], [600, 0, 5],
      [250, 258, 7], [250, 242, 7], [250, 280, 7], [250, 220, 7], [250, 251, 7],
    ],
    domain() {
      const cs = [];
      for (let pc = 0; pc <= 600; pc += 4)
        for (let by = 0; by <= 600; by += 4) cs.push([pc, by, 5]);
      for (let pc = 0; pc <= 600; pc += 17)
        for (let by = 0; by <= 600; by += 13) cs.push([pc, by, 7]);
      return cs;
    },
  },
  // ---------------- decomposition primitives (for hard contracts) ----------
  {
    name: 'hit_top',
    signature: 'fn hit_top(y: i64) -> i64',
    ref: (y) => (y <= 0 ? 1 : 0),
    examples: [[0], [1], [-1], [5], [-7], [300], [600], [2], [-30], [630]],
    domain() {
      const cs = [];
      for (let y = -30; y <= 630; y += 1) cs.push([y]);
      return cs;
    },
  },
  {
    name: 'hit_bottom',
    signature: 'fn hit_bottom(y: i64, h: i64) -> i64',
    ref: (y, h) => (y >= h ? 1 : 0),
    examples: [
      [600, 600], [599, 600], [601, 600], [610, 600], [0, 600], [300, 600],
      [400, 400], [399, 400], [405, 400],
    ],
    domain() {
      const cs = [];
      for (let y = -30; y <= 630; y += 1) cs.push([y, 600]);
      for (let y = 360; y <= 440; y += 1) cs.push([y, 400]);
      return cs;
    },
  },
  {
    name: 'moving_up',
    signature: 'fn moving_up(vy: i64) -> i64',
    ref: (vy) => (vy < 0 ? 1 : 0),
    examples: [[-1], [0], [1], [-5], [7], [-11], [11], [3], [-12], [12]],
    domain() {
      const cs = [];
      for (let v = -14; v <= 14; v += 1) cs.push([v]);
      return cs;
    },
  },
  {
    name: 'moving_down',
    signature: 'fn moving_down(vy: i64) -> i64',
    ref: (vy) => (vy > 0 ? 1 : 0),
    examples: [[-1], [0], [1], [-5], [7], [-11], [11], [3], [-12], [12]],
    domain() {
      const cs = [];
      for (let v = -14; v <= 14; v += 1) cs.push([v]);
      return cs;
    },
  },
  {
    name: 'gt',
    signature: 'fn gt(a: i64, b: i64) -> i64',
    ref: (a, b) => (a > b ? 1 : 0),
    examples: [
      [5, 3], [3, 5], [4, 4], [0, 0], [-2, 1], [1, -2], [100, 220], [220, 100],
      [99, 100], [100, 99], [-5, -5], [-6, -5], [-5, -6], [1, 0], [0, 1],
    ],
    domain() {
      const cs = [];
      for (let a = -40; a <= 640; a += 3)
        for (let b = a - 15; b <= a + 15; b += 1) cs.push([a, b]);
      for (let a = -20; a <= 20; a += 1)
        for (let b = -20; b <= 20; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'gte',
    signature: 'fn gte(a: i64, b: i64) -> i64',
    ref: (a, b) => (a >= b ? 1 : 0),
    examples: [
      [5, 3], [3, 5], [4, 4], [0, 0], [-2, 1], [1, -2], [100, 220], [220, 100],
      [99, 100], [100, 99], [-5, -5], [-6, -5], [-5, -6],
    ],
    domain() {
      const cs = [];
      for (let a = -40; a <= 640; a += 3)
        for (let b = a - 15; b <= a + 15; b += 1) cs.push([a, b]);
      for (let a = -20; a <= 20; a += 1)
        for (let b = -20; b <= 20; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'max2',
    signature: 'fn max2(a: i64, b: i64) -> i64',
    ref: (a, b) => (a > b ? a : b),
    examples: [
      [3, 5], [5, 3], [0, 0], [-2, 4], [4, -2], [-7, -3], [100, 40], [40, 100],
      [0, -1], [-1, 0], [7, 7], [-11, 5], [480, 481],
    ],
    domain() {
      const cs = [];
      for (let a = -60; a <= 660; a += 7)
        for (let b = -60; b <= 660; b += 11) cs.push([a, b]);
      for (let a = -15; a <= 15; a += 1)
        for (let b = -15; b <= 15; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'min2',
    signature: 'fn min2(a: i64, b: i64) -> i64',
    ref: (a, b) => (a < b ? a : b),
    examples: [
      [3, 5], [5, 3], [0, 0], [-2, 4], [4, -2], [-7, -3], [100, 40], [40, 100],
      [0, -1], [-1, 0], [7, 7], [-11, 5], [480, 481],
    ],
    domain() {
      const cs = [];
      for (let a = -60; a <= 660; a += 7)
        for (let b = -60; b <= 660; b += 11) cs.push([a, b]);
      for (let a = -15; a <= 15; a += 1)
        for (let b = -15; b <= 15; b += 1) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'sub2',
    signature: 'fn sub2(a: i64, b: i64) -> i64',
    ref: (a, b) => a - b,
    examples: [[10, 3], [3, 10], [0, 0], [5, -2], [-4, 6], [600, 300], [42, 42], [-7, -3]],
    domain() {
      const cs = [];
      for (let a = -60; a <= 660; a += 7)
        for (let b = -60; b <= 660; b += 11) cs.push([a, b]);
      return cs;
    },
  },
  {
    name: 'neg',
    signature: 'fn neg(v: i64) -> i64',
    ref: (v) => -v,
    examples: [[1], [-1], [0], [5], [-7], [11], [-11], [3]],
    domain() {
      const cs = [];
      for (let v = -30; v <= 30; v += 1) cs.push([v]);
      return cs;
    },
  },
  {
    name: 'abs2',
    signature: 'fn abs2(v: i64) -> i64',
    ref: (v) => (v < 0 ? -v : v),
    examples: [[1], [-1], [0], [5], [-7], [11], [-11], [3], [-12], [12]],
    domain() {
      const cs = [];
      for (let v = -100; v <= 100; v += 1) cs.push([v]);
      return cs;
    },
  },
  {
    name: 'grow',
    signature: 'fn grow(v: i64) -> i64',
    ref: (v) => (v > 0 ? v + 1 : v - 1),
    examples: [[1], [-1], [5], [-5], [10], [-10], [3], [-7], [11], [-11], [0]],
    domain() {
      const cs = [];
      for (let v = -14; v <= 14; v += 1) cs.push([v]);
      return cs;
    },
  },
  {
    name: 'speed_up',
    signature: 'fn speed_up(v: i64, vmax: i64) -> i64',
    // grow magnitude by 1 on paddle hit, capped at vmax
    ref: (v, vmax) => {
      if (Math.abs(v) >= vmax) return v;
      return v > 0 ? v + 1 : v < 0 ? v - 1 : v;
    },
    examples: [
      [5, 11], [-5, 11], [10, 11], [-10, 11], [11, 11], [-11, 11], [12, 11], [-12, 11],
      [1, 11], [-1, 11], [0, 11], [4, 8], [-4, 8], [8, 8], [-8, 8], [7, 8], [-7, 8],
    ],
    domain() {
      const cs = [];
      for (let v = -14; v <= 14; v += 1) cs.push([v, 11]);
      for (let v = -10; v <= 10; v += 1) cs.push([v, 8]);
      return cs;
    },
  },
];

// ------------------------------------------------------------------ CEGIS
const results = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const only = process.argv.slice(2); // optional rule-name filter

for (const rule of RULES) {
  if (only.length && !only.includes(rule.name)) continue;
  if (results[rule.name]?.verified) {
    console.log(`[skip] ${rule.name} already verified`);
    continue;
  }
  const t0 = Date.now();
  const domain = rule.domain();
  let examples = rule.examples.map((inputs) => ({
    inputs, expected: rule.ref(...inputs),
  }));
  let record = null;
  let lastErr = '';

  for (let iter = 1; iter <= 6; iter++) {
    // holdouts: dense deterministic sample of the domain (solver re-verifies these)
    const holdouts = sample(domain, 500, 1234 + iter).map((inputs) => ({
      inputs, expected: rule.ref(...inputs),
    }));
    process.stdout.write(`[${rule.name}] iter ${iter}: ${examples.length} examples, synthesizing... `);
    const res = synthesize({
      name: rule.name, signature: rule.signature, examples, holdouts,
    });
    if (!res.success) {
      lastErr = res.error || 'synthesis refused';
      console.log(`REFUSED (${lastErr})`);
      break;
    }
    const mog = normalizeMog(res.code);
    const ts = transpile(mog, 'typescript');
    const js = tsToJs(ts);
    let fn;
    try {
      fn = compileFn(js, rule.name);
    } catch (e) {
      lastErr = `transpiled JS failed to compile: ${e.message}`;
      console.log(`BAD-JS (${lastErr})`);
      break;
    }
    // full-domain verification of the SHIPPED JS
    const mism = [];
    for (const inputs of domain) {
      const want = rule.ref(...inputs);
      const got = fn(...inputs);
      if (got !== want) {
        mism.push(inputs);
        if (mism.length >= 4000) break;
      }
    }
    if (mism.length === 0) {
      console.log(`VERIFIED via ${res.method} (${domain.length} domain cases, ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
      record = {
        verified: true,
        signature: rule.signature,
        method: res.method,
        iterations: iter,
        mog,
        ts,
        examples,
        domainCases: domain.length,
        elapsedSec: +(((Date.now() - t0) / 1000).toFixed(1)),
      };
      break;
    }
    console.log(`${mism.length}+ mismatches (method ${res.method}); adding counterexamples`);
    // add up to 10 spread-out counterexamples
    const add = sample(mism, 10, 99 + iter);
    for (const inputs of add) {
      examples.push({ inputs, expected: rule.ref(...inputs) });
    }
  }

  results[rule.name] = record ?? { verified: false, error: lastErr, examplesTried: examples.length };
  writeFileSync(OUT, JSON.stringify(results, null, 2));
}

const ok = Object.values(results).filter((r) => r.verified).length;
console.log(`\ndone: ${ok}/${Object.keys(results).length} rules verified -> ${OUT}`);
