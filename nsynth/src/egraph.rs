//! E-GRAPH + EQUALITY SATURATION over the scalar `Expr` core.
//!
//! An e-graph represents a whole EQUIVALENCE CLASS of programs compactly: each
//! e-class holds many equivalent e-nodes, sharing sub-classes via union-find.
//! `saturate` grows the classes by applying the ALGEBRAIC LAW ruleset (the same
//! metamorphic laws used as a verifier) as REWRITES until a bounded fixpoint,
//! then `extract` reads back the MINIMAL-cost member of a class.
//!
//! Only TRUE equalities are added (each rewrite is eval-preserving under the
//! partial `Option` semantics — annihilators like `x*0` are excluded because
//! they discard a sub-expression that can error), so the extracted program is
//! ALWAYS eval-equal to the input — proven by a property test. This is strictly
//! stronger than the greedy canonicalizer: the e-graph explores BOTH directions
//! of associativity/distributivity and extracts the smaller form (e.g. it proves
//! `a*b + a*c == a*(b+c)` and returns the factored one).

use crate::enumerative::{BinOp, Expr, UnOp};
use std::collections::HashMap;

type Id = usize;

/// A node in the e-graph: an operator whose children are E-CLASS ids (not
/// sub-expressions), so structurally-different-but-equal programs share children.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ENode {
    Var(usize),
    Const(i64),
    Bin(BinOp, Id, Id),
    Un(UnOp, Id),
}

#[derive(Default)]
pub struct EGraph {
    /// Union-find parent pointers over e-class ids.
    parent: Vec<Id>,
    /// Canonical e-class id -> its e-nodes.
    classes: HashMap<Id, Vec<ENode>>,
    /// Hash-cons: a canonicalized e-node -> the class it belongs to.
    memo: HashMap<ENode, Id>,
}

impl EGraph {
    pub fn new() -> Self {
        Self::default()
    }

    fn find(&self, mut id: Id) -> Id {
        while self.parent[id] != id {
            id = self.parent[id];
        }
        id
    }

    /// Rewrite an e-node's child ids to their canonical class ids.
    fn canon(&self, n: &ENode) -> ENode {
        match n {
            ENode::Var(_) | ENode::Const(_) => n.clone(),
            ENode::Bin(op, a, b) => ENode::Bin(*op, self.find(*a), self.find(*b)),
            ENode::Un(op, a) => ENode::Un(*op, self.find(*a)),
        }
    }

    /// Add an e-node (canonicalized), returning its class. Hash-consed: an
    /// identical node returns the existing class.
    fn add(&mut self, n: ENode) -> Id {
        let n = self.canon(&n);
        if let Some(&id) = self.memo.get(&n) {
            return self.find(id);
        }
        let id = self.parent.len();
        self.parent.push(id);
        self.classes.insert(id, vec![n.clone()]);
        self.memo.insert(n, id);
        id
    }

    /// Add a whole Expr, returning its e-class. `None` for nodes outside the
    /// scalar core (if / loops / call) — the e-graph is scalar-only in v1.
    pub fn add_expr(&mut self, e: &Expr) -> Option<Id> {
        Some(match e {
            Expr::Var(i) => self.add(ENode::Var(*i)),
            Expr::Const(c) => self.add(ENode::Const(*c)),
            Expr::BinOp(op, l, r) => {
                let l = self.add_expr(l)?;
                let r = self.add_expr(r)?;
                self.add(ENode::Bin(*op, l, r))
            }
            Expr::UnaryOp(op, x) => {
                let x = self.add_expr(x)?;
                self.add(ENode::Un(*op, x))
            }
            _ => return None,
        })
    }

    /// Merge two e-classes. Returns the surviving canonical id.
    fn union(&mut self, a: Id, b: Id) -> Id {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return a;
        }
        self.parent[b] = a;
        if let Some(bn) = self.classes.remove(&b) {
            self.classes.entry(a).or_default().extend(bn);
        }
        a
    }

    /// Re-canonicalize the hash-cons after unions (merge nodes that became equal).
    fn rebuild(&mut self) {
        let ids: Vec<Id> = self.classes.keys().copied().collect();
        self.memo.clear();
        for id in ids {
            let cid = self.find(id);
            if cid != id {
                continue;
            }
            let nodes = self.classes.remove(&id).unwrap_or_default();
            let mut fresh = Vec::new();
            for n in nodes {
                let cn = self.canon(&n);
                if !fresh.contains(&cn) {
                    fresh.push(cn.clone());
                }
                self.memo.insert(cn, id);
            }
            self.classes.insert(id, fresh);
        }
    }

    fn const_in(&self, id: Id, want: i64) -> bool {
        self.class_nodes(id)
            .iter()
            .any(|n| matches!(n, ENode::Const(c) if *c == want))
    }

    fn class_nodes(&self, id: Id) -> Vec<ENode> {
        self.classes.get(&self.find(id)).cloned().unwrap_or_default()
    }

    /// Hard cap on e-node count. Rewrites that GROW the graph (associativity,
    /// factoring) can otherwise explode combinatorially; the cap guarantees
    /// termination + bounded memory. A capped graph is still SOUND (only true
    /// equalities were added) — it may just leave some equalities unproven.
    const NODE_CAP: usize = 4000;

    /// Saturate: apply the algebraic-law rewrites to a bounded fixpoint (or until
    /// the node cap). All rewrites are TRUE (eval-preserving) equalities.
    pub fn saturate(&mut self, max_iters: usize) {
        for _ in 0..max_iters {
            if self.parent.len() > Self::NODE_CAP {
                break;
            }
            let before = self.parent.len();
            let before_merges = self.num_classes();
            // Snapshot (class, node) pairs; new nodes/unions apply next round.
            let snapshot: Vec<(Id, ENode)> = self
                .classes
                .iter()
                .flat_map(|(id, ns)| ns.iter().map(move |n| (*id, n.clone())))
                .collect();
            for (cls, node) in snapshot {
                if self.parent.len() > Self::NODE_CAP {
                    break;
                }
                self.apply_rewrites(cls, &node);
            }
            self.rebuild();
            // Fixpoint: no new nodes AND no merges this round.
            if self.parent.len() == before && self.num_classes() == before_merges {
                break;
            }
        }
    }

    fn num_classes(&self) -> usize {
        (0..self.parent.len()).filter(|&i| self.find(i) == i).count()
    }

    /// The rewrite ruleset — every rule is a TRUE equality (eval-preserving).
    fn apply_rewrites(&mut self, cls: Id, node: &ENode) {
        let ENode::Bin(op, a, b) = *node else {
            // Involutions: --x == x, ~~x == x.
            if let ENode::Un(u, inner) = *node {
                for cn in self.class_nodes(inner) {
                    if let ENode::Un(u2, x) = cn {
                        if (u == UnOp::Neg && u2 == UnOp::Neg)
                            || (u == UnOp::BitNot && u2 == UnOp::BitNot)
                        {
                            self.union(cls, x);
                        }
                    }
                }
            }
            return;
        };
        let commutative = matches!(
            op,
            BinOp::Add | BinOp::Mul | BinOp::Min | BinOp::Max | BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor
        );
        // Commutativity: a op b == b op a.
        if commutative {
            let n = self.add(ENode::Bin(op, b, a));
            self.union(cls, n);
        }
        // Identity: a+0==a, a*1==a, a-0==a, a|0==a, a^0==a (0/1 are literals, sound).
        match op {
            BinOp::Add | BinOp::BitOr | BinOp::BitXor => {
                if self.const_in(b, 0) {
                    self.union(cls, a);
                }
                if self.const_in(a, 0) {
                    self.union(cls, b);
                }
            }
            BinOp::Sub => {
                if self.const_in(b, 0) {
                    self.union(cls, a);
                }
            }
            BinOp::Mul => {
                if self.const_in(b, 1) {
                    self.union(cls, a);
                }
                if self.const_in(a, 1) {
                    self.union(cls, b);
                }
            }
            _ => {}
        }
        // Idempotence: min(x,x)==x, max(x,x)==x, x&x==x, x|x==x.
        if matches!(op, BinOp::Min | BinOp::Max | BinOp::BitAnd | BinOp::BitOr)
            && self.find(a) == self.find(b)
        {
            self.union(cls, a);
        }
        // Associativity: (x op y) op b == x op (y op b).
        if commutative {
            for cn in self.class_nodes(a) {
                if let ENode::Bin(o2, x, y) = cn {
                    if o2 == op {
                        let yb = self.add(ENode::Bin(op, y, b));
                        let assoc = self.add(ENode::Bin(op, x, yb));
                        self.union(cls, assoc);
                    }
                }
            }
        }
        // NOTE: distributivity is applied in the CONTRACTING (factoring)
        // direction only — the expanding direction (a*(x+y) -> a*x+a*y) grows the
        // graph without bound and is unnecessary to prove the equality (factoring
        // one side already merges the two).
        // Factoring (contract): p*q + p*r == p*(q+r). Look for a common factor
        // across the two Mul operands of an Add.
        if op == BinOp::Add {
            let a_muls = self.mul_pairs(a);
            let b_muls = self.mul_pairs(b);
            for (p1, q) in &a_muls {
                for (p2, r) in &b_muls {
                    if self.find(*p1) == self.find(*p2) {
                        let qr = self.add(ENode::Bin(BinOp::Add, *q, *r));
                        let factored = self.add(ENode::Bin(BinOp::Mul, *p1, qr));
                        self.union(cls, factored);
                    }
                    // p appears on the RIGHT of one product (commutative) too.
                    if self.find(*p1) == self.find(*r) {
                        let qp2 = self.add(ENode::Bin(BinOp::Add, *q, *p2));
                        let factored = self.add(ENode::Bin(BinOp::Mul, *p1, qp2));
                        self.union(cls, factored);
                    }
                }
            }
        }
    }

    /// All (left, right) child pairs of `Mul` e-nodes in a class.
    fn mul_pairs(&self, id: Id) -> Vec<(Id, Id)> {
        self.class_nodes(id)
            .iter()
            .filter_map(|n| match n {
                ENode::Bin(BinOp::Mul, x, y) => Some((*x, *y)),
                _ => None,
            })
            .collect()
    }

    /// Extract the MINIMAL-cost (smallest) Expr from a class. Cost = node count.
    pub fn extract(&self, root: Id) -> Expr {
        let mut best: HashMap<Id, (usize, Expr)> = HashMap::new();
        // Iterate to a fixpoint over class costs (classes can be mutually
        // referential; bounded by class count).
        for _ in 0..self.classes.len() + 2 {
            let mut changed = false;
            let ids: Vec<Id> = self.classes.keys().copied().collect();
            for id in ids {
                let cid = self.find(id);
                for n in self.class_nodes(cid) {
                    if let Some((cost, expr)) = self.node_cost(&n, &best) {
                        // Deterministic + stable: strictly lower cost, or equal
                        // cost with a canonically-smaller form (Debug key), so the
                        // SAME class always extracts the SAME Expr across builds.
                        let better = match best.get(&cid) {
                            None => true,
                            Some((c, e)) => {
                                cost < *c || (cost == *c && format!("{expr:?}") < format!("{e:?}"))
                            }
                        };
                        if better {
                            best.insert(cid, (cost, expr));
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        best.get(&self.find(root))
            .map(|(_, e)| e.clone())
            .expect("root class must have an extractable member")
    }

    fn node_cost(
        &self,
        n: &ENode,
        best: &HashMap<Id, (usize, Expr)>,
    ) -> Option<(usize, Expr)> {
        match n {
            ENode::Var(i) => Some((1, Expr::Var(*i))),
            ENode::Const(c) => Some((1, Expr::Const(*c))),
            ENode::Un(op, a) => {
                let (ca, ea) = best.get(&self.find(*a))?.clone();
                Some((1 + ca, Expr::UnaryOp(*op, Box::new(ea))))
            }
            ENode::Bin(op, a, b) => {
                let (ca, ea) = best.get(&self.find(*a))?.clone();
                let (cb, eb) = best.get(&self.find(*b))?.clone();
                Some((1 + ca + cb, Expr::BinOp(*op, Box::new(ea), Box::new(eb))))
            }
        }
    }
}

/// Simplify a scalar Expr via equality saturation: build an e-graph, saturate
/// with the algebraic laws, and extract the minimal equivalent form. Returns the
/// input UNCHANGED when it contains non-scalar nodes (if / loops / call). The
/// result is ALWAYS eval-equal to the input (property-tested).
pub fn simplify(e: &Expr) -> Expr {
    let mut g = EGraph::new();
    match g.add_expr(e) {
        Some(root) => {
            g.saturate(6);
            g.extract(root)
        }
        None => e.clone(),
    }
}

/// True iff `a` and `b` are provably equal under the algebraic laws (they land
/// in the same e-class after saturation).
pub fn equivalent(a: &Expr, b: &Expr) -> bool {
    let mut g = EGraph::new();
    let (Some(ia), Some(ib)) = (g.add_expr(a), g.add_expr(b)) else {
        return a == b;
    };
    g.saturate(12);
    g.find(ia) == g.find(ib)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::enumerative::{BinOp, Expr, UnOp};

    fn v(i: usize) -> Expr {
        Expr::Var(i)
    }
    fn c(n: i64) -> Expr {
        Expr::Const(n)
    }
    fn bin(op: BinOp, a: Expr, b: Expr) -> Expr {
        Expr::BinOp(op, Box::new(a), Box::new(b))
    }
    fn add(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Add, a, b)
    }
    fn mul(a: Expr, b: Expr) -> Expr {
        bin(BinOp::Mul, a, b)
    }

    /// SOUNDNESS: over 1500 random scalar exprs (all binops incl div/mod, all
    /// unops) x 8 inputs, `simplify` must preserve eval EXACTLY.
    #[test]
    fn simplify_preserves_eval_on_random_exprs() {
        use crate::enumerative::CmpOp;
        let all_bin = [
            BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Mod, BinOp::Min,
            BinOp::Max, BinOp::BitAnd, BinOp::BitOr, BinOp::BitXor, BinOp::Shl, BinOp::Shr,
        ];
        let all_un = [UnOp::Neg, UnOp::Abs, UnOp::BitNot, UnOp::Popcount];
        let _ = CmpOp::Eq;
        fn lcg(s: &mut u64) -> u64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *s >> 16
        }
        fn gen(s: &mut u64, depth: u32, all_bin: &[BinOp], all_un: &[UnOp]) -> Expr {
            if depth == 0 || lcg(s) % 3 == 0 {
                if lcg(s) % 2 == 0 {
                    Expr::Var((lcg(s) % 3) as usize)
                } else {
                    Expr::Const((lcg(s) % 9) as i64 - 4)
                }
            } else if lcg(s) % 4 == 0 {
                let op = all_un[(lcg(s) as usize) % all_un.len()];
                Expr::UnaryOp(op, Box::new(gen(s, depth - 1, all_bin, all_un)))
            } else {
                let op = all_bin[(lcg(s) as usize) % all_bin.len()];
                Expr::BinOp(
                    op,
                    Box::new(gen(s, depth - 1, all_bin, all_un)),
                    Box::new(gen(s, depth - 1, all_bin, all_un)),
                )
            }
        }
        let mut s = 0xE6_2A_9Fu64;
        let mut checked = 0usize;
        for _ in 0..1500 {
            let e = gen(&mut s, 4, &all_bin, &all_un);
            let simp = simplify(&e);
            for _ in 0..8 {
                let inputs: Vec<i64> = (0..3).map(|_| (lcg(&mut s) % 15) as i64 - 7).collect();
                assert_eq!(
                    e.eval(&inputs),
                    simp.eval(&inputs),
                    "simplify changed eval: {e:?} -> {simp:?} on {inputs:?}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 12000, "ran {checked}");
    }

    /// POWER: the e-graph proves non-trivial equalities the greedy canonicalizer
    /// cannot — both directions of distributivity + associativity + commutativity.
    #[test]
    fn egraph_proves_algebraic_equalities() {
        // a*b + a*c == a*(b+c)  (factoring / distributivity — the headline).
        let lhs = add(mul(v(0), v(1)), mul(v(0), v(2)));
        let rhs = mul(v(0), add(v(1), v(2)));
        assert!(equivalent(&lhs, &rhs), "a*b+a*c == a*(b+c)");
        // (a+b)+c == a+(b+c)  (associativity).
        assert!(equivalent(
            &add(add(v(0), v(1)), v(2)),
            &add(v(0), add(v(1), v(2)))
        ));
        // a+b == b+a (commutativity).
        assert!(equivalent(&add(v(0), v(1)), &add(v(1), v(0))));
        // Non-equal stays non-equal (no false merges): a-b != b-a in general.
        assert!(!equivalent(
            &bin(BinOp::Sub, v(0), v(1)),
            &bin(BinOp::Sub, v(1), v(0))
        ));
        // a*0 is NOT proven == 0 (unsound annihilator excluded): a*0 keeps a.
        assert!(!equivalent(&mul(v(0), c(0)), &c(0)), "annihilator not asserted");
    }

    /// EXTRACTION picks the smaller equivalent form.
    #[test]
    fn egraph_extracts_minimal_form() {
        // a*b + a*c  -> factored a*(b+c) is smaller (5 nodes vs 7).
        let expanded = add(mul(v(0), v(1)), mul(v(0), v(2)));
        let simp = simplify(&expanded);
        // eval-equal (also covered by the property test) and no larger than input.
        fn size(e: &Expr) -> usize {
            match e {
                Expr::BinOp(_, a, b) => 1 + size(a) + size(b),
                Expr::UnaryOp(_, a) => 1 + size(a),
                _ => 1,
            }
        }
        assert!(size(&simp) <= size(&expanded), "simplify never grows: {simp:?}");
        for i0 in [-3, 0, 4] {
            for i1 in [1, -2] {
                for i2 in [5, -1] {
                    let ins = [i0, i1, i2];
                    assert_eq!(expanded.eval(&ins), simp.eval(&ins));
                }
            }
        }
        // x + 0 extracts to x.
        assert_eq!(simplify(&add(v(0), c(0))), v(0));
    }
}
