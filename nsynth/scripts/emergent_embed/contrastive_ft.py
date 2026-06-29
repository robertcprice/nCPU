#!/usr/bin/env python3
"""Coding-synonym contrastive fine-tune of the exported morpheme embedding, with a
HELD-OUT synonym-separation gate.

Goal: test whether the (measured-incoherent) nsynth embedding can be made to
separate coding synonyms from non-synonyms WITHOUT a full lg-neural corpus
retrain — by contrastively fine-tuning the exported 5110x256 matrix on
synonym-cluster pairs. The gate holds out a fraction of each cluster's words so
we measure GENERALIZATION to unseen synonyms, not memorization of trained pairs.

Collision-safe: reads linguigenesis/data, writes a NEW artifact
token_emb_coding_ft.safetensors. Never touches lg-neural/src or the original.
"""
import json, sys, itertools, random
import numpy as np
import torch
from safetensors.numpy import load_file, save_file

ROOT = "/Users/bobbyprice/projects/linguigenesis"
EMB = f"{ROOT}/data/nsynth_embeddings/token_emb.safetensors"
TOKJSON = f"{ROOT}/data/nsynth_embeddings/morpheme_tokenizer.json"
CODEREG = f"{ROOT}/data/coding_registry.json"
WNEDGES = f"{ROOT}/data/wordnet_coding_edges.json"
OUT = f"{ROOT}/data/nsynth_embeddings/token_emb_coding_ft.safetensors"

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
random.seed(SEED)
torch.manual_seed(SEED)

tok = json.load(open(TOKJSON))
T2I = tok["token_to_id"]
SUFFIXES = tok.get("suffixes", [])
# longest tokens first for greedy match (exclude single chars from greedy morpheme pass)
MORPHS = sorted([k for k, v in T2I.items() if v > tok["char_end"]], key=len, reverse=True)


def word_to_ids(w):
    w = w.strip().lower()
    if not w:
        return []
    if w in T2I:
        return [T2I[w]]
    # try suffix strip
    for suf in SUFFIXES:
        if w.endswith(suf) and w[: -len(suf)] in T2I:
            return [T2I[w[: -len(suf)]]]
    # greedy longest morpheme match, char fallback
    ids, i = [], 0
    while i < len(w):
        matched = None
        for m in MORPHS:
            if len(m) > 1 and w.startswith(m, i):
                matched = m
                break
        if matched:
            ids.append(T2I[matched]); i += len(matched)
        else:
            ids.append(T2I.get(w[i], tok["unk_id"])); i += 1
    return ids or [tok["unk_id"]]


# ---- build synonym clusters: seed_op -> set(surface words incl seed) ----
def add_cluster(clusters, seed, words):
    s = clusters.setdefault(seed, set())
    s.add(seed.lower())
    for w in words:
        s.add(w.lower())


clusters = {}
creg = json.load(open(CODEREG))
ents = creg["entities"] if isinstance(creg, dict) and "entities" in creg else creg
items = ents.items() if isinstance(ents, dict) else [(e.get("word"), e) for e in ents]
for k, e in items:
    rel = e.get("relations", {})
    syn = rel.get("synonym") or []
    for seed in syn:
        add_cluster(clusters, seed, [k])
for f in (WNEDGES,):
    try:
        d = json.load(open(f))
        de = d["entities"] if isinstance(d, dict) and "entities" in d else d
        di = de.items() if isinstance(de, dict) else [(e.get("word"), e) for e in de]
        for k, e in di:
            for seed in (e.get("relations", {}).get("synonym") or []) + (
                e.get("relations", {}).get("similar") or []
            ):
                add_cluster(clusters, seed, [k])
    except FileNotFoundError:
        pass

# keep clusters with >=3 words so we can hold out and still train
clusters = {k: sorted(v) for k, v in clusters.items() if len(v) >= 3}
print(f"[clusters] {len(clusters)} usable (>=3 words):")
for k, v in clusters.items():
    print(f"   {k}: {v}")

# ---- train/test split: hold out ~1 word per cluster (>=3) for the gate ----
train_words, test_pairs_pos, all_words = {}, [], set()
held = {}
for c, ws in clusters.items():
    ws = ws[:]
    random.shuffle(ws)
    n_hold = max(1, len(ws) // 3)
    held[c] = ws[:n_hold]
    train_words[c] = ws[n_hold:]
    all_words.update(ws)
# held-out positive pairs: (held word, a train word in same cluster) — unseen-to-seen
for c in clusters:
    for h in held[c]:
        for t in train_words[c]:
            test_pairs_pos.append((h, t))
# held-out negative pairs: held word vs train word in a DIFFERENT cluster
cluster_list = list(clusters)
test_pairs_neg = []
for c in clusters:
    for h in held[c]:
        for oc in cluster_list:
            if oc == c:
                continue
            for t in train_words[oc][:2]:
                test_pairs_neg.append((h, t))

vocab = sorted(all_words)
ids_cache = {w: word_to_ids(w) for w in vocab}
multi = sum(1 for w in vocab if len(ids_cache[w]) > 1)
print(f"[tok] {len(vocab)} words, {multi} tokenize to >1 token (compositional), "
      f"{len(vocab)-multi} direct single-token")


def sep_metrics(Wt):
    def vec(w):
        idx = torch.tensor(ids_cache[w], dtype=torch.long)
        v = Wt[idx].mean(0)
        return v / (v.norm() + 1e-8)
    cache = {w: vec(w) for w in vocab}
    pos = [float(torch.dot(cache[a], cache[b])) for a, b in test_pairs_pos]
    neg = [float(torch.dot(cache[a], cache[b])) for a, b in test_pairs_neg]
    pos, neg = np.array(pos), np.array(neg)
    # AUC: P(random pos cos > random neg cos)
    wins = sum((p > n) for p in pos for n in neg)
    auc = wins / (len(pos) * len(neg))
    return pos.mean(), neg.mean(), pos.mean() - neg.mean(), auc


W0 = torch.tensor(load_file(EMB)["token_emb.weight"], dtype=torch.float32)
pm, nm, margin, auc = sep_metrics(W0)
print(f"\n[BASELINE held-out] pos_cos={pm:.3f} neg_cos={nm:.3f} "
      f"margin={margin:.3f} AUC={auc:.3f}")

# ---- contrastive fine-tune on TRAIN words only ----
W = W0.clone().requires_grad_(True)
opt = torch.optim.Adam([W], lr=0.05)
# train pos pairs (within cluster, train words only), neg pairs (cross cluster)
tr_pos = [(a, b) for c in clusters for a, b in itertools.combinations(train_words[c], 2)]
tr_neg = []
for c in clusters:
    for oc in cluster_list:
        if oc == c:
            continue
        for a in train_words[c][:3]:
            for b in train_words[oc][:3]:
                tr_neg.append((a, b))
print(f"[train] {len(tr_pos)} pos pairs, {len(tr_neg)} neg pairs (TRAIN words only)")

NEG_MARGIN = 0.15


def wvec(Wt, w):
    v = Wt[torch.tensor(ids_cache[w], dtype=torch.long)].mean(0)
    return v / (v.norm() + 1e-8)


for step in range(400):
    opt.zero_grad()
    pl = torch.stack([1 - torch.dot(wvec(W, a), wvec(W, b)) for a, b in tr_pos]).mean()
    nl = torch.stack(
        [torch.clamp(torch.dot(wvec(W, a), wvec(W, b)) - NEG_MARGIN, min=0) for a, b in tr_neg]
    ).mean()
    # anchor to original so we don't destroy the space
    anchor = 0.001 * ((W - W0) ** 2).mean()
    loss = pl + nl + anchor
    loss.backward()
    opt.step()
    if step % 100 == 0 or step == 399:
        with torch.no_grad():
            # train separation (memorization check)
            tp = np.array([float(torch.dot(wvec(W, a), wvec(W, b))) for a, b in tr_pos[:200]])
            tn = np.array([float(torch.dot(wvec(W, a), wvec(W, b))) for a, b in tr_neg[:200]])
            pm2, nm2, mg2, auc2 = sep_metrics(W.detach())
        print(f"  step {step:3d} loss={float(loss):.3f} | TRAIN pos={tp.mean():.3f} "
              f"neg={tn.mean():.3f} | HELD-OUT margin={mg2:.3f} AUC={auc2:.3f}")

with torch.no_grad():
    pm2, nm2, mg2, auc2 = sep_metrics(W.detach())
print(f"\n[AFTER FT held-out] pos_cos={pm2:.3f} neg_cos={nm2:.3f} "
      f"margin={mg2:.3f} AUC={auc2:.3f}")
print(f"[GATE] need margin>=0.20 AND AUC>=0.85 -> "
      f"{'PASS' if (mg2>=0.20 and auc2>=0.85) else 'FAIL'}")

save_file({"token_emb.weight": W.detach().numpy().astype(np.float32)}, OUT)
print(f"[export] wrote {OUT}")
