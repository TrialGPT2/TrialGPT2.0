#!/usr/bin/env python3
# scripts/eval_trialgpt_rankings_qrels.py

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from tqdm import tqdm


# -----------------------------
# IO / utilities
# -----------------------------
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_pid(obj: Dict[str, Any], fallback: str = "") -> str:
    pid = str(obj.get("patient_id") or obj.get("_id") or obj.get("id") or "").strip()
    return pid or fallback


def _base_tid(tid: str) -> str:
    """
    Robust base NCT extractor:
      - "NCT01315769-G1" -> "NCT01315769"
      - "NCT01315769_G1" -> "NCT01315769"
    """
    s = str(tid or "").strip()
    m = re.match(r"^(NCT\d+)", s)
    if m:
        return m.group(1)
    # fallback split
    return re.split(r"[-_]", s)[0].strip()


def extract_candidate_base_ids_from_trial_dir(trial_dir: Path) -> List[str]:
    """
    Candidate universe: base NCTIDs from filenames like NCTxxxxxxx.json in trial_dir.
    Ignores modified_*.json.
    """
    ids: set[str] = set()
    for fp in trial_dir.glob("*.json"):
        if fp.name.startswith("modified_"):
            continue
        m = re.match(r"^(NCT\d+)\.json$", fp.name)
        if m:
            ids.add(m.group(1))
    return sorted(ids)


# -----------------------------
# queries.jsonl loading (patient IDs)
# -----------------------------
def load_patient_ids_from_queries_jsonl(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"queries.jsonl not found: {path}")
    pids: List[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            pid = _extract_pid(obj, "")
            if pid and pid not in seen:
                seen.add(pid)
                pids.append(pid)
    return pids


# -----------------------------
# Qrels loading
# -----------------------------
@dataclass
class Qrels:
    # pid -> base_tid -> label (0/1/2)
    labels: Dict[str, Dict[str, int]]
    # pid -> pool base_tids
    pool: Dict[str, List[str]]


def load_qrels(path: Path) -> Qrels:
    """
    Accepts:
      - 3-col TSV:  <query-id> <corpus-id> <score>
      - 4-col TREC: <qid> <iter> <docno> <rel>
    """
    if not path.exists():
        raise FileNotFoundError(f"qrels not found: {path}")

    labels: Dict[str, Dict[str, int]] = defaultdict(dict)
    pool_set: Dict[str, set[str]] = defaultdict(set)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue

            # skip header-ish rows
            p0 = parts[0].strip().lower()
            p1 = parts[1].strip().lower()
            if p0 in {"query-id", "qid", "queryid"} or p1 in {"corpus-id", "docno", "corpusid", "iter"}:
                continue

            if len(parts) >= 4:
                pid = str(parts[0]).strip()
                tid = _base_tid(str(parts[2]).strip())
                rel_s = parts[3]
            else:
                pid = str(parts[0]).strip()
                tid = _base_tid(str(parts[1]).strip())
                rel_s = parts[2]

            try:
                rel = int(rel_s)
            except Exception:
                rel = 0

            if not pid or not tid:
                continue

            labels[pid][tid] = rel
            pool_set[pid].add(tid)

    pool = {pid: sorted(list(s)) for pid, s in pool_set.items()}
    return Qrels(labels=dict(labels), pool=pool)


# -----------------------------
# Load TrialGPT results (full_email_*.json)
# -----------------------------
def load_trialgpt_ranked_base_list(
    fp: Path,
    agg: str = "first",  # "first" or "maxscore"
) -> List[str]:
    """
    Reads one full_email_*.json and returns a base-NCT ranking list.

    cover_page_list entries look like:
      {"trial_id":"NCT01315769-G1","score":100,...}

    agg="first":
      keep first occurrence of each base NCT in the (already-ranked) list.

    agg="maxscore":
      for each base NCT, pick the best score (tie-breaker: earliest rank),
      then sort base trials by (-score, best_rank, tid).
    """
    obj = _read_json(fp)
    rows = obj.get("cover_page_list", [])
    if not isinstance(rows, list) or not rows:
        return []

    if agg == "first":
        out: List[str] = []
        seen: set[str] = set()
        for r in rows:
            if not isinstance(r, dict):
                continue
            tid = str(r.get("trial_id") or r.get("trialId") or "").strip()
            bt = _base_tid(tid)
            if bt and bt not in seen:
                out.append(bt)
                seen.add(bt)
        return out

    # agg == "maxscore"
    best: Dict[str, Tuple[float, int]] = {}  # base -> (best_score, best_rank)
    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        tid = str(r.get("trial_id") or r.get("trialId") or "").strip()
        bt = _base_tid(tid)
        if not bt:
            continue
        sc_raw = r.get("score", 0.0)
        try:
            sc = float(sc_raw)
        except Exception:
            sc = 0.0

        if bt not in best:
            best[bt] = (sc, idx)
        else:
            prev_sc, prev_rank = best[bt]
            # prefer higher score; tie-break: earlier rank
            if (sc > prev_sc) or (sc == prev_sc and idx < prev_rank):
                best[bt] = (sc, idx)

    items = [(bt, sc, rk) for bt, (sc, rk) in best.items()]
    items.sort(key=lambda x: (-x[1], x[2], x[0]))
    return [bt for bt, _, _ in items]


def load_all_results_rankings(results_dir: Path, agg: str) -> Dict[str, List[str]]:
    """
    Loads all full_email_*.json under results_dir (non-recursive by default; easy to change).
    Returns pid -> ranked base NCT list.
    """
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    out: Dict[str, List[str]] = {}
    for fp in sorted(results_dir.glob("full_email_*.json")):
        try:
            obj = _read_json(fp)
        except Exception:
            continue

        pid = _extract_pid(obj, fallback=fp.stem.replace("full_email_", ""))
        if not pid:
            continue

        ranked = load_trialgpt_ranked_base_list(fp, agg=agg)
        if ranked:
            out[pid] = ranked
    return out


def mrr(ranked: List[str], truth: Dict[str, int], thr: int) -> float:
    rel_set = {tid for tid, v in truth.items() if v >= thr}
    for i, tid in enumerate(ranked):
        if tid in rel_set:
            return 1.0 / (i + 1)
    return 0.0


# -----------------------------
# Metrics
# -----------------------------
def graded_precision_at_k(ranked: List[str], truth: Dict[str, int], k: int, max_rel: int) -> float:
    if k <= 0:
        return 0.0
    topk = ranked[:k]
    s = 0.0
    for tid in topk:
        rel = float(truth.get(tid, 0))
        s += rel / float(max_rel)
    return s / k


def _dcg_graded(rels: List[int]) -> float:
    return sum((float(rel) / math.log2(i + 2) for i, rel in enumerate(rels)))


def ndcg_graded_at_k(ranked: List[str], truth: Dict[str, int], k: int) -> float:
    topk = ranked[:k]
    rels = [int(truth.get(tid, 0)) for tid in topk]
    dcg_val = _dcg_graded(rels)
    ideal_rels = sorted([int(v) for v in truth.values()], reverse=True)[:k]
    idcg = _dcg_graded(ideal_rels)
    return (dcg_val / idcg) if idcg > 0 else 0.0


def graded_average_precision(ranked: List[str], truth: Dict[str, int], max_rel: int) -> float:
    gains = [float(truth.get(tid, 0)) / float(max_rel) for tid in ranked]
    total_gain = sum(gains)
    if total_gain <= 0:
        return 0.0
    cum_gain = 0.0
    s = 0.0
    for i, g in enumerate(gains, start=1):
        if g <= 0:
            continue
        cum_gain += g
        prec_i = cum_gain / float(i)
        s += g * prec_i
    return s / total_gain


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--queries-jsonl", default="data/trec_2022/queries.jsonl", help="queries.jsonl containing patient_id (used to define evaluation set).")
    ap.add_argument("--qrels", default="data/trec_2022/qrels/test.tsv", help="qrels TSV (3-col or 4-col TREC). Defines per-patient pool.")
    ap.add_argument("--results-dir", required=True, help="Directory containing TrialGPT outputs full_email_*.json")
    ap.add_argument("--trial-dir", default="data/trec_2022/trials", help="Trial JSON dir (used to filter candidate base NCTs)")

    ap.add_argument("--k", type=int, default=10, help="Top-K for GradedPrecision and GradednDCG.")
    ap.add_argument("--max-rel", type=int, default=2, help="Max relevance label for graded metrics.")
    ap.add_argument("--mrr-relevance-threshold", type=int, default=2, help="Relevance label threshold for MRR.")

    ap.add_argument(
        "--subtrial-agg",
        choices=["first", "maxscore"],
        default="first",
        help="How to collapse subtrials into base trial ranking.",
    )
    ap.add_argument(
        "--pad-to-pool",
        action="store_true",
        help="Append any missing pool trials to the end to make a full pool ranking for GradedAP.",
    )

    args = ap.parse_args()

    queries_pids = set(load_patient_ids_from_queries_jsonl(Path(args.queries_jsonl)))
    qrels = load_qrels(Path(args.qrels))
    ranked_by_pid = load_all_results_rankings(Path(args.results_dir), agg=args.subtrial_agg)

    # candidate ids from trial_dir (optional sanity filter)
    candidates = set(extract_candidate_base_ids_from_trial_dir(Path(args.trial_dir)))
    if not candidates:
        raise RuntimeError(f"No candidate NCT*.json found in trial_dir={args.trial_dir}")

    # evaluation set: must exist in queries, qrels, and ranked outputs
    pids = sorted([pid for pid in qrels.pool.keys() if pid in queries_pids and pid in ranked_by_pid])
    if not pids:
        raise RuntimeError("No overlapping patients among queries.jsonl, qrels, and results_dir.")

    acc: Dict[str, List[float]] = {
        "gP": [],
        "gNDCG": [],
        "gAP": [],
        "MRR": [],
    }

    it: Iterable[str] = pids
    it = tqdm(it, total=len(pids), desc="Evaluating TrialGPT (qrels pool)", dynamic_ncols=True)

    for pid in it:
        # pool/truth from qrels (per-patient search space)
        pool_ids0 = qrels.pool[pid]
        truth0 = qrels.labels.get(pid, {})

        # filter pool/truth to trials that exist in trial_dir
        pool_ids = [t for t in pool_ids0 if t in candidates]
        pool_set = set(pool_ids)
        truth = {t: truth0.get(t, 0) for t in pool_ids}

        # ranked base list collapsed from subtrials
        rank0 = ranked_by_pid.get(pid, [])
        # restrict to pool
        rank = [t for t in rank0 if t in pool_set]

        # Ensure full coverage of pool so GradedAP is measured over the intended search space.
        if args.pad_to_pool and len(rank) < len(pool_ids):
            seen = set(rank)
            for t in pool_ids:
                if t not in seen:
                    rank.append(t)
                    seen.add(t)

        k = int(args.k)
        max_rel = int(args.max_rel)
        mrr_thr = int(args.mrr_relevance_threshold)

        acc["gP"].append(graded_precision_at_k(rank, truth, k, max_rel))
        acc["gNDCG"].append(ndcg_graded_at_k(rank, truth, k))
        acc["gAP"].append(graded_average_precision(rank, truth, max_rel))
        acc["MRR"].append(mrr(rank, truth, mrr_thr))

    def mean(xs: List[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    print("\n==============================")
    print("Evaluation summary (TrialGPT full_email_*.json; qrels defines per-patient pool; subtrials collapsed to base)")
    print("==============================")
    print(
        f"(patients={len(pids)}, K={args.k}, max_rel={args.max_rel}, "
        f"MRR_relevant=label>={args.mrr_relevance_threshold}, "
        f"subtrial_agg={args.subtrial_agg}, pad_to_pool={bool(args.pad_to_pool)})"
    )

    print("\n[Reported metrics]")
    print(f"  MRR:                     {mean(acc['MRR']):.4f}")
    print(f"  GradedAP:                {mean(acc['gAP']):.4f}")
    print(f"  GradedPrecision@{args.k}: {mean(acc['gP']):.4f}")
    print(f"  GradednDCG@{args.k}:      {mean(acc['gNDCG']):.4f}")


if __name__ == "__main__":
    main()
