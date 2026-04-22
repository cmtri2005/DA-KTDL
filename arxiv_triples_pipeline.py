"""
=============================================================================
arXiv Triples Extraction & Data Formatting Pipeline
=============================================================================
Paper reference:
    "Triples and Knowledge-Infused Embeddings for Clustering and
     Classification of Scientific Documents" – Mihael Arcan (2025)

Pipeline stages:
    1. Load & filter arXiv metadata (JSONL)
    2. Extract (Subject, Relation, Object) triples from abstracts using spaCy
    3. Linearize triples → natural language statements
    4. Format 4 text representations per document:
        - Abstract        : raw cleaned abstract
        - Triples         : linearized triple statements only
        - Concatenate     : abstract + triples (flat concat)
        - Hybrid [STEP]   : abstract [SEP] triples (structured / segmented)
    5. Save outputs as JSONL + CSV (one file per format + combined)

Usage:
    python arxiv_triples_pipeline.py
        [--input  PATH]          default: arxiv-metadata-oai-snapshot.json
        [--output DIR]           default: ./output_triples
        [--n_cluster INT]        docs for clustering split  (default 5000)
        [--n_classify INT]       docs for classifier split  (default 10000)
        [--categories STR ...]   top-level cats to keep     (default: cs math physics)
        [--min_triples INT]      drop docs with < N triples (default: 1)
        [--batch_size INT]       spaCy batch size            (default: 256)
        [--seed INT]             random seed                 (default: 42)
        [--max_total INT]        hard cap: read at most N lines from JSONL
=============================================================================
"""

import argparse
import json
import logging
import os
import random
import re
import sys
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1. Data Loading & Filtering
# ===========================================================================

TOP_LEVEL_MAP = {
    # arXiv category prefix → readable label
    "cs":      "cs",
    "math":    "math",
    "physics": "physics",
    "hep":     "physics",    # high-energy physics sub-categories
    "quant":   "physics",
    "astro":   "physics",
    "cond":    "physics",
    "eess":    "eess",
    "econ":    "econ",
    "stat":    "stat",
    "q-bio":   "q-bio",
    "q-fin":   "q-fin",
}


def get_top_level(categories: str) -> Optional[str]:
    """Return the normalised top-level category label for a space-separated
    arXiv categories string, or None if not in our chosen set."""
    for cat in categories.split():
        prefix = cat.split(".")[0].lower()
        if prefix in TOP_LEVEL_MAP:
            return TOP_LEVEL_MAP[prefix]
    return None


def clean_abstract(text: str) -> str:
    """Lowercase, strip LaTeX noise, normalise whitespace."""
    text = text.replace("\n", " ")
    # remove inline math: $...$ or $$...$$
    text = re.sub(r"\$\$?.+?\$\$?", " ", text)
    # remove latex commands like \alpha \beta etc.
    text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", " ", text)
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    # remove curly braces
    text = re.sub(r"[{}]", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_arxiv(
    path: str,
    keep_categories: List[str],
    n_cluster: int,
    n_classify: int,
    seed: int,
    max_total: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Stream arXiv JSONL, filter by category, sample split sets."""
    rng = random.Random(seed)
    needed = n_cluster + n_classify

    by_cat: Dict[str, List[Dict]] = defaultdict(list)
    total_read = 0

    log.info(f"Streaming {path} …")
    with open(path, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, desc="Reading JSONL", unit=" lines"):
            if max_total and total_read >= max_total:
                break
            total_read += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            abstract = record.get("abstract", "") or ""
            if len(abstract.strip()) < 80:          # too short, skip
                continue

            cats = record.get("categories", "") or ""
            label = get_top_level(cats)
            if label not in keep_categories:
                continue

            by_cat[label].append({
                "id":         record.get("id"),
                "title":      record.get("title", "").replace("\n", " ").strip(),
                "abstract":   clean_abstract(abstract),
                "categories": cats,
                "label":      label,
                "update_date": record.get("update_date", ""),
            })

            # Early stopping: once every category has way more than needed, stop
            if all(len(v) >= needed for v in by_cat.values()):
                log.info("Collected sufficient records per category – stopping early.")
                break

    log.info(f"Loaded {sum(len(v) for v in by_cat.items())  if False else sum(len(v) for v in by_cat.values())} docs across {len(by_cat)} categories")

    # Build a balanced pool
    pool: List[Dict] = []
    for docs in by_cat.values():
        rng.shuffle(docs)
        pool.extend(docs)

    rng.shuffle(pool)

    # Remove duplicates by id
    seen = set()
    deduped = []
    for d in pool:
        if d["id"] not in seen:
            seen.add(d["id"])
            deduped.append(d)

    cluster_set  = deduped[:n_cluster]
    classify_set = deduped[n_cluster: n_cluster + n_classify]

    log.info(f"  Clustering split : {len(cluster_set)}  docs")
    log.info(f"  Classification split: {len(classify_set)} docs")
    return cluster_set, classify_set


# ===========================================================================
# 2. Triple Extraction (spaCy dependency parsing)
# ===========================================================================

def extract_triples(doc) -> List[Tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from a spaCy Doc.

    Strategy (faithful to the paper's §3.2):
      - Identify verbs as relational anchors
      - For each verb, find nsubj (subject) and dobj/pobj (object)
      - When direct object absent, fall back to prepositional object
      - Retain noun-phrase chunks for richer readability
    """
    triples = []
    np_chunks = {token.i: chunk.text for chunk in doc.noun_chunks
                 for token in chunk}

    for token in doc:
        if token.pos_ not in ("VERB", "AUX"):
            continue

        relation = token.lemma_.lower()

        # --- subject ---
        subj = None
        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                subj = np_chunks.get(child.i, child.text)
                break

        if subj is None:
            # try going up to parent verb and finding its subject
            if token.head.pos_ in ("VERB", "AUX"):
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = np_chunks.get(child.i, child.text)
                        break

        if subj is None:
            continue   # no subject found → skip

        # --- object ---
        obj = None
        for child in token.children:
            if child.dep_ in ("dobj", "attr"):
                obj = np_chunks.get(child.i, child.text)
                break

        if obj is None:
            # fallback: prepositional object
            for child in token.children:
                if child.dep_ == "prep":
                    for grandchild in child.children:
                        if grandchild.dep_ in ("pobj", "pcomp"):
                            obj = np_chunks.get(grandchild.i, grandchild.text)
                            break
                if obj:
                    break

        if obj is None:
            continue   # no object found → skip

        # --- clean up ---
        subj = subj.strip().lower()
        obj  = obj.strip().lower()
        relation = relation.strip().lower()

        # minimal length guards
        if len(subj) < 2 or len(obj) < 2 or len(relation) < 2:
            continue

        triples.append((subj, relation, obj))

    return triples


def linearize_triples(triples: List[Tuple[str, str, str]]) -> str:
    """
    Convert list of (s, r, o) triples to a single natural language string.
    e.g. ('transformer', 'improves', 'accuracy')
      → 'Transformer improves accuracy.'
    Multiple triples are joined with a space.
    """
    sentences = []
    for s, r, o in triples:
        sent = f"{s.capitalize()} {r} {o}."
        sentences.append(sent)
    return " ".join(sentences)


def run_triple_extraction(
    docs: List[Dict],
    nlp,
    batch_size: int,
    min_triples: int,
) -> List[Dict]:
    """
    Run spaCy pipeline over all doc abstracts, extract triples,
    filter docs with fewer than min_triples extracted triples.
    Adds keys: 'triples_raw', 'triples_text' to each doc dict.
    Returns the filtered list.
    """
    abstracts = [d["abstract"] for d in docs]
    results = []

    log.info(f"Extracting triples from {len(abstracts)} abstracts …")
    for i, spacy_doc in enumerate(
        tqdm(
            nlp.pipe(abstracts, batch_size=batch_size),
            total=len(abstracts),
            desc="spaCy triple extraction",
        )
    ):
        doc = docs[i]
        triples = extract_triples(spacy_doc)
        doc["triples_raw"] = triples                       # list of (s,r,o) tuples
        doc["triples_text"] = linearize_triples(triples)  # linearised string

        if len(triples) >= min_triples:
            results.append(doc)

    log.info(f"  Kept {len(results)}/{len(docs)} docs (≥{min_triples} triple(s))")
    return results


# ===========================================================================
# 3. Four Representation Formats
# ===========================================================================

SEP_TOKEN = "[SEP]"
STEP_TOKEN = "[STEP]"


def build_representations(doc: Dict) -> Dict:
    """
    Return a dict with all 4 representation strings for a document.

    Format definitions (§3.3 of the paper):
    ─────────────────────────────────────────────────────────────────────────
    1. Abstract        →  cleaned natural language abstract only
    2. Triples         →  linearised triple statements only
    3. Concatenate     →  abstract + " " + triples  (flat, no separator)
    4. Hybrid [STEP]   →  abstract [SEP] triples [STEP] (explicit segmentation
                          using [SEP] between fields and [STEP] as end marker,
                          matching the paper's segmented hybrid format that
                          achieved 92.6% accuracy in classification)
    ─────────────────────────────────────────────────────────────────────────
    """
    abstract      = doc["abstract"]
    triples_text  = doc["triples_text"]

    fmt_abstract     = abstract
    fmt_triples      = triples_text
    fmt_concatenate  = f"{abstract} {triples_text}".strip()
    fmt_hybrid       = f"{abstract} {SEP_TOKEN} {triples_text} {STEP_TOKEN}"

    return {
        "id":               doc["id"],
        "title":            doc["title"],
        "label":            doc["label"],
        "categories":       doc["categories"],
        "update_date":      doc["update_date"],

        # triples metadata
        "n_triples":        len(doc["triples_raw"]),
        "triples_raw":      json.dumps(doc["triples_raw"]),  # serialise list

        # 4 representation formats
        "fmt_abstract":     fmt_abstract,
        "fmt_triples":      fmt_triples,
        "fmt_concatenate":  fmt_concatenate,
        "fmt_hybrid":       fmt_hybrid,
    }


# ===========================================================================
# 4. Output Writing
# ===========================================================================

FORMATS = ["abstract", "triples", "concatenate", "hybrid"]
FMT_KEYS = {
    "abstract":    "fmt_abstract",
    "triples":     "fmt_triples",
    "concatenate": "fmt_concatenate",
    "hybrid":      "fmt_hybrid",
}


def save_outputs(records: List[Dict], split_name: str, out_dir: Path):
    """
    Save outputs in three ways:
      a) combined JSONL (all 4 formats per line)
      b) one JSONL per format
      c) one CSV per format (for easy inspection)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- a) Combined JSONL ---
    combined_path = out_dir / f"{split_name}_combined.jsonl"
    with open(combined_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(records)} records → {combined_path}")

    # --- b & c) Per-format files ---
    for fmt in FORMATS:
        key = FMT_KEYS[fmt]
        fmt_path_jsonl = out_dir / f"{split_name}_{fmt}.jsonl"
        fmt_path_csv   = out_dir / f"{split_name}_{fmt}.csv"

        slim_records = [
            {
                "id":         r["id"],
                "label":      r["label"],
                "n_triples":  r["n_triples"],
                "text":       r[key],
            }
            for r in records
        ]

        # JSONL
        with open(fmt_path_jsonl, "w", encoding="utf-8") as fh:
            for rec in slim_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # CSV
        with open(fmt_path_csv, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["id", "label", "n_triples", "text"])
            writer.writeheader()
            writer.writerows(slim_records)

        log.info(f"    [{fmt}] {fmt_path_jsonl.name}  &  {fmt_path_csv.name}")


def print_sample(records: List[Dict], n: int = 3):
    """Print n example docs showing all 4 formats."""
    print("\n" + "=" * 78)
    print("SAMPLE OUTPUTS")
    print("=" * 78)
    for i, rec in enumerate(records[:n]):
        print(f"\n── Record {i+1} ── id={rec['id']}  label={rec['label']}  n_triples={rec['n_triples']}")
        for fmt, key in FMT_KEYS.items():
            preview = rec[key][:220].replace("\n", " ")
            print(f"\n  [{fmt.upper()}]\n  {preview}")
        print()


# ===========================================================================
# 5. Main Pipeline
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="arXiv → Triples extraction & 4-format data pipeline"
    )
    parser.add_argument("--input",      default=r"d:\DataMining\arxiv-metadata-oai-snapshot.json")
    parser.add_argument("--output",     default=r"d:\DataMining\output_triples")
    parser.add_argument("--n_cluster",  type=int, default=5000)
    parser.add_argument("--n_classify", type=int, default=10000)
    parser.add_argument(
        "--categories", nargs="+",
        default=["cs", "math", "physics"],
        help="Top-level arXiv categories to include"
    )
    parser.add_argument("--min_triples", type=int, default=1,
                        help="Minimum triples per document (others discarded)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="spaCy pipe batch size")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max_total",  type=int, default=200_000,
                        help="Read at most N lines from the JSONL (0=unlimited)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output)

    log.info("=" * 60)
    log.info("arXiv Triples Extraction Pipeline")
    log.info("=" * 60)
    log.info(f"Input       : {args.input}")
    log.info(f"Output dir  : {out_dir}")
    log.info(f"Categories  : {args.categories}")
    log.info(f"n_cluster   : {args.n_cluster}")
    log.info(f"n_classify  : {args.n_classify}")
    log.info(f"min_triples : {args.min_triples}")
    log.info(f"seed        : {args.seed}")
    log.info(f"max_total   : {args.max_total if args.max_total else 'unlimited'}")

    # ── Step 1: Load & Split ─────────────────────────────────────────────────
    cluster_docs, classify_docs = load_arxiv(
        path=args.input,
        keep_categories=args.categories,
        n_cluster=args.n_cluster,
        n_classify=args.n_classify,
        seed=args.seed,
        max_total=args.max_total,
    )

    # ── Step 2: Load spaCy model ─────────────────────────────────────────────
    log.info("Loading spaCy model: en_core_web_sm …")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])
    # Increase max length guard for long abstracts
    nlp.max_length = 2_000_000

    # ── Step 3: Triple extraction ────────────────────────────────────────────
    cluster_docs  = run_triple_extraction(cluster_docs,  nlp, args.batch_size, args.min_triples)
    classify_docs = run_triple_extraction(classify_docs, nlp, args.batch_size, args.min_triples)

    # ── Step 4: Build 4 representations ─────────────────────────────────────
    log.info("Building 4 text representations …")
    cluster_recs  = [build_representations(d) for d in cluster_docs]
    classify_recs = [build_representations(d) for d in classify_docs]

    # ── Step 5: Save outputs ─────────────────────────────────────────────────
    log.info(f"Saving outputs to {out_dir} …")
    save_outputs(cluster_recs,  "cluster",   out_dir)
    save_outputs(classify_recs, "classify",  out_dir)

    # ── Step 6: Print samples ────────────────────────────────────────────────
    sys.stdout.reconfigure(encoding="utf-8")
    print_sample(cluster_recs, n=2)

    # ── Step 7: Stats summary ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    for split_name, recs in [("cluster", cluster_recs), ("classify", classify_recs)]:
        cats = defaultdict(int)
        for r in recs:
            cats[r["label"]] += 1
        total_triples = sum(r["n_triples"] for r in recs)
        avg_triples   = total_triples / len(recs) if recs else 0
        print(f"\n[{split_name.upper()}]  {len(recs)} documents")
        print(f"  Category distribution : {dict(cats)}")
        print(f"  Total triples         : {total_triples}")
        print(f"  Avg triples / doc     : {avg_triples:.2f}")
    print()
    log.info("Done ✓")


if __name__ == "__main__":
    main()
