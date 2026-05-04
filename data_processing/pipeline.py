"""Main orchestration for the arXiv Phase 1 pipeline."""

import logging
import sys

from .cli import parse_args
from .loading import load_arxiv
from .output import save_outputs
from .representations import build_representations, print_sample, summarize_records
from .triples import load_spacy_model, run_triple_extraction


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    configure_logging()
    log = logging.getLogger(__name__)

    args = parse_args()
    out_dir = args.output

    log.info("=" * 60)
    log.info("arXiv Triples Extraction Pipeline")
    log.info("=" * 60)
    log.info("Input       : %s", args.input)
    log.info("Output dir  : %s", out_dir)
    log.info("n_cluster   : %s", args.n_cluster)
    log.info("n_classify  : %s", args.n_classify)
    log.info("spaCy model : %s", args.spacy_model)
    log.info("year_from   : %s", args.year_from)
    log.info("year_to     : %s", args.year_to)
    log.info("seed        : %s", args.seed)
    if args.year_from is None and args.year_to is None:
        log.warning(
            "No year filter provided. The paper mentions a filtered recent-years subset "
            "but does not publish exact bounds."
        )

    cluster_docs, classify_docs = load_arxiv(
        path=args.input,
        n_cluster=args.n_cluster,
        n_classify=args.n_classify,
        seed=args.seed,
        year_from=args.year_from,
        year_to=args.year_to,
    )

    nlp = load_spacy_model(args.spacy_model)

    cluster_docs = run_triple_extraction(cluster_docs, nlp, args.batch_size)
    classify_docs = run_triple_extraction(classify_docs, nlp, args.batch_size)

    log.info("Building 4 text representations ...")
    cluster_records = [build_representations(doc) for doc in cluster_docs]
    classify_records = [build_representations(doc) for doc in classify_docs]

    log.info("Saving outputs to %s ...", out_dir)
    save_outputs(cluster_records, "cluster", out_dir)
    save_outputs(classify_records, "classify", out_dir)

    sys.stdout.reconfigure(encoding="utf-8")
    print_sample(cluster_records, n=2)

    print("\n" + "=" * 78)
    print("PIPELINE SUMMARY")
    print("=" * 78)
    for split_name, records in [("cluster", cluster_records), ("classify", classify_records)]:
        summary = summarize_records(records)
        print(f"\n[{split_name.upper()}]  {summary['num_documents']} documents")
        print(f"  Label distribution    : {summary['label_distribution']}")
        print(f"  Total triples         : {summary['total_triples']}")
        print(f"  Avg triples / doc     : {summary['avg_triples']:.2f}")

    log.info("Done")


if __name__ == "__main__":
    main()
