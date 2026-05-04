"""Shared constants for the Phase 1 data-processing pipeline."""

SEP_TOKEN = "[SEP]"
FORMATS = ["abstract", "triples", "concatenate", "hybrid"]
FMT_KEYS = {
    "abstract": "fmt_abstract",
    "triples": "fmt_triples",
    "concatenate": "fmt_concatenate",
    "hybrid": "fmt_hybrid",
}

