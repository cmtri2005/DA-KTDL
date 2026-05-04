"""Triple extraction helpers for the arXiv Phase 1 pipeline."""

import logging
import re
from typing import Dict, List

import spacy
from tqdm import tqdm

log = logging.getLogger(__name__)


def load_spacy_model(model_name: str):
    """Load the configured spaCy model for triple extraction."""
    log.info("Loading spaCy model: %s ...", model_name)
    nlp = spacy.load(model_name, disable=["ner", "textcat"])
    nlp.max_length = 2_000_000
    return nlp


def extract_triples(doc) -> List[Dict[str, str]]:
    """
    Extract (subject, relation, object) triples from a spaCy Doc.

    Strategy aligned with the paper:
      - verbs act as relational anchors
      - subject via nsubj / nsubjpass
      - object via dobj / attr
      - fallback via prep -> pobj / pcomp
      - retain sentence context with each extracted triple
    """
    triples: List[Dict[str, str]] = []
    np_chunks = {token.i: chunk.text for chunk in doc.noun_chunks for token in chunk}

    for sent in doc.sents:
        sentence_text = re.sub(r"\s+", " ", sent.text).strip()
        for token in sent:
            if token.pos_ not in ("VERB", "AUX"):
                continue

            relation = token.lemma_.strip().lower()
            subj = None
            obj = None
            rule_tag = None

            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subj = np_chunks.get(child.i, child.text)
                    break

            if subj is None and token.head.pos_ in ("VERB", "AUX"):
                for child in token.head.children:
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj = np_chunks.get(child.i, child.text)
                        break

            if subj is None:
                continue

            for child in token.children:
                if child.dep_ in ("dobj", "attr"):
                    obj = np_chunks.get(child.i, child.text)
                    rule_tag = child.dep_
                    break

            if obj is None:
                for child in token.children:
                    if child.dep_ != "prep":
                        continue
                    for grandchild in child.children:
                        if grandchild.dep_ in ("pobj", "pcomp"):
                            obj = np_chunks.get(grandchild.i, grandchild.text)
                            rule_tag = f"prep->{grandchild.dep_}"
                            break
                    if obj is not None:
                        break

            if obj is None:
                continue

            subj = subj.strip().lower()
            obj = obj.strip().lower()
            if len(subj) < 2 or len(obj) < 2 or len(relation) < 2:
                continue

            triples.append(
                {
                    "subj": subj,
                    "rel": relation,
                    "obj": obj,
                    "source_sentence": sentence_text,
                    "rule_tag": rule_tag or "dependency",
                }
            )

    return triples


def linearize_triples(triples: List[Dict[str, str]]) -> str:
    """Convert extracted triples to a flat natural-language string."""
    sentences = []
    for triple in triples:
        sentence = f"{triple['subj'].capitalize()} {triple['rel']} {triple['obj']}."
        sentences.append(sentence)
    return " ".join(sentences)


def triples_to_kg_edges(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Represent triples as document-level KG edges with provenance."""
    return [
        {
            "source": triple["subj"],
            "relation": triple["rel"],
            "target": triple["obj"],
            "source_sentence": triple["source_sentence"],
            "rule_tag": triple["rule_tag"],
        }
        for triple in triples
    ]


def run_triple_extraction(
    docs: List[Dict],
    nlp,
    batch_size: int,
) -> List[Dict]:
    """Run spaCy over all abstracts and attach extracted triples to each doc."""
    abstracts = [doc["abstract"] for doc in docs]

    log.info("Extracting triples from %d abstracts ...", len(abstracts))
    for i, spacy_doc in enumerate(
        tqdm(
            nlp.pipe(abstracts, batch_size=batch_size),
            total=len(abstracts),
            desc="spaCy triple extraction",
        )
    ):
        triples = extract_triples(spacy_doc)
        docs[i]["triples"] = triples
        docs[i]["kg_edges"] = triples_to_kg_edges(triples)
        docs[i]["triples_text"] = linearize_triples(triples)

    log.info("  Added triples to %d docs", len(docs))
    return docs
