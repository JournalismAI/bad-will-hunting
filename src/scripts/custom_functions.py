from functools import partial
from pathlib import Path
from typing import Iterator, Iterable, Callable
import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from spacy.kb import KnowledgeBase, Candidate
import rapidfuzz

def get_candidates_from_fuzzy_matching(kb: KnowledgeBase, span_text, matching_thres=80):
    """
    Return a list of candidate entities for an alias based on fuzzy string matching.
    Each candidate defines the entity, the original alias,
    and the prior probability of that alias resolving to that entity.  
    If the alias is not known in the KB, an empty list is returned instead.
    """
    
    aliases = kb.get_alias_strings()
    # Get candidates with a rapidfuzz score match over similarity threshold
    candidate_aliases=rapidfuzz.process.extract(
        span_text, 
        aliases, 
        scorer=rapidfuzz.fuzz.WRatio, 
        limit=None, 
        score_cutoff=matching_thres
    )

    # Try to find and return only candidates containing exact matches to the entity name
    candidates=[candidate for candidate in candidate_aliases if span_text in candidate[0]]
    if candidates:
        return [kb.get_alias_candidates(tuple_[0])[0] for tuple_ in candidates]
    
    # Allow for spelling variations when no matching names are found (e.g. zelensky vs zelenskyy) 
    return [kb.get_alias_candidates(tuple_[0])[0] for tuple_ in candidate_aliases]

def get_custom_candidates(kb: KnowledgeBase, span) -> Iterator[Candidate]:
    """
    Return candidate entities for a given span by using the text of the span as the alias
    and fetching appropriate entries from the index.
    If no exact match candidates are found, return candidates based on fuzzy matching.  
    """
         
    # Get candidates based on whole span text
    label = span.label_
    if label !='PERSON':
        return []
    span_text=span.text
    candidates = kb.get_alias_candidates(span_text)
    
    # Get candidates based on partial matches 
    if not candidates:
        candidates = get_candidates_from_fuzzy_matching(kb, span_text)
        
    return candidates

@spacy.registry.readers("MyCorpus.v1")
def create_docbin_reader(file: Path) -> Callable[["Language"], Iterable[Example]]:
    return partial(read_files, file)
    
def read_files(file: Path, nlp: "Language") -> Iterable[Example]:
    # we run the full pipeline and not just nlp.make_doc to ensure we have entities and sentences
    # which are needed during training of the entity linker
    with nlp.select_pipes(disable="entity_linker"):
        doc_bin = DocBin().from_disk(file)
        docs = doc_bin.get_docs(nlp.vocab)
        for doc in docs:
            yield Example(nlp(doc.text), doc)

@spacy.registry.misc("gu.CandidateGenerator.v1")
def get_candidates() -> Callable[[KnowledgeBase, "Span"], Iterable[Candidate]]:
    return get_custom_candidates
