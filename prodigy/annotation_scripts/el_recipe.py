"""
Custom Prodigy recipe to perform manual annotation of entity links,
given an existing NER model and a knowledge base performing candidate generation.
"""

import string
from pathlib import Path
import nltk
import pandas as pd
import numpy as np
import prodigy
import rapidfuzz
import spacy
from nltk.tokenize import word_tokenize
from prodigy.components.filters import filter_duplicates
from prodigy.components.loaders import JSONL
from prodigy.models.ner import EntityRecognizer
from prodigy.util import set_hashes
from spacy.kb import KnowledgeBase, Candidate


# Utility functions
def get_candidates_from_fuzzy_matching(span, kb, single_name=False, matching_thres=60):
    """
    Return a list of candidate entities for an alias `span` and a knowledge base `kb`,
    based on fuzzy string matching.
    If the alias span is a single name (eg. first name only), set `single_name` to `True`
    in order to get better matches.
    The fuzzy matching's tolerance can be adjusted using the `matching_thres` parameter.
    Each candidate comprises an entity, original alias,
    and the prior probability of that alias resolving to that entity.
    If the alias is not known in the KB, and empty list is returned.
    """
    aliases = kb.get_alias_strings()
    matches = {}
    for al in aliases:
        if single_name:
            # For single name mentions, e.g. Trump, use partial_ratio
            fuzzy_ratio = rapidfuzz.fuzz.partial_ratio(span.lower(), al.lower())
        else:
            # For multi name mentions, e.g. Donald Trump, use WRatio
            fuzzy_ratio = rapidfuzz.fuzz.WRatio(span.lower(), al.lower())
        if fuzzy_ratio >= matching_thres:
            matches[al] = fuzzy_ratio
    candidates = []
    for match in matches:
        candidates.extend(kb.get_alias_candidates(match))
    return candidates, matches


def relevant_lexicon(text, stopwords):
    """ Create a set of words from `text` omitting `stopwords`. """
    punctuation_rm = str.maketrans('', '', string.punctuation)
    text = text.translate(punctuation_rm)
    word_set = set(word_tokenize(text))
    return word_set.difference(stopwords)


def _add_options(stream, kb, nlp, id_dict, kb_entities):
    """Create the UI options the annotator will be given, using the candidates from the `kb`
    for each NER span in `stream` using a bespoke logic to only surface relevant candidates.
    """
    for task in stream:
        for mention in filter(lambda m: m["label"] in ["PERSON"], task["spans"]):
            start_char = int(mention["start"])
            end_char = int(mention["end"])
            span_text = task["text"][start_char: end_char]
            single_name = len(span_text.split(' ')) <= 1
            # Retrieve a wide list of candidates based on fuzzy string matching and similarity score
            candidates, matches = get_candidates_from_fuzzy_matching(span_text, kb, single_name)

            # Convert candidate and scores to df
            matches_df = pd.DataFrame.from_dict(matches, orient='index').reset_index().rename(
                columns={'index': 'name', 0: 'score'})
            # Merge kb entity description
            matches_df = matches_df.merge(kb_entities, on=['name'], how='left')
            # Retain candidates with scores in top decile
            top_decile = 0.9
            score_thres = matches_df['score'].quantile(top_decile)
            matches_df = matches_df[matches_df['score'] >= score_thres]
            # Sort based on score and length of description text
            matches_df = matches_df.sort_values(by=['score', 'desc_len'], ascending=False).reset_index(drop=True)
            # Increase score in candidates with a wikiID
            matches_df.loc[matches_df['id'].str.match('Q\d'), 'score'] = matches_df.loc[matches_df['id'].str.match(
                'Q\d'), 'score'] * 1.05
            # Calculate normalised min/max description length.
            # Descriptions over 300 characters long are capped at 1.
            max_len = 300
            min_len = matches_df['desc_len'].min()
            matches_df['normalised_desc_len'] = (matches_df['desc_len'] - min_len) / (max_len - min_len)
            matches_df['normalised_desc_len'] = matches_df['normalised_desc_len'].apply(lambda x: 1 if x > 1 else x)

            # Find common words between paragraph and candidate descriptions to improve candidate selection
            stopwords = set(nltk.corpus.stopwords.words())
            desc_ents_d = dict()
            column_order = matches_df.reset_index().columns.values
            id_loc, = np.where(column_order == 'id')
            desc_loc, = np.where(column_order == 'desc')
            for tuple_ in matches_df.itertuples():
                qid = tuple_[id_loc[0]]
                desc = tuple_[desc_loc[0]]
                desc_ents = relevant_lexicon(desc, stopwords)
                desc_ents_d[qid] = desc_ents

            common_ent_count_d = {}
            text = task["text"]
            sentence_ents = [str(ent) for ent in nlp(text).ents]
            sentence_ents = set(' '.join(sentence_ents).split())
            for qid, ent in desc_ents_d.items():
                common_ent_counts = len(sentence_ents.intersection(ent))
                common_ent_count_d[qid] = common_ent_counts

            common_ent_counts = pd.DataFrame.from_dict(common_ent_count_d, orient='index').reset_index().rename(
                columns={'index': 'id', 0: 'common_ent_counts'})

            matches_df = matches_df.merge(common_ent_counts, on='id')

            matches_df = matches_df.sort_values(by=['common_ent_counts', 'score', 'normalised_desc_len'],
                                                ascending=False).reset_index(drop=True)

            n_candidates = 10
            matches_df = matches_df.head(n_candidates)

            candidates = matches_df['id'].map({candidate.entity_: candidate for candidate in candidates}).to_list()

            if candidates:
                options = []
                # add in a few additional options in case a correct ID cannot be picked
                options.append({"id": "NER_WrongType", "text": "Incorrect entity type returned by NER model."})
                options.append({"id": "NEL_MoreContext", "text": "Need more context to decide."})
                options.append({"id": "NEL_NoCandidate", "text": "No viable candidate."})

                options.extend([
                    {"id": c.entity_, "html": _print_info(c.entity_, id_dict, kb_entities)}
                    for c in candidates
                ])

                task["options"] = options
                task["config"] = {"choice_style": "multiple"}
                yield task


def shorten_description(descr, n_splits=2, min_desc_len=50, max_desc_len=500):
    """Keep description output length within `min_desc_len` and `max_desc_len` number of characters
        NB: This only affects the Prodigy UI text, not the original KB desc."""
    short_descr = '. '.join(descr.split('. ')[:n_splits])
    while len(short_descr) < min_desc_len:
        # Keep extending description length
        n_splits += 1
        new_desc_str = '. '.join(descr.split('. ')[:n_splits])
        if len(new_desc_str) == len(short_descr):
            # Max possible len break clause
            break
        short_descr = new_desc_str
        if len(short_descr) > max_desc_len:
            # Keep description length up to max limit
            n_splits -= 1
            short_descr = '. '.join(descr.split('. ')[:n_splits])
            break
    if short_descr.replace(' ', '')[-1] != '.':
        # Ensure stop mark at the end of description
        short_descr += '.'
    return short_descr


def _print_info(entity_id, id_dict, kb_entities):  # _url):
    """ Create a description paragraph for each candidate, adding a link to the corresponding knowledge base
    for further loop-up."""
    name, descr = id_dict.get(entity_id)
    url = kb_entities.loc[kb_entities['id'] == str(entity_id), 'kb_url'].values[0]
    # Tailor output description length
    short_descr = shorten_description(descr)
    # Prodigy option display text
    option = "<a class=\"entLink\" href='" + f'{url}' + "' target='_blank' onclick=\"clickMe(event)\">" + entity_id + "</a>"
    option += ": " + name + '; \n' + short_descr
    return option


# Prodigy recipes
@prodigy.recipe(
    "entity_linker.manual",
    dataset=("The Prodigy dataset to write to", "positional", None, str),
    source=("The input data (JSON-lines format)", "positional", None, Path),
    nlp_loc=(
            "Path to OR name of the NLP model with a pretrained NER component",
            "positional",
            None,
            str,
    ),
    kb_loc=("Path to the Knowledge Base", "positional", None, Path),
    entity_loc=(
            "Path to the file with additional information about the entities",
            "positional",
            None,
            Path,
    ),
)
def entity_linker_manual(dataset, source, nlp_loc, kb_loc, entity_loc):
    # Load the NLP and KB objects from file
    nlp = spacy.load(nlp_loc, exclude=["tagger", "parser", "lemmatizer", "attribute_ruler", "tok2vec", "transformer"])
    kb = KnowledgeBase(vocab=nlp.vocab, entity_vector_length=1)
    kb.from_disk(kb_loc)
    model = EntityRecognizer(nlp)

    # Read the pre-defined CSV file into dictionaries mapping QIDs to the full names and descriptions
    kb_entities = pd.read_csv(entity_loc, index_col=0)
    # Select and format relevant columns
    kb_entities = kb_entities[['id', 'name', 'desc', 'kb_url']]
    kb_entities['id'] = kb_entities['id'].astype(str)
    kb_entities['name'] = kb_entities['name'].astype(str)
    kb_entities['desc'] = kb_entities['desc'].astype(str)
    kb_entities['kb_url'] = kb_entities['kb_url'].astype(str)
    # Add description length column
    kb_entities['desc_len'] = kb_entities['desc'].str.len()
    # Define indices to access values across columns
    kb_column_order = kb_entities.reset_index().columns.values
    id_loc, = np.where(kb_column_order == 'id')
    name_loc, = np.where(kb_column_order == 'name')
    desc_loc, = np.where(kb_column_order == 'desc')
    # Create id to name, description mapping
    id_dict = dict()
    for tuple_ in kb_entities.itertuples():
        qid = tuple_[id_loc[0]]
        name = tuple_[name_loc[0]]
        desc = tuple_[desc_loc[0]]
        id_dict[qid] = (name, desc)
    # Initialize the Prodigy stream by running the NER model
    ## Faster reading by using generators
    stream = JSONL(source)
    stream = (set_hashes(eg) for eg in stream)
    stream = (eg for score, eg in model(stream))

    # For each NER mention, add the candidates from the KB to the annotation task as well as the url of the article
    stream = _add_options(stream, kb, nlp, id_dict, kb_entities)
    stream = filter_duplicates(stream, by_input=False, by_task=True)

    blocks = [{"view_id": "html",
               "html_template": "<a href=https://{{url}} target=\"_blank\">Guardian Article URL</a>"
               },
              {"view_id": "choice"},
              ]

    return {
        "dataset": dataset,
        "stream": stream,
        "view_id": "blocks",
        "config": {"blocks": blocks, "choice_auto_accept": False, "buttons": ["accept", "undo"], }
    }


