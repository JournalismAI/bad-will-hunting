import pandas
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin, Span


def find_no_candidates(row, no_candidate_sub_str=["NEL", "NER"]):
    return any(sub_str in " ".join(row["accept"]) for sub_str in no_candidate_sub_str)


def find_best_candidate(row):
    best_candidate_d = {}
    for option_id in row["accept"]:
        for candidate in list(filter(lambda x: option_id == x["id"], row["options"])):
            candidate_len = len(candidate["html"].split("a>:")[1])
            best_candidate_d[option_id] = candidate_len
    return max(best_candidate_d, key=best_candidate_d.get)


def clean_data(df):
    """Implement data cleaning."""
    df = df[df["accept"].apply(len) != 0]
    df = df[~df.apply(find_no_candidates, axis=1)]
    return df


def make_doc(example, nlp):
    """Make spacy document from prodigy example (dict)"""
    sentence = example["text"]
    gold_ids = []
    if example["answer"] == "accept":
        QID = example["accept"]
        doc = nlp.make_doc(sentence)
        gold_ids.append(QID)
        # we assume only 1 annotated span per sentence, and only 1 KB ID per span
        entity = doc.char_span(
            example["spans"][0]["start"],
            example["spans"][0]["end"],
            label=example["spans"][0]["label"],
            kb_id=QID,
        )
        doc.ents = [entity]
        for i, t in enumerate(doc):
            doc[i].is_sent_start = i == 0
        return doc


def main(input_files, out_stem, nlp_model="en_core_web_lg", subset=None, verbose=False):
    """
    Prepare datasets to train/test EL model
    """

    inputs = [pandas.read_json(inp, lines=True) for inp in input_files]
    df = pandas.concat(inputs)

    # Clean data
    df = clean_data(df)

    # Surface only one (the best candidate)
    df["accept"] = df.apply(find_best_candidate, axis=1)

    ## Only use a subset of entities?
    if subset is not None:
        subset = [_.strip() for _ in subset[0].split(",")]
        df = df[df["accept"].isin(subset)]

    # Split datasets and ensure no paragraph is split between train and dev
    index_train, index_test = train_test_split(
        df["_input_hash"].unique(), test_size=0.4, random_state=14
    )

    df_train = df[df["_input_hash"].isin(index_train)]
    df_test = df[df["_input_hash"].isin(index_test)]
    print(f"Size of datasets: Train ({df_train.shape[0]}), Dev ({df_test.shape[0]})")
    # Make docs
    nlp = spacy.load(nlp_model, exclude="parser, tagger")
    train_docs = df_train.apply(make_doc, args=(nlp,), axis=1)
    test_docs = df_test.apply(make_doc, args=(nlp,), axis=1)

    # Create DocBins for exporting to Spacy format
    train_docbin = DocBin()
    test_docbin = DocBin()

    for doc in train_docs:
        train_docbin.add(doc)
    for doc in test_docs:
        test_docbin.add(doc)

    # Output training data in .spacy format
    train_corpus = f"{out_stem}_train.spacy"
    test_corpus = f"{out_stem}_test.spacy"

    train_docbin.to_disk(train_corpus)
    test_docbin.to_disk(test_corpus)


if __name__ == "__main__":
    import argparse

    verbose = False

    parser = argparse.ArgumentParser(
        prog="prepare_training",
        description="Prepare training dataset from annotations.",
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        default=["data/el_session_1.jsonl", "data/el_session_2.jsonl", "data/el_session_3.jsonl"],
    )
    parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        help="Name structure of output files.",
        default="data/el",
    )
    parser.add_argument(
        "-n",
        "--nlp",
        nargs="?",
        help="Name of the Spacy model",
        default="en_core_web_lg",
    )
    parser.add_argument(
        "-s",
        "--subset",
        nargs="+",
        help='Subset of entity IDS (comma-seperated and in QUOTES) to be used.\n Example: "Q22686, Q180589, 15108, 67986, Q272201, 13191"',
    )

    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    main(args.input_files, args.out, args.nlp, args.subset, args.verbose)
