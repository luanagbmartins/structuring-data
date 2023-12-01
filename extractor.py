import pandas as pd
import numpy as np
import spacy
import jsonlines
import nmslib
import os
import json
from keybert import KeyBERT
from datetime import datetime


def extract_and_index(
    documents,
    remove_stopwords=True,
    embedding_model="pt_core_news_lg",
    out_dir="indexed_docs",
    **kwargs
):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    print("----- Strating at " + now + " -----")
    out_dir = os.path.join(out_dir, now)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(
        datetime.now().strftime("%H-%M-%S-%f") + "----- Loading Embedding Model -----"
    )
    if isinstance(embedding_model, str):
        nlp = spacy.load(embedding_model)
    else:
        nlp = embedding_model

    kw_model = KeyBERT(model=nlp)

    lower_doc = [d.lower() for d in documents]

    params = {}
    if remove_stopwords:
        stop_words = []
        for text in lower_doc:
            sw = [token.text for token in nlp(text) if token.is_stop]
            stop_words.extend(sw)
        stop_words_singles = dict.fromkeys(stop_words)
        stop_words_singles = list(stop_words_singles)
        params["stop_words"] = stop_words_singles

    params["keyphrase_ngram_range"] = kwargs.get("keyphrase_ngram_range", (2, 2))
    params["top_n"] = kwargs.get("top_n", 10)
    params["stop_words"] = stop_words_singles
    params["min_df"] = kwargs.get("min_df", int(len(lower_doc) * 0.005))
    params["use_maxsum"] = kwargs.get("use_maxsum", False)
    params["nr_candidates"] = kwargs.get("nr_candidates", None)
    params["use_mmr"] = kwargs.get("use_mmr", True)
    params["diversity"] = kwargs.get("diversity", 0.2)

    print(datetime.now().strftime("%H-%M-%S-%f") + "----- Extracting Keywords -----")
    kw = kw_model.extract_keywords(lower_doc, **params)

    print(
        datetime.now().strftime("%H-%M-%S-%f")
        + "----- Collecting Representations -----"
    )
    embs = []
    for k in kw:
        local_embs = []
        for key, val in k:
            doc = nlp(key)
            local_embs.append(doc.vector)
        final_emb = np.array(local_embs).mean(axis=0)
        if final_emb.size:
            embs.append(final_emb)

    index = nmslib.init(method="hnsw", space="angulardist")
    index.addDataPointBatch(embs)
    index.createIndex({"post": 2}, print_progress=True)

    print(datetime.now().strftime("%H-%M-%S-%f") + "----- Saving Index and Data -----")
    data = []
    for doc, k in zip(lower_doc, kw):
        data.append({"doc": doc, "keys": k})

    df = pd.DataFrame(data)

    index.saveIndex(os.path.join(out_dir, "nms_index.index"), save_data=True)
    print(
        datetime.now().strftime("%H-%M-%S-%f")
        + "----- Index Saved At: "
        + os.path.join(out_dir, "nms_index.index")
    )
    df.to_pickle(os.path.join(out_dir, "docs.pkl"))
    print(
        datetime.now().strftime("%H-%M-%S-%f")
        + "----- Docs Saved At: "
        + os.path.join(out_dir, "docs.pkl")
    )
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(params, f)
