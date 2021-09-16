from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os
import spacy
from tqdm import tqdm
import nmslib
import spacy_universal_sentence_encoder
from datetime import datetime
from copy import deepcopy
from numpy.linalg import norm


def cos_sim(a, b):
    cos_sim = np.dot(a, b) / (norm(a) * norm(b))
    return cos_sim


def extract_topics(
    documents,
    remove_stop_words=True,
    out_dir="topic_models",
    extraction_search_times=10,
    **kwargs
):

    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    print("----- Strating at " + now + " -----")
    out_dir = os.path.join(out_dir, now)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    documents = [d.lower() for d in documents]

    print(datetime.now().strftime("%H-%M-%S-%f") + " ----- Loading Models -----")
    nlp = spacy_universal_sentence_encoder.load_model("xx_use_lg")

    if remove_stop_words:

        word_nlp = spacy.load("pt_core_news_lg")

        stop_words = []
        for text in documents:
            sw = [token.text for token in word_nlp(text.lower()) if token.is_stop]
            stop_words.extend(sw)

        stop_words_singles = dict.fromkeys(stop_words)
        stop_words_singles = list(stop_words_singles)

        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words=stop_words_singles
        )

    print(
        datetime.now().strftime("%H-%M-%S-%f")
        + " ----- Searching for Most Representative Topics -----"
    )
    highest_ent = -100
    for i in tqdm(range(extraction_search_times)):

        topic_model = BERTopic(
            language=None,
            embedding_model=nlp,
            n_gram_range=(1, 2),
            vectorizer_model=vectorizer_model,
            calculate_probabilities=True,
            verbose=True,
        )

        fit_topics, fit_probs = topic_model.fit_transform(documents)

        all_p = topic_model.get_topic_info().Count.to_list()

        s = np.sum(all_p)
        ent = 0
        for p in all_p:
            norm_p = p / s
            ent += norm_p * np.log(norm_p)

        if -ent > highest_ent:
            highest_ent = -ent
            best_model = deepcopy(topic_model)
            best_topics = fit_topics

    all_dists = []
    for d in documents:
        emb = []
        doc = nlp(d)
        for t in best_model.topic_embeddings:
            emb.append(cos_sim(doc.vector, t))
        all_dists.append(np.asarray(emb))

    index = nmslib.init(method="hnsw", space="cosinesimil")
    index.addDataPointBatch(all_dists)
    index.createIndex({"post": 2}, print_progress=True)

    data = []
    for doc, t, d in zip(documents, best_topics, all_dists):
        data.append({"doc": doc, "topic": t, "distances": d})

    df = pd.DataFrame(data)

    print(datetime.now().strftime("%H-%M-%S-%f") + " ----- Saving Index and Data -----")

    index.saveIndex(os.path.join(out_dir, "nms_index.index"))
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

    best_model.save(os.path.join(out_dir, "topic_model"), save_embedding_model=False)
    print(
        datetime.now().strftime("%H-%M-%S-%f")
        + "----- Topic Model Saved At: "
        + os.path.join(out_dir, "topic_model")
    )

    print(best_model.get_topic_info())
    print(df.head(5))


class Finder:
    def __init__(self, topic_model_folder_path) -> None:

        nlp = spacy_universal_sentence_encoder.load_model("xx_use_lg")

        from bertopic.backend._spacy import SpacyBackend

        model = SpacyBackend(nlp)

        self.topic_model = BERTopic.load(
            os.path.join(topic_model_folder_path, "topic_model"), embedding_model=model
        )

        self.docs = pd.read_pickle(os.path.join(topic_model_folder_path, "docs.pkl"))

        self.index = nmslib.init(method="hnsw", space="cosinesimil")
        self.index.loadIndex(os.path.join(topic_model_folder_path, "nms_index.index"))

    def get_similar(self, document, k=10):

        vec = self.topic_model._extract_embeddings(document.lower())
        topic, _ = self.topic_model.transform(document.lower())

        emb = []
        for t in self.topic_model.topic_embeddings:
            emb.append(cos_sim(vec, t))

        ids, ds = self.similar(emb, k=k)
#         print(ds)
        out = self.docs.iloc[ids].copy()
        out["dists"] = ds

        return out, topic[0]

    def similar(self, emb, k=10):
        # query for the nearest neighbours
        ids, distances = self.index.knnQuery(emb, k=k)
        # translate into words given it's index
        return ids, distances
