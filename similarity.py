import nmslib
import os
import spacy
import json
import pandas as pd
import numpy as np
from keybert import KeyBERT
from datetime import datetime


class Finder:
    def __init__(self, index_folder_path, embedding_model="pt_core_news_lg", diversity=0):

        if isinstance(embedding_model, str):
            self.nlp = spacy.load(embedding_model)
        else:
            self.nlp = embedding_model

        self.index = nmslib.init(method="hnsw", space="angulardist")
        self.index.loadIndex(
            os.path.join(index_folder_path, "nms_index.index"), load_data=True
        )

        self.docs = pd.read_pickle(os.path.join(index_folder_path, "docs.pkl"))

        with open(os.path.join(index_folder_path, "params.json")) as f:
            self.params = json.load(f)
        self.params["min_df"] = 1
        if diversity > 0 and self.params['diversity'] > 0:
            self.params['diversity'] = diversity

        self.kw_model = KeyBERT(model=self.nlp)

    def get_similar(self, document, k=10):
        document = document.lower()

        new_kw = self.kw_model.extract_keywords(document, **self.params)

        local_embs = []
        for key, val in new_kw:
            doc = self.nlp(key)
            local_embs.append(doc.vector)
        final_emb = np.array(local_embs).mean(axis=0)
        ids, ds = self.similar_raw(final_emb, k)

        out = self.docs.iloc[ids].copy()
        out["distances"] = ds

        return out, new_kw

    def similar_raw(self, emb, k=10):
        # query for the nearest neighbours
        ids, distances = self.index.knnQuery(emb, k=k)
        # translate into words given it's index
        return ids, distances
