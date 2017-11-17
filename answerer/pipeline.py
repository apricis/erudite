import re
import json
import numpy as np
import pickle
import string
import logging
from qclass import q2bow
from extractor import NGramTiler, MajorityVoter
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch, helpers
from data_reader import read_trec_format
from urllib.parse import urlparse, parse_qs


class QAPipeline(object):
    def __init__(self):
        self.punctuation_re = r'[{}]'.format(string.punctuation)
        logging.info("loading words from wikipedia dump...")
        self.words = dict([(w, i) for i, w in enumerate(json.load(open('../data/count.json')))])
        logging.info("loading question classification models...")
        self.qclf = pickle.load(open('svm_bow.clf', 'rb'))
        self.qle = pickle.load(open('svm.le', 'rb'))
        logging.info("loading document frequencies...")
        self.df = json.load(open('../data/df.json'))

        logging.info("loading document frequencies...")
        self.en_stopwords = stopwords.words('english')

        logging.info("initializing elasticsearch instance...")
        self.es = Elasticsearch(timeout=500)

    def answer(self, original_question):
        question = re.sub(self.punctuation_re, '', original_question).lower()
        q_vec = re.findall(r"[\w']+|{}".format(self.punctuation_re), original_question)
        pred_enc_label = self.qclf.predict(q2bow(q_vec, self.words).reshape(1, -1))
        qclass = np.asscalar(self.qle.inverse_transform(pred_enc_label))
        query = {
            "query": {
                "match": {
                    "content": question
                }
            }
        }
        
        pages = self.es.search(index="wiki", body=query, filter_path=['hits.hits'], size=10)
        snippets, page_ids = [], {}
        for p in pages['hits']['hits']:
            snippets.append(p['_source']['content'])
            url_params = parse_qs(urlparse(p['_source']['url']).query)
            page_ids[url_params['curid'][0]] = p['_source']['title']

        # mvoter = MajorityVoter(exp_answer_type=qclass)
        # answer = mvoter.extract(snippets)
        ngram_tiler = NGramTiler(max_n=4, question=question, exp_answer_type=qclass, stopwords=self.en_stopwords)
        answer = ngram_tiler.extract(snippets, self.df)
        return answer


if __name__ == '__main__':
    pipeline = QAPipeline()

    success, total = 0, 0
    for q, a in read_trec_format('../data/curated-train.tsv'):
        fqa = pipeline.answer(q)
        if fqa and re.match(a.lower(), fqa.lower()):
            success += 1
        total += 1
    print(success, total)