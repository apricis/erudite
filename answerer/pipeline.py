import re
import json
import numpy as np
import pickle
import string
import logging
from qclass import q2bow, q2vec
from extractor import NGramTiler, MajorityVoter
from query import Reformulator
from nltk.corpus import stopwords
from elasticsearch import Elasticsearch, helpers
from data_reader import read_trec_format
import pymysql
import configparser


class QAPipeline(object):
    def __init__(self):
        self.punctuation_re = r'[{}]'.format(string.punctuation)
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.__conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                                      db=config['db']['name'], password=config['db']['password'])
        logging.info("loading question classification models...")
        self.qclf = pickle.load(open('svm_wv.clf', 'rb'))
        self.qle = pickle.load(open('svm.le', 'rb'))
        logging.info("loading document frequencies...")
        self.en_stopwords = stopwords.words('english')
        # self.vecs = pickle.load(open('../../data/glove.6B/glove.300d.pkl', 'rb'))

        logging.info("initializing elasticsearch instance...")
        self.es = Elasticsearch(timeout=500)

    def answer(self, original_question):
        query = Reformulator(original_question, self.en_stopwords).reformulate()
        q_vec = re.findall(r"[\w']+|{}".format(self.punctuation_re), original_question)
        pred_enc_label = self.qclf.predict(q2bow(q_vec, self.__conn).reshape(1, -1))
        # pred_enc_label = self.qclf.predict(q2vec(q_vec, self.vecs).reshape(1, -1))
        qclass = np.asscalar(self.qle.inverse_transform(pred_enc_label))
        print(qclass)
        
        es_query = {
            "query": {
                "match": {
                    "content": query
                }
            }
        }
        
        pages = self.es.search(index="wiki", body=es_query, filter_path=['hits.hits'], size=10)
        snippets = []
        for p in pages['hits']['hits']:
            snippets.append(p['_source']['content'])

        # mvoter = MajorityVoter(exp_answer_type=qclass)
        # answer = mvoter.extract(snippets)
        ngram_tiler = NGramTiler(max_n=4, connection=self.__conn, question=original_question, 
                                 exp_answer_type=qclass, stopwords=self.en_stopwords)
        answer = ngram_tiler.extract(snippets)
        return answer


if __name__ == '__main__':
    logging.disable(logging.ERROR)
    pipeline = QAPipeline()

    success, total = 0, 0
    for q, a in read_trec_format('../data/curated-train.tsv'):
        print(q)
        fqa = pipeline.answer(q)
        print("Pipeline answer:", fqa)
        print("Ground truth:", a)
        print(fqa and (re.match(a.lower(), fqa.lower()) or (fqa.lower() in a.lower())))
        print("=============================")
        if fqa and (re.match(a.lower(), fqa.lower()) or (fqa.lower() in a.lower())):
            success += 1
        total += 1
    print(success, total)