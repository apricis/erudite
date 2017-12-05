import re
import json
import numpy as np
import pickle
import string
import logging
import argparse
from qclass import q2bow, q2vec
from extractor import NGramTiler, MajorityVoter
from query import Reformulator
from ranker import PassageRanker
from nltk.corpus import stopwords
from urllib.parse import urlparse, parse_qs
from operator import itemgetter
from elasticsearch import Elasticsearch, helpers
from data_reader import read_trec_format
from annoy import AnnoyIndex
import pymysql
import configparser



class QAPipeline(object):
    def __init__(self, qc_method, qc_features, qc_dimension, ranker_method,
                 ranker_dimension=None, reformulate=True):
        self.__reformulate = reformulate
        self.punctuation_re = r'[{}]'.format(string.punctuation)
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.__conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                                      db=config['db']['name'], password=config['db']['password'])
        logging.info("loading question classification models...")
        self.__qc_method = qc_method
        self.__qc_features = qc_features
        if qc_features == 'wv':
            clf_name = "{}_{}_{}.clf".format(qc_method, qc_features, qc_dimension)
        else:
            clf_name = "{}_{}.clf".format(qc_method, qc_features)
        le_name = '{}.le'.format(qc_method)
        self.qclf = pickle.load(open(clf_name, 'rb'))
        self.qle = pickle.load(open(le_name, 'rb'))
        logging.info("loading document frequencies...")
        self.en_stopwords = stopwords.words('english')

        logging.info("initializing elasticsearch instance...")
        self.es = Elasticsearch(timeout=500)

        self.qc_vecs_index = AnnoyIndex(qc_dimension)
        self.qc_vecs_index.load('../data/glove{}.ann'.format(qc_dimension))

        if ranker_dimension is None:
            self.rk_vecs_index = self.qc_vecs_index
        else:
            self.rk_vecs_index = AnnoyIndex(ranker_dimension)
            self.rk_vecs_index.load('../data/glove{}.ann'.format(ranker_dimension))
        self.rk_method = ranker_method


    def answer(self, original_question, top5=False):
        q_tokens = re.findall(r"[\w']+|[{}]".format(string.punctuation), original_question)
        if self.__qc_features == 'wv':
            q_vec = q2vec(q_tokens, self.qc_vecs_index, self.__conn)
        else:
            q_vec = q2bow(q_tokens, self.__conn)
        pred_enc_label = self.qclf.predict(q_vec.reshape(1, -1))
        qclass = np.asscalar(self.qle.inverse_transform(pred_enc_label))
        print("EAT {}".format(qclass))
        
        index_name = 'wiki'
        reformulator = Reformulator(original_question, qclass, self.en_stopwords)
        no_punctuation_question = reformulator.question()
        
        if self.__reformulate:
            query = reformulator.reformulate()
        else:
            query = no_punctuation_question
        print("Query: {}".format(query))
        es_query = {
            "query": {
                "match": {
                    "content": {
                        "query": query,
                        "operator": "and"
                    }
                }
            }
        }
        pages = self.es.search(index=index_name, body=es_query, filter_path=['hits.hits'], size=30)

        snippets, page_ids, article_names, scores = [], {}, [], []
        for p in pages['hits']['hits']:
            snippets.append(p['_source']['content'])
            url_params = parse_qs(urlparse(p['_source']['url']).query)
            page_ids[url_params['curid'][0]] = p['_source']['title']
            article_names.append(p['_source']['title'])
            scores.append(p['_score'])
        # ranked_pages = []

        ranker = PassageRanker(self.rk_vecs_index, self.__conn)
        discounts = ranker.rank_articles(article_names, query)
        scores = [(i, s * d) for i, (s, d) in enumerate(zip(scores, discounts))]
        scores = sorted(scores, key=itemgetter(1), reverse=True)
        ranked_pages = [pages['hits']['hits'][i] for i, _ in scores[:10]]

        snippets = [snippets[i] for i, _ in scores[:10]]
        if self.rk_method == 'wv':
            ranked_snippets = ranker.rank_snippets_glove(snippets, query)[:100]
        else:
            ranked_snippets = ranker.rank_snippets_tf_idf(snippets, query)[:100]
        snippets = [s for s, d in ranked_snippets]

        # mvoter = MajorityVoter(exp_answer_type=qclass)
        # answer = mvoter.extract(snippets)
        ngram_tiler = NGramTiler(max_n=3, connection=self.__conn, question=no_punctuation_question, 
                                 exp_answer_type=qclass, stopwords=self.en_stopwords)
        answer = ngram_tiler.extract(snippets, top5=top5)
        return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-qm', '--qc-method', type=str, choices=['svm', 'rvm', 'hsvm'], default='svm')
    parser.add_argument('-qf', '--qc-features', type=str, choices=['bow', 'wv'], default='wv')
    parser.add_argument('-qd', '--qc-dimension', type=int, default=200)
    parser.add_argument('-rd', '--rk-dimension', type=int, default=200)
    parser.add_argument('-rm', '--rk-method', type=str, choices=['tfidf', 'wv'], default='wv')
    parser.add_argument('-r', '--reformulate', action='store_true')
    args = parser.parse_args()

    logging.disable(logging.ERROR)
    pipeline = QAPipeline(args.qc_method, args.qc_features, args.qc_dimension, args.rk_method,
                          ranker_dimension=args.rk_dimension, reformulate=args.reformulate)

    success, top5_success, total = 0, 0, 0
    for q, a in read_trec_format('../data/curated-train.tsv'):
        print(q)
        final_answers = pipeline.answer(q, top5=True)
        print("Pipeline answer(s):", final_answers)
        print("Ground truth:", a)
        if final_answers:
            for i, fa in enumerate(final_answers):
                fa = " ".join(fa)
                print(fa and (re.match(a.lower(), fa.lower()) or (fa.lower() in a.lower())))
                if fa and (re.match(a.lower(), fa.lower()) or (fa.lower() in a.lower())):
                    top5_success += 1
                    if i == 0:
                        success += 1
                    break
        total += 1
        print("=============================")
    print(success, top5_success, total)