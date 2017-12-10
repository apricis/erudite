import re
import json
import time
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
from operator import itemgetter
from elasticsearch import Elasticsearch, helpers
from data_reader import read_trec_format
from annoy import AnnoyIndex
import pymysql
import configparser



class QAPipeline(object):
    def __init__(self, qc_method, qc_features, qc_dimension, ranker_method, answer_extractor,
                 ranker_dimension=None, reformulate=True, exact_query=False,
                 nlc_index=False, article_reranking=True, lang='en'):
        self.__reformulate = reformulate
        self.punctuation_re = r'[{}]'.format(string.punctuation)
        config = configparser.ConfigParser()
        config.read('../config.ini')
        self.__conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                                      db=config['db']['name'], password=config['db']['password'])
        logging.info("loading question classification models...")
        self.__lang = lang
        self.__lang_map = {
            'en': ('english', 'en_US.UTF-8'),
            'sv': ('swedish', 'sv_SE.UTF-8')
        }
        self.__qc_method = qc_method
        self.__qc_features = qc_features
        if qc_features == 'wv':
            clf_name = "{}_{}_{}_{}.clf".format(lang, qc_method, qc_features, qc_dimension)
        else:
            clf_name = "{}_{}_{}.clf".format(lang, qc_method, qc_features)
        le_name = '{}_{}.le'.format(lang, qc_method)
        self.qclf = pickle.load(open(clf_name, 'rb'))
        self.qle = pickle.load(open(le_name, 'rb'))
        logging.info("loading document frequencies...")
        self.stopwords = stopwords.words(self.__lang_map[lang][0])

        logging.info("initializing elasticsearch instance...")
        self.es = Elasticsearch(timeout=500)

        self.qc_vecs_index = AnnoyIndex(qc_dimension)
        self.qc_vecs_index.load('../data/glove.{}.{}.ann'.format(lang, qc_dimension))

        if ranker_dimension is None:
            self.rk_vecs_index = self.qc_vecs_index
        else:
            self.rk_vecs_index = AnnoyIndex(ranker_dimension)
            self.rk_vecs_index.load('../data/glove.{}.{}.ann'.format(lang, ranker_dimension))
        self.rk_method = ranker_method
        self.do_exact = exact_query
        self.average_query_time = 0
        self.number_of_queries = 0
        self.nlc_index = nlc_index
        self.article_reranking = article_reranking
        self.answer_extractor = answer_extractor


    def answer(self, original_question, top5=False):
        start_time = time.time()
        q_tokens = re.findall(r"[\w']+|[{}]".format(string.punctuation), original_question)
        if self.__qc_features == 'wv':
            q_vec = q2vec(q_tokens, self.qc_vecs_index, self.__conn, lang=self.__lang)
        else:
            q_vec = q2bow(q_tokens, self.__conn, lang=self.__lang)
        pred_enc_label = self.qclf.predict(q_vec.reshape(1, -1))
        qclass = np.asscalar(self.qle.inverse_transform(pred_enc_label))
        print("EAT {}".format(qclass))
        
        # change here to language specific as well
        index_name = 'wiki_nlc' if self.nlc_index else 'svwiki'
        print("Using index {}".format(index_name))
        reformulator = Reformulator(original_question, qclass, lang=self.__lang, stopwords=self.stopwords)
        no_punctuation_question = reformulator.question()
        
        if self.do_exact:
            query = reformulator.reformulate_exact()
            exact_es_query = {
                "query": {
                    "match_phrase": {
                        "content": {
                            "query": query
                        }
                    }
                }
            }
            pages = self.es.search(index=index_name, body=exact_es_query, filter_path=['hits.hits'], size=10)
            executed_query_type = 'phrase'

        if not self.do_exact or (len(pages['hits']['hits']) < 2):
            if self.__reformulate:
                query = reformulator.reformulate()
            else:
                query = no_punctuation_question
            inexact_es_query = {
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "and"
                            # "low_freq_operator": "and",
                            # "cutoff_frequency": 0.05
                        }
                    }
                }
            }
            pages = self.es.search(index=index_name, body=inexact_es_query, filter_path=['hits.hits'], size=30)
            executed_query_type = 'intersection'
        print("Query: {}".format(query))
        print("Executed query type: {}".format(executed_query_type))
        snippets, article_names, scores = [], [], []
        for p in pages['hits']['hits']:
            snippets.append(p['_source']['content'])
            article_names.append(p['_source']['title'])
            scores.append(p['_score'])
        ranked_pages, ranked_snippets = [], []

        ranker = PassageRanker(self.rk_vecs_index, self.__conn, lang=self.__lang)
        if executed_query_type == 'intersection' and self.article_reranking:
            discounts = ranker.rank_articles(article_names, query)
            scores = [(i, s * d) for i, (s, d) in enumerate(zip(scores, discounts))]
            scores = sorted(scores, key=itemgetter(1), reverse=True)
            ranked_pages = [pages['hits']['hits'][i] for i, _ in scores[:10]]
            snippets = [snippets[i] for i, _ in scores[:10]]
        
        if self.rk_method != 'none':
            if self.rk_method == 'wv':
                ranked_snippets = ranker.rank_snippets_glove(snippets, query)[:50]
            else:
                ranked_snippets = ranker.rank_snippets_tf_idf(snippets, query)[:50]
            snippets = [s for s, d in ranked_snippets]
        
        if self.answer_extractor == 'ngt':
            ngram_tiler = NGramTiler(max_n=4, connection=self.__conn, question=no_punctuation_question, 
                                     exp_answer_type=qclass, stopwords=self.stopwords, lang=self.__lang,
                                     used_locale=self.__lang_map[self.__lang][1])
            answer = ngram_tiler.extract(snippets, top5=top5)
        else:
            mvoter = MajorityVoter(question=no_punctuation_question, exp_answer_type=qclass, lang=self.__lang,
                                   used_locale=self.__lang_map[self.__lang][1])
            answer = mvoter.extract(snippets, top5=top5)

        query_time = time.time() - start_time
        self.number_of_queries += 1
        self.average_query_time = (self.average_query_time * (self.number_of_queries - 1) + query_time) / self.number_of_queries
        return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-qm', '--qc-method', type=str, choices=['svm', 'rvm', 'hsvm'], default='svm')
    parser.add_argument('-qf', '--qc-features', type=str, choices=['bow', 'wv'], default='wv')
    parser.add_argument('-l', '--lang', type=str, choices=['en', 'sv'], default='en')
    parser.add_argument('-qd', '--qc-dimension', type=int, default=200)
    parser.add_argument('-rd', '--rk-dimension', type=int, default=200)
    parser.add_argument('-rm', '--rk-method', type=str, choices=['none', 'tfidf', 'wv'], default='wv')
    parser.add_argument('-r', '--reformulate', action='store_true')
    parser.add_argument('-ar', '--article-reranking', action='store_true')
    parser.add_argument('-eq', '--exact-query', action='store_true')
    parser.add_argument('-nlci', '--nlc-index', action='store_true')
    parser.add_argument('-d', '--dataset', type=str, default='../data/curated-train.tsv')
    parser.add_argument('-ae', '--answer-extractor', type=str, choices=['ngt', 'mv'], default='ngt')
    args = parser.parse_args()
    print(args)

    logging.disable(logging.ERROR)
    pipeline = QAPipeline(args.qc_method, args.qc_features, args.qc_dimension, args.rk_method,
                          args.answer_extractor, ranker_dimension=args.rk_dimension,
                          reformulate=args.reformulate, exact_query=args.exact_query,
                          nlc_index=args.nlc_index, article_reranking=args.article_reranking,
                          lang=args.lang)

    success, top5_success, total, mrr = 0, 0, 0, 0
    for q, a in read_trec_format(args.dataset):
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
                    mrr += 1 / (i + 1)
                    break
        total += 1
        print("=============================")
    mrr /= total
    print(success, top5_success, total, mrr)
    print("Average query time: {}s".format(pipeline.average_query_time))