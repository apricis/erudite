import json
import string
import time
import logging
import re
import pickle
import configparser
from operator import itemgetter
import numpy as np
import pymysql
from answerer.qclass import q2bow, q2vec
from answerer.query import Reformulator
from answerer.extractor import NGramTiler, MajorityVoter
from answerer.ranker import PassageRanker
from flask import Flask, render_template, request, session, g
from elasticsearch import Elasticsearch, helpers
from nltk.corpus import stopwords
from annoy import AnnoyIndex
from neo4j.v1 import GraphDatabase, basic_auth


config = configparser.ConfigParser()
config.read('config.ini')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logging.disable(logging.ERROR)

app = Flask(__name__)

language = 'swedish'
lang_code = 'sv'
locale_code = 'sv_SE.UTF-8'

logging.info("loading question classification models...")

qclf = pickle.load(open('answerer/{}_svm_wv_200.clf'.format(lang_code), 'rb'))
qle = pickle.load(open('answerer/{}_svm.le'.format(lang_code), 'rb'))

logging.info("loading document frequencies...")
allow_stopwords = ['m']
lang_stopwords = set(stopwords.words(language)) - set(allow_stopwords)

logging.info("initializing elasticsearch instance...")
es = Elasticsearch(timeout=500)

logging.info("initializing DB connection...")
conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                       db=config['db']['name'], password=config['db']['password'])

# this should be taken from somewhere
logging.info("initializing Annoy index...")
vecs_index = AnnoyIndex(200)
vecs_index.load('data/glove.{}.200.ann'.format(lang_code))

# neo4j_auth = basic_auth(config['neo4j']['login'], config['neo4j']['password'])
# neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=neo4j_auth)
# neo4j_session = neo4j_driver.session()


@app.before_request
def before_request():
    g.request_start_time = time.time()
    g.request_time = lambda: "%.5fs" % (time.time() - g.request_start_time)


@app.route("/")
def main():
    return render_template('main.html')


@app.route("/search", methods=["POST"])
def search():
    original_question = request.form["question"]
    index_name = 'wiki_nlc' if request.form.get('index', False) else 'svwiki'
    
    q_tokens = re.findall(r"[\w']+|[{}]".format(string.punctuation), original_question)
    # pred_enc_label = qclf.predict(q2bow(q_tokens, conn).reshape(1, -1))
    q_vec = q2vec(q_tokens, vecs_index, conn, lang=lang_code)
    pred_enc_label = qclf.predict(q_vec.reshape(1, -1))
    qclass = np.asscalar(qle.inverse_transform(pred_enc_label))
    
    reformulator = Reformulator(original_question, qclass, lang=lang_code, stopwords=lang_stopwords)
    no_punctuation_question = reformulator.question()
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
    pages = es.search(index=index_name, body=exact_es_query, filter_path=['hits.hits'], size=10)
    executed_query_type = 'phrase'
    print("Got ES exact results at %.5fs"  % (time.time() - g.request_start_time))

    if len(pages['hits']['hits']) < 2:
        query = reformulator.reformulate()
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
        pages = es.search(index=index_name, body=inexact_es_query, filter_path=['hits.hits'], size=30)
        executed_query_type = 'intersection'
        print("Got ES inexact results at %.5fs"  % (time.time() - g.request_start_time))

    snippets, article_names, scores = [], [], []
    for p in pages['hits']['hits']:
        snippets.append(p['_source']['content'])
        article_names.append(p['_source']['title'])
        scores.append(p['_score'])
    ranked_pages, ranked_snippets = [], []

    print("Collected articles at %.5fs"  % (time.time() - g.request_start_time))

    ranker = PassageRanker(vecs_index, conn, lang=lang_code)
    # if executed_query_type == 'intersection':
    #     # should we re-rank like that?
    #     discounts = ranker.rank_articles(article_names, query)
    #     scores = [(i, s * d) for i, (s, d) in enumerate(zip(scores, discounts))]
    #     scores = sorted(scores, key=itemgetter(1), reverse=True)
    #     ranked_pages = [pages['hits']['hits'][i] for i, _ in scores[:10]]
    #     snippets = [snippets[i] for i, _ in scores[:10]]
    #     print("Re-ranked articles at %.5fs"  % (time.time() - g.request_start_time))

    ranked_snippets = ranker.rank_snippets_glove(snippets, query)[:50]
    snippets = [s for s, d in ranked_snippets]
    print("Ranked snippets at %.5fs"  % (time.time() - g.request_start_time))
    
    # mvoter = MajorityVoter(exp_answer_type=qclass)
    # answer = mvoter.extract(snippets)
    ngram_tiler = NGramTiler(max_n=4, connection=conn, question=no_punctuation_question, used_locale=locale_code,
                             lang=lang_code, exp_answer_type=qclass, stopwords=lang_stopwords)
    print("Initialized NGramTiler at %.5fs"  % (time.time() - g.request_start_time))
    answer = ngram_tiler.extract(snippets)
    print("Extracted answer at %.5fs"  % (time.time() - g.request_start_time))
    print('----------------------------------------')
    answer_string = " ".join(answer) if answer else "I don't know"
    return render_template('search_results.html', question=original_question, pages=pages['hits']['hits'],
                           ranked_pages=ranked_pages, passages=ranked_snippets, answer_type=qclass,
                           answer=answer_string, query=query, query_type=executed_query_type)
