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
from urllib.parse import urlparse, parse_qs
from annoy import AnnoyIndex
from neo4j.v1 import GraphDatabase, basic_auth


config = configparser.ConfigParser()
config.read('config.ini')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logging.disable(logging.ERROR)

app = Flask(__name__)

logging.info("loading question classification models...")
qclf = pickle.load(open('answerer/svm_wv_300.clf', 'rb'))
qle = pickle.load(open('answerer/svm.le', 'rb'))

logging.info("loading document frequencies...")
en_stopwords = stopwords.words('english')

logging.info("initializing elasticsearch instance...")
es = Elasticsearch(timeout=500)

logging.info("initializing DB connection...")
conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                       db=config['db']['name'], password=config['db']['password'])

# this should be taken from somewhere
logging.info("initializing Annoy index...")
vecs_index = AnnoyIndex(300)
vecs_index.load('data/glove300.ann')

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
    index_name = 'enwiki_trigram' if request.form.get('index', False) else 'wiki'
    
    q_tokens = re.findall(r"[\w']+|[{}]".format(string.punctuation), original_question)
    # pred_enc_label = qclf.predict(q2bow(q_tokens, conn).reshape(1, -1))
    q_vec = q2vec(q_tokens, vecs_index, conn)
    pred_enc_label = qclf.predict(q_vec.reshape(1, -1))
    qclass = np.asscalar(qle.inverse_transform(pred_enc_label))
    
    reformulator = Reformulator(original_question, qclass, en_stopwords)
    no_punctuation_question = reformulator.question()
    query = reformulator.reformulate()
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
    pages = es.search(index=index_name, body=es_query, filter_path=['hits.hits'], size=30)
    print("Got ES results at %.5fs"  % (time.time() - g.request_start_time))

    snippets, page_ids, article_names, scores = [], {}, [], []
    for p in pages['hits']['hits']:
        snippets.append(p['_source']['content'])
        url_params = parse_qs(urlparse(p['_source']['url']).query)
        page_ids[url_params['curid'][0]] = p['_source']['title']
        article_names.append(p['_source']['title'])
        scores.append(p['_score'])
    ranked_pages, ranked_snippets = [], []

    print("Collected articles at %.5fs"  % (time.time() - g.request_start_time))

    # should we re-rank like that?
    ranker = PassageRanker(vecs_index, conn)
    discounts = ranker.rank_articles(article_names, query)
    scores = [(i, s * d) for i, (s, d) in enumerate(zip(scores, discounts))]
    scores = sorted(scores, key=itemgetter(1), reverse=True)
    ranked_pages = [pages['hits']['hits'][i] for i, _ in scores[:10]]
    print("Re-ranked articles at %.5fs"  % (time.time() - g.request_start_time))

    snippets = [snippets[i] for i, _ in scores[:10]]
    ranked_snippets = ranker.rank_snippets_tf_idf(snippets, query)[:100]
    snippets = [s for s, d in ranked_snippets]
    print("Ranked snippets at %.5fs"  % (time.time() - g.request_start_time))
    
    # mvoter = MajorityVoter(exp_answer_type=qclass)
    # answer = mvoter.extract(snippets)
    ngram_tiler = NGramTiler(max_n=3, connection=conn, question=no_punctuation_question,
                             exp_answer_type=qclass, stopwords=en_stopwords)
    print("Initialized NGramTiler at %.5fs"  % (time.time() - g.request_start_time))
    answer = ngram_tiler.extract(snippets)
    print("Extracted answer at %.5fs"  % (time.time() - g.request_start_time))
    print('----------------------------------------')
    answer_string = " ".join(answer) if answer else "I don't know"
    return render_template('search_results.html', question=original_question, pages=pages['hits']['hits'],
                           ranked_pages=ranked_pages, passages=ranked_snippets, answer_type=qclass, answer=answer_string, query=query)
