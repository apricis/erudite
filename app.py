import json
import string
import time
import logging
import re
import pickle
import configparser
import numpy as np
import pymysql
from answerer.qclass import q2bow, q2vec
from answerer.query import Reformulator
from answerer.extractor import NGramTiler, MajorityVoter
from flask import Flask, render_template, request, session, g
from elasticsearch import Elasticsearch, helpers
from nltk.corpus import stopwords
from urllib.parse import urlparse, parse_qs
from neo4j.v1 import GraphDatabase, basic_auth


config = configparser.ConfigParser()
config.read('config.ini')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# logging.disable(logging.ERROR)

app = Flask(__name__)

logging.info("loading question classification models...")
qclf = pickle.load(open('answerer/svm_wv.clf', 'rb'))
qle = pickle.load(open('answerer/svm.le', 'rb'))

logging.info("loading document frequencies...")
en_stopwords = stopwords.words('english')

logging.info("initializing elasticsearch instance...")
es = Elasticsearch(timeout=500)

logging.info("initializing DB connection...")
conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                       db=config['db']['name'], password=config['db']['password'])

vecs = pickle.load(open('../data/glove.6B/glove.300d.pkl', 'rb'))

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
    # query = Reformulator(original_question, en_stopwords).reformulate()
    query = original_question
    
    q_vec = re.findall(r"[\w']+|[{}]".format(string.punctuation), original_question)
    # pred_enc_label = qclf.predict(q2bow(q_vec, conn).reshape(1, -1))
    pred_enc_label = qclf.predict(q2vec(q_vec, vecs).reshape(1, -1))
    qclass = np.asscalar(qle.inverse_transform(pred_enc_label))
    
    es_query = {
        "query": {
            "match": {
                "content": query
            }
        }
    }
    pages = es.search(index=index_name, body=es_query, filter_path=['hits.hits'], size=10)
    snippets, page_ids = [], {}
    for p in pages['hits']['hits']:
        snippets.append(p['_source']['content'])
        url_params = parse_qs(urlparse(p['_source']['url']).query)
        page_ids[url_params['curid'][0]] = p['_source']['title']

    # non_utility_categories = """SELECT * FROM categorylinks 
    #                             WHERE cl_from IN ({}) AND cl_to NOT REGEXP 
    #                             '[Aa]rticles|[Ww]iki(pedia|data)|CS[0-9]_|[Tt]emplates?|[Pp]ages|[Uu]se_|Accuracy_disputes_|Engvar[A-Z]_|_stubs?';
    #                             """.format(",".join(page_ids.keys()))

    # with conn.cursor() as cursor:
    #     cursor.execute(non_utility_categories)
    #     res = cursor.fetchall()
    #     cat2root_query = """
    #         MATCH path=shortestPath((c:Category {{catName: "{}"}})-[:SUBCAT_OF*0..]->(r:RootCategory))
    #         UNWIND nodes(path) as n
    #         MATCH (n:Level1Category)
    #         RETURN n
    #         """
    #     print(page_ids)
    #     for page_id, cat_name, page_type in res:
    #         result = neo4j_session.run(cat2root_query.format(cat_name.decode('unicode_escape')))
    #         for record in result:
    #             print(page_ids[str(page_id)], record['n']['catName'])
    
    # mvoter = MajorityVoter(exp_answer_type=qclass)
    # answer = mvoter.extract(snippets)
    ngram_tiler = NGramTiler(max_n=4, connection=conn, question=original_question,
                             exp_answer_type=qclass, stopwords=en_stopwords)
    answer = ngram_tiler.extract(snippets)
    return render_template('search_results.html', question=original_question, pages=pages['hits']['hits'],
                           answer_type=qclass, answer=answer, query=query)
