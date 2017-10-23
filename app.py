import json
import time
import logging
import re
import pickle
import numpy as np
from answerer.qclass import q2bow
from answerer.extractor import NGramTiler
from flask import Flask, render_template, request, session, g
from elasticsearch import Elasticsearch, helpers
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
# too memory consuming to load count.json
words = dict([(w, i) for i, w in enumerate(json.load(open('../wiki_indexer/count.json')))])
qclf = pickle.load(open('answerer/svm_bow.clf', 'rb'))
qle = pickle.load(open('answerer/svm.le', 'rb'))
es = Elasticsearch(timeout=500)
en_stopwords = stopwords.words('english')

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
	question = re.sub(r'[.,!?;()]', '', original_question).lower()
	q_vec = re.findall(r"[\w']+|[.,!?;()]", original_question)
	pred_enc_label = qclf.predict(q2bow(q_vec, words).reshape(1, -1))
	qclass = np.asscalar(qle.inverse_transform(pred_enc_label))
	query = {
		"query": {
			"match": {
				"passage": question
			}
		}
	}
	pages = es.search(index="wiki", body=query, filter_path=['hits.hits'], size=100)
	snippets = [p['_source']['passage'] for p in pages['hits']['hits']]
	ngram_tiler = NGramTiler(question=question, exp_answer_type=qclass, stopwords=en_stopwords)
	answer = ngram_tiler.extract(snippets)
	return render_template('search_results.html', question=original_question, pages=pages['hits']['hits'],
		answer_type=qclass, answer=answer)
