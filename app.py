import json
import re
import pickle
import numpy as np
from answerer.qclass import q2bow
from flask import Flask, render_template, request, session
from elasticsearch import Elasticsearch, helpers

app = Flask(__name__)
# too memory consuming to load count.json
words = dict([(w, i) for i, w in enumerate(json.load(open('../wiki_indexer/count.json')))])
qclf = pickle.load(open('answerer/svm_bow.clf', 'rb'))
qle = pickle.load(open('answerer/svm.le', 'rb'))
es = Elasticsearch(timeout=500)

@app.route("/")
def main():
    return render_template('main.html')

@app.route("/search", methods=["POST"])
def search():
	query_string = request.form["query"]
	q_vec = re.findall(r"[\w']+|[.,!?;()]", query_string)
	print(q_vec)
	pred_enc_label = qclf.predict(q2bow(q_vec, words).reshape(1, -1))
	qclass = np.asscalar(qle.inverse_transform(pred_enc_label))
	query = {
		"query": {
			"match": {
				"passage": query_string
			}
		}
	}
	pages = es.search(index="wiki", body=query, filter_path=['hits.hits'], size=100)
	return render_template('search_results.html', query=query_string, pages=pages['hits']['hits'],
		answer_type=qclass)
