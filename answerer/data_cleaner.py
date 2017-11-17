import logging
import json


logging.info("loading words from wikipedia dump...")
wiki_words = dict([(w, i) for i, w in enumerate(json.load(open('../data/count.json')))])


def uiuc2clean(fname):
    answer_types, questions = [], []
    with open(fname, encoding="ISO-8859-1") as f:
        for line in f:
            data = line.split()
            for w in data[1:]:
            	if w.lower() not in wiki_words:
            		print(w)


if __name__ == '__main__':
	uiuc2clean('../../data/uiuc_corrected/test.label')