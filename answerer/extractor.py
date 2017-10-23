import re
import operator
from itertools import tee, islice
from collections import defaultdict
from polyglot.text import Text


SNIPPETS = [
	"Oslo is the capital of Norway.",
	"Turner Broadcasting System Norway AS is one of Turner's European divisions. The company is based Oslo, the capital of Norway. The company owns and operates Cartoon Network Norway, Boomerang Norway, and TCM Norway.",
	"Eastern Norway is by far the most populous region of Norway. It contains the country's capital, Oslo, which is Norway's most populous city.",
	"In 1988 Norway signed on to protocol 6 of the European Convention on Human Rights which bans the use of capital punishment in peacetime and ratified protocol 13 which bans all use of capital punishment whatsoever in 2005. Norway generally opposes capital punishment outside of the country as well. The government has banished Mullah Krekar from Norway, but has not sent him to Iraq due to the possibility of him being charged with capital crimes in his home county. In the Martine Vik Magnussen case, Norway has declined to cooperate with the Yemenese government unless a guarantee is made that the death penalty is off the table.",
	"Oslo, founded in 1000, is the largest city and the capital of Norway.",
	"Nordberg is a neighbourhood in Nordre Aker in Oslo, the capital of Norway.",
	"Norway maintains embassies in 86 countries. 60 countries maintain an embassy in Norway, all of them in the capital, Oslo.",
	"Drammen is a city in Buskerud, Norway. The port and river city of Drammen is centrally located in the eastern and most populated part of Norway. Drammen is the capital of the county of Buskerud.",
	"During the late Middle Ages and until the breakup of the union between Sweden and Norway Inderøy was the seat of the Governor, Judge, and Tax Collector of Nordre Trondhjems amt, thus it was the county capital of what now is known as Nord-Trøndelag. The district court for central Nord-Trøndelag is still named after Inderøy.",
	"Drammen Fjernvarme District Heating is a district heating system in Drammen, Norway, a regional capital some 65km west of Oslo."
]


def nwise(iterable, n=2):                                                      
    iters = tee(iterable, n)                                                     
    for i, it in enumerate(iters):                                               
        next(islice(it, i, i), None)                                               
    return list(zip(*iters))


class NGramTiler(object):
	def __init__(self, max_n=3, question='', exp_answer_type='', stopwords=[]):
		self.max_n = 3
		self.__question = question.lower().split()
		self.__eat = exp_answer_type
		self.__ngram_names = ['uni', 'bi', 'tri', 'tetra', 'peta']
		self.__entities = None
		self.__test_nec = {
			'LOC': 'LOC' in self.__eat,
			'ORG': self.__eat == 'HUM:gr',
			'PER': self.__eat in ['HUM:ind', 'HUM:title'],
			None: False
		}
		self.__stopwords = stopwords

	def __test_eat(self, word):
		if 'NUM' in self.__eat:
			return word.isnumeric()
		elif 'ABBR' in self.__eat:
			return True
		elif 'DESC' in self.__eat:
			return True
		elif 'ENTY' in self.__eat:
			return True
		else:
			return self.__test_nec[self.__entities.get(word, None)]

	def mine(self, snippets):
		snippets = [re.sub(r'[.,!?;()]', '', snippet.lower()) for snippet in snippets]
		entities = Text(" ".join(snippets)).entities
		self.__entities = dict([(" ".join(ent), ent.tag[2:]) for ent in entities])
		snippets = [snippet.split() for snippet in snippets]
		ng_snippets = {}
		for n in range(self.max_n):
			ng_snippets["{}grams".format(self.__ngram_names[n])] = [nwise(snippet, n + 1) 
																	for snippet in snippets]
		return self.__n_gram_stats(ng_snippets)

	def __n_gram_stats(self, ng_snippets):
		stats = defaultdict(set)
		for ng_type in ng_snippets:
			for i, ng_snippet in enumerate(ng_snippets[ng_type]):
				for ng in ng_snippet:
					stats[ng].add(i)
		stats = {k:len(v) for k, v in stats.items()}
		return stats

	def filter(self, votes):
		filtered_votes = {}
		for k, v in votes.items():
			if k[0] not in self.__stopwords and k[-1] not in self.__stopwords:
				for word in k:
					# focus words exception should be added
					if self.__test_eat(word) and word not in self.__question:
						filtered_votes[k] = v
						break

		return sorted(filtered_votes.items(), key=operator.itemgetter(1), reverse=True)

	def tile(self, votes):
		pass

	def extract(self, snippets):
		votes = self.filter(self.mine(snippets))
		print(votes)
		return " ".join(votes[0][0])


if __name__ == '__main__':
	from nltk.corpus import stopwords
	en_stopwords = stopwords.words('english')
	ngram_tiler = NGramTiler(question='What is the capital of Norway', exp_answer_type='LOC:city', stopwords=en_stopwords)
	print(ngram_tiler.extract(SNIPPETS))