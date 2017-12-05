import re
import string


class Reformulator(object):
    def __init__(self, question, qclass, stopwords=None):
        self.__original_question = question
        self.__punctuation_re = r'[{}]'.format(string.punctuation)
        # this substitution should deal good with U.S. case
        self.__question = re.sub(self.__punctuation_re, '', question).lower()
        self.__stopwords = stopwords
        self.__qclass = qclass.split(':')[1]
        self.__expansion_rules = {
            'cremat': 'creative',
            'dismed': 'disease',
            'instru': 'instrument',
            'lang': 'language',
            'other': '',
            'techmeth': 'technique',
            'termeq': 'term',
            'veh': 'vehicle',
            'dist': 'distance',
            'ord': 'order',
            'perc': 'percentage',
            'temp': 'temperature',
            'volsize': 'size'
        }

    def question(self):
        return self.__question

    def reformulate(self):
        without_stopwords = [w for w in self.__question.split() if w not in self.__stopwords]
        query = without_stopwords
        query.append(self.__expansion_rules.get(self.__qclass, ''))
        return " ".join(query)
