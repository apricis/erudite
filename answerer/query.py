import re
import string


class Reformulator(object):
    def __init__(self, question, stopwords):
        self.__original_question = question
        self.__punctuation_re = r'[{}]'.format(string.punctuation)
        self.__question = re.sub(self.__punctuation_re, '', question).lower()
        self.__stopwords = stopwords

    def reformulate(self):
        return " ".join([w for w in self.__question.split() if w not in self.__stopwords])
