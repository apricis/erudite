import re
import string


class Reformulator(object):
    def __init__(self, question, qclass, lang='en', stopwords=None):
        self.__original_question = question
        punctuation = re.sub(r"[-+/&']", '', string.punctuation)
        self.__punctuation_re = r'[{}]'.format(punctuation)
        question = question[0].lower() + question[1:]
        question = re.sub(r'(?<=[A-Z])\.', 'QQQ', question)
        question = re.sub(self.__punctuation_re, '', question)
        self.__question = re.sub(r'QQQ', '.', question)
        self.__stopwords = stopwords
        self.__qclass = qclass.split(':')[1]
        if lang == 'en':
            question_words = ['what', 'which', 'who', 'whom', 'when', 'where', 'why', 'how']
            conj_prep_words = ['of', 'not']
        elif lang == 'sv':
            question_words = ['vilket', 'vilken', 'vem', 'whom', 'när', 'var', 'varför', 'hur']
            conj_prep_words = ['av', 'inte', 'ej']
        else:
            raise NotImplemented('This language is not available')

        self.__exact_stop_words = set(stopwords) - set(conj_prep_words)
        self.__expansion_rules = {
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
            'speed': 'speed',
            'temp': 'temperature',
            'volsize': 'size'
        }
        if qclass == 'ABBR:abb':
            try:
                self.__stopwords.append('abbreviation')
            except:
                self.__stopwords.add('abbreviation')
            self.__exact_stop_words.append('abbreviation')  

    def question(self):
        return self.__question

    def reformulate(self):
        without_stopwords = [w for w in self.__question.split() 
                             if w not in self.__stopwords]
        query = without_stopwords
        query.append(self.__expansion_rules.get(self.__qclass, ''))
        return " ".join(query)

    def reformulate_exact(self):
        without_exact_stopwords = [w for w in self.__question.split() 
                                   if w not in self.__exact_stop_words]
        query = without_exact_stopwords
        query.append(self.__expansion_rules.get(self.__qclass, ''))
        return " ".join(query)

