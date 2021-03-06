import re
import time
import locale
import calendar
import math
import operator
from itertools import tee, islice
from collections import defaultdict
from polyglot.text import Text
import dateparser
import logging
import string
import datetime as dt
from measurement.base import MeasureBase
from measurement.measures import Weight, Distance, Speed, Temperature
from word2number import w2n
from matplotlib import colors as mcolors
from nltk.tokenize import sent_tokenize


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


class EntityRecognizer(object):
    def __init__(self):
        self.colors = self.__build_colors()

    def __build_colors(self):
        shades = [None, "dark", "light"]
        colors = list(mcolors.CSS4_COLORS.keys())
        return ["{} {}".format(s, c) if s else c for c in colors for s in shades]

    def check_color(self, word):
        return " ".join(word) in self.colors



class NumericRecognizer(object):
    def __init__(self, used_locale):
        self.si_prefixes = MeasureBase.SI_PREFIXES.values()
        self.dist_units = self.__build_distance_units()
        self.temp_units = self.__build_temperature_units()
        self.weight_units = self.__build_weight_units()
        self.speed_units = self.__build_speed_units()
        self.__locale = used_locale
        locale.setlocale(locale.LC_ALL, self.__locale)
        self.month_names = [calendar.month_name[i] for i in range(1, 13)]

    def __build_temperature_units(self):
        units = list(Temperature.UNITS.keys())
        aliases = Temperature.ALIAS.keys()
        units.extend(aliases)
        return units

    def __build_distance_units(self):
        units = list(Distance.UNITS.keys())
        si_units = Distance.SI_UNITS
        prefixed_si_units = ['{}{}'.format(p, u) for u in si_units for p in self.si_prefixes]
        singular_aliases = Distance.ALIAS.keys()
        plural_aliases = ["{}s".format(a) for a in singular_aliases if a[-1] != 's']
        units.extend(prefixed_si_units)
        units.extend(singular_aliases)
        units.extend(plural_aliases)
        # works only for english
        units.append('feet')
        return units

    def __build_weight_units(self):
        units = list(Weight.UNITS.keys())
        si_units = Weight.SI_UNITS
        prefixed_si_units = ['{}{}'.format(p, u) for u in si_units for p in self.si_prefixes]
        singular_aliases = Weight.ALIAS.keys()
        plural_aliases = ["{}s".format(a) for a in singular_aliases if a[-1] != 's']
        units.extend(prefixed_si_units)
        units.extend(singular_aliases)
        units.extend(plural_aliases)
        return units

    def __build_speed_units(self):
        units = list(Speed.ALIAS.keys())
        si_units = Distance.SI_UNITS
        si_speed_units = [['{}{}/h'.format(p, u), '{}{} per h'.format(p, u)]
                          for u in si_units for p in self.si_prefixes]
        for si_unit in si_speed_units:
            units.extend(si_unit)
        return units

    def check_float(self, word):
        try:
            float(word)
            return True
        except ValueError:
            return False

    def check_verbose(self, word):
        try:
            w2n.word_to_num(word)
            return True
        except ValueError:
            return False

    def check_date(self, word):
        fmts = ('%Y','%b %d, %Y','%b %d, %Y','%B %d, %Y',
                '%B %d %Y','%m/%d/%Y','%m/%d/%y','%b %Y',
                '%B%Y','%b %d,%Y')
        cand = " ".join(word)
        
        for fmt in fmts:
            try:
                t = dt.datetime.strptime(cand, fmt)
                return True
            except ValueError as err:
                pass
        return False

    def check_speed(self, word):
        if len(word) != 2: return False
        return self.check_numeric(word[0]) and word[1] in self.speed_units

    def check_numeric(self, word):
        return word.isnumeric() or self.check_float(word) or self.check_verbose(word)

    def check_distance(self, word):
        # word is a tuple containing all unigram components of ngram
        if len(word) == 1: return False
        word = [x.lower() for x in word]
        return all([self.check_numeric(w) for w in word[:-1]]) and word[-1] in self.dist_units

    def check_temperature(self, word):
        # degree sign?
        if len(word) == 1: return False
        word = [x.lower() for x in word]
        return all([self.check_numeric(w) for w in word[:-1]]) and word[-1] in self.temp_units

    def check_weight(self, word):
        if len(word) == 1: return False
        word = [x.lower() for x in word]
        return all([self.check_numeric(w) for w in word[:-1]]) and word[-1] in self.weight_units
        

class NGramTiler(object):
    def __init__(self, max_n=3, connection=None, mine_with_tf=False, question='', 
                 exp_answer_type='', stopwords=[], lang='en', used_locale='en_US.UTF-8'):
        self.max_n = max_n
        self.__question = question.split()
        self.__eat = exp_answer_type
        self.__lang = lang
        self.__ngram_names = ['uni', 'bi', 'tri', 'tetra', 'peta']
        self.__entities = {}
        self.__stopwords = stopwords
        self.__er = EntityRecognizer()
        self.__nr = NumericRecognizer(used_locale)
        self.__conn = connection
        self.__mine_with_tf = mine_with_tf
        self.__abbr = None
        if self.__eat == 'ABBR:exp':
            for w in self.__question:
                if w.isupper():
                    self.__abbr = list(w)
                    break

    def __test_nec(self, ner_tag):
        if 'LOC' in ner_tag:
            return 'LOC' in self.__eat
        elif 'ORG' in ner_tag:
            return self.__eat == 'HUM:gr'
        elif 'PER' in ner_tag:
            return self.__eat in ['HUM:ind', 'HUM:title']
        else:
            return False

    def __test_eat(self, word):
        # word here is a tuple containing all unigram components of ngram
        if 'NUM' in self.__eat:
            if self.__eat == 'NUM:dist':
                return self.__nr.check_distance(word)
            elif self.__eat == 'NUM:temp':
                return self.__nr.check_temperature(word)
            elif self.__eat == 'NUM:weight':
                return self.__nr.check_weight(word)
            elif self.__eat == 'NUM:date':
                return self.__nr.check_date(word)
            elif self.__eat == 'NUM:speed':
                return self.__nr.check_speed(word)
            else:
                return all([self.__nr.check_numeric(w) for w in word])
        elif 'ABBR' in self.__eat:
            if self.__eat == 'ABBR:abb':
                return all([re.match(r'([A-Z]{1,2}\.?)+(?=[A-Z])', w) for w in word])
            else:
                if not self.__abbr: return True
                # some expressions for abbreviations might contain additional words
                # like "of" "and" etc
                w_len, abbr_len = len(word), len(self.__abbr)
                if w_len < abbr_len - 2 and w_len > abbr_len + 2:
                    return False
                # some expressions contain '&', like R&B
                for i, w in enumerate(self.__abbr):
                    if w == '&':
                        self.__abbr[i] = 'and'
                match, abbr_match = 0, -1
                for w in word:
                    if abbr_match < 0 and w[0] in self.__abbr:
                        abbr_match = self.__abbr.index(w[0]) + 1
                        match += 1
                    elif abbr_match >= 0:
                        if abbr_match >= abbr_len: break
                        if w[0] == self.__abbr[abbr_match]:
                            match += 1
                            abbr_match += 1
                        else:
                            continue
                return match == abbr_len
        elif 'ENTY' in self.__eat:
            if self.__eat == 'ENTY:color':
                return self.__er.check_color(word)
            else:
                is_not_ne = " ".join(word) not in self.__entities
                is_not_numeric = not all([self.__nr.check_numeric(w) for w in word])
                return is_not_ne and is_not_numeric
        elif 'DESC' in self.__eat:
            is_not_ne = " ".join(word) not in self.__entities
            is_not_numeric = not all([self.__nr.check_numeric(w) for w in word])
            is_not_all_capitalized = not all([w.istitle() for w in word])
            return is_not_ne and is_not_numeric and is_not_all_capitalized
        else:
            return self.__test_nec(self.__entities.get(word, ''))

    def __not_question_words(self, words):
        for word in words:
            for qw in self.__question:
                if word == qw or qw.startswith(word):
                    return False
        return True

    def mine(self, snippets):
        sentences = []
        for snippet in snippets:
            for sent in sent_tokenize(snippet):
                sentences.append(sent)
                text = Text(re.sub('\n', '<br>', sent), hint_language_code=self.__lang)
                entities = text.entities
                for ent in entities:
                    e = tuple([x for x in ent])
                    if " ".join(e) not in self.__stopwords:
                        if e not in self.__entities:
                            self.__entities[e] = defaultdict(int)
                        self.__entities[e][ent.tag] += 1
        # majority vote on NER tags
        for ent in self.__entities:
            ngram_length = len(ent)
            self.__entities[ent] = max(self.__entities[ent].items(), 
                                       key=operator.itemgetter(1))[0]

        sentences = [re.sub(r'[.,!?;()]', '', s) for s in sentences]
        punctuation = re.sub('[-+/&]', '', string.punctuation)
        token_re = r"[\w/]+|'[\w]+|[{}]".format(punctuation)
        sentences = [re.findall(token_re, s) for s in sentences]

        ng_snippets = {}
        for n in range(self.max_n):
            ng_type = "{}grams".format(self.__ngram_names[n])
            ng_snippets[ng_type] = [nwise(snippet, n + 1) for snippet in sentences]

        return self.__n_gram_stats(ng_snippets)

    def __n_gram_stats(self, ng_snippets):
        stats = {}
        for ng_type in ng_snippets:
            stats[ng_type] = defaultdict(list) if self.__mine_with_tf else defaultdict(set)
            for i, ng_snippet in enumerate(ng_snippets[ng_type]):
                if self.__mine_with_tf:
                    for ng in ng_snippet:
                        stats[ng_type][ng].append(i)
                else:
                    for ng in ng_snippet:
                        stats[ng_type][ng].add(i)
            stats[ng_type] = {k:len(v) for k, v in stats[ng_type].items()}
        return stats

    def __check_stopwords(self, word):
        return word[0] not in self.__stopwords and word[-1] not in self.__stopwords

    def __check_punctuation(self, word):
        punctuation = '{}—\''.format(string.punctuation)
        bounds_not_punctuation = word[0] not in punctuation and word[-1] not in punctuation
        first_word_not_from_punctuation = word[0].strip()[0] not in punctuation
        return bounds_not_punctuation and first_word_not_from_punctuation

    def filter(self, votes):
        filtered_votes = {}
        for ng_type in votes:
            filtered_votes[ng_type] = {}
            for k, v in votes[ng_type].items():
                if self.__check_stopwords(k) and self.__check_punctuation(k):
                    if self.__test_eat(k) and self.__not_question_words(k):
                        filtered_votes[ng_type][k] = v
        return filtered_votes

    def tile(self, votes):
        # Boosting n-grams with their unigram components
        # re-scoring with normalized idf sum
        rescored_votes = []
        with self.__conn.cursor() as cursor:
            res = cursor.execute("SELECT COUNT(*) FROM {}_words;".format(self.__lang))
            N = cursor.fetchone()[0]
            
            unigrams = [v[0].lower() for v in votes['unigrams'].keys()]
            
            if unigrams:
                placeholders = ("%s," * len(unigrams))[:-1]
                res = cursor.execute("""
                    SELECT word, df FROM {}_words WHERE word IN ({})
                    """.format(self.__lang, placeholders), unigrams)
                dfs = dict([(w.decode("utf-8"), df) for w, df in cursor.fetchall()])
            else:
                dfs = {}

            for ng_type in votes:
                for ng, score in votes[ng_type].items():
                    ng_idf = 0
                    for unigram in ng:
                        if ng_type != 'unigrams':
                            score += votes['unigrams'].get((unigram,), 0)

                        if unigram in dfs:
                            ng_idf += math.log(N / dfs[unigram])
                        else:
                            res = cursor.execute("""
                                SELECT df FROM {}_words WHERE word='{}'
                                """.format(self.__lang, self.__conn.escape_string(unigram.lower())))
                            res = cursor.fetchone()
                            if res:
                                df = res[0]
                                ng_idf += math.log(N / df)
                    score *= ng_idf / len(ng)
                    rescored_votes.append((ng, score))
        return sorted(rescored_votes, key=operator.itemgetter(1), reverse=True)

    def extract(self, snippets, top5=False):
        mined = self.mine(snippets)
        filtered = self.filter(mined)
        votes = self.tile(filtered)
        print(votes)
        if votes:
            return [v[0] for v in votes[:5]] if top5 else votes[0][0]


class MajorityVoter(object):
    def __init__(self, question='', exp_answer_type='', lang='en', used_locale='en_US.UTF-8'):
        self.__eat = exp_answer_type
        self.__question = question.split()
        self.__lang = lang
        self.__er = EntityRecognizer()
        self.__nr = NumericRecognizer(used_locale)
        self.__extractor = self.__auto_define_extractor()
        self.__test_nec = {
            'LOC': 'LOC' in self.__eat,
            'ORG': self.__eat == 'HUM:gr',
            'PER': self.__eat in ['HUM:ind', 'HUM:title'],
            None: False
        }

    def __auto_define_extractor(self):
        if 'NUM' in self.__eat:
            if self.__eat == 'NUM:dist':
                return lambda x: self.__numeric_pairs_extractor(x, method=self.__nr.check_distance)
            elif self.__eat == 'NUM:temp':
                return lambda x: self.__numeric_pairs_extractor(x, method=self.__nr.check_temperature)
            elif self.__eat == 'NUM:weight':
                return lambda x: self.__numeric_pairs_extractor(x, method=self.__nr.check_weight)
            elif self.__eat == 'NUM:date':
                return self.__date_extractor
            elif self.__eat == 'NUM:speed':
                return lambda x: self.__numeric_pairs_extractor(x, method=self.__nr.check_speed)
            else:
                return self.__numeric_extractor
        elif self.__eat == 'ENTY:color':
            return self.__color_extractor
        else:
            return self.__entity_extractor

    def __numeric_pairs_extractor(self, snippet, method=None):
        punct_replace_re = r'[.,!?;()]'
        punctuation = re.sub('[-+/&]', '', string.punctuation)
        token_re = r"[\w/]+|'[\w]+|[{}]".format(punctuation)
        for sent in sent_tokenize(snippet):
            sent = re.sub(punct_replace_re, '', sent)
            sent = re.findall(token_re, sent)
            N = len(sent)
            for i in range(N):
                if i + 1 < N and method and method((sent[i], sent[i + 1])):
                    return "{} {}".format(sent[i], sent[i + 1])

    def __date_extractor(self, snippet):
        punct_replace_re = r'[.,!?;()]'
        punctuation = re.sub('[-+/&]', '', string.punctuation)
        token_re = r"[\w/]+|'[\w]+|[{}]".format(punctuation)
        cand = None
        for sent in sent_tokenize(snippet):
            sent = re.sub(punct_replace_re, '', sent)
            sent = re.findall(token_re, sent)
            N = len(sent)
            for i in range(N):
                if cand is not None: return cand
                if sent[i] in self.__question: continue
                if self.__nr.check_date((sent[i], )):
                    cand = sent[i]

                if i + 1 >= N or sent[i + 1] in self.__question: continue
                if i + 1 < N and self.__nr.check_date((sent[i], sent[i+1])):
                    cand = "{} {}".format(sent[i], sent[i + 1])
                
                if i + 2 >= N or sent[i + 2] in self.__question: continue
                if i + 2 < N and self.__nr.check_date((sent[i], sent[i+1], sent[i+2])):
                    cand = "{} {} {}".format(sent[i], sent[i + 1], sent[i + 2])

    def __color_extractor(self, snippet):
        punct_replace_re = r'[.,!?;()]'
        punctuation = re.sub('[-+/&]', '', string.punctuation)
        token_re = r"[\w/]+|'[\w]+|[{}]".format(punctuation)
        for sent in sent_tokenize(snippet):
            sent = re.sub(punct_replace_re, '', sent)
            sent = re.findall(token_re, sent)
            N = len(sent)
            for i in range(N):
                if sent[i] not in self.__question and self.__er.check_color((sent[i],)):
                    return sent[i]

    def __numeric_extractor(self, snippet):
        groups = re.search(r'\d+', snippet)
        return groups.group(0) if groups else None

    def __entity_extractor(self, snippet):
        text = Text(re.sub('\n', '<br>', snippet), hint_language_code=self.__lang)
        entities = text.entities
        for entity in entities:
            if " ".join(entity) not in self.__question and self.__test_nec[entity.tag[2:]]:
                return " ".join(entity)

    def extract(self, snippets, top5=False):
        cand_answers = defaultdict(int)
        for snippet in snippets:
            answer = self.__extractor(snippet)
            cand_answers[answer] += 1
        # print(cand_answers)
        if top5:
            cand = sorted(cand_answers.items(), key=operator.itemgetter(1), reverse=True)
            return [(str(c[0]),) for c in cand[:5]]
        else:
            return (max(cand_answers.items(), key=operator.itemgetter(1))[0],)



if __name__ == '__main__':
    from nltk.corpus import stopwords
    en_stopwords = stopwords.words('english')
    ngram_tiler = NGramTiler(question='What is the capital of Norway', exp_answer_type='LOC:city', stopwords=en_stopwords)
    print(ngram_tiler.extract(SNIPPETS))
