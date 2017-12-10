import re
import numpy as np
import string
from operator import itemgetter
from collections import defaultdict
from scipy.spatial.distance import cosine


class PassageRanker(object):
    def __init__(self, vec_index, db_conn, lang='en'):
        self.__index = vec_index
        self.__conn = db_conn
        self.__lang = lang

    def phrase2vec(self, phrase):
        with self.__conn.cursor() as cursor:
            words = [x if self.__lang == 'sv' else x.lower() for x in phrase]
            res = cursor.execute("""
            SELECT annoy_id, word FROM {}_words2annoy_200 WHERE word IN ({})
            """.format(self.__lang, ",".join(["%s"]*len(phrase))), words)
            vecs = dict([(x[1].decode("utf-8"), self.__index.get_item_vector(x[0])) 
                        for x in cursor.fetchall()])

        if self.__lang == 'sv':
            # didn't lowercase for Swedish when training GloVe vectors
            p_vec = [vecs[x] for x in phrase if x in vecs]
        else:
            p_vec = [vecs[x.lower()] for x in phrase if x.lower() in vecs]
        return np.average(p_vec, axis=0) if len(p_vec) > 0 else None

    def rank_articles(self, article_names, query):
        similarities = []
        token_re = r"\w+|'[\w]+|[{}]".format(string.punctuation)
        query_tokens = re.findall(token_re, query)
        q_vec = self.phrase2vec(query_tokens)
        for title in article_names:
            tokens = re.findall(token_re, title)
            p_vec = self.phrase2vec(tokens)
            similarity = 1 if p_vec is None else 2 - cosine(q_vec, p_vec)
            similarities.append(similarity)
        return similarities

    def rank_snippets_glove(self, snippets, query):
        distances = []
        token_re = r"\w+|'[\w]+|[{}]".format(string.punctuation)
        if self.__lang == 'sv':
            query_tokens = re.findall(token_re, query)
        else:
            query_tokens = re.findall(token_re, query.lower())
        q_vec = self.phrase2vec(query_tokens)
        for snippet in snippets:
            for paragraph in snippet.split('\n'):
                if self.__lang == 'sv':
                    paragraph_tokens = re.findall(token_re, paragraph)
                else:
                    paragraph_tokens = re.findall(token_re, paragraph.lower())
                if not paragraph_tokens: continue
                p_vec = self.phrase2vec(paragraph_tokens)
                if p_vec is None: continue
                dist = cosine(q_vec, p_vec)
                if dist:
                    distances.append((paragraph, dist))
        distances = sorted(distances, key=itemgetter(1))
        return distances

    def rank_snippets_tf_idf(self, snippets, query):
        token_re = r"\w+|'[\w]+|[{}]".format(string.punctuation)
        if self.__lang == 'sv':
            query_tokens = re.findall(token_re, query)
        else:
            query_tokens = re.findall(token_re, query.lower())
        with self.__conn.cursor() as cursor:
            res = cursor.execute("SELECT COUNT(*) FROM {}_words;".format(self.__lang))
            N = cursor.fetchone()[0]
            placeholders = ("%s," * len(query_tokens))[:-1]
            res = cursor.execute("SELECT word, df FROM {}_words WHERE word IN ({})".format(self.__lang, placeholders), query_tokens)
            dfs = dict([(w.decode("utf-8"), df) for w, df in cursor.fetchall()])
        
        qlen, qtf, idfs = 0, defaultdict(int), {}
        for token in query_tokens:
            qtf[token] += 1
            qlen += 1
            if token not in idfs and token in dfs:
                idfs[token] = np.log(N / dfs[token])

        scores = []
        for snippet in snippets:
            for paragraph in snippet.split('\n'):
                if self.__lang == 'sv':
                    paragraph_tokens = re.findall(token_re, paragraph)
                else:
                    paragraph_tokens = re.findall(token_re, paragraph.lower())
                if not paragraph_tokens: continue
                ptf = defaultdict(int)
                par_len = 0
                for ptoken in paragraph_tokens:
                    if ptoken in qtf:
                        ptf[ptoken] += 1
                    par_len += 1
                score = 0
                for qtoken in query_tokens:
                    if qtoken in idfs:
                        score += qtf[qtoken] * idfs[qtoken] * ptf[qtoken] * idfs[qtoken]
                score /= np.sqrt(par_len) * np.sqrt(qlen)

                scores.append((paragraph, score))
        scores = sorted(scores, key=itemgetter(1), reverse=True)
        return scores
