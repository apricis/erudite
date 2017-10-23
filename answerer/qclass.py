import argparse
import itertools
import os
import json
import pickle
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.stats import describe
from scipy.sparse import lil_matrix
from elasticsearch import Elasticsearch


LOGGING_LEVEL = logging.DEBUG
UIUC_DATA = '../../data/uiuc'

logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s - %(process)d -  %(levelname)s - %(message)s")


##
## @brief      { function_description }
##
## @param      q        The question
## @param      vecs     The vecs
## @param      weights  The weights
##
## @return     { description_of_the_return_value }
##
def q2vec(q, vecs, weights=None, default_weight=1):
    if weights:
        return np.sum([weights.get(x.lower(), default_weight) * np.array(vecs[x.lower()]) 
                       for x in q if x.lower() in vecs], axis=0)
    else:
        return np.average([vecs[x.lower()] for x in q if x.lower() in vecs], axis=0)


def q2bow(q, words):
    v = np.zeros((len(words), 1))
    for w in q:
        i = words.get(w.lower(), -1)
        if i != -1:
            v[i] = 1
    return v


def q2bowmatrix(questions, words):
    V, N = len(words), len(questions)
    logging.debug("Creating a BoW matrix {}x{}".format(N, V))
    bow_matrix = lil_matrix((N, V), dtype=np.bool)
    questions = [[w.lower() for w in q] for q in questions]
    
    for i, q in enumerate(questions):
        for w in q:
            j = words.get(w, -1)
            if j != -1:
                bow_matrix[i, j] = 1
    return bow_matrix


##
## @brief      { function_description }
##
## @param      vecs       The vecs
## @param      atypes     The atypes
## @param      clf        The clf
## @param      desc       The description
## @param      questions  The questions
##
## @return     { description_of_the_return_value }
##
def print_stats(vecs, atypes, clf, desc, questions=None):
    pred_atypes = le.inverse_transform(clf.predict(vecs))
    correct, total = np.sum(pred_atypes == atypes), len(atypes)
    percentage = round(correct / total * 100, 2)
    logging.info("{}: correctly classified -- {}/{} -- {}%".format(desc, correct, total, percentage))


##
## @brief      { function_description }
##
## @param      q     The question as string
## @param      clf   The clf
##
## @return     { description_of_the_return_value }
##
def predict(q, clf):
    print(q, end=' -- ')
    if args.word_vectors:
        pred = clf.predict(q2vec(q.split(), vecs).reshape(1, -1))
    elif args.bag_of_words:
        pred = clf.predict(q2bow(q.split(), words).reshape(1, -1))
    print(le.inverse_transform(pred))


##
## @brief      { Display histogram for a data encoded with sklearn LabelEncoder
##             }
##
## @param      enc_data    { The encoded data }
## @param      le          { LabelEncoder instance }
## @param      plot_title  The plot title
##
## @return     { description_of_the_return_value }
##
def display_le_histogram(enc_data, le, plot_title):
    n_classes = len(le.classes_)
    min_class, max_class  = 0, n_classes - 1
    plt.hist(enc_data, bins=n_classes, range=(min_class, max_class+1), rwidth=0.9,
             orientation='horizontal', align='left')
    plt.title(plot_title)
    plt.yticks(range(len(le.classes_)), le.classes_)
    plt.show()


def load_data(fname):
    answer_types, questions = [], []
    with open(fname, encoding="ISO-8859-1") as f:
        for line in f:
            data = line.split()
            answer_types.append(data[0])
            questions.append(data[1:])
    return questions, answer_types


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 color="white" if cm[i, j] > thresh else "black",
                 ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-wv', '--word-vectors', action='store_true')
    parser.add_argument('-bow', '--bag-of-words', action='store_true')
    parser.add_argument('-val', '--validation', action='store_true')
    parser.add_argument('-t', '--test', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-hist', '--histogram', action='store_true')
    parser.add_argument('--min-df', default=100, type=int)
    args = parser.parse_args()

    training_file = os.path.join(UIUC_DATA, 'train_5500.label')
    test_file = os.path.join(UIUC_DATA, 'test.label')
    questions, answer_types = load_data(training_file)
    le = preprocessing.LabelEncoder()
    le.fit(answer_types)
    enc_atypes = le.transform(answer_types)

    if args.save:
        pickle.dump(le, open("svm.le", "wb"))

    if args.word_vectors:
        logging.info("Using word vectors. Loading...")
        vecs = pickle.load(open('../../data/glove.6B/glove.300d.pkl', 'rb'))
        q_vecs = np.array([q2vec(q, vecs) for q in questions])
        logging.info("Finished loading word vectors.")
    elif args.bag_of_words:
        logging.info("Using bag-of-words. Loading...")
        words = dict([(w, i) for i, w in enumerate(json.load(open('../../wiki_indexer/count.json')))])
        # if args.min_df:
        #     words = []
        #     for w in df:
        #         if df[w] >= args.min_df:
        #             words.append(w)
        # else:
        #     words = df.keys()
        # N = len(words)
        # logging.warning("There are {} words in vocabulary, which might result in MemoryError".format(N))
        q_vecs = q2bowmatrix(questions, words)
        logging.info("Finished loading bag-of-words.")
    else:
        logging.error("Please specify the text representation to be used")
        exit(1)
    
    # Getting IDF
    # N = 5438663 # counted via python script
    # logging.info("Computing IDF...")
    # weights = json.load(open('../wiki_indexer/df.json'))
    # for w in weights:
    #     weights[w] = np.log10(N / weights[w])
    # logging.info("Finished IDF computation.")

    if args.validation:
        logging.info("Preparing data...")
        train_q_vecs, val_q_vecs, train_atypes, val_atypes = train_test_split(
            q_vecs, answer_types, test_size=0.1, random_state=29)

        train_enc_atypes, val_enc_atypes = le.transform(train_atypes), le.transform(val_atypes)
        if args.histogram:
            display_le_histogram(train_enc_atypes, le, "Distribution of answer types in training data")
            display_le_histogram(val_enc_atypes, le, "Distribution of answer types in validation data")
        logging.info("Finished preparing data.")
    elif args.histogram:
        display_le_histogram(enc_atypes, le, "Distribution of answer types in training data")

    # After CV on RBF kernel the value of gamma is 0.25
    logging.info("Training SVM classifier...")
    clf = svm.SVC(kernel='linear')
    # clf = RandomForestClassifier(n_estimators=30, max_depth=11, criterion='entropy', random_state=0)
    logging.info(clf)
    # clf = GridSearchCV(svc, {
    #     'C': np.arange(0.4, 0.81, 0.1),
    #     'gamma': np.arange(0.1, 0.21, 0.01)}, cv=5)
    if args.validation:
        clf.fit(train_q_vecs, train_enc_atypes)
    else:
        clf.fit(q_vecs, enc_atypes)
    
    if args.save:
        if args.bag_of_words: save_label = "bow"
        elif args.word_vectors: save_label = "wv"

        pickle.dump(clf, open("svm_{}.clf".format(save_label), "wb"))

    # train_enc_pred = clf.predict(train_q_vecs)
    # plot_confusion_matrix(confusion_matrix(train_enc_atypes, train_enc_pred), le.classes_)

    # Cross-validation
    # print("Best parameters set found on development set:")
    # print()
    # print(clf.best_params_)
    # print()
    # print("Grid scores on development set:")
    # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # print()
    
    if args.validation:
        print_stats(train_q_vecs, train_atypes, clf, "TRAINING DATA")
        print_stats(val_q_vecs, val_atypes, clf, "VALIDATION DATA")
    else:
        print_stats(q_vecs, answer_types, clf, "TRAINING DATA")

    if args.test:
        test_q, test_atypes = load_data(test_file)
        if args.word_vectors:
            test_q_vecs = np.array([q2vec(q, vecs) for q in test_q])
        elif args.bag_of_words:
            test_q_vecs = q2bowmatrix(test_q, words)
        print_stats(test_q_vecs, test_atypes, clf, "TEST DATA")

    predict("What 's the capital of Sweden ?", clf)
    predict("What city is the capital of Great Britain ?", clf)
    predict("What is the capital of Ukraine ?", clf)
    predict("Who is the president of Ukraine ?", clf)
    predict("When was the second world war ?", clf)
    predict("What is chemical formula ?", clf)

