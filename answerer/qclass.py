import argparse
import itertools
import os
import json
import pickle
import numpy as np
import pickle
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter, mul
from sklearn import svm, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from scipy.stats import describe
from scipy.sparse import lil_matrix, vstack
try:
    from .rvc import RVC, linear_kernel
except:
    from rvc import RVC, linear_kernel
from annoy import AnnoyIndex


LOGGING_LEVEL = logging.DEBUG
UIUC_DATA = '../../data/uiuc'
EAT_CLASSES = ['ABBR:abb', 'ABBR:exp', 'DESC:def', 'DESC:desc', 'DESC:manner',
               'DESC:reason', 'ENTY:animal', 'ENTY:body', 'ENTY:color',
               'ENTY:cremat', 'ENTY:currency', 'ENTY:dismed', 'ENTY:event',
               'ENTY:food', 'ENTY:instru', 'ENTY:lang', 'ENTY:letter',
               'ENTY:other', 'ENTY:plant', 'ENTY:product', 'ENTY:religion',
               'ENTY:sport', 'ENTY:substance', 'ENTY:symbol', 'ENTY:techmeth',
               'ENTY:termeq', 'ENTY:veh', 'ENTY:word', 'HUM:desc', 'HUM:gr',
               'HUM:ind', 'HUM:title', 'LOC:city', 'LOC:country', 'LOC:mount',
               'LOC:other', 'LOC:state', 'NUM:code', 'NUM:count', 'NUM:date',
               'NUM:dist', 'NUM:money', 'NUM:ord', 'NUM:other', 'NUM:perc',
               'NUM:period', 'NUM:speed', 'NUM:temp', 'NUM:volsize', 'NUM:weight']

logging.basicConfig(level=LOGGING_LEVEL, format="%(asctime)s - %(process)d -  %(levelname)s - %(message)s")


##
## @brief      { function_description }
##
## @param      q            The question
## @param      vecs_index   AnnoyIndex instance with all wordvecs in it
## @param      weights      The weights
##
## @return     { description_of_the_return_value }
##
def q2vec(q, vecs_index, conn, weights=None, default_weight=1):
    with conn.cursor() as cursor:
        res = cursor.execute("""
        SELECT annoy_id, word FROM words2annoy_100 WHERE word IN ({})
        """.format(",".join(["%s"]*len(q))), [x.lower() for x in q])
        vecs = dict([(x[1].decode("utf-8"), vecs_index.get_item_vector(x[0])) for x in cursor.fetchall()])

    if weights:
        return np.sum([weights.get(x.lower(), default_weight) * np.array(vecs[x.lower()]) 
                       for x in q if x.lower() in vecs], axis=0)
    else:
        return np.average([vecs[x.lower()] for x in q if x.lower() in vecs], axis=0)


def q2bow(q, conn):
    with conn.cursor() as cursor:
        res = cursor.execute("SELECT COUNT(*) FROM words;")
        N = cursor.fetchone()[0]
        v = np.zeros((N, 1))
        res = cursor.execute("""
        SELECT id FROM words WHERE word IN ({});
        """.format(",".join(["%s"]*len(q))), q)
        ids = [x[0] for x in cursor.fetchall()]
        v[ids] = 1
        return v


def q2bowmatrix(questions, conn):
    with conn.cursor() as cursor:
        res = cursor.execute("SELECT COUNT(*) FROM words;")
        V = cursor.fetchone()[0]
        N = len(questions)
        logging.debug("Creating a BoW matrix {}x{}".format(N, V))
        bow_matrix = lil_matrix((N, V), dtype=np.bool)
        questions = [[w.lower() for w in q] for q in questions]
        
        for i, q in enumerate(questions):
            res = cursor.execute("""
            SELECT id FROM words WHERE word IN ({});
            """.format(",".join(["%s"]*len(q))), q)
            ids = [x[0] for x in cursor.fetchall()]
            bow_matrix[i, ids] = 1
        return bow_matrix


def get_coarse_class(labels):
    return np.array(list(map(itemgetter(slice(0, 3)), labels)))


##
## @brief      { function_description }
##
## @param      vecs       The vecs
## @param      atypes     The atypes
## @param      desc       The description
##
## @return     { description_of_the_return_value }
##
def print_stats(pred_atypes, atypes, desc):
    fine_correct, total = np.sum(pred_atypes == atypes), len(atypes)
    coarse_correct = np.sum(get_coarse_class(pred_atypes) == get_coarse_class(atypes))
    fine_percentage = round(fine_correct / total * 100, 2)
    coarse_percentage = round(coarse_correct / total * 100, 2)
    logging.info("{}: correctly classified (coarse-grained) -- {}/{} -- {}%".format(
                 desc, coarse_correct, total, coarse_percentage))
    logging.info("{}: correctly classified (fine-grained) -- {}/{} -- {}%".format(
                 desc, fine_correct, total, fine_percentage))


##
## @brief      { function_description }
##
## @param      q     The question as string
## @param      clf   The clf
##
## @return     { description_of_the_return_value }
##
def predict(q, clf, conn, vecs_index):
    print(q, end=' -- ')
    if args.word_vectors:
        pred = clf.predict(q2vec(q.split(), vecs_index, conn).reshape(1, -1))
    elif args.bag_of_words:
        pred = clf.predict(q2bow(q.split(), conn).reshape(1, -1))
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


def coarse_fine_split(vecs, labels):
    fine_grained = defaultdict(list)
    labels_coarse = []
    for v, l in zip(vecs, labels):
        c_label = l.split(':')[0]
        fine_grained[c_label].append((v, l))
        labels_coarse.append(c_label)
    return vecs, labels_coarse, fine_grained


def only_confused_matrix(cmtr):
    R, C = cmtr.shape
    rows2keep, cols2keep = [], []
    for i in range(R):
        row_confused, col_confused = 0, 0
        for j in range(C):
            if i != j:
                row_confused += cmtr[i][j]
                col_confused += cmtr[j][i]
        if row_confused > 0:
            rows2keep.append(i)
        if col_confused > 0:
            cols2keep.append(i)
    dim2keep = rows2keep if len(rows2keep) > len(cols2keep) else cols2keep
    return cmtr[dim2keep, :][:, dim2keep], dim2keep



if __name__ == '__main__':
    import pymysql
    import configparser

    config = configparser.ConfigParser()
    config.read('../config.ini')

    conn = pymysql.connect(host='127.0.0.1', user=config['db']['user'], charset='utf8',
                           db=config['db']['name'], password=config['db']['password'])

    parser = argparse.ArgumentParser()
    parser.add_argument('--hsvm', action='store_true')
    parser.add_argument('--svm', action='store_true')
    parser.add_argument('--rvm', action='store_true')
    parser.add_argument('-wv', '--word-vectors', action='store_true')
    parser.add_argument('-d', '--dimension', default=300, type=int)
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

    if args.word_vectors:
        logging.info("Using word vectors. Loading...")
        vecs_index = AnnoyIndex(args.dimension)
        vecs_index.load('../data/glove{}.ann'.format(args.dimension))
        q_vecs = np.array([q2vec(q, vecs_index, conn) for q in questions])
        logging.info("Finished loading word vectors.")
    elif args.bag_of_words:
        logging.info("Using bag-of-words. Loading...")
        q_vecs = q2bowmatrix(questions, conn)
        logging.info("Finished loading bag-of-words.")
    else:
        logging.error("Please specify the text representation to be used")
        exit(1)

    if args.validation:
        logging.info("Preparing data...")
        train_q_vecs, val_q_vecs, train_atypes, val_atypes = train_test_split(
            q_vecs, answer_types, test_size=0.1, random_state=29)

        train_enc_atypes, val_enc_atypes = le.transform(train_atypes), le.transform(val_atypes)
        if args.histogram:
            plt.figure()
            display_le_histogram(train_enc_atypes, le, "Distribution of answer types in training data")
            plt.figure()
            display_le_histogram(val_enc_atypes, le, "Distribution of answer types in validation data")
        logging.info("Finished preparing data.")
    else:
        train_q_vecs, train_enc_atypes, train_atypes = q_vecs, enc_atypes, answer_types

    if args.histogram:
        display_le_histogram(enc_atypes, le, "Distribution of answer types in training data")
    
    if args.hsvm:
        logging.info("Training hierarchical SVM classifier -- 2 stages")
        logging.info("Preparing data...")
        train_coarse, atypes_coarse, fine_grained = coarse_fine_split(q_vecs, answer_types)

        logging.info("1) Training coarse-grained SVM classifier...")
        coarse_le = preprocessing.LabelEncoder()
        coarse_le.fit(atypes_coarse)
        enc_atypes_coarse = coarse_le.transform(atypes_coarse)
        clf_coarse = svm.SVC(kernel='linear')
        clf_coarse.fit(train_coarse, enc_atypes_coarse)

        logging.info("2) Training fine-grained SVM classifier...")
        clf_fine_grained = {}
        for coarse_eat in fine_grained:
            f_clf = svm.SVC(kernel='linear')
            try:
                f_vecs = vstack(map(itemgetter(0), fine_grained[coarse_eat]))
            except:
                f_vecs = list(map(itemgetter(0), fine_grained[coarse_eat]))
            f_atypes = list(map(itemgetter(1), fine_grained[coarse_eat]))
            f_enc_atypes = le.transform(f_atypes)
            f_clf.fit(f_vecs, f_enc_atypes)
            clf_fine_grained[coarse_eat] = f_clf

        pred_enc_coarse = clf_coarse.predict(train_q_vecs)
        pred_coarse = coarse_le.inverse_transform(pred_enc_coarse)
        final_pred_enc_atypes = []
        for v, cl in zip(train_q_vecs, pred_coarse):
            total_shape = mul(*v.shape) if len(v.shape) == 2 else v.shape[0]
            final_pred_enc_atypes.extend(clf_fine_grained[cl].predict(v.reshape((1, total_shape))))
        final_pred_atypes = le.inverse_transform(final_pred_enc_atypes)
        print_stats(final_pred_atypes, train_atypes, "TRAINING DATA")
    else:
        if args.svm:
            # After CV on RBF kernel the value of gamma is 0.25
            logging.info("Training SVM classifier...")
            clf = svm.SVC(kernel='linear')
        elif args.rvm:
            logging.info("Training RVM classifier...")
            clf = OneVsRestClassifier(RVC(kernel=linear_kernel), n_jobs=-1)

        logging.info(clf)
        # clf = GridSearchCV(svc, {
        #     'C': np.arange(0.4, 0.81, 0.1),
        #     'gamma': np.arange(0.1, 0.21, 0.01)}, cv=5)
        clf.fit(train_q_vecs, train_enc_atypes)

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

        method_name = 'SVM' if args.svm else 'RVM'
        features = 'bow' if args.bag_of_words else 'wv {}d'.format(args.dimension) 
        cplot_title = 'Confusion matrix ({}, {})'.format(method_name, features)
        pred_train_enc_atypes = clf.predict(train_q_vecs)
        pred_train_atypes = le.inverse_transform(pred_train_enc_atypes)
        # confusion_mtr = confusion_matrix(train_enc_atypes, pred_train_enc_atypes)
        # plt.figure()
        # plot_confusion_matrix(confusion_mtr, le.classes_, title=cplot_title)
        # only_confused_mtr, dim2keep = only_confused_matrix(confusion_mtr)
        # plt.figure()
        # plot_confusion_matrix(only_confused_mtr, le.classes_[dim2keep], title=cplot_title)
        print_stats(pred_train_atypes, train_atypes, "TRAINING DATA")
        if args.validation:
            pred_val_atypes = le.inverse_transform(clf.predict(val_q_vecs))
            print_stats(pred_val_atypes, val_atypes, "VALIDATION DATA")
    
    if args.save:
        if args.hsvm: method_name = 'hsvm'
        elif args.svm: method_name = 'svm'
        elif args.rvm: method_name = 'rvm'

        # need to dump 7 classifiers for hsvm
        if args.word_vectors:
            pickle.dump(clf, open("{}_wv_{}.clf".format(method_name, args.dimension), "wb"))
        elif args.bag_of_words:
            pickle.dump(clf, open("{}_bow.clf".format(method_name), "wb"))
        # need to dump 2 label encoders for hsvm
        pickle.dump(le, open("{}.le".format(method_name), "wb"))

    if args.test:
        test_q, test_atypes = load_data(test_file)
        # plt.figure()
        # display_le_histogram(le.transform(test_atypes), le, "Distribution of answer types in test data")
        if args.word_vectors:
            test_q_vecs = np.array([q2vec(q, vecs_index, conn) for q in test_q])
        elif args.bag_of_words:
            test_q_vecs = q2bowmatrix(test_q, conn)

        if args.hsvm:            
            pred_enc_coarse = clf_coarse.predict(test_q_vecs)
            pred_coarse = coarse_le.inverse_transform(pred_enc_coarse)
            final_pred_enc_atypes = []
            for v, cl in zip(test_q_vecs, pred_coarse):
                total_shape = mul(*v.shape) if len(v.shape) == 2 else v.shape[0]
                final_pred_enc_atypes.extend(clf_fine_grained[cl].predict(v.reshape((1, total_shape))))
            final_pred_atypes = le.inverse_transform(final_pred_enc_atypes)
            print_stats(final_pred_atypes, test_atypes, "TEST DATA")
        else:
            pred_atypes = le.inverse_transform(clf.predict(test_q_vecs))
            print_stats(pred_atypes, test_atypes, "TEST DATA")

    vecs_index = locals().get('vecs_index', None)
    # predict("What 's the capital of Sweden ?", clf, conn, vecs_index)
    # predict("What city is the capital of Great Britain ?", clf, conn, vecs_index)
    # predict("What is the capital of Ukraine ?", clf, conn, vecs_index)
    # predict("Who is the president of Ukraine ?", clf, conn, vecs_index)
    # predict("When was the second world war ?", clf, conn, vecs_index)
    # predict("What is chemical formula ?", clf, conn, vecs_index)
    plt.show()
