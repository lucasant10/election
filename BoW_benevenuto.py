import sys
import argparse
import configparser
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.utils import shuffle
import gensim
import sklearn
from collections import defaultdict
from text_processor import TextProcessor
from sklearn.grid_search import GridSearchCV
import pymongo

param_grid = {
    'random_forest': {
        'bootstrap': [True],
        'max_depth': [80, 90, 100],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [100, 200, 300]
    },
    'logistic': {},
    'gradient_boosting': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'svm':{'kernel': ['rbf', 'linear'], 'C': [1, 10], 'gamma': [0.001, 0.0001]}
}

# Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids


# logistic, gradient_boosting, random_forest, svm_linear, svm_rbf
W2VEC_MODEL_FILE = None
EMBEDDING_DIM = None
MODEL_TYPE = None
TOKENIZER = None
NO_OF_FOLDS = 10
SEED = 42
MAX_NB_WORDS = None


# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
tweets = {}


word2vec_model = None


def gen_data(tweets, tw_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tw_class))):
        y_map[v] = i
    print(y_map)

    X, y = [], []
    for i, tweet in enumerate(tweets):
        emb = np.zeros(EMBEDDING_DIM)
        for word in tweet:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(tweet)
        X.append(emb)
        y.append(y_map[tw_class[i]])
    return X, y


def select_tweets(tweets):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    X, Y = [], []
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        for w in tweet:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb += 1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    return tweet_return


def get_model(m_type=None):
    if not m_type:
        print("ERROR: Please specify a model type!")
        return None
    if m_type == 'logistic':
        logreg = LogisticRegression()
    elif m_type == "gradient_boosting":
        logreg = GradientBoostingClassifier()
    elif m_type == "random_forest":
        logreg = RandomForestClassifier()
    elif m_type == "svm":
        logreg = SVC()
    elif m_type == "svm_linear":
        logreg = LinearSVC()
    else:
        print("ERROR: Please specify a correct model")
        return None

    return logreg


def classification_model(X, Y, model_type=None):
    X, Y = shuffle(X, Y, random_state=SEED)
    print("Model Type:", model_type)
    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    model = GridSearchCV(estimator=get_model(model_type),
                         param_grid=param_grid[model_type], n_jobs=-1, verbose=2)

    scores1 = cross_val_score(model.fit(X, Y), X, Y,
                              cv=NO_OF_FOLDS, scoring='precision_weighted')
    print("Precision(avg): %0.3f (+/- %0.3f)" %
          (scores1.mean(), scores1.std() * 2))

    scores2 = cross_val_score(
        model, X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print("Recall(avg): %0.3f (+/- %0.3f)" %
          (scores2.mean(), scores2.std() * 2))

    scores3 = cross_val_score(
        model, X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print("F1-score(avg): %0.3f (+/- %0.3f)" %
          (scores3.mean(), scores3.std() * 2))

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='BagOfWords model for politics twitter')
    parser.add_argument('-m', '--model', choices=[
                        'logistic', 'gradient_boosting', 'random_forest', 'svm', 'svm_linear'], required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)

    args = parser.parse_args()
    MODEL_TYPE = args.model
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)

    print('Word2Vec embedding: %s' % (W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' % (EMBEDDING_DIM))

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                                     binary=False,
                                                                     unicode_errors="ignore")
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client.twitterdb

    tweets = list()
    tw_class = list()

    tmp = db.politics.find()
    for tw in tmp:
        tweets.append(tw['text_processed'].split(' '))
        tw_class.append('politics')

    tmp = db.non_politics.find()
    for tw in tmp:
        tweets.append(tw['text_processed'].split(' '))
        tw_class.append('non-politics')

    tweets = select_tweets(tweets)

    X, Y = gen_data(tweets, tw_class)

    model = classification_model(X, Y, MODEL_TYPE)
    joblib.dump(model, dir_in + MODEL_TYPE + '_ben.skl')

    # python BoW_benevenuto.py --model logistic --seed 42 -f cbow_s300.txt -d 300
    # python BoW_benevenuto.py --model gradient_boosting --seed 42 -f cbow_s300.txt -d 300
    # python BoW_benevenuto.py --model random_forest --seed 42 -f cbow_s300.txt -d 300
    # python BoW_benevenuto.py --model svm_linear --seed 42 -f cbow_s300.txt -d 300
    # python BoW_benevenuto.py --model svm --seed 42 -f cbow_s300.txt -d 300
