
import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn
import gc
import datetime
import sys
import argparse
import configparser
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support, roc_curve, auc
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.utils import shuffle
from scipy import interp
import gensim
import sklearn
from collections import defaultdict
from text_processor import TextProcessor
from sklearn.grid_search import GridSearchCV
from utils import save_report_to_csv
from time import gmtime, strftime

def log (text):
    print('{} -> {}'.format(strftime("%Y-%m-%d %H:%M:%S", gmtime()), text) ) 

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
    'svm':{'kernel': ['rbf', 'linear'], 'C': [1, 10], 'gamma': [0.001, 0.0001], 'probability':[True, True]}
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
POLITICS_FILE = 'politics.txt' 
NON_POLITICS_FILE = 'non-politics.txt' 

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)
texts = {}


word2vec_model = None


def gen_data(texts, tx_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tx_class))):
        y_map[v] = i
    print(y_map)

    X, y = [], []
    for i, text in enumerate(texts):
        emb = np.zeros(EMBEDDING_DIM)
        for word in text:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(text)
        X.append(emb)
        y.append(y_map[tx_class[i]])
    return X, y


def select_texts(texts):
    # selects the texts as in embedding method
    # Processing
    X, Y = [], []
    text_return = []
    for text in texts:
        _emb = 0
        for w in text:
            if w in word2vec_model:  # Check if embeeding there in embedding model
                _emb += 1
        if _emb:   # Not a blank text
            text_return.append(text)
    print('texts selected:', len(text_return))
    return text_return


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

def generate_roc_curve (classifier, X, y, model_name=None):
    cv = StratifiedKFold(n_splits=NO_OF_FOLDS)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    for train, test in cv.split(X, y):
        x_train = np.array([X[i] for i in train])
        y_train = np.array([y[i] for i in train])

        x_test = np.array([X[i] for i in test])
        y_test = np.array([y[i] for i in test])

        #probas_ = classifier.fit(x_train, y_train).predict_proba(x_test)
        probas_ = classifier.predict_proba(x_test)
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    if model_name is None:
        model_name = MODEL_TYPE.split('_')
    
    if model_name is None:
        model_name = MODEL_TYPE
    else:
        model_name = ' '.join (model_name)

    input_file = POLITICS_FILE.replace('tmp/', '')

    plt.title('ROC Curve for '+ input_file +' - Classifier: '+ model_name.capitalize())
    plt.legend(loc="lower right")

    #plt.show()
    plt.savefig("plots/roc_curve_" + model_name + '_' + input_file +".png")
    plt.clf()

    return mean_auc, std_auc


def classification_model(X, Y, model_type=None):
    X, Y = shuffle(X, Y, random_state=SEED)
    print("Model Type:", model_type)
    #predictions = cross_val_predict(logreg, X, Y, cv=NO_OF_FOLDS)
    model = GridSearchCV(estimator=get_model(model_type),
                         param_grid=param_grid[model_type], n_jobs=-1, verbose=0)

    scores1 = cross_val_score(model.fit(X, Y), X, Y, cv=NO_OF_FOLDS, scoring='precision_weighted')
    
    #mean_auc, std_auc = generate_roc_curve (model, X, Y)
    

    print("Precision(avg): %0.3f (+/- %0.3f)" %
          (scores1.mean(), scores1.std() * 2))
    precision_score_mean = scores1.mean()
    precision_score_mean = scores1.std() * 2

    scores2 = cross_val_score(
        model, X, Y, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print("Recall(avg): %0.3f (+/- %0.3f)" %
          (scores2.mean(), scores2.std() * 2))

    recall_score_mean = scores2.mean()
    recall_score_std = scores2.std() * 2

    scores3 = cross_val_score(
        model, X, Y, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print("F1-score(avg): %0.3f (+/- %0.3f)" %
          (scores3.mean(), scores3.std() * 2))

    f1_score_mean = scores3.mean()
    f1_score_std = scores3.std() * 2

    save_report_to_csv ('training_report.csv', [
        POLITICS_FILE,
        precision_score_mean,precision_score_mean,
        recall_score_mean,recall_score_std,
        f1_score_mean,f1_score_std,
    ])

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='BagOfWords model for politics texts')
    parser.add_argument('-m', '--model', choices=[
                        'logistic', 'gradient_boosting', 'random_forest', 'svm', 'svm_linear'], required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--folds', default=NO_OF_FOLDS)
    parser.add_argument('--politicsfile', default=POLITICS_FILE)
    parser.add_argument('--nonpoliticsfile', default=NON_POLITICS_FILE)

    args = parser.parse_args()
    MODEL_TYPE = args.model
    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    NO_OF_FOLDS = int(args.folds)
    POLITICS_FILE = args.politicsfile
    NON_POLITICS_FILE = args.nonpoliticsfile

    print ('################ {} ##############'.format(MODEL_TYPE))
    log('Running {} with Word2Vec embedding: {}'.format(MODEL_TYPE, W2VEC_MODEL_FILE))
    log('Embedding Dimension: {}'.format (EMBEDDING_DIM))
    log('Politics File: {}'.format(POLITICS_FILE))
    log('Non-politics File: {}'.format (NON_POLITICS_FILE))

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                                     binary=False,
                                                                     unicode_errors="ignore")

    tp = TextProcessor()

    texts = list()
    tx_class = list()

    tmp = list()
    with open(POLITICS_FILE) as l_file:
        for line in l_file:
            tmp.append(line)
            tx_class.append('politics')
    
    texts += tp.text_process(tmp, text_only=True)

    tmp = list()
    with open(NON_POLITICS_FILE) as l_file:
        for line in l_file:
            tmp.append(line)
            tx_class.append('non-politics')

    texts += tp.text_process(tmp, text_only=True)

    texts = select_texts(texts)

    X, Y = gen_data(texts, tx_class)

    model = classification_model(X, Y, MODEL_TYPE)

    input_file = POLITICS_FILE.replace('tmp/', '')

    joblib.dump(model, dir_in + MODEL_TYPE + '_'+ input_file+'_ben.skl')

    gc.collect()

    exit(0)

    # python bow_classifier.py --model logistic --seed 42 -f cbow_s300.txt -d 300
    # python bow_classifier.py --model gradient_boosting --seed 42 -f cbow_s300.txt -d 300
    # python bow_classifier.py --model random_forest --seed 42 -f cbow_s300.txt -d 300
    # python bow_classifier.py --model svm --seed 42 -f cbow_s300.txt -d 300
