# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn

import sys
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Input
from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Dropout, Flatten, Merge, Convolution1D, MaxPooling1D, GlobalMaxPooling1D
import numpy as np
from sklearn.metrics import make_scorer, f1_score, accuracy_score, recall_score, precision_score, classification_report, precision_recall_fscore_support
from sklearn.ensemble  import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import KFold
from keras.utils import np_utils
import gensim, sklearn
from collections import defaultdict
from batch_gen import batch_gen
import os
import configparser
import json
import h5py
import math
import os
from utils import save_report_to_csv, get_model_name_by_file
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, H5_FOLDER, NPY_FOLDER

from bow_classifier import generate_roc_curve
from text_processor import TextProcessor
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

### Preparing the text data
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

# vocab generation
vocab, reverse_vocab = {}, {}
freq = defaultdict(int)

EMBEDDING_DIM = None
W2VEC_MODEL_FILE = None
NO_OF_CLASSES=2
MAX_SEQUENCE_LENGTH = 25
SEED = 42
NO_OF_FOLDS = 10
CLASS_WEIGHT = None
LOSS_FUN = None
INITIALIZE_WEIGHTS_WITH = None
LEARN_EMBEDDINGS = None
EPOCHS = 10
BATCH_SIZE = 30
SCALE_LOSS_FUN = None
MODEL_NAME = 'cnn_model_'
DICT_NAME = 'cnn_dict_'
POLITICS_FILE = 'politics.txt' 
NON_POLITICS_FILE = 'non-politics.txt' 
word2vec_model = None

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except:
            n += 1
            pass
    print("%d embedding missed" % n)
    print("%d embedding found" % len(embedding))
    return embedding


def select_texts(texts):
    # selects the texts as in embedding method
    # Processing
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


def gen_vocab(model_vec):
    vocab = dict([(k, v.index) for k, v in model_vec.vocab.items()])
    vocab['UNK'] = len(vocab) + 1
    print(vocab['UNK'])
    return vocab

def gen_sequence(vocab, texts, tx_class):
    y_map = dict()
    for i, v in enumerate(sorted(set(tx_class))):
        y_map[v] = i
    print(y_map)
    X, y = [], []
    for i, text in enumerate(texts):
        seq = []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tx_class[i]])
    return X, y

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def cnn_model(sequence_length, embedding_dim):
    model_variation = 'CNN-rand'  #  CNN-rand | CNN-non-static | CNN-static
    print('Model variation is %s' % model_variation)

    # Model Hyperparameters
    n_classes = NO_OF_CLASSES
    embedding_dim = EMBEDDING_DIM
    filter_sizes = (3, 4, 5)
    num_filters = 120
    dropout_prob = (0.25, 0.25)
    hidden_dims = 100

    # Training parameters
    # Word2Vec parameters, see train_word2vec
    #min_word_count = 1  # Minimum word count
    #context = 10        # Context window size

    graph_in = Input(shape=(sequence_length, embedding_dim))
    convs = []
    for fsz in filter_sizes:
        conv = Convolution1D(nb_filter=num_filters,
                             filter_length=fsz,
                             border_mode='valid',
                             activation='relu')(graph_in)
                             #,subsample_length=1)(graph_in)
        pool = GlobalMaxPooling1D()(conv)
        #flatten = Flatten()(pool)
        convs.append(pool)

    if len(filter_sizes)>1:
        out = Merge(mode='concat')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)

    # main sequential model
    model = Sequential()
    #if not model_variation=='CNN-rand':
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=LEARN_EMBEDDINGS))
    model.add(Dropout(dropout_prob[0]))#, input_shape=(sequence_length, embedding_dim)))
    model.add(graph)
    model.add(Dropout(dropout_prob[1]))
    model.add(Activation('relu'))
    model.add(Dense(len(set(tx_class)), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model


def train_CNN(X, y, inp_dim, model, weights, epochs=EPOCHS, batch_size=BATCH_SIZE):
    cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = [], [], []
    p1, r1, f11 = 0., 0., 0.
    p_class, r_class, f1_class = [], [], []
    sentence_len = X.shape[1]

    marcro_f1, macro_r, macro_p = [],[],[]
    
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "word2vec":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print("ERROR!")
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in range(epochs):
            for X_batch in batch_gen(X_temp, batch_size):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    for cw in range(len(set(tx_class))):
                        class_weights[cw] = np.where(y_temp == cw)[0].shape[
                            0]/float(len(y_temp))
                try:
                    y_temp = np_utils.to_categorical(
                        y_temp, num_classes=len(set(tx_class)))
                except Exception as e:
                    print(e)
                    print(y_temp)
                #print(x.shape, y.shape)
                loss, acc = model.train_on_batch(
                    x, y_temp, class_weight=class_weights)

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        #print(classification_report(y_test, y_pred))
        #print(precision_recall_fscore_support(y_test, y_pred))
        #print(y_pred)
        p.append (precision_score(y_test, y_pred, average='weighted'))
        p1 += precision_score(y_test, y_pred, average='micro')
        p_class.append(precision_score(y_test, y_pred, average=None))
        r.append(recall_score(y_test, y_pred, average='weighted'))
        r1 += recall_score(y_test, y_pred, average='micro')
        r_class.append(recall_score(y_test, y_pred, average=None))
        f1.append (f1_score(y_test, y_pred, average='weighted'))
        f11 += f1_score(y_test, y_pred, average='micro')
        f1_class.append(f1_score(y_test, y_pred, average=None))

        macro_p.append(precision_score(y_test, y_pred, average='macro'))
        macro_r.append(recall_score(y_test, y_pred, average='macro'))
        marcro_f1.append(f1_score(y_test, y_pred, average='macro'))

    print("macro results are")
    print("average precision is %f" % (np.array(p).mean()))
    print("average recall is %f" % (np.array(r).mean()))
    print("average f1 is %f" % (np.array(f1).mean()))

    save_report_to_csv (REPORT_FOLDER  +'CNN_training_report.csv', [
        'CNN', 
        get_model_name_by_file (POLITICS_FILE),
        #weighted scores
        np.array(p).mean(),
        np.array(p).std() * 2,
        np.array(r).mean(),
        np.array(r).std() * 2,
        np.array(f1).mean(),
        np.array(f1).std() * 2,

        #macro scores
        np.array(macro_p).mean(),
        np.array(macro_p).std() * 2,
        np.array(macro_r).mean(),
        np.array(macro_r).std() * 2,
        np.array(marcro_f1).mean(),
        np.array(marcro_f1).std() * 2,

        #by class scores
        np.array(p_class[:,0]).mean(),
        np.array(p_class[:,1]).mean(),
        np.array(r_class[:,0]).mean(),
        np.array(r_class[:,1]).mean(),
        np.array(f1_class[:,0]).mean(),
        np.array(f1_class[:,1]).mean()
    ])

    print("micro results are")
    print("average precision is %f" % (p1 / NO_OF_FOLDS))
    print("average recall is %f" % (r1 / NO_OF_FOLDS))
    print("average f1 is %f" % (f11 / NO_OF_FOLDS))

    #return ((p / NO_OF_FOLDS), (r / NO_OF_FOLDS), (f1 / NO_OF_FOLDS))

if __name__ == "__main__":
    print ('Starting CNN...')
    parser = argparse.ArgumentParser(description='CNN based models for politics text')
    parser.add_argument('-f', '--embeddingfile', required=True)
    parser.add_argument('-d', '--dimension', required=True)
    parser.add_argument('--epochs', default=EPOCHS, required=True)
    parser.add_argument('--batch-size', default=BATCH_SIZE, required=True)
    parser.add_argument('-s', '--seed', default=SEED)
    parser.add_argument('--learn-embeddings', action='store_true', default=False)
    parser.add_argument('--initialize-weights', choices=['random', 'word2vec'], required=True)
    parser.add_argument('--scale-loss-function', action='store_true', default=False)
    parser.add_argument('--politicsfile', default=POLITICS_FILE)
    parser.add_argument('--nonpoliticsfile', default=NON_POLITICS_FILE)

    args = parser.parse_args()

    W2VEC_MODEL_FILE = args.embeddingfile
    EMBEDDING_DIM = int(args.dimension)
    SEED = int(args.seed)
    INITIALIZE_WEIGHTS_WITH = args.initialize_weights
    EPOCHS = int(args.epochs)
    BATCH_SIZE = int(args.batch_size)
    SCALE_LOSS_FUN = args.scale_loss_function
    POLITICS_FILE = args.politicsfile
    NON_POLITICS_FILE = args.nonpoliticsfile

    np.random.seed(SEED)
    print('W2VEC embedding: %s' % (W2VEC_MODEL_FILE))
    print('Embedding Dimension: %d' % (EMBEDDING_DIM))
    print('Allowing embedding learning: %s' % (str(LEARN_EMBEDDINGS)))

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v+W2VEC_MODEL_FILE,
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

    vocab = gen_vocab(word2vec_model)
    X, y = gen_sequence(vocab, texts, tx_class)

    data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    y = np.array(y)
    data, y = sklearn.utils.shuffle(data, y)
    W = get_embedding_weights()
    model = cnn_model(data.shape[1], EMBEDDING_DIM)
    train_CNN(data, y, EMBEDDING_DIM, model, W)

    input_file = POLITICS_FILE.replace(TMP_FOLDER, '').strip()

    model.save(H5_FOLDER + MODEL_NAME + input_file + ".h5")
    np.save(NPY_FOLDER + DICT_NAME + input_file +'.npy', vocab)

    
    

#python cnn.py -f cbow_s300.txt  -d 300 --epochs 10 --batch-size 30 --initialize-weights word2vec --scale-loss-function