import sys
sys.path.append('../')
import argparse
import configparser
import numpy as np
from political_classification import PoliticalClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
from text_processor import TextProcessor
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.externals import joblib
import gensim

MODEL_FILE = ''
W2VEC_MODEL_FILE = ''
EMBEDDING_DIM = 300

def load_file():
    tweets = list()
    xl = pd.ExcelFile("/home/lucas/Dropbox/UFMG/Benevenuto/Dados Rotulados.xlsx")
    df = xl.parse("Sheet1")

    tweets = [tw for tw in df.iloc[:,1]]
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    return tweets, y_true

def gen_data(texts):
    X = []
    for text in texts:
        emb = np.zeros(EMBEDDING_DIM)
        for word in text:
            try:
                emb += word2vec_model[word]
            except:
                pass
        emb /= len(text)
        X.append(emb)
    return X

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Validation political BoW models')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-f', '--embeddingfile', required=True)

    args = parser.parse_args()
    W2VEC_MODEL_FILE = args.embeddingfile
    MODEL_FILE = args.model
    
    cf = configparser.ConfigParser()
    cf.read("../file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                                     binary=False,
                                                                     unicode_errors="ignore")
    texts, y_true = load_file()
    model = joblib.load(dir_in + MODEL_FILE)
    pol = ''
    n_pol = ''
    y_pred = list()
    tp = TextProcessor()
    texts = tp.text_process(texts, text_only=True)
    X = gen_data(texts)
    y_pred = model.predict(X)
    print(classification_report(y_true, y_pred))
    
    for i, tx in enumerate(texts):
        text = ' '.join(tx)
        if y_pred[i]: 
            pol += text + '\n'
        else:
            n_pol += text + '\n'

    f =  open(dir_in + "CSCW/politics.txt", 'w')
    f.write(pol)
    f.close()

    f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    f.write(n_pol)
    f.close()


# python benevenuto_val.py -m random_forest_ben.skl -f cbow_s300.txt