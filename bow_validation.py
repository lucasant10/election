import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import argparse
import seaborn as sn
import configparser
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from text_processor import TextProcessor
import pandas as pd
from sklearn.externals import joblib
import gensim
import gc
from bow_classifier import generate_roc_curve, generate_normal
from utils import save_report_to_csv
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, SKL_FOLDER

MODEL_FILE = ''
W2VEC_MODEL_FILE = ''
EMBEDDING_DIM = 300



def load_file():
    texts = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet1")

    texts = [tw for tw in df.iloc[:,1]]
    
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    
    return texts, y_true
def plot_confusion_matrix (confusion_matrix_array):

    df_cm = pd.DataFrame(confusion_matrix_array, range(2), range(2))

    #plt.figure(figsize = (10,7))

    plot = df_cm.plot()
    fig = plot.get_figure()
    

    ax = plt.subplot()
    
    sn.heatmap(df_cm, annot=True, fmt='g', ax = ax, annot_kws={"size": 16})# font size
    
    # labels, title and ticks
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Real')
    
    ax.yaxis.set_ticklabels(['Non Political', 'Political']) 
    ax.xaxis.set_ticklabels(['Non Political', 'Political'])

    model_name = MODEL_FILE.replace (SKL_FOLDER, '')
    model_name = model_name.replace ('.politics_ben.skl', '')
    
    ax.set_title(model_name.replace('_', ' ').upper())
    fig.add_subplot(ax)

    fig.savefig(PLOT_FOLDER + 'confusion_matrix_' + model_name + '.png', dpi=400)

def gen_data(texts):
    X = []
    i = 0
    for text in texts:
        emb = np.zeros(EMBEDDING_DIM)
        for word in text:
            try:
                emb += word2vec_model[word]
            except:
                pass
        if not len (text):
            print (texts[i-1])
            print (i)
            exit(0)
        i +=1
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
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    

    print ('Loading word2vec model...')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(dir_w2v + W2VEC_MODEL_FILE,
                                                                     binary=False,
                                                                     unicode_errors="ignore")
    print ('Loading excel validation file...')
    texts, y_true = load_file()

    print ('Loading ' + MODEL_FILE + ' file...')
    model = joblib.load( MODEL_FILE)
    pol = ''
    n_pol = ''
    y_pred = list()
    tp = TextProcessor()
    texts = tp.text_process(texts, text_only=True)
    X = gen_data(texts)

    mean_auc, std_auc = generate_roc_curve (model, X, y_true, MODEL_FILE)
    generate_normal(model, X, y_true, MODEL_FILE)
    
    print ('Predicting...')

    y_pred = model.predict(X)

    print ('Classification Report')
    print(classification_report(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)

    model_name = MODEL_FILE.replace (SKL_FOLDER, '')
    model_name = model_name.replace ('.politics_ben.skl', '')
    model_name = model_name.replace('_', ' ').upper()

    save_report_to_csv (REPORT_FOLDER + model_name +'_validation_report.csv', [
        model_name, 
        p,
        r, 
        f1,
        s,
        mean_auc, 
        std_auc
    ])

    print ('Confusion Matrix')
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    print(pd.crosstab (y_true, y_pred, rownames=['Real'], colnames=['Predict'], margins=True))

    plot_confusion_matrix (confusion_matrix(y_true, y_pred))
    
    # for i, tx in enumerate(texts):
    #     text = ' '.join(tx)
    #     if y_pred[i]: 
    #         pol += text + '\n'
    #     else:
    #         n_pol += text + '\n'

    # f =  open(dir_in + "CSCW/politics.txt", 'w')
    # f.write(pol)
    # f.close()

    # f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    # f.write(n_pol)
    # f.close()


# python bow_validation.py -m random_forest_ben.skl -f cbow_s300.txt

gc.collect()
exit(0)