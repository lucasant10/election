import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn

import configparser
import os
import argparse
import math
import numpy as np
from scipy import interp
from political_classification import PoliticalClassification
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support
from text_processor import TextProcessor
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import save_report_to_csv
from bow_classifier import generate_roc_curve
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.manifold import TSNE
import plotly.offline as py
import plotly.graph_objs as go


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

H5_FILE = 'cnn_model.h5'
NPY_FILE = 'cnn_model.npy'

def plot_words(data, start, stop, step):
    trace = go.Scatter(
        x = data[start:stop:step,0], 
        y = data[start:stop:step, 1],
        mode = 'markers',
        text= word_list[start:stop:step]
    )
    layout = dict(title= 't-SNE 1 vs t-SNE 2',
                  yaxis = dict(title='t-SNE 2'),
                  xaxis = dict(title='t-SNE 1'),
                  hovermode= 'closest')
    fig = dict(data = [trace], layout= layout)
    py.image.save_as(fig, filename='a-simple-plot.png')

def load_file():
    tweets = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet1")

    tweets = [tw for tw in df.iloc[:,1]]
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    return tweets, y_true

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Validation political CNN model')
    parser.add_argument('-h5',  default=H5_FILE)
    parser.add_argument('-npy', default=NPY_FILE)

    args = parser.parse_args()
    H5_FILE = args.h5
    NPY_FILE = args.npy

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    tweets, y_true = load_file()
    tp = TextProcessor()

    pc = PoliticalClassification(H5_FILE, NPY_FILE, 25)

    pol = ''
    n_pol = ''
    y_pred = list()
    tweets = tp.text_process(tweets, text_only=True)
    for tw in tweets:
        text = ' '.join(tw)
        if pc.is_political(text):
            pol += text + '\n'
            y_pred.append(1)
        else:
            n_pol += text + '\n'
            y_pred.append(0)

    print(classification_report(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    
    #try:
    #generate_roc_curve (pc.get_classifier(), X, y_true,  model_name='CNN')
    #except Exception as e:
     #   print (e)

    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0

    for train, test in cv.split(tweets, y_true):

        x_test = np.array([tweets[i] for i in test])
        y_test = np.array([y_true[i] for i in test])

        y_probas = list()
        for x_ in x_test:
            y_probas.append(pc.is_political_prob(x_))

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probas)
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


    plt.title('ROC Curve: CNN')
    plt.legend(loc="lower right")

    #plt.show()
    plt.savefig("plots/roc_curve_CNN.png")
    plt.clf()

    save_report_to_csv ('validation_report.csv', [ 
        'CNN',
        H5_FILE,
        p,
        r, 
        f1,
        s,
        mean_auc,
        std_auc
    ])

    # Zoom in view of the upper left corner.
    #plt.figure(2)
    #plt.xlim(0, 0.2)
    #plt.ylim(0.8, 1)
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.plot(frps, tprs, label='CNN (AUC = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.title('ROC curve (zoomed in at top left)')
    #plt.legend(loc='best')
    #plt.savefig('plots/cnn_roc_curve_zoom.png', dpi=400)

    
    
    
    # f =  open(dir_in + "CSCW/politics.txt", 'w')
    # f.write(pol)
    # f.close()

    # f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    # f.write(n_pol)
    # f.close()
