import configparser
import os
import argparse
import math
import numpy as np
from political_classification import PoliticalClassification
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support
from text_processor import TextProcessor
import pandas as pd
from utils import save_report_to_csv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

H5_FILE = 'cnn_model.h5'
NPY_FILE = 'cnn_model.npy'

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

    save_report_to_csv ('validation_report.csv', [ 
        'CNN',
        H5_FILE,
        p,
        r, 
        f1,
        s
    ])

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
    auc_keras = auc(fpr_keras, tpr_keras)

    


    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='CNN (AUC = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('plots/cnn_roc_curve.png', dpi=400)
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='CNN (AUC = {:.3f})'.format(auc_keras))
    #plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.savefig('plots/cnn_roc_curve_zoom.png', dpi=400)

    # f =  open(dir_in + "CSCW/politics.txt", 'w')
    # f.write(pol)
    # f.close()

    # f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    # f.write(n_pol)
    # f.close()
