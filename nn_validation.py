import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn

import configparser
import os
import argparse
import numpy as np
from scipy import interp
from political_classification import PoliticalClassification
from sklearn.metrics import roc_curve, auc, classification_report, precision_recall_fscore_support
from text_processor import TextProcessor
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import save_report_to_csv
from bow_classifier import generate_roc_curve
from bow_classifier import SEED
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, H5_FOLDER

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

H5_FILE = 'cnn_model.h5'
NPY_FILE = 'cnn_model.npy'

def load_file():
    texts = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet1")

    texts = [tx for tx in df.iloc[:,1]]
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    return texts, y_true


def generate_roc_curve (X, y_true):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0

    
    for train, test in cv.split(X, y_true):

        x_test = np.array([X[i] for i in test])
        y_test = np.array([y_true[i] for i in test])

        y_probas = list()
        for x_ in x_test:
            y_probas.append(pc.is_political_prob(' '.join(x_)))

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
    cnn_curve_plot = H5_FILE.replace('.h5', '')
    cnn_curve_plot = cnn_curve_plot.replace(H5_FOLDER, '')
    plt.savefig(PLOT_FOLDER + 'roc_curve_' + cnn_curve_plot + '.png')
    plt.clf()

    return mean_auc, std_auc


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

    X, y_true = load_file()
    tp = TextProcessor()

    pc = PoliticalClassification(H5_FILE, NPY_FILE, 25)

    pol = ''
    n_pol = ''
    y_pred = list()
    X = tp.text_process(X, text_only=True)
    for tx in X:
        text = ' '.join(tx)
        if pc.is_political(text):
            pol += text + '\n'
            y_pred.append(1)
        else:
            n_pol += text + '\n'
            y_pred.append(0)

    print(classification_report(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)
    
    mean_auc, std_auc = generate_roc_curve (X, y_true)

    save_report_to_csv (REPORT_FOLDER + 'validation_report.csv', [ 
        'CNN',
        H5_FILE,
        p,
        r, 
        f1,
        s,
        mean_auc,
        std_auc
    ])    
    
    
    # f =  open(dir_in + "CSCW/politics.txt", 'w')
    # f.write(pol)
    # f.close()

    # f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    # f.write(n_pol)
    # f.close()
