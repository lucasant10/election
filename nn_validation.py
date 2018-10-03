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
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_recall_fscore_support, f1_score, accuracy_score, recall_score, precision_score, confusion_matrix
from text_processor import TextProcessor
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from utils import save_report_to_csv, get_model_name_by_file, load_validation_file_csv
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, H5_FOLDER
from bow_classifier import SEED
from scipy.stats import norm

H5_FILE = 'cnn_model.h5'
NPY_FILE = 'cnn_model.npy'
VALIDATION_FILE = ''


def load_file():
    texts = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet2")

    texts = [tx for tx in df.iloc[:,1]]
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    return texts, y_true


def generate_roc_curve (X, y_true):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0

    for _, test in cv.split(X, y_true):

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
    
    plt.legend(loc="lower right")
    
    cnn_curve_plot = H5_FILE.replace('.h5', '')
    cnn_curve_plot = cnn_curve_plot.replace(H5_FOLDER, '')
    plt.title('ROC Curve: '+ cnn_curve_plot.replace('_', ' ').upper())

    plt.savefig(PLOT_FOLDER + 'roc_curve_' + cnn_curve_plot + '.png')
    plt.clf()

    return mean_auc, std_auc

def plot_confusion_matrix (confusion_matrix_array):

    print ('###### Start Confusion Matrix ####')

    print (confusion_matrix_array)

    save_report_to_csv (REPORT_FOLDER + get_model_name_by_file(VALIDATION_FILE) +'_confusion_report.csv', [
        'MultinomialNB', 
        get_model_name_by_file(MODEL_FILE),
        confusion_matrix_array[0][0],
        confusion_matrix_array[0][1],
        confusion_matrix_array[1][0],
        confusion_matrix_array[1][1]
    ])


    print ('###### End Confusion Matrix ####')


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

    model_name = MODEL_FILE
    
    model_name = model_name.replace ('.politics_ben.skl', '')
    model_name = model_name.replace (SKL_FOLDER, '')
    
    ax.set_title(model_name.replace ('_', ' ').upper())

    fig.add_subplot(ax)

    fig.savefig(PLOT_FOLDER + 'cnn_confusion_matrix_publica_'+ model_name + '.png', dpi=400)


def generate_normal(X, y_true):
    cv = StratifiedKFold(n_splits=10)
    precision = list()
    recall = list()
    f1 = list()
    for _ in range(4):
        for _, test in cv.split(X, y_true):
            x_test = np.array([X[i] for i in test])
            y_test = np.array([y_true[i] for i in test])
            y_pred = list()
            for x_ in x_test:
                y_pred.append(pc.is_political(' '.join(x_)))
            precision.append(precision_score(y_test, y_pred, average='weighted'))
            recall.append(recall_score(y_test, y_pred, average='weighted'))
            f1.append(f1_score(y_test, y_pred, average='weighted'))
    plot_save(precision, "Precision")
    plot_save(recall, "Recall")
    plot_save(f1, "F1-Score")

def plot_save(dist, label):
    sns.distplot(dist, fit=norm, kde=False, bins=8)
    plt.xlabel(label)
    plt.ylabel('Frequency')
    
    cnn_normal_plot = H5_FILE.replace('.h5', '')
    cnn_normal_plot = cnn_normal_plot.replace(H5_FOLDER, '')
    
    plt.title('Accuracy of CNN classifier: (%s)' % cnn_normal_plot.upper())
    plt.savefig(PLOT_FOLDER + "pred_%s_%s_CNN.png" % (label, cnn_normal_plot.upper()))
    plt.clf()
    save_report_to_csv (REPORT_FOLDER + 'acc_validation_report.csv', [ 
        'CNN',
        label + ' ' + cnn_normal_plot,
        np.mean(dist),
        np.std(dist), 
    ])    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Validation political CNN model')
    parser.add_argument('-h5',  default=H5_FILE)
    parser.add_argument('-npy', default=NPY_FILE)
    parser.add_argument('-vf', '--validationfile', required=True)

    args = parser.parse_args()
    H5_FILE = args.h5
    NPY_FILE = args.npy
    VALIDATION_FILE = args.validationfile

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    X, y_true = load_validation_file_csv(VALIDATION_FILE)
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

    ff1 = f1_score (y_true, y_pred, average='weighted')
    recall = recall_score (y_true, y_pred, average='weighted')
    precision = precision_score (y_true, y_pred, average='weighted')

    f1_macro = f1_score (y_true, y_pred, average='macro')
    recall_macro = recall_score (y_true, y_pred, average='macro')
    precision_macro = precision_score (y_true, y_pred, average='macro')

    accuracy = accuracy_score (y_true, y_pred)

    generate_normal(X,y_true)

    mean_auc, std_auc = generate_roc_curve (X, y_true)

    plot_confusion_matrix (confusion_matrix(y_true, y_pred))

    save_report_to_csv (REPORT_FOLDER + 'CNN_validation_report.csv', [ 
        'CNN',
        get_model_name_by_file(H5_FILE),
        get_model_name_by_file(VALIDATION_FILE),

        accuracy,
        
        p[0],
        p[1],
        r[0],
        r[1], 
        f1[0],
        f1[1],
        s[0],
        s[1],

        f1_macro,
        recall_macro,
        precision_macro,
        
        mean_auc, 
        std_auc,
        
        ff1,
        recall,
        precision
    ])  
