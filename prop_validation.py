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
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from text_processor import TextProcessor
import pandas as pd
from sklearn.externals import joblib
import gensim
import gc
from utils import save_report_to_csv
from prop_classifier import get_vectorizer
from scipy import interp


MODEL_FILE = ''
NO_OF_FOLDS = 10

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
    ax.set_title(MODEL_FILE)
    ax.yaxis.set_ticklabels(['Non Political', 'Political']) 
    ax.xaxis.set_ticklabels(['Non Political', 'Political'])

    fig.add_subplot(ax)
    
    fig.savefig('plots/confusion_matrix_'+MODEL_FILE + '.png', dpi=400)


def generate_roc_curve (classifier, X, y, model_name=None):
    cv = StratifiedKFold(n_splits=NO_OF_FOLDS)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0

    for train, test in cv.split(X, y):

        x_test = [X[i] for i in test]
        y_test = np.array([y[i] for i in test])

        x_test = vectorizer.transform(x_test)
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
        model_name = ''.join (model_name)

    model_name = model_name.strip()
    
    model_name = model_name.replace ('politics_ben.skl', '')

    plt.title('ROC Curve: '+ model_name.capitalize())
    plt.legend(loc="lower right")

    #plt.show()
    plt.savefig("plots/roc_curve_" + model_name + ".png")
    plt.clf()

    return mean_auc, std_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validation political Propublica model')
    parser.add_argument('-m', '--model', required=True)

    args = parser.parse_args()
    MODEL_FILE = args.model
    
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    print ('Loading excel validation file...')
    texts, y_true = load_file()

    print ('Loading '+MODEL_FILE+' file...')
    model = joblib.load(dir_in + MODEL_FILE)
    vectorizer = get_vectorizer()
    pol = ''
    n_pol = ''
    y_pred = list()

   
    mean_auc, std_auc = generate_roc_curve (model, texts, y_true, MODEL_FILE)
    
    print ('Predicting...')

    X = vectorizer.transform(texts)
    y_pred = model.predict(X)

    print ('Classification Report')
    print(classification_report(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)

    save_report_to_csv ('validation_report.csv', [
        MODEL_FILE, 
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
    
    gc.collect()
    exit(0) 
