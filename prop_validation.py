
import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import argparse
import seaborn as sn
import configparser
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, f1_score, recall_score, precision_score
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, train_test_split
from text_processor import TextProcessor
import pandas as pd
from sklearn.externals import joblib
import gensim
import gc
from utils import save_report_to_csv, get_model_name_by_file, load_validation_file_csv
from prop_classifier import get_vectorizer
from scipy import interp
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, SKL_FOLDER
from bow_classifier import NO_OF_FOLDS, SEED

MODEL_FILE = ''
VALIDATION_FILE = ''

def load_file():
    texts = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet2")

    texts = [tw for tw in df.iloc[:,1]]
    
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    
    return texts, y_true
def plot_confusion_matrix (confusion_matrix_array):

    print ('###### Start Confusion Matrix ####')

    print (confusion_matrix_array)

    save_report_to_csv (REPORT_FOLDER + get_model_name_by_file(VALIDATION_FILE)+'_confusion_report.csv', [
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

    fig.savefig(PLOT_FOLDER + 'confusion_matrix_publica_'+ model_name + '.png', dpi=400)


def generate_roc_curve (classifier, X, y, model_name=None, fold='fold0'):
    cv = StratifiedKFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=SEED)
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
    
    
    model_name = MODEL_FILE
    
    model_name = model_name.replace ('.politics_ben.skl', '')
    model_name = model_name.replace (SKL_FOLDER, '')

    plt.title('ROC Curve: '+ model_name.replace('_', ' ').upper() + ' ' + fold)
    plt.legend(loc="lower right")

    #plt.show()
    plt.savefig(PLOT_FOLDER + "roc_curve_propublica_" + model_name + '_'+ fold + ".png")
    plt.clf()

    return mean_auc, std_auc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Validation political Propublica model')
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-vf', '--validationfile', required=True)

    args = parser.parse_args()
    MODEL_FILE = args.model
    VALIDATION_FILE = args.validationfile
    
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_w2v = path['dir_w2v']
    dir_in = path['dir_in']

    
    texts, y_true = load_validation_file_csv(VALIDATION_FILE)

    print ('Loading '+MODEL_FILE+' file...')
    model = joblib.load(MODEL_FILE)
    vectorizer = get_vectorizer()
    pol = ''
    n_pol = ''
    y_pred = list()

   
    mean_auc, std_auc = generate_roc_curve (model, texts, y_true, MODEL_FILE, get_model_name_by_file(VALIDATION_FILE))
    
    print ('Predicting...')

    X = vectorizer.transform(texts)
    y_pred = model.predict(X)

    print ('Classification Report')
    print(classification_report(y_true, y_pred))
    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred)

    ff1 = f1_score (y_true, y_pred, average='weighted')
    recall = recall_score (y_true, y_pred, average='weighted')
    precision = precision_score (y_true, y_pred, average='weighted')

    f1_macro = f1_score (y_true, y_pred, average='macro')
    recall_macro = recall_score (y_true, y_pred, average='macro')
    precision_macro = precision_score (y_true, y_pred, average='macro')

    accuracy = accuracy_score (y_true, y_pred)

    save_report_to_csv (REPORT_FOLDER + 'validation_report.csv', [
        'MultinomialNB', 
        get_model_name_by_file(MODEL_FILE),
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

    print ('Confusion Matrix')
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    print(pd.crosstab (y_true, y_pred, rownames=['Real'], colnames=['Predict'], margins=True))

    plot_confusion_matrix (confusion_matrix(y_true, y_pred))
    
    gc.collect()
    exit(0) 
