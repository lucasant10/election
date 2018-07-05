import configparser
import json
import os
from collections import defaultdict
import math
import pymongo
import numpy as np
from political_classification import PoliticalClassification
from sklearn.metrics import classification_report, precision_recall_fscore_support
from text_processor import TextProcessor
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def load_file():
    tweets = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet1")

    tweets = [tw for tw in df.iloc[:,1]]
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    return tweets, y_true

if __name__ == "__main__":
    
    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    tweets, y_true = load_file()
    tp = TextProcessor()

    pc = PoliticalClassification('cnn_s300.h5', 'cnn_s300.npy', 18)

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
    print(precision_recall_fscore_support(y_true, y_pred))

    f =  open(dir_in + "CSCW/politics.txt", 'w')
    f.write(pol)
    f.close()

    f =  open(dir_in + "CSCW/non_politics.txt", 'w')
    f.write(n_pol)
    f.close()
