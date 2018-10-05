# -*- coding: utf-8 -*-
import csv
from run import SKL_FOLDER, REPORT_FOLDER, H5_FOLDER, NPY_FOLDER, TMP_FOLDER, INPUT_FOLDER
import datetime

def save_report_to_csv (file_name, features):
    print ('Saving %s' % (file_name))
    features.append (datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"))
    with open(file_name, 'a', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(features)

def report2dict(cr):
    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)
    
    # Store in dictionary
    measures = tmp[0]

    D_class_data = dict()
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    
    return D_class_data
        

def get_model_name_by_file (file_name):

    model = file_name.replace (SKL_FOLDER, '')
    model = model.replace (REPORT_FOLDER, '')
    model = model.replace (H5_FOLDER, '')
    model = model.replace (NPY_FOLDER, '')
    model = model.replace (TMP_FOLDER, '')
    model = model.replace (INPUT_FOLDER, '')
    model = model.replace ('.politics', '')
    model = model.replace ('_ben.skl', '')
    model = model.replace ('propublica', '')
    model = model.replace ('PROPUBLICA', '')
    model = model.replace ('GRADIENT BOOSTING', '')
    model = model.replace ('SVM', '')
    model = model.replace ('RANDOM FOREST', '')
    model = model.replace ('LOGIST', '')
    model = model.replace ('h5', '')
    model = model.replace ('cnn model ', '')
    model = model.replace ('.csv', '')
    model = model.replace ('.CSV', '')

    model = ' '.join(model.split('_')).upper()

    return model

def get_model_name (file_name):

    if 'svm' in file_name: return 'svm'
    if 'logistic' in file_name: return 'logistic'
    if 'random_forest' in file_name: return 'random_forest'
    if 'gradient_boosting' in file_name: return 'gradient_boosting'
    if 'propublica' in file_name: return 'multinomialb'

    return 'ERROR'

def load_file():
    print ('Loading excel validation file...')

    texts = list()
    xl = pd.ExcelFile("Dados Rotulados.xlsx")
    df = xl.parse("Sheet2")

    texts = [tw for tw in df.iloc[:,1]]
    
    y_true = [1 if i==u'política' else 0 for i in df.iloc[:,2]]
    
    return texts, y_true

def load_validation_file_csv(validation_file):
    print ('Loading CSV validation file...')
    
    texts = list()
    y_true = list()

    with open(validation_file, 'r', encoding="utf-8") as csvfile:
    
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
        
        for row in spamreader:
            texts.append(row[0])
            y_true.append (int(row[1]))
    
    return texts, y_true
