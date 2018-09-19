# -*- coding: utf-8 -*-
import csv
from run import SKL_FOLDER, REPORT_FOLDER, H5_FOLDER, NPY_FOLDER, TMP_FOLDER

def save_report_to_csv (file_name, features):
    print ('Saving %s' % (file_name))
    with open(file_name, 'a') as csvfile:
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
    model = model.replace ('.politics', '')
    model = model.replace ('_ben.skl', '')

    model = ' '.join(model.split('_')).upper()

    return model