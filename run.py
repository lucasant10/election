# -*- coding: utf-8 -*-
import sys

import itertools
import glob
from subprocess import call
import os
import argparse
import csv

ROOT_FOLDER = os.environ.get('ROOT_FOLDER','/Volumes/Data/eleicoes/')

#ROOT_FOLDER = '/Volumes/Data/eleicoes/'
#ROOT_FOLDER = '/home/vod/marcio.inacio/results/'
INPUT_FOLDER = ROOT_FOLDER + 'input/'
OUTPUT_FOLDER = ROOT_FOLDER + 'output/'
PLOT_FOLDER =  OUTPUT_FOLDER +'plot/'
SKL_FOLDER =  OUTPUT_FOLDER +'skl/'
H5_FOLDER =  OUTPUT_FOLDER + 'h5/'
NPY_FOLDER =  OUTPUT_FOLDER + 'npy/'
REPORT_FOLDER = ROOT_FOLDER + 'report/'
TMP_FOLDER = ROOT_FOLDER + 'tmp/' 

MODEL_TYPE = 'all'


def alreadyRanValidation (model, feature, fold):
    try:
        with open(OUTPUT_FOLDER + 'runs.csv', 'r', encoding="utf-8") as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='"')
            for row in spamreader:
                if row[0] == '_'.join([model, feature, fold]):
                    print ('%s already done!' % ('_'.join([model, feature, fold]))) 
                    return True
    except Exception as e:
        pass

    return False

def confirmValidation (model, feature, fold):
    with open(OUTPUT_FOLDER + 'runs.csv', 'a', encoding="utf-8") as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['_'.join([model, feature, fold])])
        
    return False


def pro_publica_classifier (file_in_politics, file_in_non_politics, validation_file):
    """
    " Pro Publica Classifier
    """
    
    print ('Running ProPublica Classifier:{}'.format(file_in_politics))

    call (["python", 
        "prop_classifier.py", 
        "--politicsfile", file_in_politics,
        "--nonpoliticsfile", file_in_non_politics
    ])

    input_file = file_in_politics.replace(TMP_FOLDER, '').strip()

    skl_file = SKL_FOLDER + 'propublica_'+ input_file+'_ben.skl'

    print ('Running ProPublica Validation:{}'.format(file_in_politics))

    call (["python", 
        "prop_validation.py", 
        "-vf", validation_file,
        "-m", skl_file
    ])
    
def bow_classifier (model, file_in_politics, file_in_non_politics, validation_file):
    print ('->>>> Running {} for {}'.format(model, ('_'.join(features))))

    skl_file = SKL_FOLDER + model + '_'+ file_in_politics.replace(TMP_FOLDER ,  '') +'_ben.skl'

    #if not os.path.isfile (skl_file):
    print ("Building {}".format(skl_file))

    call (["python", 
        "bow_classifier.py", 
        "--model", model, 
        "-f", "cbow_s300.txt",
        "--seed", "42",
        "-d", "300",
        "--politicsfile", file_in_politics,
        "--nonpoliticsfile", file_in_non_politics
    ])


    print ('->>>> Running validation process for {} with {}'.format(model, skl_file))

    call (["python", 
        "bow_validation.py", 
        "-vf", validation_file,
        "-m", skl_file,
        "-f", "cbow_s300.txt"
    ])

def cnn_classifier (file_in_politics, file_in_non_politics, validation_file):
    print ('->>>> Running CNN for {}'.format(('_'.join(features))))
            
    h5_file = H5_FOLDER + 'cnn_model_' + file_in_politics.replace(TMP_FOLDER ,  '').strip() + ".h5"
    
    npy_file = NPY_FOLDER + 'cnn_dict_' + file_in_politics.replace(TMP_FOLDER ,  '').strip() + ".npy"

    print ('Building H5 file: {}'.format(h5_file))
    print ('Building NPY file: {}'.format(npy_file))

    #if not os.path.isfile (h5_file):
        # python3 cnn.py -f cbow_s300.txt  -d 300 --epochs 10 --batch-size 30 --initialize-weights word2vec 
    call (["python", 
        "cnn.py", 
        "-f", "cbow_s300.txt",
        "--epochs", "10",
        "-d", "300",
        "--batch-size", "30",
        "--initialize-weights", "word2vec",
        "--politicsfile", file_in_politics,
        "--nonpoliticsfile", file_in_non_politics
    ])
        
    
    print ('->>>> Running CNN Validation for {}'.format(('_'.join(features))))
    # python3 nn_validation.py -h5 h5_file_name -npy npy_file_name 
    call (["python", 
        "nn_validation.py", 
        "-vf", validation_file,
        "-h5", h5_file,
        "-npy",npy_file
    ])
"""
" Sources
" S1 = Political Tweets
" S2 = Annotated political Ads
" S3 = Facebook Ads Explanation
" S4 = Official Facebook Political Ads
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Probublica model for politics texts')
    parser.add_argument('--rootfolder', default=ROOT_FOLDER)
    parser.add_argument('-m', '--model', choices=[
                        'logistic', 'gradient_boosting', 'random_forest', 'svm', 'svm_linear', 'cnn', 'propublica', 'bow', 'all'], required=True)

    args = parser.parse_args()
    
    ROOT_FOLDER = args.rootfolder
    MODEL_TYPE = args.model    

    if not os.path.isdir (ROOT_FOLDER):
        print ('Root Folder {} does not exist!'.format(ROOT_FOLDER))
        exit(0)

    out = glob.glob(INPUT_FOLDER + "*.txt")

    #if os.path.isfile(REPORT_FOLDER + "training_report.csv"):
     #   os.remove(REPORT_FOLDER + "training_report.csv")

    #if os.path.isfile(REPORT_FOLDER + "validation_report.csv"):
    #    os.remove(REPORT_FOLDER + "validation_report.csv")

    print ('Reading input folder: {}'.format(INPUT_FOLDER))

    tmp = [i.replace(INPUT_FOLDER, '').replace('_non-politics.txt', '').replace('_politics.txt', '') for i in out]
    tmp.sort()

    files = set(tmp)

    for k in range(1, len(files)+1):
        inputs = list(itertools.combinations (files, k))
        
        for input_ in inputs:
            
            features = sorted(list(input_))

            for i in range (0, 5):

                if alreadyRanValidation (MODEL_TYPE, '_'.join(features), 'fold' + str(i)):
                    continue

                validation_file = INPUT_FOLDER + 'fold%s.csv' % (i)

                print ('Validation file %s ' % (validation_file))
                
                file_in_politics = TMP_FOLDER + ('_'.join(features))+'.politics'
                file_in_non_politics = TMP_FOLDER + ('_'.join(features))+'.nonpolitics'
                
                # generate politics input file for model
                with open(file_in_politics, 'w', encoding="utf-8") as outfile:
                    for fname in features:
                        politics_file = INPUT_FOLDER + fname + '_politics.txt'
                        # combine multiples politics input files
                        with open(politics_file, encoding="utf-8") as infile:
                            for line in infile:
                                outfile.write(line)

                # generate non politics input file for model
                with open(file_in_non_politics, 'w', encoding="utf-8") as outfile:
                    for fname in features:
                        non_politics_file = INPUT_FOLDER + fname + '_non-politics.txt'

                        # combine multiples non-politics input files
                        with open(non_politics_file, encoding="utf-8") as infile:
                            for line in infile:
                                outfile.write(line)

                if 'S3' == ('_'.join(features)):
                    file_in_non_politics = TMP_FOLDER +'S2.nonpolitics'

                print (file_in_politics)
                print (file_in_non_politics)

                if MODEL_TYPE == 'propublica' or MODEL_TYPE == 'all':
                    
                    pro_publica_classifier (file_in_politics, file_in_non_politics, validation_file)

                if MODEL_TYPE in ['svm', 'logistic', 'gradient_boosting', 'random_forest', 'bow'] or MODEL_TYPE == 'all':
                    if MODEL_TYPE != 'all' and MODEL_TYPE != 'bow':
                        bow_classifier (MODEL_TYPE, file_in_politics, file_in_non_politics, validation_file)
                    else:
                        for model in ['svm', 'logistic', 'gradient_boosting', 'random_forest']:
                            bow_classifier (model, file_in_politics, file_in_non_politics, validation_file)

                if MODEL_TYPE == 'cnn' or MODEL_TYPE == 'all':
                    cnn_classifier (file_in_politics, file_in_non_politics, validation_file)
                
                confirmValidation (MODEL_TYPE, '_'.join(features), 'fold' + str(i))
            
    # python3 run.py -m all
    # python3 run.py -m propublica
    
    # python3 run.py -m svm
    # python3 run.py -m logistic
    # python3 run.py -m gradient_boosting
    # python3 run.py -m random_forest
    # python3 run.py -m cnn
