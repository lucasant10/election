# -*- coding: utf-8 -*-
import itertools
import glob
from subprocess import call
import os
out = glob.glob("in/*.txt")

if os.path.isfile("./training_report.csv"):
    os.remove("./training_report.csv")

if os.path.isfile("./validation_report.csv"):
    os.remove("./validation_report.csv")

files = set([i.replace('in/', '').replace('_non-politics.txt', '').replace('_politics.txt', '') for i in out])
for k in range(1, len(files)+1):
    inputs = list(itertools.combinations (files, k))
    
    for input_ in inputs:
        
        features = list(input_)
        
        file_in_politics = 'tmp/'+('_'.join(features))+'.politics'
        file_in_non_politics = 'tmp/'+('_'.join(features))+'.nonpolitics'

        # generate politics input file for model
        with open(file_in_politics, 'w') as outfile:
            for fname in features:
                politics_file = 'in/' + fname + '_politics.txt'
                # combine multiples politics input files
                with open(politics_file) as infile:
                    for line in infile:
                        outfile.write(line)

        # generate non politics input file for model
        with open(file_in_non_politics, 'w') as outfile:
            for fname in features:
                non_politics_file = 'in/' + fname + '_non-politics.txt'

                # combine multiples non-politics input files
                with open(non_politics_file) as infile:
                    for line in infile:
                        outfile.write(line)



        for model in ['logistic', 'gradient_boosting', 'random_forest', 'svm']:
            print ('->>>> Running {} for {}'.format(model, ('_'.join(features))))
            call (["python", 
                "bow_classifier.py", 
                "--model", model, 
                "-f", "cbow_s300.txt",
                "--seed", "42",
                "-d", "300",
                "--politicsfile", file_in_politics,
                "--nonpoliticsfile", file_in_non_politics
            ])

            #python bow_validation.py -m random_forest_ben.skl -f cbow_s300.txt

            skl_file = model + '_'+ file_in_politics.replace('tmp/', '') +'_ben.skl'
            
            print ('->>>> Running validation process for {} with {}'.format(model, skl_file))

            call (["python", 
                "bow_validation.py", 
                "-m", skl_file,
                "-f", "cbow_s300.txt"
            ])

        print ('->>>> Running CNN for {}'.format(('_'.join(features))))
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
            "-h5", 'cnn_model_' + file_in_politics.replace('tmp/', '').strip() + ".h5",
            "-npy",'cnn_dict_' + file_in_politics.replace('tmp/', '').strip() + ".npy"
        ])
    
# python3 run.py