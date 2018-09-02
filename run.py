# -*- coding: utf-8 -*-
import itertools
import glob
from subprocess import call
import os
import argparse

ROOT_FOLDER = '/Volumes/Data/eleicoes/'
INPUT_FOLDER = ROOT_FOLDER + 'input/'
OUTPUT_FOLDER = ROOT_FOLDER + 'output/'
PLOT_FOLDER =  OUTPUT_FOLDER +'plot/'
SKL_FOLDER =  OUTPUT_FOLDER +'skl/'
H5_FOLDER =  OUTPUT_FOLDER + 'h5/'
NPY_FOLDER =  OUTPUT_FOLDER + 'npy/'
REPORT_FOLDER = ROOT_FOLDER + 'report/'
TMP_FOLDER = ROOT_FOLDER + 'tmp/' 

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

    args = parser.parse_args()
    print (args)
    ROOT_FOLDER = args.rootfolder    

    if not os.path.isdir (ROOT_FOLDER):
        print ('Root Folder {} does not exist!'.format(ROOT_FOLDER))
        exit(0)

    out = glob.glob(INPUT_FOLDER + "*.txt")

    if os.path.isfile(REPORT_FOLDER + "training_report.csv"):
        os.remove(REPORT_FOLDER + "training_report.csv")

    if os.path.isfile(REPORT_FOLDER + "validation_report.csv"):
        os.remove(REPORT_FOLDER + "validation_report.csv")

    print ('Reading input folder: {}'.format(INPUT_FOLDER))

    tmp = [i.replace(INPUT_FOLDER, '').replace('_non-politics.txt', '').replace('_politics.txt', '') for i in out]
    tmp.sort()

    files = set(tmp)

    for k in range(1, len(files)+1):
        inputs = list(itertools.combinations (files, k))
        
        for input_ in inputs:
            
            features = list(input_)
            
            file_in_politics = TMP_FOLDER + ('_'.join(features))+'.politics'
            file_in_non_politics = TMP_FOLDER + ('_'.join(features))+'.nonpolitics'

            # generate politics input file for model
            with open(file_in_politics, 'w') as outfile:
                for fname in features:
                    politics_file = INPUT_FOLDER + fname + '_politics.txt'
                    # combine multiples politics input files
                    with open(politics_file) as infile:
                        for line in infile:
                            outfile.write(line)

            # generate non politics input file for model
            with open(file_in_non_politics, 'w') as outfile:
                for fname in features:
                    non_politics_file = INPUT_FOLDER + fname + '_non-politics.txt'

                    # combine multiples non-politics input files
                    with open(non_politics_file) as infile:
                        for line in infile:
                            outfile.write(line)


            """
            " Pro Publica Classifier
            """
            call (["python", 
                "prop_classifier.py", 
                "--politicsfile", file_in_politics,
                "--nonpoliticsfile", file_in_non_politics
            ])

            input_file = file_in_politics.replace(TMP_FOLDER, '').strip()

            skl_file = SKL_FOLDER + 'propublica_'+ input_file+'_ben.skl'

            call (["python", 
                "prop_validation.py", 
                "-m", skl_file
            ])
            
            for model in ['svm', 'logistic', 'gradient_boosting', 'random_forest']:
                print ('->>>> Running {} for {}'.format(model, ('_'.join(features))))

                skl_file = SKL_FOLDER + model + '_'+ file_in_politics.replace(TMP_FOLDER ,  '') +'_ben.skl'

                if not os.path.isfile (skl_file):
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

                #python bow_validation.py -m random_forest_ben.skl -f cbow_s300.txt

                
                
                print ('->>>> Running validation process for {} with {}'.format(model, skl_file))

                call (["python", 
                    "bow_validation.py", 
                    "-m", skl_file,
                    "-f", "cbow_s300.txt"
                ])

            print ('->>>> Running CNN for {}'.format(('_'.join(features))))
            
            h5_file = H5_FOLDER + 'cnn_model_' + file_in_politics.replace(TMP_FOLDER ,  '').strip() + ".h5"
            
            npy_file = NPY_FOLDER + 'cnn_dict_' + file_in_politics.replace(TMP_FOLDER ,  '').strip() + ".npy"

            print ('Loading H5 file: {}'.format(h5_file))
            print ('Loading NPY file: {}'.format(npy_file))

            if not os.path.isfile (h5_file):
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
                "-h5", h5_file,
                "-npy",npy_file
            ])
        
    # python3 run.py