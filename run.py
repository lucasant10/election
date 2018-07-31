# -*- coding: utf-8 -*-
import itertools
import glob
from subprocess import call

#filenames = ['file1.txt', 'file2.txt']

out = glob.glob("in/*.txt")

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



        for model in ['logistic', 'random_forest', 'gradient_boosting', 'svm']:
            print ('->>>> Running for {}'.format(('_'.join(features))))
            call (["python", 
                "bow_classifier.py", 
                "--model", model, 
                "-f", "cbow_s300.txt",
                "--seed", "42",
                "-d", "300",
                "--politicsfile", file_in_politics,
                "--nonpoliticsfile", file_in_non_politics
            ])