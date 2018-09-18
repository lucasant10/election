import warnings

warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass

warnings.warn = warn

import os
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import gc
from sklearn.externals import joblib
import argparse
import configparser
from run import PLOT_FOLDER, REPORT_FOLDER, TMP_FOLDER, SKL_FOLDER
from bow_classifier import SEED, NO_OF_FOLDS, save_report_to_csv

POLITICS_FILE = 'politics.txt' 
NON_POLITICS_FILE = 'non-politics.txt' 

def equalize_classes(predictor, response):
    """
    Equalize classes in training data for better representation.
    https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis#SMOTE
    """
    return SMOTE().fit_sample(predictor, response)


def get_vectorizer():
    """
    Return a HashingVectorizer, which we're using so that we don't
    need to serialize one.
    """
    return HashingVectorizer(
        alternate_sign=False,
        n_features=500000,
        ngram_range=(1, 3)
    )

def train_classifier(classifier, vectorizer, data):
    train, test = train_test_split(data, test_size=0.1, random_state=SEED, shuffle=True)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train, y_train = equalize_classes(x_train, y_train)
    print("final size of training data: %s" % x_train.shape[0])
    classifier.fit(x_train, y_train)
    print(classification_report(y_test, classifier.predict(x_test)))

    scores1 = cross_val_score(classifier, x_train, y_train, cv=NO_OF_FOLDS, scoring='precision_weighted')
    
    print("Precision(avg): %0.3f (+/- %0.3f)" %
          (scores1.mean(), scores1.std() * 2))
    precision_score_mean = scores1.mean()
    precision_score_std = scores1.std() * 2

    scores2 = cross_val_score(
        classifier, x_train, y_train, cv=NO_OF_FOLDS, scoring='recall_weighted')
    print("Recall(avg): %0.3f (+/- %0.3f)" %
          (scores2.mean(), scores2.std() * 2))

    recall_score_mean = scores2.mean()
    recall_score_std = scores2.std() * 2

    scores3 = cross_val_score(
        classifier, x_train, y_train, cv=NO_OF_FOLDS, scoring='f1_weighted')
    print("F1-score(avg): %0.3f (+/- %0.3f)" %
          (scores3.mean(), scores3.std() * 2))

    f1_score_mean = scores3.mean()
    f1_score_std = scores3.std() * 2

    save_report_to_csv (REPORT_FOLDER + 'MultinomialNB_training_report.csv', [
        'MultinomialNB', 
        POLITICS_FILE.replace(TMP_FOLDER, ''),
        precision_score_mean,
        precision_score_std,
        recall_score_mean,
        recall_score_std,
        f1_score_mean,
        f1_score_std,
    ])

    return classifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Probublica model for politics texts')
    parser.add_argument('--politicsfile', default=POLITICS_FILE)
    parser.add_argument('--nonpoliticsfile', default=NON_POLITICS_FILE)

    args = parser.parse_args()

    POLITICS_FILE = args.politicsfile
    NON_POLITICS_FILE = args.nonpoliticsfile

    cf = configparser.ConfigParser()
    cf.read("file_path.properties")
    path = dict(cf.items("file_path"))
    dir_in = path['dir_in']

    data = list()

    with open(POLITICS_FILE) as l_file:
        for line in l_file:
            data.append((line, 1.0))

    with open(NON_POLITICS_FILE) as l_file:
        for line in l_file:
            data.append((line, 0.0))


    classifier = MultinomialNB()
    vectorizer = get_vectorizer()
    model = train_classifier(classifier, vectorizer, data)

    input_file = POLITICS_FILE.replace(TMP_FOLDER, '').strip()

    joblib.dump(model, SKL_FOLDER + 'propublica_'+ input_file+'_ben.skl')


    

    gc.collect()
    exit(0) 











