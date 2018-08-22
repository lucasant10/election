import os
from imblearn.over_sampling import SMOTE
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import gc
from sklearn.externals import joblib
import argparse
import configparser

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
    train, test = train_test_split(data, test_size=0.1)
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    x_train = vectorizer.transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_train, y_train = equalize_classes(x_train, y_train)
    print("final size of training data: %s" % x_train.shape[0])
    classifier.fit(x_train, y_train)
    print(classification_report(y_test, classifier.predict(x_test)))
    return classifier

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Probublica model for politics texts')
    parser.add_argument('--politicsfile', default=POLITICS_FILE)
    parser.add_argument('--nonpoliticsfile', default=NON_POLITICS_FILE)

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

    input_file = POLITICS_FILE.replace('tmp/', '').strip()
    joblib.dump(model, dir_in + 'propublica_'+ input_file+'_ben.skl')

    gc.collect()
    exit(0) 











