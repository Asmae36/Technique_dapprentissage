from model.KNN import KNN
from model.MLP import MLP
from model.NaiveBayesGaussienne import NaiveBayesGaussienne
from model.RandomForest import RandomForest
from model.Adaboost import Adaboost
from model.SVM import SVM
from sklearn.preprocessing import LabelEncoder

import argparse
import pandas as pd
import numpy as np


def ReadData():

    train = pd.read_csv('Technique_dapprentissage-main/src/data/train.csv')
    test = pd.read_csv('Technique_dapprentissage-main/src/data/test.csv')

    Data = LabelEncoder().fit(train.species)
    Labels = Data.transform(train.species)
    Classes = np.array(Data.classes_)
    Ids_Test = test.id

    Train = train.drop(['species','id'], axis=1)
    Test = test.drop(['id'], axis=1)

    return Train, Test, Labels, Ids_Test, Classes

def Parser():
    parser = argparse.ArgumentParser(description='leaf classification using 6 different algorithms.')
    parser.add_argument('--method', type=str, default='all',choices=['SVM','KNN','MLP','RandomForest','NaiveBayesGaussienne','Adaboost','all'])
    parser.add_argument('--hidden_layer', type=tuple, default=(20,))
    return parser.parse_args()