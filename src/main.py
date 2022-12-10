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


if __name__ == '__main__':
    arguments = Parser()
    method = arguments.method

    Train, Test, Labels, Ids_Test, Classes = ReadData()
    models = []

    if method == 'SVM':
        c = SVM(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)
    elif method == 'KNN':
        c = KNN(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)
    elif method == 'MLP':
        c = MLP(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)    
    elif method == 'RandomForest':
        c = RandomForest(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)
    elif method == 'NaiveBayesGaussienne':
        c = NaiveBayesGaussienne(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)
    elif method == 'Adaboost':
        c = Adaboost(Train, Test, Labels, Ids_Test, Classes)
        models.append(c)    
    elif method == 'all':
        Clfs = [SVM,KNN,MLP,RandomForest,NaiveBayesGaussienne,Adaboost]
        for c in Clfs:
            classifier = c(Train, Test, Labels, Ids_Test, Classes)
            models.append(classifier)
    else:
        raise Exception('method not valid')

    for c_model in models:
        c_model.hyperparameters_research()
        c_model.Train()
        c_model.get_Accuracies()
         
         
         
 #code vérifié
