#Classifier number 2: k-nearest neighbors

from model.Classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier

class KNN(Classifier):

    def _init_(self, Train, Test, Labels, Ids_Test, Classes):
        super(KNN, self)._init_(Train, Test, Labels, Ids_Test, Classes)
        self.name = KNeighborsClassifier._name_
        self._classifier = KNeighborsClassifier()
        self._param_grid = {'n_neighbors': [1, 2, 3, 4, 5],'weights': ['distance','uniform'],'algorithm': ['ball_tree','brute' ,'auto','kd_tree'],'leaf_size': [10, 20, 30, 40, 50],'p': [1, 2]}
