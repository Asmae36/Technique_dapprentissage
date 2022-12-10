from sklearn.svm import SVC
from model.Classifier import Classifier


class SVM(Classifier):


    def __init__(self, Train, Test, Labels, Ids_Test, Classes):

        super(SVM, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = SVC.__name__
        self._classifier = SVC()
        self._param_grid = {'C': [50, 100,1000, 10000],'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}


#code véifié
