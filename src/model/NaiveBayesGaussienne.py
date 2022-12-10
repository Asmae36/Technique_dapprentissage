from sklearn.naive_bayes import GaussianNB
from model.Classifier import Classifier


class NaiveBayesGaussienne(Classifier):

    def _init_(self, Train, Test, Labels, Ids_Test, Classes):

        super(NaiveBayesGaussienne, self)._init_(Train, Test, Labels, Ids_Test, Classes)
        self.name = GaussianNB._name_
        self._classifier = GaussianNB()
        self._param_grid = {'var_smoothing': [0.01,0.001, 0.0001,0.00001,0.02, 0.002, 0.0002,0.00002]}
