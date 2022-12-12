from model.Classifier import Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA(Classifier):
    def __init__(self, Train, Test, Labels, Ids_Test, Classes):
   
        super(LDA, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = LinearDiscriminantAnalysis.__name__
        self._classifier = LinearDiscriminantAnalysis()
        self._param_grid = {'solver':['svd','lsqr','eigen'],'tol':[0.0001,0.001,0.01,0.1]}
