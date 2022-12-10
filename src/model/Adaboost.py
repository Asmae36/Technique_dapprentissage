from model.Classifier import Classifier
from sklearn.ensemble import AdaBoostClassifier

class Adaboost(Classifier):

    def __init__(self, Train, Test, Labels, Ids_Test, Classes):
  
        super(Adaboost, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = AdaBoostClassifier.__name__
        self._classifier = AdaBoostClassifier()
        self._param_grid = {'n_estimators':[40,50,60,70],'learning_rate':[1,0.1,0.01,0.001,0.0001]}



   
#code vérifié
