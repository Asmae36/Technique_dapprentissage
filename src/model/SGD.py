from model.Classifier import Classifier
from sklearn.linear_model import SGDClassifier

class SGD(Classifier):

    def __init__(self, Train, Test, Labels, Ids_Test, Classes):
  
        super(SGD, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = SGDClassifier.__name__
        self._classifier = SGDClassifier()
        self._param_grid = {'penalty':['l1','l2','elasticnet'],'loss':['hinge','perceptron','huber']}


   
#code vérifié
