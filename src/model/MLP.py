from sklearn.neural_network import MLPClassifier
from model.Classifier import Classifier

class MLP(Classifier):
    
    def __init__(self, Train, Test, Labels, Ids_Test, Classes):
   
        super(MLP, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = MLPClassifier.__name__
        self._classifier = MLPClassifier()
        self._param_grid = {'hidden_layer_sizes': [(50,), (60,), (70,),(80,), (90,), (100,)],'learning_rate_init': [0.01, 0.001, 0.0001,0.00001],'solver': ['adam', 'sgd'],'activation': ['relu', 'logistic','tanh']}

        
#code vérifié
