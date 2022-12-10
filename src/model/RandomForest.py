from sklearn.ensemble import RandomForestClassifier
from model.Classifier import Classifier


class RandomForest(Classifier):

    def __init__(self, Train, Test, Labels, Ids_Test, Classes):

        super(RandomForest, self).__init__(Train, Test, Labels, Ids_Test, Classes)
        self.name = RandomForestClassifier.__name__
        self._classifier = RandomForestClassifier(n_jobs=-1)
        self._param_grid = {'n_estimators': [250, 300, 350],'max_depth': [15,20,35,30]}
        
        
#code vérifié
