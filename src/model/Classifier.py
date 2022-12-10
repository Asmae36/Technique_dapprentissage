"""
We will start with a parent class 'Classifier.py' that will contain the search oh hyperparameters.
The use of this class will facilitate the implementation of our 6 algorithms ; we will each time inherit the methods of this class.
"""

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score



class Classifier:
    
    def _init_(self,Train, Test, Labels, Ids_Test, Classes):

        self._Train =Train
        self._Test = Test
        self._Labels = Labels
        self._Ids_Test = Ids_Test
        self._Classes = np.array(Classes)

        self._X_Train, self._Y_Train,self._X_Valid, self._Y_Valid = self._Data_splitting()

        self._best_model = None
        self._best_pair = None
        self._best_score = None
        """
          parameter Train: Training game, will be subdivided to validate the training
          parameter Test: Data to classify
          parameter Labels:annotations
          parameter Ids_Test: Id of the test dataframe for the leaf-classification dataset
          parameter Classes: Plant species names
        """

    def Train(self):
       
        self._best_model.fit(self._X_Train, self._Y_Train)
       
        #This function train the model with the provided dataset
