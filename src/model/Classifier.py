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
    def Training_Accuracy(self):
         
        return accuracy_score(self._Y_Train, self._best_model.predict(self._X_Train))
        #This function allow us to calculates training accuracy and returns the training accuracy

    def Validation_Accuracy(self):
        
        return accuracy_score(self._Y_Valid, self._best_model.predict(self._X_Valid))

        #This function allow us to calculates the validation accuracy and return validation accuracy
        

    def get_Accuracies(self):
        print(f'Training accuricies: {self.Training_Accuracy():.2%}')
        print(f'Validation accuricies: {self.Validation_Accuracy():.2%}')
        
        #This function displays both validation and training accuricies
        
    def hyperparameters_research(self):
   
        grid = GridSearchCV(self._classifier, self._param_grid, scoring='accuracy', n_jobs=-1, verbose=1)
        grid.fit(self._X_Train, self._Y_Train)

        self._best_model = grid.best_estimator_
        self._best_score = grid.best_score_
        self._best_pair = grid.best_params_
        print(f'The best parameters identified for {self.name} are {self._best_pair} for an accuracy '
              f'of {self._best_score:.2%}')
        
        """
         This function can perform a hyper-parameter search. The best trained model found is saved in
         self._best_model and the best hyper-parameters found in self._best_pair.
         """
    def _Data_splitting(self):
       
        Split_stratified = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=30)

        for Train_indice, Valid_indice in Split_stratified.split(self._Train, self._Labels):
            X_Train,X_Valid = self._Train.values[Train_indice],self._Train.values[Valid_indice]
            Y_Train,Y_Valid = self._Labels[Train_indice],self._Labels[Valid_indice]

        return X_Train,Y_Train,X_Valid,Y_Valid
     #This function divides the annotated dataset into training and validation subset and returns the 4 subsets 
     
#code vérifié
