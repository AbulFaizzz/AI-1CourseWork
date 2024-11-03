import numpy as np
import math

class simple_KNN:
    def __init__(self, K, verbose=False): #initial state from class 
        self.K = K 
        self.verbose = verbose 
       
    def euclidean_distance(self,a,b):

                    
        assert a.shape[0] == b.shape[0] # a and b should have same dimensions and features
        distance=0.0
        for feature in range( a.shape[0]):
            difference = a[feature] - b[feature]
            distance= distance + difference*difference
        return math.sqrt(distance)

    def fit(self, X, y): #it is used for storing the training dataset and oraganises it in useful ways
        self.modelX = X
        self.modelY = y
        self.numtraining= X.shape[0] #rows
        self.featuresnum= X.shape[1] #columns
        self.labelsPresent = np.unique(y)

    def predict(self, newItems): #make predictions and returns them as an array
        numPredict = newItems.shape[0]
        predictions = np.empty(numPredict)  

        for index in range (numPredict):
            predictions[index]=self.predict_new_item(newItems[index])
        return predictions    

    def predict_new_item(self, newItem): #predicts a label for a test item, makes a prediction
        distances = []
        for index in range (self.numtraining):
            distance= self.euclidean_distance(newItem, self.modelX[index])
            distances.append(distance)
        closestK = self.get_ids_of_K_closest(self.K, distances) 

        for index in range(len(closestK)):
            closestK[index] = self.modelY[closestK[index]]
        
        counter = {}

        for closest in closestK:
            counter[closest] = counter.get(closest, 0) + 1  #counting how many times we get 0 and 1
         
        high = 0
        high_count = 0
        for key,value in counter.items():
            if high_count < value:
                high = key 
                high_count = value
        return high
        
    def get_ids_of_K_closest(self, K, distFromNewItem): #function to get the k-closest training items near the item
        closestK= []
        for thisK in range(K):
            thisclosest= 0
            for index in range (self.numtraining):
                if distFromNewItem[index]< distFromNewItem[thisclosest]:
                    thisclosest= index
            closestK.append(thisclosest)
            distFromNewItem[thisclosest]= 98769876542345678978965
        return closestK