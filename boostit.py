import numpy as np
import math
       
#    //////////      Boosting Alg     ////////////

class BoostingClassifier:
    
    # initialize the parameters here
    def __init__(self):

        self.T = 5  # ensemble size, total number of classifiers
        self.Tt = self.T
        self.a = [] # confidence factor 𝛼 sub t

        # arrays to hold each centroid
        self.posCentroid = []
        self.negCentroid = []

        # array to hold each interations model for prediction 
        self.M = []

    def fit(self, X, y):
        # print("\nln28 - X training data input:\n", X)
        # print("\nln29 - y label training data input:\n", y)
        # print("\nfeatures test:\n", features)

        # Sort input training data X into the two label classes 1 and -1
        posClass = [] # y label 1
        negClass = [] # y label -1
        # print("\nln35 - empty pos class:\n", posClass)

        # for loop to do sorting 
        for i in range(len(X)):
            if y[i] == 1: # positive class
                posClass.append(X[i])
            else:   # negative class
                negClass.append(X[i])
        
        # print("\nln43 - pos class:\n", posClass)
        # print("\nln44 - neg class:\n", negClass)

        # TO-DO -- Initialize all weights equally as described in the boosting algorithm. 
        # self.w = np.zeros(len(X))
        # for i in range(len(X)): 
        #     self.w[i] = None
         
        # each weight is initialized to 1/|D|, where |D| is the number of training points
        self.D = abs(len(X))
        self.w = np.full((len(X)), 1 / self.D)
        self.wInc = 1 / self.D
        self.wDec = 1 / self.D

        # print("\nln55 w:\n", self.w)

        # actual iterations of model
        for t in range(self.T): 
            print("\nIteration:", t + 1)

            # Sort the weights just like the input training data to match a weight for every point
            posW = []
            negW = []

            # for loop to do sorting 
            for i in range(len(self.w)):
                if y[i] == 1: # positive class
                    posW.append(self.w[i])
                else:   # negative class
                    negW.append(self.w[i])

            # print("\nln72 - pos weights:\n", posW)
            # print("\nln73 - neg weights:\n", negW)
            
            # TO-DO -- Create weighted centroids. This is where you apply the weighted average. I did this in one line, but you might not be able to! 
            posWCentroid = []
            wx = []

            # loop to do wi * xi for all of i - pos class
            for i in range(len(posW)):
                wx.append(posW[i] * posClass[i])

            # rest of equation with the 1/ sum of weights * sum of wi and xi for all of i
            posWCentroid.append(np.sum(wx, axis = 0) / np.sum(posW, axis = 0))

            negWCentroid = []
            wx = []

            # loop to do wi * xi for all of i - neg class
            for i in range(len(negW)):
                wx.append(negW[i] * negClass[i])

            # rest of equation with the 1/ sum of weights * sum of wi and xi for all of i
            negWCentroid.append(np.sum(wx, axis = 0) / np.sum(negW, axis = 0))

            self.posCentroid.append(posWCentroid)
            self.negCentroid.append(negWCentroid)
            # print("\nln98 - pos weights centroids:\n", self.posCentroid)
            # print("\nln99 - neg weights centroids:\n", self.negCentroid)

            predictions = []    # array to hold predicitons

            #loop through and create centroids based on the class and comapre the distances to make a prediciton  
            for i in range(len(X)): 

                # neg 
                dis0 = np.linalg.norm(X[i] - self.negCentroid)

                # pos
                dis1 = np.linalg.norm(X[i] - self.posCentroid)

                # print("\nln114 - dis0:\n", dis0)
                # print("\nln115 - dis1:\n", dis1)

                if dis0 < dis1: 
                    predictions.append(-1)
                else: 
                    predictions.append(1)

            # print("\nln124 - predicitons:\n", predictions)

            et = 0  # error rate

            # calculate error rate
            for j, pred, true in zip(range(len(predictions)), predictions, y): 
                if pred != true: # if the predicition is not equal to the true label 
                    et += self.w[j]
            
            print("Error rate = ", et)

            # If the error rate >= 0.5, end the loop
            if et >= 0.5: 
                self.T = t
                break

            # Calculate confidence factor 𝛼 sub t
            self.a.append(0.5 * math.log((1 - et) / et))
            # print(t)
            print("Alpha: ", self.a[t])

            
            # inc and dec boosting weights based on performance 
            self.wInc = self.wInc / (2 * et)
            self.wDec = self.wDec / (1 / (1 - et))

            for k, pred, true in zip(range(len(predictions)), predictions, y): 
                if pred == true: # if prediciton matches true label, dec the weight of the point
                    self.w[k] = self.w[k] * self.wDec
                else:  # in the weight of the point
                    self.w[k] = self.w[k] * self.wInc
                    # print("Hello World")

            print("Factor to increase weights = ", self.wInc)
            print("Factor to decrease weights = ", self.wDec)
        
        # if the loop was broken bc of a high et before the end, fix the centroid arrays
        if self.T != self.Tt: 
            self.a = self.a[:self.T]
            self.negCentroid = self.negCentroid[:self.T]
            self.posCentroid = self.posCentroid[:self.T]
        
        self.a = np.array(self.a, dtype=object)
        self.negCentroid = np.array(self.negCentroid, dtype=object)
        self.posCentroid = np.array(self.posCentroid, dtype=object)
        
        # to create the ensemble centroids, multiply together the confidence factors and the negative centroids
        posM = []
        for i in range(len(self.posCentroid)):
            posM.append(self.a[i] * self.posCentroid[0])

        negM = []
        for i in range(len(self.negCentroid)):
            negM.append(self.a[i] * self.negCentroid[0])

        # print("\nln178 alpha test:\n", self.a)
        
        # divide by the sum of the confidence factors
        self.M.append(sum(posM) / sum(self.a))
        self.M.append(sum(negM) / sum(self.a))

        self.M = np.array(self.M)
        # print("\nln 185 - M:\n", self.M)    # [[posM], [negM]]

        return self

    def predict(self, X):
        predictions = []
            #loop through and create centroids based on the class and comapre the distances to make a prediciton  
        for i in range(len(X)): 

            # neg 
            dis0 = np.linalg.norm(X[i] - self.M[1])

            # pos
            dis1 = np.linalg.norm(X[i] - self.M[0])

            # print("\nln114 - dis0:\n", dis0)
            # print("\nln115 - dis1:\n", dis1)

            if dis0 < dis1: 
                predictions.append(-1)
            else: 
                predictions.append(1)

        # print("\nln124 - predicitons:\n", predictions)
        return predictions



#   ///////////////    main to test output /////////////////////
if __name__ == "__main__": 
    #!/usr/bin/env python3
    import sys
    import os
    import numpy as np
    import time

    # evaluation on your local machine only
    dataset_dir = 'dataset1'
    train_set = os.path.join(dataset_dir, 'train.npy')
    test_set = os.path.join(dataset_dir, 'test.npy')

    def evaluation_score(y_pred, y_test):
        y_pred = np.squeeze(y_pred)
        assert y_pred.shape == y_test.shape, "Error: the shape of your prediction doesn't match the shape of ground truth label."

        TP = 0    # truth positive
        FN = 0    # false negetive
        TN = 0    # true negetive
        FP = 0     # false positive

        for i in range(len(y_pred)):
            pred_label = y_pred[i]
            gt_label = y_test[i]

            if int(pred_label) == -1:
                if pred_label == gt_label:
                    TN += 1
                else:
                    FN += 1
            else:
                if pred_label == gt_label:
                    TP += 1
                else:
                    FP += 1

        accuracy = round((TP + TN) / (TP + FN + FP + TN),4)
        precision = round(TP / (TP + FP) if ((TP + FP) > 0) else 0,4)
        recall = round(TP / (TP + FN) if ((TP + FN)) > 0 else 0,4)
        f1 = round(2 * precision * recall / (precision + recall) if ((precision + recall) > 0) else 0,4)
        final_score = round(50 * accuracy + 50 * f1,4)

        print("\nTesting:")
        print("TP: {}\nFP: {}\nTN: {}\nFN: {}\nError rate: {}".format(TP, FP, TN, FN, (FP+FN)/(TP+FP+TN+FN)))

        return accuracy, precision, recall, f1, final_score

    # load dataset
    with open(train_set, 'rb') as f:
        X_train = np.load(f)
        y_train = np.load(f)

    with open(test_set, 'rb') as f:
        X_test = np.load(f)
        y_test = np.load(f)

    # print("\nx train:\n", len(X_train))
    # print("\ny train:\n", len(y_train))
    # print("\nxtrain:\n", X_train)
    # print("\nxtrain:\n", X_train)

    # # make data smaller for faster compile time
    # smallPercent = int(0.50 * len(X_train))
    # for i in range(len(X_train) - smallPercent):
    #     X_train = np.delete(X_train, smallPercent, 0)
    #     y_train = np.delete(y_train, smallPercent,)
    # # print("\nx train smaller:\n", len(X_train))
    # # print("\ny train smaller:\n", len(y_train))

    # # make data smaller for faster compile time
    # smallPercent = int(0.50 * len(X_test))
    # for i in range(len(X_test) - smallPercent):
    #     X_test = np.delete(X_test, smallPercent, 0)
    #     y_test = np.delete(y_test, smallPercent,)
    # # print("\nx train smaller:\n", len(X_train))
    # # print("\ny train smaller:\n", len(y_train))


    clf = BoostingClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc, precision, recall, f1, final_score = evaluation_score(y_pred, y_test)
    print("Accuracy: {}, F-measure: {}, Precision: {}, Recall: {}, Final_Score: {}".format(acc, f1, precision, recall, final_score))
