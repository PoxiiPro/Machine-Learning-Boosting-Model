# import the required packages here
import numpy as np

def mostFrequent(list):
    count = 0
    num = list[0]
     
    for i in list:
        curr_frequency = list.count(i)
        if(curr_frequency > count):
            count = curr_frequency
            num = i
 
    return num

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):

    # read in our data files
    xTrain = np.loadtxt(Xtrain_file, delimiter = ",")
    yTrain = np.loadtxt(Ytrain_file)

    # print("print test after reading in files, xTrain: ", xTrain, "\nyTrain: ", yTrain)
    # print("ytrain test index", yTrain[10]) 

    testData = np.loadtxt(test_data_file, delimiter = ",")
    # print("testData: ", testData)

    # create zero arrays for the distances and the distance labels
    d = np.zeros((len(xTrain), len(testData)))
    dLabels = np.zeros_like(d)

    # print("distance arrays test, distances: ", d, "\ndLabels: ", dLabels)

    # for each point in the test data, find the euclidean distance between each testData point and training point x
    # row i is test data point
    # col j is xTrain data point
    for i, t in enumerate(testData): 
        for j, x in enumerate(xTrain): 
            # euclidean distance between testData and xTrain data
            ed = np.linalg.norm(t - x)

            # store the euclidean distances
            d[j][i] = ed

            # Set the corresponding label in d_labels as the corresponding value in train_labels - different classes 0 - 10 based on the j point - x train data point
            dLabels[j][i] = yTrain[j]

    # print("d:\n", d)
    # print("dLabels:\n", dLabels)

    #  k nearest neighbors, k can vary depending on what we want to use and changes underfitting/overfitting
    k = 10

    # empty list for predicitons
    # pred = []
    prediction = []

    # loop through test data
    for i, t in enumerate(testData):

        # set only col value to the variables
        distances = d[:, i]
        labels = dLabels[:, i]

        # print("distances: ", distances)
        # print("labels: ", dLabels)

        # Sort distances and labels from least distance to greatest distance. 
        distances, labels = zip(*sorted(zip(distances, labels)))

        pred = []
        # Grab the k lowest values in distances list aka nearest neighbors and get their labels
        for i in range(k):
            pred.append(labels[i])
        
        # frequent = mostFrequent(pred)
        # for i in range(k):
        #     if frequent == distances[i]:
        #         neededLabel = labels[i]
        #         break
    
        prediction.append(mostFrequent(pred))
        # prediction.append(sum(pred) / k)

    # print("\ndistances index test:\n", distances[1])
    # print("\ndistances after sort:\n", distances)

    # print("labels after sort:\n", dLabels)
    print("\npred test:\n", pred)

    # print("\nprediciton test:\n", prediction)
    # print(prediction)

    np.savetxt(pred_file, prediction, fmt = '%1d', delimiter = ",")

if __name__ == "__main__": 
    Xtrain_file = 'Xtrain.csv'
    Ytrain_file = 'Ytrain.csv'
    pred_file = 'predictions.txt'
    test_input_dir = 'Xtrain.csv'


    xTrain = np.loadtxt(Xtrain_file, delimiter = ",")
    
    # use only half of the data
    # for i in range(20):
    #     xTrain = np.delete(xTrain, 20, 0)
    # print(xTrain)
    half_Of_X_Train = 'halfOfxTrain'
    # np.savetxt(half_Of_X_Train, xTrain, fmt = '%1d', delimiter = ",")
    
    # //////////////////////// Accuracy Test //////////////////////
    run(Xtrain_file, Ytrain_file, half_Of_X_Train, pred_file)
    pred = np.loadtxt(pred_file, skiprows = 0)
    actual = np.loadtxt(Ytrain_file, delimiter = ",")

    tp = 0
    tn = 0

    fp = 0
    fn = 0

    for a, pre in zip(actual, pred): 
      if a == 1 and pre == 1: 
        tp += 1
      elif a == 1 and pre == 0: 
        fn += 1
      elif a == 0 and pre == 1: 
        fp += 1
      else: 
        tn += 1
        
    accuracy = round(100 * (tp + tn) / (tp + fp + tn + fn), 4)

    print("\n---- Dataset ----")
    print("Accuracy: %s" % accuracy)
    print("TP : %i ; FP : %i" % (tp, fp))
    print("TN : %i ; FN : %i" % (tn, fn))
    print("\n")