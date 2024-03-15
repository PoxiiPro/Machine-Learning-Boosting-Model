import numpy as np
# from numpy import sqrt

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def discriminant(dataPoint, centroid0, centroid1):

    # Euclidean distance between each centroid and data point using the 

    # class0Discriminant = sqrt(pow(dataPoint[0] - centroid0[0], 2) + pow(dataPoint[1] - centroid0[1], 2) + pow(dataPoint[2] - centroid0[2], 2) + pow(dataPoint[3] - centroid0[3], 2))
    # class1Discriminant = sqrt(pow(dataPoint[0] - centroid1[0], 2) + pow(dataPoint[1] - centroid1[1], 2) + pow(dataPoint[2] - centroid1[2], 2) + pow(dataPoint[3] - centroid1[3], 2))
    # class2Discriminant = sqrt(pow(dataPoint[0] - centroid2[0], 2) + pow(dataPoint[1] - centroid2[1], 2) + pow(dataPoint[2] - centroid2[2], 2) + pow(dataPoint[3] - centroid2[3], 2))
    # class3Discriminant = sqrt(pow(dataPoint[0] - centroid3[0], 2) + pow(dataPoint[1] - centroid3[1], 2) + pow(dataPoint[2] - centroid3[2], 2) + pow(dataPoint[3] - centroid3[3], 2))

    # using a diff method to get the L2 norm
    class0Discriminant = np.linalg.norm(centroid0 - dataPoint)
    class1Discriminant = np.linalg.norm(centroid1 - dataPoint)
    # class2Discriminant = np.linalg.norm(centroid2 - dataPoint)

    # checks for a 4th feature
    # if centroid0[0][0] != 0:

    # print(class0Discriminant) 

    # variable to see how many cnetroids are being passed into the function, first arg is alwasy data, all the following are centroids
    # numCentroids = len(args - 1)

    # creating lists to hold distance of centroids
    # centroidDistance = [[] for x in range(numCentroids)]

    # loop to calculate the Euclidean distance for every centroid we have 
    # for i in range(numCentroids):
    #   centroidDistance[i] = sqrt(pow(args[0][0] - args[i][0], 2) + pow(args[0][1] - args[i][1], 2) + pow(args[0][2] - args[i][2], 2))

    # print("class 0 dis: ", class0Discriminant, "class 1 dis: ", class1Discriminant, "class 2 dis: ", class2Discriminant)

    # if the distance between the class 0 centroid and the data point is the smallest distance then return 0
    if  class0Discriminant < class1Discriminant:
        # print("if 1")
        # print(class0Discriminant, "<", class1Discriminant, "and", class0Discriminant, "<", class2Discriminant)
        return 0
    else:  # else if the distance between the class 1 centroid and the data point is < distance between the class 2 centroid and data point then return 1
        # print("elif 2")
        # print(class1Discriminant, "<", class2Discriminant)
        return 1 
    # else:
    #     return 2

def run(train_input_dir, train_label_dir, test_input_dir, pred_file):
    
    # using np.loadtxt (np arrays), load the data from data files: train_input_dir and train_label_dir
    train_input = np.loadtxt(train_input_dir, skiprows = 0)
    train_label = np.loadtxt(train_label_dir, skiprows = 0)

    # test printing
    # print("list from files test np")
    # print(train_input)
    # print("Train label")
    # print(train_label)

    # # convert the np arrays into lists
    # train_input = list(train_input)
    # train_label = list(train_label)

    # # test printing
    # print("list from files test")
    # print(train_input)
    # print(train_label)

    # creating 2 separate arrays to store data based on same label type
    class0 = []
    class1 = []
    # class2 = []   - hw 4 doesnt have 3 classes

    # print(len(train_label))
    # loop to go thru training label array and sort the data into the proper label array
    for i in range(len(train_label)):
        if train_label[i] == 0:
            class0.append(train_input[i])
        elif train_label[i] == 1:
            class1.append(train_input[i])
        # elif train_label[i] == 2:
        #     class2.append(train_input[i])

    # test printing
    # print("print sorted lists test")
    # print("Class 0: \n", class0, "\n")
    # print("Class 1: \n", class1, "\n")
    # print("Class 2: \n", class2, "\n")

    # convert class lists into np arrays
    class0 = np.array(class0)
    class1 = np.array(class1)
    # class2 = np.array(class2)

    # test printing
    # print("print sorted lists test np")
    # print("Class 0: \n", class0, "\n")
    # print("Class 1: \n", class1, "\n")
    # print("Class 2: \n", class2, "\n")

    # centroid: sum of all values in a class, then divide by total number of values in that class cluster aka the mean
    centroidClass0 = np.mean(class0, axis = 0)
    centroidClass1 = np.mean(class1, axis = 0)
    # centroidClass2 = np.mean(class2, axis = 0)

    # centroid test printing
    # print("print sorted lists test np")
    # print("Class 0 Centroid: \n", centroidClass0, "\n")
    # print("Class 1 Centroid: \n", centroidClass1, "\n")
    # print("Class 2 Centroid: \n", centroidClass2, "\n")

    # usw the discriminant function to calculate the prediction for each point in the given testing data by calculating the distance between the point and the three centroids
    # Reading data
    test_data = np.loadtxt(test_input_dir , skiprows = 0)

    prediction = np.array([discriminant(data, centroidClass0, centroidClass1) for data in test_data], dtype = object)
    # print(prediction)
    [num, _] = test_data.shape

    prediction = np.zeros((num, 1), dtype = np.int16)

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter = ",")
    
# if statement to test code
if __name__ == "__main__":
    train_input_dir = 'training1.txt'
    train_label_dir = 'training1_label.txt'
    test_input_dir = 'testing1.txt'
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)