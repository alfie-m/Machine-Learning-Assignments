from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def main():
    print("********************************************************************************************")
    print("********************************COMP219 Assignment 1****************************************")
    print("********************************************************************************************")
    print("Enter f1 to display dataset information")
    print("Enter f2 to call a machine learning algorithm from sklearn")
    print("Enter f3 to call my imlpementation of a k-nearest neighbor algorithm")
    print("Enter f4 to evaluate the models")
    print("OR enter q to quit")
    select = input("Enter option: ")

    if select == "f1":
        if_f1_selected()

    if select == "f2":
        if_f2_selected()

    elif select == "f3":
        if_f3_selected()

    elif select == "f4":
        if_f4_selected()

    elif select == "q":
        if_q_selected()

    else:
        print("\nInvalid option")
        main()

def if_f1_selected():
    digits = datasets.load_digits()
    print(digits.keys())
    print(digits.DESCR)
    #print(digits.images.shape)
    #print(digits.data.shape)

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_f2_selected():
    print("********************************************************************************************")
    print("Calling machine learning algorithm from sklearn with 0.7/0.3 train to test split")

    digits = datasets.load_digits()

    #use training data set
    X = digits.data
    Y = digits.target
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3) #training with 70% of data

    #create k-NN clasifier with 7 neighbours
    knn = KNeighborsClassifier(n_neighbors=7)

    #fit the classifier to training data
    knn.fit(X_train,Y_train)

    #print accuracy
    accuracy = knn.score(X_test,Y_test) * 100
    print("Training gives a ",accuracy,"percent accurate result")

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()
    else:
        print("\nINVALID INPUT")
        back_to_menu = input("\n\nEnter q to return to menu: ")
        main()

def if_f3_selected():
    print("********************************************************************************************")
    print("Calling my implemtation of a k-nearest neighbor algroithm with 0.7/0.3 train to test split")

    digits = datasets.load_digits()

    #use first 100 images as training data set
    x = 1696 #training with 70% of dataset
    X_train = digits.data[0:x]
    Y_train = digits.target[0:x]

    #testing the data set
    pred = 15
    X_test = digits.data[pred]
    print ("Data has real value of %d" %digits.target[pred])

    #function to calculate distance
    def dist(x,y):
        return np.sqrt(np.sum((x-y)**2))

    #predicting the test data
    l = len(X_train)
    distance = np.zeros(l)
    for i in range(l):
        distance[i] = dist(X_train[i],X_test)
    min_index = np.argmin(distance)
    print ("My knn algorithm redicted value is %d" %Y_train[min_index])

    #accuracy of prediction
    l = len(X_train)
    no_err = 0
    distance = np.zeros(l)
    for j in range(1697,1797): #next 100 data points after training
        X_test = digits.data[j]
        for i in range(l):
            distance[i] = dist(X_train[i], X_test)
        min_index = np.argmin(distance)
        if Y_train[min_index] != digits.target[j]:
            no_err += 1
    accuracy = 100 - no_err
    print ("Training gives a",accuracy,"percent accurate result")

    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()

def if_f4_selected():
    print("********************************************************************************************")
    #allow user to query CNN_models
    print("********************************************************************************************")
    back_to_menu = input("Enter q to return to the menu: ")
    if back_to_menu == "q":
        main()

def if_q_selected():
    exit()

main()
