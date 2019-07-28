import pandas as pd
from sklearn.model_selection import train_test_split
import math



def importdata():
    balance_data = pd.read_csv('data.txt', delimiter="\t", header=None, names=[
        "PATIENT ID", "CHEST PAIN", "MALE", "SMOKES", "EXERCISES", "ATTACK"])
    # Printing the dataswet shape
    # print("Dataset Lenght: ", len(balance_data))
    # print("Dataset Shape: ", balance_data.shape)
    # Printing the dataset obseravtions
    # print("Dataset: ", balance_data)
    # print specific column unique data
    # print("Male:", balance_data['MALE'].unique())
    return balance_data


def splitdataset(balance_data):
    # Seperating the target variable
    X = balance_data.values[:, 0:5]
    Y = balance_data.values[:, 5]
    # Spliting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test


def sum(*numbers):
    addition = 0
    for number in numbers:
        addition += number
    return addition


def probablity(*numbers):
    total = sum(*numbers)
    resultantProbablity = []
    for number in numbers:
        resultantProbablity.append(number/total)
    return resultantProbablity


def entropy(*numbers):
    probablities = probablity(*numbers)
    resultEntropy = 0
    for p in probablities:
        if p == 0:
            continue
        else:
            resultEntropy += p*math.log(p, 2)
    return -1*resultEntropy


def findEntropy(data, attribute, resultAttribute):
    attributeData = data[attribute]
    resultData = data[resultAttribute]
    no_of_options = len(attributeData.unique())

    dataForEntropy = [[0 for x in range(no_of_options)] for y in range(2)]

    # assigning yes as 0 and no as 1 informally
    i = 0
    for selection in attributeData:
        if(selection == "yes"):
            if(resultData[i] == "yes"):
                dataForEntropy[0][0] += 1
            else:
                dataForEntropy[0][1] += 1
        else:
            if(resultData[i] == "yes"):
                dataForEntropy[1][0] += 1
            else:
                dataForEntropy[1][1] += 1
        i += 1

    option1 = dataForEntropy[0][0] + dataForEntropy[0][1]
    option2 = dataForEntropy[1][0] + dataForEntropy[1][1]
    prob = probablity(option1, option2)
    return prob[0]*entropy(dataForEntropy[0][0], dataForEntropy[0][1]) + prob[1]*entropy(dataForEntropy[1][0], dataForEntropy[1][1])


# def conditionalEntropy(uniques, y_train):
#     YesCount = 0
#     NoCount = 0
#     for y in y_train:
#         if(y == 'yes'):
#             YesCount += 1
#         else:
#             NoCount += 1


def main():
    data = importdata()
    # X, Y, x_train, x_test, y_train, y_test = splitdataset(data)
    # print(probablity(1, 3, 6))
    # print(entropy(2, 4))
    # print(findEntropy(data, "MALE", "ATTACK"))
    data.columns


# Calling main function
if __name__ == "__main__":
    main()
