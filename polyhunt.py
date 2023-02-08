# This is a sample Python script.
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import argparse
from pathlib import Path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def k_fold_cross_validation(M, gamma, data, autofit):
    avg_accuracy = 0
    train_data = []
    test_data = []
    for i in range(0, 5):
        for index, j in data.iterrows():
            if index % 5 == i:
                temp = [j[0],j[1]]
                test_data.append(temp)
            else:
                temp = [j[0], j[1]]
                train_data.append(temp)
        w = calc_weights(M, gamma, train_data)
        avg_accuracy += ERMS(M,w, test_data, gamma)
        #avg_accuracy += squared_error(M,w, test_data, gamma)
    avg_accuracy = avg_accuracy * 0.2
    print("model of degree "+str(M))
    print(w)
    if not autofit: plot_model(M, w, np.array(data))
    return avg_accuracy


def squared_error(M,w, data, gamma):
    error = 0
    for i in data:
       model_val = predict(M,w,i[0])
       real_val = i[1]
       temp1 =(model_val-real_val)**2
       temp2 = gamma* np.linalg.norm(w)
       error += temp1 + temp2

       # error += (pow((predict(w, i[0]) - i[1]), 2) + (gamma * np.dot(w.transpose(), w)))
    return 0.5 * error

def ERMS(M,w,data,gamma):
    return math.sqrt(2*squared_error(M,w,data,gamma)/len(data))

def predict(M,w, data):
   phi = build_phi(M,data)
   return np.dot(w,phi.transpose())

def print_model(w):
    print(w)

def train(M, gamma, trainpath,autofit):
    data = get_data(trainpath)
    return (k_fold_cross_validation(M,gamma,data,autofit))

def build_phi(M, data_point):
    phi = []
    for i in range (0,M+1):
        phi.append(data_point**i)
    phi = np.array(phi)
    return phi
def build_dm(M, data):
    dm = []
    for i in range(0,M+1):
        row = []
        for j in data:
            row.append(pow(j[0],i))
        dm.append(row)
    design_matrix = np.matrix(dm)
    return design_matrix


def get_data(trainpath):
    data = pd.read_csv(trainpath, header=None,
                 names=["X", "Y"], dtype = float)
    return data


def calc_weights(M, gamma, data):
    y = data
    t = np.matrix(y)
    design_matrix = build_dm(M, data)
    regularizer = np.identity(M+1) * gamma
    inverse = (regularizer + design_matrix * design_matrix.transpose())
    inverse = np.linalg.inv(inverse)
    w = inverse * design_matrix * t[:,1]
    ## w = (λI + (Φ^T)Φ)^-1 (Φ^T)t.
    w = np.ravel(w)
    return w


def fit_model(info, autofit, M, gamma, trainpath):
    if info:
        print("Gus Vietze: gvietze@u.rochester.edu")
    if autofit:
        error = []
        min_error = np.inf
        min_error_M = 0
        best_model = []
        for i in range(1, M + 1):
            mth_error = train(i, gamma, trainpath,autofit)
            error.append(mth_error)
            if mth_error < min_error:
                min_error =mth_error
                min_error_M = i
        plot_error(error,M)
        data = get_data(trainpath)
        data = np.array(data)
        plot_model(min_error_M,calc_weights(min_error_M,gamma,data),data)
    else:
            train(M, gamma, trainpath,autofit)

def get_output(M,w,data):
    output = []
    for point in data:
        output.append(predict(M,w,point[0]))
    return output

def plot_model(M, w, data):
   output =  get_output(M,w,data)
   data = np.array(data)
   plt.plot(data[:,0],output, '-')
   plt.plot(data[:,0],data[:,1],'.')
   plt.xlabel("X")
   plt.ylabel("Y")
   plt.title("Model Function of Degree "+ str(M))
   plt.show()
def plot_error(error, M):
    degree = list(range(1,M+1))
    plt.plot(degree, error, '-o')
    plt.xlabel("Polynomial Degree")
    plt.ylabel("ERMS")
    plt.title("Error by M")
    plt.show()


# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type = Path)
    parser.add_argument("M", type = int)
    parser.add_argument("gamma", type = float)
    parser.add_argument("auftofit", type = bool)
    parser.add_argument("info", type = bool)
    p = parser.parse_args()
    fit_model(p.info,p.auftofit, p.M , p.gamma, p.file_path)

# See PyCharm help at https://www.sjetbrains.com/help/pycharm/
