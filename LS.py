from matplotlib import interactive
interactive(True)
import sqlite3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys
import os
from sklearn import preprocessing
from sklearn.model_selection import KFold
sys.path.append(os.path.abspath("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\cudart64_101.dll"))

#Create a connection with the database
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)
    return conn

#Run sql queries
def Make_Query(conn,query):
    cur = conn.cursor()
    cur.execute(query)

    rows = cur.fetchall()
    results = []
    for row in rows:
        results.append(row)
    return results

#Vecrors Ψκ=[] to rows
#and labels= [H,D,A,A,D...]
#where H = home team win
#D = draw
#A = away team win
def Data_Labels(data):
    data_ = []
    labels = []
    k = []
    for i in range(len(data)):
        data_.append([data[i][4], data[i][5], data[i][6]])
        k.append("B365")
        data_.append([data[i][7], data[i][8], data[i][9]])
        k.append("BW")
        data_.append([data[i][10], data[i][11], data[i][12]])
        k.append("IW")
        data_.append([data[i][13], data[i][14], data[i][15]])
        k.append("LB")
        if data[i][2] > data[i][3]:
            labels.append("H")
            labels.append("H")
            labels.append("H")
            labels.append("H")
        elif data[i][2] < data[i][3]:
            labels.append("A")
            labels.append("A")
            labels.append("A")
            labels.append("A")
        else:
            labels.append("D")
            labels.append("D")
            labels.append("D")
            labels.append("D")
    return data_, labels,k

#First all Ψκ vectors with H label then with D and then with A
#Create Y array where H=[1,0,0],D=[0,1,0],A=[0,0,1]...Υ=[[1,0,0]...,[0,1,0]...,[0,0,1],...]
def Least_square_X_Y(data, labels):
    x_array_home = []
    x_array_away = []
    x_array_draw = []
    y_array_home = []
    y_array_away = []
    y_array_draw = []
    for i in range(len(data)):
        if labels[i] is 'H':
            x_array_home.append(data[i])
            y_array_home.append([1, 0, 0])
        elif labels[i] is 'D':
            x_array_draw.append(data[i])
            y_array_draw.append([0, 1, 0])
        else:
            x_array_away.append(data[i])
            y_array_away.append([0, 0, 1])
    return x_array_home+x_array_draw+x_array_away, y_array_home+y_array_draw+y_array_away

#Create a plot with the points and the 3 lines.
def plot(XT,YT,AT):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = []
    y = []
    z = []
    count = 0
    for i in range(len(XT)):
        if YT[i][1] == 1 and count==0:
            ax.scatter(x, y, z, c='r', marker='o')
            x=[]
            y=[]
            z=[]
            count +=1
        if YT[i][2] == 1 and count==1:
            ax.scatter(x, y, z, c='g', marker='v')
            x=[]
            y=[]
            z=[]
            count +=1
        x.append(XT[i][0])
        y.append(XT[i][1])
        z.append(XT[i][2])
    plt.interactive(True)
    ax.scatter(x,y,z,c='b',marker='.')
    w1 = np.linspace(-15, 35, 100)
    w2 = np.linspace(-15, 35, 100)
    y1 = (AT[0][0]*w1 + AT[0][1]*w2  + AT[0][3])/(-AT[0][2])
    y2 = (AT[1][0]*w1 + AT[1][1]*w2  + AT[1][3])/(-AT[1][2])
    y3 = (AT[2][0]*w1 + AT[2][1]*w2  + AT[2][3])/(-AT[2][2])
    plt.plot(w1, w2, y1, '--y', label='y1',linewidth=4)
    plt.plot(w1, w2, y2, '-.k', label='y2',linewidth=4)
    plt.plot(w1, w2, y3, ':m', label='y3',linewidth=4)
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    plt.legend(loc='upper left')
    plt.show(10)

#Implamantation of least square using tensorflow
def Least_Square_Classifier(X,Y):
    data1 = np.array(X)
    data2 = np.array(Y)
    X_tensor = tf.constant(data1,dtype=tf.float32)
    Y_tensor = tf.constant(data2,dtype=tf.float32)
    tX_X = tf.matmul(tf.transpose(X_tensor), X_tensor)
    X_inv = tf.linalg.inv(tf.transpose(tX_X))
    product = tf.matmul(X_inv, tf.transpose(X_tensor))
    A = tf.matmul(product, Y_tensor)
    print("W=",A)
    tmp = A.numpy()
    print("Line H vs D,A is: "+(str(tmp[0][0])+"*x1 + "+str(tmp[1][0])+"*x2 + "+ str(tmp[2][0])+"*x3 + "+str(tmp[3][0])))
    print("Line D vs H,A is: "+(str(tmp[0][1])+"*x1 + "+str(tmp[1][1])+"*x2 + "+ str(tmp[2][1])+"*x3 + "+str(tmp[3][1])))
    print("Line A vs H,D is: "+(str(tmp[0][2])+"*x1 + "+str(tmp[1][2])+"*x2 + "+ str(tmp[2][2])+"*x3 + "+str(tmp[3][2])))
    #plot(X,Y,tf.transpose(A))
    tmp = tf.transpose(A).numpy()
    return tmp

#Add one more dimension to the vectors of the matrix X.
def Add_Ones(vectors_array):
    for i in range(len(vectors_array)):
        vectors_array[i].append(1)
    return vectors_array
def get_w(W):
    w1 = [W[0][0], W[0][1], W[0][2], W[0][3]]
    w2 = [W[1][0], W[1][1], W[1][2], W[1][3]]
    w3 = [W[2][0], W[2][1], W[2][2], W[2][3]]
    return w1,w2,w3
def x_per_company(data,labels,k,company):
    x = []
    y = []
    for i in range(len(data)):
        if k[i] is company:
            x.append(data[i])
            y.append(labels[i])
    return x,y

#Calculate of MSE (sum of error squares)
#for each line
def LS_JW(X,Y,W):
    w1,w2,w3 = get_w(W)
    jw1 = 0
    jw2 = 0
    jw3 = 0
    for i in range(len(X)):
        jw1 += (Y[i][0] - np.matmul(np.array(X[i]),np.array(w1)))**2
        jw2 += (Y[i][1] - np.matmul(np.array(X[i]),np.array(w2)))**2
        jw3 += (Y[i][2] - np.matmul(np.array(X[i]),np.array(w3)))**2
    print ("J(w) for line w1: "+str(jw1))
    print ("J(w) for line w2: "+str(jw2))
    print ("J(w) for line w3: "+str(jw3))
    print("Sum: "+str(jw1+jw2+jw3))
    return jw1, jw2, jw3

#Return the accuracy of the classifier
def ls_accuracy(X,W,Y):
        predictions = np.dot(np.array(X),np.transpose(np.array(W)))
        print(predictions)
        preds_correct_boolean = np.argmax(predictions, 1) == np.argmax(Y, 1)
        correct_predictions = np.sum(preds_correct_boolean)
        accuracy = 100.0 * correct_predictions / predictions.shape[0]
        print("-----------------------------------------------------")
        print("The accuracy is: ", accuracy, "%")
        print("-----------------------------------------------------")
        return accuracy

#Implement 10 K fold validation
def K_Fold_Validation(data,labels,N):
    data_array = np.array(data)
    labels_array = np.array(labels)
    kf = KFold(n_splits=N)
    lines = []
    best_train_data_x = []
    best_train_data_y = []
    fold_accuracy = []
    for train_index, test_index in kf.split(data):
        train_data_x, test_data_x = data_array[train_index],data_array[test_index]
        train_data_y, test_data_y = labels_array[train_index],labels_array[test_index]
        X, Y = Least_square_X_Y(np.array(train_data_x).tolist(),np.array(train_data_y).tolist())
        best_train_data_x.append(X)
        best_train_data_y.append(Y)
        print("Training...")
        W = Least_Square_Classifier(Add_Ones(X), Y)
        ls_accuracy(X, W, Y)
        LS_JW(X, Y, W)
        lines.append(W)
        print("Testing...")
        X_test,Y_test= Least_square_X_Y(np.array(test_data_x).tolist(),np.array(test_data_y).tolist())
        LS_JW(Add_Ones(X_test),Y_test,W)
        fold_accuracy.append(ls_accuracy(X_test, W, Y_test))
    print("The best of 10 folds with the best accuracy is: ",fold_accuracy.index(max(fold_accuracy))+1)
    print("With accuracy ",fold_accuracy[fold_accuracy.index(max(fold_accuracy))],"%")
    print("With weight matrix W= ",lines[fold_accuracy.index(max(fold_accuracy))])
    plot(best_train_data_x[fold_accuracy.index(max(fold_accuracy))], best_train_data_y[fold_accuracy.index(max(fold_accuracy))], lines[fold_accuracy.index(max(fold_accuracy))])
    folds_average = 0
    for i in range(len(fold_accuracy)):
        folds_average += fold_accuracy[i]
    return folds_average/len(fold_accuracy)

#Find the company with the best 
#clasification accuracy based on the mean
#accuracy of the 10 k fold validation
def best_company(avrg_jw,names):
    for i in range(len(avrg_jw)):
        print("For the company ",names[i])
        print("The clasification accuracy was:",avrg_jw[i],"%")
    print("The company with the best clasification accurasy is ",names[avrg.index(max(avrg))])

sess = tf.compat.v1.Session()
Match_data = Make_Query(create_connection("database.sqlite"),"SELECT home_team_api_id, away_team_api_id, home_team_goal, away_team_goal,B365H, B365D, B365A, BWH, BWD, BWA, IWH, IWD, IWA, LBH, LBD, LBA FROM Match WHERE B365H != 0 AND B365D != 0 AND B365A != 0 AND BWH != 0 AND BWD != 0 AND BWA != 0 AND IWH != 0 AND IWD != 0 AND IWA != 0 AND LBH != 0 AND LBD != 0 AND LBA;")
data,labels,k = Data_Labels(Match_data)
company_names = ["B365", "BW", "IW", "LB"]
avrg = []
for i in range(len(company_names)):
    print("For company with name: ",company_names[i])
    x,y = x_per_company(preprocessing.scale(data),labels,k,company_names[i])
    avrg.append(K_Fold_Validation(x,y,10))
best_company(avrg,company_names)

