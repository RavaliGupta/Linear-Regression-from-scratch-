# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 19:10:41 2016

@author: mamid
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_error_for_graph_given_points(b, m, x,y):
    totalError = 0
    error_features=[]
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    for i in range(1, len(x)):
      feature_x= (x[i]-mean_x)/(max_x-min_x)
      totalError += ((m * feature_x + b)-y[i]) ** 2
     
      error_features.append(totalError / (2*(float(len(x)))))
        #print("length of points:::",float(len(points)))
        #print ("total points are::::",totalError)
       # print("error here is....",totalError/ (2*(float(len(points)))))
    return error_features

def compute_error_for_line_given_points(b, m, x,y):
    totalError = 0
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    for i in range(1, len(x)):
      feature_x= (x[i]-mean_x)/(max_x-min_x)
      totalError += ((m * feature_x + b)-y[i]) ** 2
      #error_features.append(totalError / (2*(float(len(x)))))
        #print("length of points:::",float(len(points)))
        #print ("total points are::::",totalError)
       # print("error here is....",totalError/ (2*(float(len(points)))))
    return totalError / (2*(float(len(x))))
    
    
def step_gradient(b_current, m_current, x,y, learningRate):
    b_gradient = 0
    m_gradient = 0
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    N = float(len(x))
    for i in range(1, len(x)):
        feature_x= (x[i]-mean_x)/(max_x-min_x)
        b_gradient +=  ( ((m_current * feature_x) + b_current)-y[i])
        m_gradient +=  feature_x * (((m_current * feature_x) + b_current)-y[i])
        #print("after calculating b_gradient is:::",b_gradient)
        #print("after calculating m_gradient is:::",m_gradient)
        #print("current value of b is:::",b_current)
        #print("current value of b is:::",m_current)
    new_b = b_current - ((learningRate/N) * b_gradient)
    new_m = m_current - ((learningRate/N) * m_gradient)
   
    return new_b, new_m
    
def gradient_descent_runner(x,y, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, x,y, learning_rate)    
    
    return b, m

def run():
    points = pd.read_csv("BreastCancerData.csv", delimiter=",")
    x = np.array(points['Perimeter'])
    y = np.array(points['Compactness'])
    #print (x,y)
    learning_rate = 0.1
    initial_b = 0.1
    initial_m = 0.1
    num_iterations = 500
    plt.scatter(x, y)

    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, x,y)))
    print ("Running...")  
    b, m = gradient_descent_runner(x,y, initial_b, initial_m, learning_rate, num_iterations)    
    #print (b,m)
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
   
    feature_x= (x-mean_x)/(max_x-min_x)
    
    plt.plot(x, m*feature_x+b, color='r')
    plt.show()
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, x,y)))
#    #  print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations,b,m,compute_error_for_line_given_points(b,m,points)))
#     print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
   # error_values=compute_error_for_graph_given_points(b, m, x,y)
    #print("error values::::", len(error_values))
    error_values= []
    b_cost = 0.1
    m_cost = 0.1
    for i in range(1,num_iterations):
      b_cost, m_cost = step_gradient(b_cost, m_cost, x,y, learning_rate) 
      #print(b_cost,m_cost)
      error_values.append(compute_error_for_line_given_points(b_cost, m_cost, x,y))
      print("error values::::", len(error_values))
      print("error value is::::", error_values[i-1])
    #for i in range(1,num_iterations):
    plt.plot(num_iterations,error_values,color='r')
    plt.show()
    
    
   
if __name__ == '__main__':
    run()

