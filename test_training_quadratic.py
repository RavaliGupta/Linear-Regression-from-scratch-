from numpy import *
import matplotlib.pyplot as plt
import numpy as np

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(thetha0, thetha1, thetha2, x,y):
    totalError = 0
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    print("max value of x is::",max_x)
    print("min value of x is::",min_x)
    print("mean value of x is::",mean_x)
    for i in range(1, len(x)):
       feature_x= (x[i]-mean_x)/(max_x-min_x)
       # print("total error before:::",totalError)
       # print("error calculated:::",( (m * x + b)-y) ** 2)
       totalError += (y[i] - (thetha2*(feature_x ** 2)+thetha1 * feature_x + thetha0)) ** 2
        #print("length of points:::",float(len(points)))
        #print ("total points are::::",totalError)
       #print("error here is....",totalError/ (2*(float(len(x)))))
    return totalError / (2*(float(len(x))))

def step_gradient(thetha0_current, thetha1_current, thetha2_current, x,y, learningRate):
    thetha0_gradient = 0
    thetha1_gradient = 0
    thetha2_gradient = 0
    N = float(len(x))
    max_x=np.amax(x)
    min_x=np.amin(x)
    mean_x=np.mean(x)
    for i in range(1, len(x)):
        feature_x= (x[i]-mean_x)/(max_x-min_x)
      #  print("brfore calculating b_gradient is:::",b_gradient)
       # print("brfore calculating m_gradient is:::",m_gradient)
        thetha0_gradient +=  ( (((thetha2_current*(feature_x ** 2))) +(thetha1_current * feature_x) + thetha0_current)-y[i])
        thetha1_gradient +=  feature_x * (((thetha2_current*(feature_x ** 2)) +(thetha1_current * feature_x) + thetha0_current)-y[i])
        thetha2_gradient +=  (feature_x ** 2) * (((thetha2_current*(feature_x ** 2)) +(thetha1_current * feature_x) + thetha0_current)-y[i])
        #print("after calculating b_gradient is:::",b_gradient)
        #print("after calculating m_gradient is:::",m_gradient)
        #print("current value of b is:::",b_current)
        #print("current value of b is:::",m_current)
    new_thetha0 = thetha0_current - ((learningRate/N) * thetha0_gradient)
    new_thetha1 = thetha1_current - ((learningRate/N) * thetha1_gradient)
    new_thetha2 = thetha2_current - ((learningRate/N) * thetha2_gradient)
    #new_m = m_current - ((learningRate/N) * m_gradient)
   
    return [new_thetha0, new_thetha1, new_thetha2]

def gradient_descent_runner(x,y, starting_thetha0, starting_thetha1, starting_thetha2, learning_rate, num_iterations):
    thetha0 = starting_thetha0
    thetha1 = starting_thetha1
    thetha2 = starting_thetha2
    for i in range(num_iterations):
        thetha0, thetha1, thetha2 = step_gradient(thetha0, thetha1, thetha2, x,y, learning_rate)
              
    return [thetha0, thetha1, thetha2]

def run():
    points = pd.read_csv("testdata.csv", delimiter=",")
    points_training = pd.read_csv("trainingdata.csv", delimiter=",")
    x = np.array(points['Perimeter'])
    y = np.array(points['Compactness'])
    x_training = np.array(points_training['Perimeter'])
    y_training = np.array(points_training['Compactness'])
    #print (x,y)
    learning_rate = 0.1
    initial_thetha0 = 0.15
    initial_thetha1 = 0.15
    initial_thetha2 = 0.15
    num_iterations = 1000
    plt.scatter(x_training, y_training)

    print ("Starting gradient descent at thetha0 = {0}, thetha1 = {1}, thetha2 = {2}, error = {3}".format(initial_thetha0, initial_thetha1, initial_thetha2, compute_error_for_line_given_points(initial_thetha0, initial_thetha1, initial_thetha2, x,y)))
    print ("Running...")
    [thetha0, thetha1, thetha2] = gradient_descent_runner(x_training,y_training, initial_thetha0, initial_thetha1, initial_thetha2, learning_rate, num_iterations)
    
    max_x_training=np.amax(x_training)
    min_x_training=np.amin(x_training)
    mean_x_training=np.mean(x_training)
   
    feature_x_training= (x-mean_x_training)/(max_x_training-min_x_training)
    plt.plot(x, thetha2*feature_x_training*feature_x_training+thetha1*feature_x_training+thetha0, color='r')
    plt.show()
    print ("After {0} iterations thetha0 = {1}, thetha1 = {2}, thetha2 = {3}, error = {4}".format(num_iterations, thetha0, thetha1,thetha2, compute_error_for_line_given_points(thetha0, thetha1, thetha2, x,y)))

if __name__ == '__main__':
    run()

print("hello, inside gradient descent");
