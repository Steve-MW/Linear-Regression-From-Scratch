from numpy import *
import pandas as pd
import matplotlib.pyplot as plt

# Here we are going to compute the cost function given by
# J(theta0, theta1) = (1/2m)*sum from 1 to m of( (h(x(i))-y(i))^2 )
# That is for each i, we take the difference between the hypothesis h(x) and the actuall
# point in the data set y. This will give us the distance between our predicted point 
# and the actuall point, i.e. the ith point. Also h(x) is essentially the equation for 
# a line i.e. h(x) = y = theta1 * x + theta0
# And finally we the difference for each point and take the sum of it that is Sigma.
# Finally we take the average of it i.e. (1/2m) to give the the totall distance of our
# line h(x) from all the points. Essentially we reduce the cost function to find the best fit. 
def costFunction(theta0, theta1, points):
    J = 0
    m = len(points)
    for i in range (0, len(points)):
        x = points[i,0]
        y = points[i,1]
        #This is essentially the intution of the sigma function, i.e. sum from 1 to m, the differences
        #Between all the points.
        J += ((theta1*x + theta0) - y)**2

    # Note :- never take the avaerage during the computation of the sum 
    # i.e. before taking the Sigma. You have to take the totall sum of all the differences 
    # then you havee to take the average i.e. (1/2m)
    J = J/float(2*m)

    return J

#This where all the magic happens !
'''
    The gradient descent formula used here is 

    repeat till convergance {

        theta0 = theta0 - alpha * ( (1/m) * Sigma from 1 to m ( theta0 + theta1 * x ) - y ) )
        theta1 = theta1 - alpha * ( (1/m) * Sigma from 1 to m ( theta0 + theta1 * x ) - y ) * x )
    }

   or in short 

    repeat till convergance {

        theta0 = theta0 - alpha * ( (1/m) * Sigma from 1 to m ( h(x) ) - y ) )
        theta1 = theta1 - alpha * ( (1/m) * Sigma from 1 to m ( h(x) ) - y ) * x )
    }

    or to be even more short 

    repeat till convergance {

        theta0 = theta0 - alpha * partial differenciation w.r.t theta0 ( J (theta0 , theta1) )
        theta0 = theta0 - alpha * partial differenciation w.r.t theta1 ( J (theta0 , theta1) )
    }

    Note :- perfect definition of this formula in andrew NG course at coursera for Machine Leanring, Cost funtion and Gradient descent 

    so the repeat till convergance part is happening in the gradientDecent function 
    and since we are not doing dynamic convergace i.e. stop the iteration when the distance 
    is the smallest but insted we fixed a total of 1000 iterations and we also fixed the alpha
    at 0.0001. 
    Hence, in the oneGradient function we are doing only a single gradient or to be more specific 
    we find the partial differentitation of J(theta0, theta1) and multiply it with the learning rate.
    Then we multiply it with alpha.
    After this we subtract it with theta0 and theta1 for their respective values. 
    Hence we have new and updated values for theta0 and theta1 which when applied to the intitial cost 
    function, will give us one that is closer to the local minima.
    We will do this over and over again in the gradientDescent function to finally arrive at the local 
    minima. Hence once these local minima values when applied to the function h(x) i.e. the hypothesis, 
    we will get the best fit, or the line that is closest to all the points.
    
    Note:- perfect explaination of this on Coursera, Andrew NG, Machine Learning, Gradient descent with Linear regression <3

    '''

def oneGradient(tempTheta0, tempTheta1,  points, alpha):
    
    tempSigma0 = 0
    tempSigma1 = 0
    m = float(len(points))
    for i in range (0, len(points)):
        x = points[i,0]
        y = points[i,1]
        tempSigma0 += (tempTheta0 + tempTheta1 * x - y )    #This variable can also be called theta0Gradient for better understanding
        tempSigma1 += (tempTheta0 + tempTheta1 * x - y )*x  #This variable can also be called theta1Gradient for better understanding 
    tempSigma0 = (alpha *tempSigma0)/m
    tempSigma1 = (alpha *tempSigma1)/m

    tempTheta0 = tempTheta0 - tempSigma0
    tempTheta1 =  tempTheta1 - tempSigma1

    print("After iteration ", i , " :")
    print(costFunction(tempTheta0, tempTheta1, points))



    return [tempTheta0, tempTheta1]



        
def gradientDescent(points, initialTheta0, initialTheta1, alpha, iterations):
    
    theta0 = initialTheta0
    theta1 = initialTheta1
    for i in range(iterations):
        [theta0,theta1] = oneGradient(theta0,theta1,array(points), alpha)
        

    return theta0,theta1





def run():
    #here we are splitting or simply put parsing the data i.e the relation between test scores and the number of hours studied
    points = genfromtxt('data.csv', delimiter=',')
    #here we are using a static learning rate i.e. Alpha. Keeping a smaller learning rate will prevent the process of 
    #over shooting the convergance point 

    #print(points.shape)
    #print(points)

    alpha = 0.0001
    initialTheta0 = 0 #starting with an initial guess value, better to start with 0
    initialTheta1 = 0 #same case for this variable 

    #These variables are of the form y = mx + b, or to be more matematical y = theta1 * x + b

    #This variable defines the number of iterations we make on the gradient decent update 
    #This is usually done dynamically i.e. done till the point if convergance 

    iterations = 1000

    #Now we feed these variables into the gradient decent function to get the optimun values for 
    #the parameters Theta0 and Theta1

    [theta0, theta1] = gradientDescent(points, initialTheta0, initialTheta1,  alpha,  iterations)
    #print("optimal values for the pareameters after gradient decent :- \n")
    #print("Theta0 = ",theta0,"\n")
    #print("Theta1 = ",theta1,"\n")

    return theta0,theta1






#if '__name__' == '__main__':
def main():
    [theta0, theta1] = run()
    predicting = float(input("Enter the number of hours studied !"))
    print("optimal values for the pareameters after gradient decent :- \n")
    print("Theta0 = ",theta0,"\n")
    print("Theta1 = ",theta1,"\n")
    predicted = theta0 + theta1*predicting
    print("Hours studied = ")
    print(predicted)
    '''
    fig = plt.figure(figsize=(18,9))
    df = pd.read_csv('data2.csv')

    plt.scatter(df.hours, df.marks)
    plt.plot(predicting,predicted,'--')
    plt.show()
    '''

main() 
