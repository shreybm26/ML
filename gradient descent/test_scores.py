import numpy as np
import pandas as pd

def gradient_descent(x,y):
    m_curr=np.random.rand()
    b_curr=np.random.rand()
    iterations=1000
    n=len(x)
    learning_rate=0.001

    for i in range(iterations):
        y_predicted=m_curr*x+b_curr
        cost=1/n*sum(val**2 for val in (y-y_predicted))
        md=-2/n*sum(x*(y-y_predicted))
        bd=-2/n*sum(y-y_predicted)
        m_curr=m_curr-learning_rate*md
        b_curr=b_curr-learning_rate*bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
    
df=pd.read_csv("D:\Programs\Code\ML\gradient descent\scores_data.csv")
x=np.array(df['math'])
y=np.array(df['cs'])

gradient_descent(x,y)