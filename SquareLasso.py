import pandas as pd
import numpy as np
import picos
import math
import scipy.stats 
import seaborn as sns
from picos import RealVariable, BinaryVariable
from picos import Constant
from picos import Problem
import cvxopt as cvx
import matplotlib.pyplot as plt

def Threshold(z,gamma):
    if abs(z)<=gamma:
        y=0
    elif (abs(z)>gamma) & (z>0):
        y=z-gamma
    else:
        y=z+gamma
    return y

def square_CD(X,y,lambda_):
# square root lasso for linear regression using coodinate descent method
# input:X,y,penalty level lambda
# ouput:coef of the model
    n=X.shape[0]
    p=X.shape[1]
    beta=np.zeros(p)
    eps=1e-6
    max_iter=50
    temp=beta.copy()
    for i in range(max_iter):
        for j in range(p):
            s1=np.linalg.norm(X[:,j])
            s2=np.dot((y-np.dot(X,beta)),X[:,j])+s1**2*beta[j]
            s3=np.linalg.norm(y-np.dot(X,beta))*lambda_
            beta[j]=Threshold(s2/s1**2,s3/s1**2)
        dbeta=np.linalg.norm(beta-temp)
        if dbeta<eps:
            break
        temp=beta.copy()
    return beta


def square_lasso(X,y,lambda_):
# square root lasso for linear regression using SOCP
# input:X,y,penalty level-lambda
# ouput:coef of the model   
    P2=Problem()
    n=X.shape[0]
    p=X.shape[1]
    belta_plus=RealVariable("belta+",p)
    belta_minus=RealVariable("belta-",p)
    y=cvx.matrix(y)
    t=RealVariable("t",1)
    x=[Constant('x[{0}].T'.format(i), X[i,:]) for i in range(n)]
    y=[Constant('y[{0}]'.format(i), y[i]) for i in range(n)]
    v = RealVariable('V',n)
    P2.add_list_of_constraints([(y[i]-x[i]*belta_plus+x[i]*belta_minus==v[i])  for i in range(n)])
    P2.add_constraint(abs(v)<=t)
    P2.add_constraint(belta_plus>=0)
    P2.add_constraint(belta_minus>=0)
    list1=np.zeros(p)
    list2=np.zeros(p)
    list1[::]=lambda_/n
    l1=Constant('Lambda/n',list1)
    P2.set_objective('min',(t/10+(l1|belta_plus)+(l1|belta_minus)))
    P2.solve()
    belta=belta_plus-belta_minus
    belta=np.array([belta[i].value for i in range(p)])
    return belta
