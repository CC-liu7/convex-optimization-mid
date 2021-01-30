import numpy as np
def Threshold(z,gamma):
    if abs(z)<=gamma:
        y=0
    elif (abs(z)>gamma) & (z>0):
        y=z-gamma
    else:
        y=z+gamma
    return y


def Lasso(X,y,lambda_):
# input:X,y,penalty level-lambda
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
            beta[j]=Threshold(s2/s1**2,lambda_/s1**2)
        dbeta=np.linalg.norm(beta-temp)
        if dbeta<eps:
            break
        temp=beta.copy()
    return beta
 
