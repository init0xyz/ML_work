import numpy as np

def computeCost(X,y,theta):
    m = X.size
    now = np.ones(m)
    X = np.append(now,X)
    X = X.reshape((m,2),order = 'F')
    ### 注意X这个matrix需要添加一层bias
    h_val = np.dot(X,theta)
    dif = h_val - y
    val = np.dot(dif.transpose(),dif)
    return (val[0][0])/(2*m)

def gradientDescent(X,y,theta,alpha,num_iters):
    m = X.size
    now = np.ones(m)
    X = np.append(now,X)
    X = X.reshape((m,2),order = 'F')
    for i in range(num_iters):
        h_val = np.dot(X,theta) - y
        theta -= ((alpha)/(m))*np.dot(X.transpose(),h_val)
    return theta

if __name__ == "__main__":
    pass