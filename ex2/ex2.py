import numpy as np

def sigmoid(z : np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + np.exp(-z))

X = np.array([])
y = np.array([])

def init(X_matrix,y_matrix):
    global X,y
    (m,n) = X_matrix.shape
    X_matrix = X_matrix.reshape((1,X_matrix.size),order = 'F')    
    X_matrix = np.append(np.ones((1,m)),X_matrix)
    X_matrix = X_matrix.reshape((m,n+1),order = 'F')    
    X = X_matrix
    y = y_matrix
    y = y.reshape((y.size,1))


def costFunction(theta):
    theta = theta.reshape((theta.size,1))
    (m,n) = X.shape
    h_theta = sigmoid(np.dot(X,theta))
    cost = sum(sum(-y*np.log(h_theta) - (1-y)*np.log(1-h_theta)))/m
    return cost

def Gradient(theta):
    theta = theta.reshape((theta.size,1))
    (m,n) = X.shape
    h_theta = sigmoid(np.dot(X,theta))
    grad = np.dot(X.transpose(),h_theta-y)/m
    grad = grad.reshape(n)
    return grad

    
if __name__ == "__main__":
    pass