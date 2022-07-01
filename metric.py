import numpy as np

def RMSE(R, pred):
    
    rx, ry = R.nonzero() 
    loss = 0
        
    for x, y in zip(rx, ry):
        loss += pow(R[x, y] - pred[x, y], 2)
                
    return np.sqrt(loss/len(rx))