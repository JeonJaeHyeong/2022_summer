import numpy as np
import torch

def RMSE(R, pred):
    
    rx, ry = R.nonzero() 
    loss = 0
        
    for x, y in zip(rx, ry):
        loss += pow(R[x, y] - pred[x, y], 2)
                
    return np.sqrt(loss/len(rx))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def hit_ndcg(model, test_data):
    
    hit = 0
    ndcg = 0
    
    for i in range(len(test_data)):
        test_pred = torch.FloatTensor(model(test_data[i][0], test_data[i][1])).view(-1, 1)
        neg_pred = torch.FloatTensor(model(test_data[i][0].expand(99), test_data[i][2]))
        concat = torch.cat([test_pred, neg_pred]).view(-1)
            
        _, indices = torch.topk(concat, 10)
        indices = indices.numpy().tolist()
        if 0 in indices:
            hit += 1
            index = indices.index(0)
            ndcg += np.reciprocal(np.log2(index+2))
            
    hit_ratio = hit / len(test_data)
    ndcg = ndcg / len(test_data)
    
    return hit_ratio, ndcg