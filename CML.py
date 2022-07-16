import numpy as np
import pandas as pd
import load_rating_data as ld
from utils import RMSE 
import time
import random
import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim, nn
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split


class CML(nn.Module):
    
    def __init__(self, args):
        
        super(CML, self).__init__()
        self.args = args
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.latent_dim = args.latent_dim
        self.margin = args.margin
        self.lambda_c = args.lambda_c
        self.neg_item_dic = {}
        self.n_neg_samples = args.n_neg_samples
        
        self.user_embedding = nn.Embedding(self.n_users, self.latent_dim, max_norm = 1) # restrict norms
        self.item_embedding = nn.Embedding(self.n_items, self.latent_dim, max_norm = 1)
        
    
    def distance_loss(self, i, j, k):
        """
        compute distance loss
        """
        user = self.user_embedding(i).view(len(i), 1, self.latent_dim) # (batch_size, 1, latent_dim)
        item = self.item_embedding(j).view(len(j), 1, self.latent_dim) # (batch_size, 1, latent_dim)
        neg_item = self.item_embedding(k)   # (batch_size, n_neg_samples, latent_dim)
        d_ij = torch.cdist(user, item).view(-1, 1)**2 # (batch_size, 1)
        d_ik = torch.cdist(user, neg_item).view(-1, self.n_neg_samples)**2  # (batch_size, n_neg_samples)
        
        metric = self.margin + d_ij - d_ik # (batch_size, n_neg_samples)
        loss = 0
        for i in range(len(metric)):
            temp_metric = metric[i][metric[i]>0]    # []+
            rank_d_ij = self.n_items * len(temp_metric) / self.n_neg_samples  # J x M / N
            w_ij = np.log(rank_d_ij + 1)
            loss +=  (w_ij * temp_metric).sum()
        
        return loss
    
    
    def cov_loss(self):
        U = self.user_embedding(torch.LongTensor([x for x in range(self.n_users)]))
        V = self.item_embedding(torch.LongTensor([x for x in range(self.n_items)]))
        
        matrix = torch.cat([U, V])
        n_rows = matrix.shape[0]
        matrix = matrix - torch.mean(matrix, dim=0)
        cov = torch.matmul(matrix.T, matrix) / n_rows
        loss = (torch.linalg.norm(cov) - torch.linalg.norm(torch.diagonal(cov),2))/self.n_users
        
        return loss * self.lambda_c
    
    
    def create_train_dataset(self, train):
        
        pos_item_dic = {}
        total_item = np.arange(self.n_items)
        
        neg_item = []
        for row in train.itertuples():
            if row.user_id not in pos_item_dic.keys():
                mask = (train.user_id == row.user_id)
                pos_item_dic[row.user_id] = np.array(train.loc[mask, :].item_id)
                self.neg_item_dic[row.user_id] = np.setdiff1d(total_item, pos_item_dic[row.user_id])
            neg = np.random.choice(self.neg_item_dic[row.user_id], self.n_neg_samples)
            neg_item.append(neg)

        dataset = RatingDataset(user_tensor = torch.LongTensor(train.iloc[:, 0]),
                            item_tensor = torch.LongTensor(train.iloc[:, 1]),
                            neg_item_list = torch.LongTensor(neg_item))

        return dataset
    
    
    def evaluate(self, train, test):
        U = self.user_embedding(torch.LongTensor([x for x in range(self.n_users)]))
        V = self.item_embedding(torch.LongTensor([x for x in range(self.n_items)]))
        dist = torch.cdist(U, V)
        
        #for row in train.itertuples():
        #    dist[int(row.user_id), int(row.item_id)] = 1000
            
        top50_id = torch.topk(dist, k=50, dim=1, largest=False)[1].numpy()
        top100_id = torch.topk(dist, k=100, dim=1, largest=False)[1].numpy()
        hit_50 = 0
        hit_100 = 0
        
        for i in range(len(test)):
            if int(test.iloc[i, 1]) in top50_id[int(test.iloc[i, 0])]:
                hit_50 += 1
            if int(test.iloc[i, 1]) in top100_id[int(test.iloc[i, 0])]:
                hit_100 += 1
        
        r50, r100 =  hit_50/len(test), hit_100/len(test)
        return r50, r100
    
class RatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, neg_item_list):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.neg_items = neg_item_list
        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.neg_items[index]
    
    
def run(args):
    
    # Basic settings
    from torch.utils.tensorboard import SummaryWriter 
    writer = SummaryWriter("cml_logs")
    
    df = ld.load_rating_1m()
    args.n_users, args.n_items = len(df.user_id.unique()), len(df.item_id.unique())
    ratio = 0.8
    train, test = train_test_split(df, test_size=1-ratio)
    val, test = train_test_split(test, test_size=0.5)
    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
        
    model = CML(args) #.to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=0.01)

    # Start training
    # Save the starting time
    start_time = time.time()

    for epoch in range(0, args.epoch):
        # Here starts the train loop.
        dataset = model.create_train_dataset(train)
        train_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True)
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            user, item, neg_items = batch[0], batch[1], batch[2]
            loss = model.distance_loss(user, item, neg_items) + model.cov_loss()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        t = time.time()-start_time
        model.eval()
        recall_50, recall_100 = model.evaluate(train, test)
        writer.add_scalar("training_loss", total_loss, epoch)
        writer.add_scalar("recall@50", recall_50, epoch)
        writer.add_scalar("recall@100", recall_100, epoch)
        print("epoch = {:d}, total_loss = {:.4f}, recall@50 = {:.4f}, recall@100 = {:.4f}, epoch_time = {:.4f}sec".format(epoch, total_loss, recall_50, recall_100, time.time()-start_time))


if __name__ == '__main__':
    from easydict import EasyDict as edict
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = edict()
    # training options
    args.latent_dim = 32
    args.margin = 0.5
    args.lambda_c = 10
    args.epoch = 10                      # training epoch.
    args.n_neg_samples = 10
    args.batch_size = 1024
    args.device = device

    run(args) 