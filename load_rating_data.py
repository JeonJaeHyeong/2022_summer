import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import copy
import torch


class RatingDataset(Dataset):
    def __init__(self, user_tensor, item_tensor, values):
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.values = values
        
    def __len__(self):
        return self.user_tensor.size(0)
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.values[index]



# load ml-100k rating data
def load_rating_data():
    
    file_path = "data/ml-100k/u.data"
    prefer = []
    for line in open(file_path, 'r'):  
        (userid, movieid, rating, ts) = line.split('\t')
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        t = int(ts)
        prefer.append([uid, mid, rat, t])
    data = np.array(prefer)
    df = pd.DataFrame(data).reset_index(drop=True)
    df.columns = ['uid', 'mid', 'rating', 'timestamp']
    
    return rename(df)


# load ml-1m rating data
def load_rating_1m():
    
    file_path = "data/ml-1m/ratings.dat"
    rating = pd.read_csv(file_path, sep='::', engine='python')
    rating.columns = ['uid', 'mid', 'rating', 'timestamp']
    
    return rename(rating)

def rename(df):
    
    user_id = df[['uid']].drop_duplicates().reindex()
    user_id['user_id'] = np.arange(len(user_id))
    item_id = df[['mid']].drop_duplicates()
    item_id['item_id'] = np.arange(len(item_id))

    df = pd.merge(df, user_id, on=['uid'], how='left')
    df = pd.merge(df, item_id, on=['mid'], how='left')

    df = df[['user_id', 'item_id', 'rating', 'timestamp']]

    return df 

def one_hot_data(df, original):
    users = original.user_id.unique()
    items = original.item_id.unique()
    user_1hot = np.zeros((len(df), len(users)))
    for i in range(len(df)):
        user_1hot[i, int(df.user_id[i]) - 1] = 1    
        
    item_1hot = np.zeros((len(df), len(items)))
    for i in range(len(df)):
        item_1hot[i, int(df.item_id[i]) - 1] = 1    
    
    time_vec = np.zeros((len(df), 1))
    for i in range(len(df)):
        time_vec[i] = df.timestamp[i]
        
    rating_mat = np.zeros((len(users), len(items)))
    for i in range(len(df)): #df.itertuples():
        rating_mat[int(df.user_id[i])-1, int(df.item_id[i])-1] = df.rating[i]
    rating_mat = np.array(np.vectorize(lambda x: 0 if x==0 else 1)(rating_mat))
    
    
    # Build movie rated dataset
    movie_rated = np.zeros((len(df), len(items)))
    for i in range(len(df)):
        movie_rated[i] = rating_mat[df["user_id"][i]-1]

    movie_rated = movie_rated / movie_rated.sum(axis = 1)[:, np.newaxis]
    
    X = np.concatenate([user_1hot, item_1hot, movie_rated], axis = 1)
    Y  = np.zeros((len(df), 1))
    for i in range(len(df)):
        Y[i] = df.rating[i]
    
    return X, Y
    

class information():

    def __init__(self, args):
        mask = (args.df.user_id <= 50)
        self.df = args.df.loc[mask, :]     # memory issue
        self.n_users = args.n_users = len(self.df.user_id.unique())
        self.n_items = args.n_items = len(self.df.item_id.unique())
        self.train, self.test = self.split_df()
        
        self.train_neg_df = self.get_neg_items(self.train, args.train_n_neg)
        self.train_dataset = self.get_dataset(self.train, self.train_neg_df, "implicit_neg")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size = args.batch_size, shuffle = True)

        self.test_neg_df = self.get_neg_items(self.test, args.test_n_neg)
        self.test_dataset = self.get_dataset(self.test, self.test_neg_df, "neg_samples")
        self.test_dataloader = DataLoader(self.test_dataset, batch_size = args.batch_size, shuffle = True)
        
    def split_df(self):
        
        ratio = 0.8
        train, test = train_test_split(self.df, test_size=1-ratio)
        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)        
        return train, test

    def get_neg_items(self, dataframe, n_neg_samples):
        pos_item_dic = {}
        neg_item_dic = {}
        total_item = np.arange(self.n_items)
        neg_item = []
        for row in dataframe.itertuples():
            if row.user_id not in pos_item_dic.keys():
                mask = (dataframe.user_id == row.user_id)
                pos_item_dic[row.user_id] = np.array(dataframe.loc[mask, :].item_id)
                neg_item_dic[row.user_id] = np.setdiff1d(total_item, pos_item_dic[row.user_id])
            neg = np.random.choice(neg_item_dic[row.user_id], n_neg_samples)
            neg_item.append([neg])
        neg_df = pd.DataFrame(neg_item)
        neg_df.columns = ["negative"]       
        return pd.concat([dataframe.user_id, neg_df], axis=1)


    def get_dataset(self, dataframe, neg_df,  dataset_type):
        
        merged_df = pd.merge(dataframe, neg_df, on= "user_id")
        if dataset_type == "neg_samples":
            train_dataset = RatingDataset(user_tensor = torch.LongTensor(merged_df.user_id),
                                item_tensor = torch.LongTensor(merged_df.item_id),
                                values = torch.LongTensor(merged_df.negative))
        
        elif dataset_type == "implicit_neg":
            users, items, implicit = [], [], []
            
            for row in merged_df.itertuples():
                users.append(int(row.user_id))
                items.append(int(row.item_id))
                implicit.append(float(1))
                
                for neg in row.negative:
                    users.append(int(row.user_id))
                    items.append(int(neg))
                    implicit.append(float(0))
            
            
            train_dataset = RatingDataset(user_tensor = torch.LongTensor(users),
                                item_tensor = torch.LongTensor(items),
                                values = torch.FloatTensor(implicit))            
        return train_dataset
            