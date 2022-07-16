import numpy as np
import pandas as pd


# load ml-100k rating data
def load_rating_data():
    
    file_path = "data/ml-100k/u.data"
    prefer = []
    for line in open(file_path, 'r'):  
        (userid, movieid, rating, ts) = line.split('\t')
        uid = int(userid)
        mid = int(movieid)
        rat = float(rating)
        prefer.append([uid, mid, rat])
    data = np.array(prefer)
    df = pd.DataFrame(data).reset_index(drop=True)
    df.columns = ['user_id', 'item_id', 'rating']
    return df


# load ml-1m rating data
def load_rating_1m():
    
    file_path = "data/ml-1m/ratings.dat"
    rating = pd.read_csv(file_path, sep='::', engine='python')
    rating.columns = ['uid', 'mid', 'rating', 'timestamp']
    user_id = rating[['uid']].drop_duplicates().reindex()
    user_id['user_id'] = np.arange(len(user_id))
    item_id = rating[['mid']].drop_duplicates()
    item_id['item_id'] = np.arange(len(item_id))

    rating = pd.merge(rating, user_id, on=['uid'], how='left')
    rating = pd.merge(rating, item_id, on=['mid'], how='left')

    rating = rating[['user_id', 'item_id', 'rating', 'timestamp']]
    
    return rating