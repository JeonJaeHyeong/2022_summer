import numpy as np
import pandas as pd

file_path = "data/ml-100k/u.data"

def load_rating_data():
    
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