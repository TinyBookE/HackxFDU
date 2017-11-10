import pandas as pd
import numpy as np
import os

genre_list = ["fork", "rock", "electric", "classical", "jazz", "rb"]

if not os.path.exists('./data/database0.csv'):

    fpath = './music'
    features = np.load('music.npy').reshape(-1,38*100)
    #labels = np.load('labels.npy').reshape(-1,1)
    
    for index,genre in enumerate(genre_list):
        file = os.path.join(fpath,genre)
        start = 0
        
        database_pd = pd.Series(os.listdir(file))
        database_pd.to_csv('./data/database'+str(index)+'.csv',encoding = 'utf-8')

        if index == 0 :
            database_np = np.hstack((np.arange(20).reshape(20,1),features[:start+20]))
            database_np = np.hstack((database_np,np.ones(20).reshape(20,1)))
        else:
            database_np = np.hstack((np.arange(20).reshape(20,1),features[start:start+20]))
            database_np = np.hstack((database_np,np.ones(20).reshape(20,1)))
        np.save('./data/database'+str(index)+'.npy',database_np)
        
        start += 20

        
        
    
    
    
