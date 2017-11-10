from model import predict
from input_data import read_mp3
import numpy as np
import pandas as pd
from numpy import linalg as la
import os



def euclidSimilar(inA,inB):  
    return 1.0/(1.0+la.norm(inA-inB))

def find(mp3, num = 1, respond = 0):

    genre_list = ["fork", "rock", "electric", "classical", "jazz", "rb"]
    seq = read_mp3(mp3)
    index = predict(seq)[0]
    print(index)
    database_pd = pd.read_csv('./data/database' +str(index)+'.csv', encoding = 'utf-8')
    print(database_pd)
    database_np = np.load('./data/database' +str(index)+'.npy').reshape(-1,3802)
    print(database_np)
    seq = seq.reshape(38*100)
    filename_list = []
    
    for i in range(20):
        database_np[i,38*100+1] = euclidSimilar(database_np[i,0:38*100],seq)

    database_np = database_np[np.argsort(database_np[:,38*100+1])]
    print(database_np)
    for i in range(num):
        search_index = int(database_np[i,0])
        print(search_index)
        print (database_pd.iloc[search_index,1])
        filename_list.append(os.path.join('music',genre_list[index],database_pd.iloc[search_index,1]))
    return filename_list

    
    
    
        

