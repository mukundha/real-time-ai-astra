from keras.utils import Sequence
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import numpy as np
import os
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def pandas_factory(colnames, rows):
    return pd.DataFrame(rows, columns=colnames)
      
class AstraDataGenerator(Sequence):
    def __init__(self):
        self.keyspace = os.environ.get('KEYSPACE')
        self.table = os.environ.get('TABLE')
        self.batch_size = int(os.environ.get('BATCH_SIZE'))
        cloud_config= {
            'secure_connect_bundle': os.environ.get('SECURE_CONNECT_BUNDLE_PATH')
        }
        auth_provider = PlainTextAuthProvider(os.environ.get('CLIENT_ID'), os.environ.get('CLIENT_SECRET'))
        self.cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
        self.session = self.cluster.connect(self.keyspace)
        self.session.row_factory = pandas_factory
        self.fetch_size = self.batch_size
        self.paging_state = None
        self.total_samples = self._get_total_samples()
        
    def __len__(self):
        return int(np.ceil(self.total_samples / self.batch_size))
    
    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size        
        df = self._fetch_data_chunk()
        X,Y = self._preprocess_data(df)              
        return X,Y        
    
    def on_epoch_end(self):
        print('Epoch reached')
        pass  # Add any necessary functionality at the end of each epoch
    
    def _get_total_samples(self):
        query = f"SELECT COUNT(*) FROM {self.keyspace}.{self.table}"
        result = self.session.execute(query)        
        total_samples = result._current_rows['count'][0]
        return total_samples
    
    def _fetch_data_chunk(self):
        query = f"SELECT * FROM {self.keyspace}.{self.table}"
        statement = SimpleStatement(query, fetch_size=self.fetch_size)                
        result = self.session.execute(statement,paging_state=self.paging_state)
        df = result._current_rows
        self.paging_state = result.paging_state        
        return df
    
    def _preprocess_data(self, ratings_df):
        train_data = ratings_df
        scaler = MinMaxScaler()
        train_data['rating'] = scaler.fit_transform(train_data['rating'].values.reshape(-1, 1))
        train_data['last_rating'] = scaler.fit_transform(train_data['last_rating'].values.reshape(-1, 1))
        train_user = train_data['userId'].values
        train_movie = train_data['movieId'].values
        train_last_watched_movie = train_data['last_watched_movie'].values
        train_last_rating = train_data['last_rating'].values
        train_ratings = train_data['rating'].values
        return [train_user, train_movie, train_last_watched_movie, train_last_rating], train_ratings
