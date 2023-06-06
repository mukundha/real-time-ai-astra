from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from generator import AstraDataGenerator
import os 

# Define the number of unique users and movies
num_users = 610
num_movies = 9724

# Define the input layers
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))
last_watched_movie_input = Input(shape=(1,))
last_rating_input = Input(shape=(1,))

# Embedding layers for users, movies, last_watched_movie
user_embedding = Embedding(num_users, 50, embeddings_regularizer=l2(1e-5))(user_input)
movie_embedding = Embedding(num_movies, 50, embeddings_regularizer=l2(1e-5))(movie_input)
last_watched_embedding = Embedding(num_movies, 50, embeddings_regularizer=l2(1e-5))(last_watched_movie_input)

# Flatten the embeddings
user_flat = Flatten()(user_embedding)
movie_flat = Flatten()(movie_embedding)
last_watched_flat = Flatten()(last_watched_embedding)

# Concatenate the flattened embeddings and last_rating input
concat = Concatenate()([user_flat, movie_flat, last_watched_flat, last_rating_input])

# Dense layers
dense1 = Dense(64, activation='relu', kernel_regularizer=l2(1e-5))(concat)
dense2 = Dense(32, activation='relu', kernel_regularizer=l2(1e-5))(dense1)
output = Dense(1, activation='sigmoid')(dense2)

# Create the model
model = Model(inputs=[user_input, movie_input, last_watched_movie_input, last_rating_input], outputs=output)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

model.fit(AstraDataGenerator(),epochs=5)
model.save('movie_recommendation_realtime.h5')

