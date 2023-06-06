### Real Time AI with Astra

Recommended review: [Understand Real-Time AI](https://github.com/mukundha/real-time-ai-demo)

Here, We review the role of Astra in ML lifecycle: Training [will follow soon with Inference, Feature store, Prediction store and Model monitoring]

Use the [Data Loader in Astra Portal](https://docs.datastax.com/en/astra-serverless/docs/develop/dev-upload-data.html
) to upload the `data/ratings_realtime.csv` to Astra DB

For reference, auto-created table. Make sure to check the datatypes
```
CREATE TABLE demo.ratings_realtime (
    "userId" int,
    "movieId" int,
    last_rating decimal,
    last_watched_movie int,
    rating decimal,
    timestamp timestamp,
    PRIMARY KEY ("userId", "movieId")
)

```
#### Pre-requisites

```
pip install numpy
pip install pandas
pip install tensorflow
pip install scikit-learn
```

```
export CLIENT_ID=<ASTRA_CLIENT_ID>
export CLIENT_SECRET=<ASTRA_CLIENT_SECRET>
export KEYSPACE=<ASTRA_KEYSPACE>
export TABLE=<ASTRA_TRAINING_TABLE_NAME>
export BATCH_SIZE=<BATCH_SIZE>
export SECURE_CONNECT_BUNDLE_PATH=<ASTRA_SECURE_CONNECT_BUNDLE>
```

#### Training

`python train.py`

Trained model will be saved as `movie_recommendation_realtime.h5`

#### 

Citation
========

To acknowledge use of the dataset in publications, please cite the following paper:

> F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. <https://doi.org/10.1145/2827872>
