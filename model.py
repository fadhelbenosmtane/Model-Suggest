import pandas as pd

# Load the MovieLens dataset
ratings_data = pd.read_csv('ratings.csv')
ratings_pivot = ratings_data.pivot(index='userId', columns='movieId', values='rating').fillna(0)
from sklearn.neighbors import NearestNeighbors

# Initialize the model
model = NearestNeighbors(metric='cosine', algorithm='brute')

# Fit the model using the ratings pivot table
model.fit(ratings_pivot)
# Find the k-nearest neighbors to user 1
_, indices = model.kneighbors(ratings_pivot.loc[1].values.reshape(1, -1), n_neighbors=10)
# Get the movie IDs rated highly by the similar users
movie_ids = []
for idx in indices[0]:
    high_ratings = ratings_pivot.loc[idx][ratings_pivot.loc[idx] > 4].index.tolist()
    movie_ids.extend(high_ratings)

# Remove duplicates from the list of movie IDs
movie_ids = list(set(movie_ids))

# Print the recommended movies
print("Recommended movies for user 1:")
for movie_id in movie_ids:
    print(movie_id)

