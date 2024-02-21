
#############################################
# PROJE: Hybrid Recommender System
#############################################

# Make predictions using item-based and user-based recommender methods for the user with the given ID.
# Consider 5 recommendations from the user-based model and 5 recommendations from the item-based model, then ultimately provide 10 recommendations from both models

#############################################
# Task 1: Data preparation
#############################################

import pandas as pd
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.width', 300)

# Step 1: Read Movie and Rating datasets.
# Dataset including movieId, movie name and kind of movie  
movie = pd.read_csv('datasets/movie.csv')
movie.head()
movie.shape

# UserID, movie name, dataset including rate to the film and time for film 
rating = pd.read_csv('datasets/rating.csv')
rating.head()
rating.shape
rating["userId"].nunique()


# Task 2: By using movie film set, add movie names and movie types to rating dataset.
# The movies that users voted for in the Rating dataset only have their IDs.
# We are adding the movie names and genres corresponding to the IDs from the Movie dataset.
df = movie.merge(rating, how="left", on="movieId")
df.head()
df.shape


# Step 3: Calculate the number of people giving rate for each movie. Remove the movies from the dataset that have a total vote count less than 1000.
# Calculate the number of people giving rate for each movie. 
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts

# We save the names of movies with a total vote count less than 1000 in the rare_movies
# And remove from dataset. 
rare_movies = comment_counts[comment_counts["title"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape


# Step 4: # Build a pivot table for dataframe that user ID's are in index and Movie names are in the columns and rating as value 

user_movie_df = common_movies.pivot_table(index="userId", columns=["title"], values="rating")




# Step 5: Let's make above all steps into a function.
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


#############################################
# Task 2: Determination the movies watched by the user for whom recommendations will be made
#############################################

# Step 1: Choose randomly a user ID.
random_user = 108170

# Step 2: Build a new dataframe named "random_user_df" including observations belonging to randomly selected user.    
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
random_user_df.shape

# Step 3: Save the movies that selected user rated in a new "movies_watched" list. 
movies_watched = random_user_df.columns[random_user_df.notna().any()].to_list()
movies_watched

movie.columns[movie.notna().any()].to_list()

#############################################
# Task 3:Access the data and IDs of other users who have watched the same movies
#############################################
# Step 1: Select the columns related to the movies watched by the chosen user from 'user_movie_df' and create a new dataframe named 'movies_watched_df
user_movie_df.head()
a = [1.0,2.0]
user_movie_df[a]
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape

# Step 2: Create a new dataframe named 'user_movie_count' that carries information about how many movies each user has watched from the movies watched by the selected user.

user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head(5)
user_movie_count.sort_values(by="movie_count", ascending=False)

# Step 3: We consider those who have watched 60% or more of the movies that the selected user has rated as similar users. Create a list named 'users_same_movies' from the IDs of these users.

perc = len(movies_watched) * 60 / 100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
len(users_same_movies)



#############################################
# Task 4: Determination the most similar users to the user for whom recommendations will be made
#############################################

# Step 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
user_movie_df[movies_watched].head()
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head()
final_df.shape
len(set(final_df.index))

# Step 2: Create a new dataframe named 'corr_df' where the correlations between users will be found.
corr_df = final_df.T.corr().unstack().sort_values()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.head()
corr_df.index
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df = corr_df.reset_index()

corr_df.head()

#corr_df[corr_df["user_id_1"] == random_user]



# Step 3: Filter users with high correlation (above 0.65) with the selected user and create a new dataframe named 'top_users'.
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
top_users
top_users = top_users.sort_values(by='corr', ascending=False)
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.shape
top_users.head()

# Step 4:  Merge rating dataset to top_users dataframe.
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings["userId"].unique()
top_users_ratings.head()



#############################################
# Task 5: First 5 Movies for the calculation of Weighted Average Recommendation Score
#############################################

# Step 1: Create a new variable named 'weighted_rating' consisting of the product of each user's correlation and rating values
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
top_users_ratings.head()

# Step 2: Create a new dataframe named 'recommendation_df' containing the average of weighted ratings for each movie ID and all users associated with each movie.
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
recommendation_df.head()

# Step 3: Select the movies from 'recommendation_df' where the weighted rating is greater than 3.5 and sort them based on the weighted rating.
# Save the 5 observation as movies_to_be_recommend
recommendation_df[recommendation_df["weighted_rating"] > 3.5]
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)


# Step 4:  Show the names of the top 5 recommended movies.
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"][:5]

# 0    Mystery Science Theater 3000: The Movie (1996)
# 1                               Natural, The (1984)
# 2                             Super Troopers (2001)
# 3                         Christmas Story, A (1983)
# 4                       Sonatine (Sonachine) (1993)



#############################################
# Step 6: Item-Based Recommendation
#############################################

# Provide item-based recommendations based on the movie title that the user most recently watched and rated the highest.
user = 108170

# Step 1: Read movie and rating datasets.
movie = pd.read_csv('Modül_4_Tavsiye_Sistemleri/datasets/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

# Step 2: Retrieve the ID of the movie with the most recent rating among the movies that the user for whom recommendations will be made has rated with a score of 5.
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]

# Step 3 :Filter the 'user_movie_df' dataframe created in the User-Based Recommendation section based on the selected movie ID.
movie[movie["movieId"] == movie_id]["title"].values[0]
movie_df = user_movie_df[movie[movie["movieId"] == movie_id]["title"].values[0]]
movie_df

# Step 4: Using the filtered dataframe, find and sort the correlation between the selected movie and other movies.
user_movie_df.corrwith(movie_df).sort_values(ascending=False).head(10)

# The function applying the last two steps 
def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

# Step 5: Provide the top 5 films, excluding the selected movie itself, as recommendations
movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index


# 'My Science Project (1985)',
# 'Mediterraneo (1991)',
# 'Old Man and the Sea,
# The (1958)',
# 'National Lampoon's Senior Trip (1995)',
# 'Clockwatchers (1997)']



