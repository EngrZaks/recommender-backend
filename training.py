from scipy import sparse
from surprise import Reader, Dataset
import surprise
import numpy as np
from surprise import SVD
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import xgboost as xgb
start = datetime.now()
data = pd.read_csv("ratings.csv")
data = data.drop('timestamp', axis=1)
train_data = data.iloc[:int(data.shape[0]*0.80)]
test_data = data.iloc[int(data.shape[0]*0.80):]

reader = Reader(rating_scale=(1, 5))

# create the traindata from the dataframe...
train_data_mf = Dataset.load_from_df(
    train_data[['userId', 'movieId', 'rating']], reader)

# build the trainset from traindata..
trainset = train_data_mf.build_full_trainset()

# test set from test data
reader = Reader(rating_scale=(1, 5))

# create the traindata from the dataframe...
test_data_mf = Dataset.load_from_df(
    test_data[['userId', 'movieId', 'rating']], reader)

# build the trainset from traindata.., It is of dataset format from surprise library..
testset = test_data_mf.build_full_trainset()
svd = SVD(n_factors=100, biased=True, random_state=15, verbose=True)
svd.fit(trainset)

# storing train predictions
train_preds = svd.test(trainset.build_testset())
train_pred_mf = np.array([pred.est for pred in train_preds])
test_preds = svd.test(testset.build_testset())
test_pred_mf = np.array([pred.est for pred in test_preds])

# Creating a sparse matrix
train_sparse_matrix = sparse.csr_matrix(
    (train_data.rating.values, (train_data.userId.values, train_data.movieId.values)))

train_averages = dict()
# get the global average of ratings in our train set.
train_global_average = train_sparse_matrix.sum()/train_sparse_matrix.count_nonzero()
train_averages['global'] = train_global_average

def get_average_ratings(sparse_matrix, of_users):
    # average ratings of user/axes
    ax = 1 if of_users else 0  # 1 - User axes,0 - Movie axes
    
    # ".A1" is for converting Column_Matrix to 1-D numpy array
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # Boolean matrix of ratings ( whether a user rated that movie or not)
    is_rated = sparse_matrix != 0
    # no of ratings that each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1
    
    # max_user  and max_movie ids in sparse matrix
    u, m = sparse_matrix.shape
    # creae a dictonary of users and their average ratigns..
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i]
                       for i in range(u if of_users else m)
                       if no_of_ratings[i] != 0}

    # return that dictionary of average ratings
    return average_ratings


# Average ratings given by a user
train_averages['user'] = get_average_ratings(
    train_sparse_matrix, of_users=True)
print('\nAverage rating of user 10 :', train_averages['user'][10])

# Average ratings given for a movie
train_averages['movie'] = get_average_ratings(
    train_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :', train_averages['movie'][15])

# get users, movies and ratings from our samples train sparse matrix
train_users, train_movies, train_ratings = sparse.find(train_sparse_matrix)

final_data = pd.DataFrame()
count = 0
for (user, movie, rating) in zip(train_users, train_movies, train_ratings):
    st = datetime.now()
    #     print(user, movie)
    # compute the similar Users of the "user"
    user_sim = cosine_similarity(
        train_sparse_matrix[user], train_sparse_matrix).ravel()
    # we are ignoring 'The User' from its similar users.
    top_sim_users = user_sim.argsort()[::-1][1:]
    # get the ratings of most similar users for this movie
    top_ratings = train_sparse_matrix[top_sim_users, movie].toarray(
    ).ravel()
    top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5]) # top 5
    top_sim_users_ratings.extend(
        [train_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
    #     print(top_sim_users_ratings, end=" ")

    # compute the similar movies of the "movie"
    movie_sim = cosine_similarity(
        train_sparse_matrix[:, movie].T, train_sparse_matrix.T).ravel()
    # we are ignoring 'The User' from its similar users.
    top_sim_movies = movie_sim.argsort()[::-1][1:]
    # get the ratings of most similar movie rated by this user..
    top_ratings = train_sparse_matrix[user,
                                      top_sim_movies].toarray().ravel()
    top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5]) # length, 5
    top_sim_movies_ratings.extend(
        [train_averages['user'][user]]*(5-len(top_sim_movies_ratings)))
    #     print(top_sim_movies_ratings, end=" : -- ")

    #-----------------prepare the row to be stored in a file-----------------#
    row = [user, movie]
    row.append(train_averages['global'])  # first feature
    row.extend(top_sim_users_ratings) # similar user ratings
    row.extend(top_sim_movies_ratings) # similar movies ratings
    row.append(train_averages['user'][user]) # Avg_user rating
    row.append(train_averages['movie'][movie])# Avg_movie rating

    # finalley, The actual Rating of this user-movie pair...
    row.append(rating)
    final_data = pd.concat([final_data, pd.Series(row)], axis=1)
    count += 1
    print(count)
    if count % 10000 == 0:
        print("Done for {} rows ----- {}".format(count, datetime.now() - start))
    # count = count + 1
    # final_data = final_data.append([row])
    # print(count)

    # if (count) % 10000 == 0:
    #     # print(','.join(map(str, row)))
    #     print("Done for {} rows----- {}".format(count, datetime.now() - start))

final_data.columns = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4',
                      'sur5', 'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']
test_sparse_matrix = sparse.csr_matrix(
    (test_data.rating.values, (test_data.userId.values, test_data.movieId.values)))
# Global avg of all movies by all users
test_averages = dict()
# get the global average of ratings in our train set.
test_global_average = test_sparse_matrix.sum()/test_sparse_matrix.count_nonzero()
test_averages['global'] = test_global_average
test_averages

# get the user averages in dictionary (key: user_id/movie_id, value: avg rating)
def get_average_ratings(sparse_matrix, of_users):
    # average ratings of user/axes
    ax = 1 if of_users else 0  # 1 - User axes,0 - Movie axes

    # convert to 1-D numpy array
    sum_of_ratings = sparse_matrix.sum(axis=ax).A1
    # matrix of ratings 
    is_rated = sparse_matrix != 0
    # no of ratings of each user OR movie..
    no_of_ratings = is_rated.sum(axis=ax).A1

    # max_user  and max_movie ids in sparse matrix
    u, m = sparse_matrix.shape
    # dictonary of users and their average ratigns..
    average_ratings = {i: sum_of_ratings[i]/no_of_ratings[i]
                       for i in range(u if of_users else m)
                       if no_of_ratings[i] != 0}
    return average_ratings

# Average ratings given by a user
test_averages['user'] = get_average_ratings(test_sparse_matrix, of_users=True)
#print('\nAverage rating of user 10 :',test_averages['user'][10])

# Average ratings given for a movie

test_averages['movie'] = get_average_ratings(
    test_sparse_matrix, of_users=False)
print('\n AVerage rating of movie 15 :', test_averages['movie'][15])

# get users, movies and ratings from our samples train sparse matrix
test_users, test_movies, test_ratings = sparse.find(test_sparse_matrix)

final_test_data = pd.DataFrame()
count = 0
for (user, movie, rating) in zip(test_users, test_movies, test_ratings):
    st = datetime.now()
    #     print(user, movie)
    # --------------------- Ratings of "movie" by similar users of "user" ---------------------
    # compute the similar Users of the "user"
    user_sim = cosine_similarity(
        test_sparse_matrix[user], test_sparse_matrix).ravel()
    # we are ignoring 'The User' from its similar users.
    top_sim_users = user_sim.argsort()[::-1][1:]
    # get the ratings of most similar users for this movie
    top_ratings = test_sparse_matrix[top_sim_users, movie].toarray().ravel()
    # we will make it's length "5" by adding movie averages to .
    top_sim_users_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_users_ratings.extend(
        [test_averages['movie'][movie]]*(5 - len(top_sim_users_ratings)))
    #     print(top_sim_users_ratings, end=" ")

    # --------------------- Ratings by "user"  to similar movies of "movie" ---------------------
    # compute the similar movies of the "movie"
    movie_sim = cosine_similarity(
        test_sparse_matrix[:, movie].T, test_sparse_matrix.T).ravel()
    # we are ignoring 'The User' from its similar users.
    top_sim_movies = movie_sim.argsort()[::-1][1:]
    # get the ratings of most similar movie rated by this user..
    top_ratings = test_sparse_matrix[user, top_sim_movies].toarray().ravel()
    # we will make it's length "5" by adding user averages to.
    top_sim_movies_ratings = list(top_ratings[top_ratings != 0][:5])
    top_sim_movies_ratings.extend(
        [test_averages['user'][user]]*(5-len(top_sim_movies_ratings)))
    #     print(top_sim_movies_ratings, end=" : -- ")

    #-----------------prepare the row to be stores in a file-----------------#
    row = list()
    row.append(user)
    row.append(movie)
    # Now add the other features to this data...
    row.append(test_averages['global'])  # first feature
    # next 5 features are similar_users "movie" ratings
    row.extend(top_sim_users_ratings)
    # next 5 features are "user" ratings for similar_movies
    row.extend(top_sim_movies_ratings)
    # Avg_user rating
    row.append(test_averages['user'][user])
    # Avg_movie rating
    row.append(test_averages['movie'][movie])

    # finalley, The actual Rating of this user-movie pair...
    row.append(rating)
    count = count + 1
    final_test_data = final_test_data.append([row])
    print(count)

    if (count) % 10000 == 0:
        # print(','.join(map(str, row)))
        print("Done for {} rows----- {}".format(count, datetime.now() - start))

final_test_data.columns = ['user', 'movie', 'GAvg', 'sur1', 'sur2', 'sur3', 'sur4', 'sur5',
                                   'smr1', 'smr2', 'smr3', 'smr4', 'smr5', 'UAvg', 'MAvg', 'rating']
# creating XGBoost


def get_error_metrics(y_true, y_pred):
    rmse = np.sqrt(
        np.mean([(y_true[i] - y_pred[i])**2 for i in range(len(y_pred))]))
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return rmse, mape


# prepare train data
x_train = final_data.drop(['user', 'movie', 'rating'], axis=1)
y_train = final_data['rating']

# Prepare Test data
x_test = final_test_data.drop(['user', 'movie', 'rating'], axis=1)
y_test = final_test_data['rating']


# initialize XGBoost model...
xgb_model = xgb.XGBRegressor(
    silent=False, n_jobs=13, random_state=15, n_estimators=100)
# dictionaries for storing train and test results
train_results = dict()
test_results = dict()


# fit the model
print('Training the model..')
start = datetime.now()
xgb_model.fit(x_train, y_train, eval_metric='rmse')
print('Done. Time taken : {}\n'.format(datetime.now()-start))
print('Done \n')

# from the trained model, get the predictions....
print('Evaluating the model with TRAIN data...')
start = datetime.now()
y_train_pred = xgb_model.predict(x_train)
# get the rmse and mape of train data...
rmse_train, mape_train = get_error_metrics(y_train.values, y_train_pred)

# store the results in train_results dictionary..
train_results = {'rmse': rmse_train,
                 'mape': mape_train,
                 'predictions': y_train_pred}

#######################################
# get the test data predictions and compute rmse and mape
print('Evaluating Test data')
y_test_pred = xgb_model.predict(x_test)
rmse_test, mape_test = get_error_metrics(
    y_true=y_test.values, y_pred=y_test_pred)
# store them in our test results dictionary.
test_results = {'rmse': rmse_test,
                'mape': mape_test,
                'predictions': y_test_pred}

print("FINAL TEST", test_results)
