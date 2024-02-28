'''Case 3'''
from pyspark import SparkContext
#from pyspark.sql import SparkSession
import json
import sys
from itertools import combinations
import time
import random
import math

#sc.stop()

'''folder = sys.argv[1]
test_filepath = sys.argv[2]
output_filepath = sys.argv[3]
'''
folder = '/content/drive/MyDrive/Colab Notebooks/553_hw3_data/'
test_filepath = '/content/drive/MyDrive/Colab Notebooks/553_hw3_data/yelp_val_in.csv'
output_filepath = '/content/out3.csv'


sc= SparkContext(appName='Case3')
start = time.time()
# train data
lines = sc.textFile(folder+'yelp_train.csv') # load train file into rdd
# Skip the header
header = lines.first()
lines = lines.filter(lambda x: x!= header)
# change each line to list
train_data = lines.map(lambda x: x.strip().split(",")).cache()
#train_d = train_data

# test data
lines_ = sc.textFile(test_filepath) # load train file into rdd
# Skip the header
header = lines_.first()
lines_ = lines_.filter(lambda x: x!= header)
# change each line to list
test_data = lines_.map(lambda x: x.strip().split(",")).cache()

# prepare train and test rdd for Pearson calculation
# get businese_id and user_id, format {bus_id: (user_id, rating)}
train_bus = train_data.map(lambda x: (x[1],(x[0],float(x[2])))).groupByKey().mapValues(list).collectAsMap()
#get train and average ratine for each user
# '3MntE_HWbNNoyiLGxywjYA': 3.4
train_bus_avg = train_data.map(lambda x: (x[1],float(x[2]))).mapValues(lambda x: (x, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1])
train_bus_avg = train_bus_avg.collectAsMap()
#get a train user id and business id buckets, format {user_id, (business_id, rating)}
train_user = train_data.map(lambda x: (x[0],(x[1],float(x[2])))).groupByKey().mapValues(list).collectAsMap()
train_u_avg = train_data.map(lambda x: (x[0],float(x[2]))).mapValues(lambda x: (x, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])).mapValues(lambda x: x[0] / x[1])
train_u_avg = train_u_avg.collectAsMap()
# get businese_id and user_id, format (bus_id,user_id)
test = test_data.map(lambda x: (x[1],x[0]))
'''item based CF'''
#Pearson and Prediction
def calculate_predict(b, u):
    # check if both business and user are new
    if b not in train_bus and u not in train_user:
        return (b, u, 3.75)  # Assign a default value

    # check if business is new but user is old
    if b not in train_bus and u in train_user:
        return (b, u, train_u_avg.get(u, 3.5))  # Use user's average rating

    # check if user is new but business is old
    if u not in train_user:
        return (b, u, train_bus_avg.get(b, 3.5))  # Use business's average rating

    # initialize numerator and denominator for Pearson correlation calculation
    pred_nu = 0
    pred_de = 0

    # find all business this user has rated and users for the target business
    all_bus_ratings = train_user[u]  # List of (business_id, rating) pairs for the user
    b2_ratings = dict(train_bus[b])  # List of (user_id, rating) pairs for the business

    # calculate average ratings for the target business
    b2_avg = train_bus_avg.get(b, 3.5)

    top_similar_businesses = []
    # iterate through each business rated by the user
    for b1, b1_rating in all_bus_ratings:

        # check if the user also rated the same business for which we are predicting
      if b1 in train_bus:
          b1_ratings = train_bus[b1]  # list of (user_id, rating) pairs for business b1

            # find common users who rated both business b1 and the target business b
          common_users = [user_id for user_id, _ in b1_ratings if user_id in b2_ratings]

          #if len(common_users)==2: # not enough information
            # continue
            #top_similar_businesses.append((b1, w))
            #continue
          #elif len(common_users)==2:

          if len(common_users)>2:
              #print('couser',common_users)
                #print('b1',b1)
                # Calculate Pearson correlation for common users
              w_nu = sum((rating - train_bus_avg.get(b1, 3.5)) * (b2_ratings[user_id] - b2_avg)
                           for user_id, rating in b1_ratings if user_id in common_users)
              w_de_1 = sum((rating - train_bus_avg.get(b1, 3.5)) ** 2
                             for _, rating in b1_ratings if _ in common_users)
              w_de_2 = sum((b2_ratings[user_id] - b2_avg) ** 2
                             for user_id, _ in b1_ratings if user_id in common_users)
                #print(w_nu)
              if w_de_1 != 0 and w_de_2 != 0:
                  w = w_nu / (math.sqrt(w_de_1) * math.sqrt(w_de_2))
                  top_similar_businesses.append((b1, w))
          else: # If no or only one co_user, calculate a pearson corelation based on the nusiness avg
            w = (5-abs(train_bus_avg.get(b1,3.5)-train_bus_avg.get(b,3.5)))/5
            top_similar_businesses.append((b1, w))
    top_similar_businesses.sort(key=lambda x: x[1], reverse=True)
    top_similar_businesses = top_similar_businesses[:15] # get the top 15 neighbers
    #print(top_similar_businesses)


    # Calculate the final prediction using only the top N similar businesses
    for b1, w in top_similar_businesses:
     b1_rating = dict(train_user[u])[b1]
     #print(b1_rating)  # Rating given by user u for business b1
     pred_nu += w * (b1_rating)
     #print('nu',pred_nu)
     pred_de += abs(w)
     #print('de',pred_de)

    # Calculate the final prediction
    if pred_de == 0:
     predict = 3.5  # Assign a default value if no correlation found
    else:
     predict = pred_nu / pred_de
    #print(predict)
    return (b, u, predict)

cf_results = test.map(lambda x: calculate_predict(x[0],x[1])).collect()
cf_predictions = []
for i in cf_results:
  cf_predictions.append(i[2])

'''model-based'''

# get features
# business features: bus_id, stars, review_count
bus = sc.textFile(folder+'business.json')
bus_RDD= bus.map(lambda x: json.loads(x)) # load json file to RDD
bus_f = bus_RDD.map(lambda x: (x['business_id'],(x['stars'],x['review_count']))).collectAsMap()
# user features: user_id, review_cnt, average_stars
user = sc.textFile(folder+'user.json')
user_RDD= user.map(lambda x: json.loads(x)) # load json file to RDD
user_f = user_RDD.map(lambda x: (x['user_id'],(x['review_count'],x['average_stars']))).collectAsMap()
# tip: use positive and negative in tips to get the sentiment scores
sc = SparkContext.getOrCreate()
# load tips data
tips_rdd = sc.textFile(folder+'tip.json').map(json.loads)

positive_words = set(["good", "great", "excellent", "happy", "love", "best", "fantastic","like"])
negative_words = set(["bad", "worst", "poor", "sad", "hate", "terrible", "awful","Don't"])

# calculate sentiment score
def sentiment_score(text):
    words = text.lower().split()
    pos_count = sum(word in positive_words for word in words)
    neg_count = sum(word in negative_words for word in words)
    return pos_count - neg_count

# Map the tips RDD to include sentiment score
tips_with_sentiment = tips_rdd.map(lambda x: (x['business_id'], x['user_id'], x['text'], sentiment_score(x['text']))).cache()

# Perform aggregations or other transformations as needed
business_sentiment = tips_with_sentiment.map(lambda x: (x[0], x[3])).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()
user_sentiment = tips_with_sentiment.map(lambda x: (x[1], x[3])).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

# checkin features: total checkin amount
ci = sc.textFile(folder+'checkin.json')
ci_RDD= ci.map(lambda x: json.loads(x)) # load json file to RDD
total_checkins = ci_RDD.flatMap(lambda x: [(x['business_id'], count) for _, count in x['time'].items()]).reduceByKey(lambda a, b: a + b).collectAsMap()
# encode cities and states into numerical value
cities = bus_RDD.map(lambda x: x.get('city', '')).distinct().collect()
states = bus_RDD.map(lambda x: x.get('state', '')).distinct().collect()

city_map = {city: idx for idx, city in enumerate(cities)}
state_map = {state: idx for idx, state in enumerate(states)}


# extract as many business features as possible
def extract_business_features(business):
    features = {
        'stars': 0,
        'review_count': 0,
        'is_open': 0,
        'city': 0,
        'state': 0,
        'bike_parking': 0,
        'credit_cards': 0,
        'good_for_kids':0,
        'hasTV':0,
        'rest_delivery':0,
        'rest_reserve':0,
        'rest_takeout':0
    }

    features['stars'] = business.get('stars', 0)
    features['review_count'] = business.get('review_count', 0)
    features['is_open'] = business.get('is_open', 0)
    city = business.get('city', '')
    state = business.get('state', '')
    features['city'] = city_map.get(city, -1)  # -1 for unknown city
    features['state'] = state_map.get(state, -1)

    attributes = business.get('attributes', {})
    if attributes:
        features['bike_parking'] = 1 if attributes.get('BikeParking') == 'True' else 0
        features['credit_cards'] = 1 if attributes.get('BusinessAcceptsCreditCards') == 'True' else 0
        features['good_for_kids'] = 1 if attributes.get('GoodForKids') == 'True' else 0
        features['hasTV'] = 1 if attributes.get('HasTV') == 'True' else 0
        features['rest_delivery'] = 1 if attributes.get('RestaurantsDelivery') == 'True' else 0
        features['rest_reserve'] = 1 if attributes.get('RestaurantsReservations') == 'True' else 0
        features['rest_takeout'] = 1 if attributes.get('RestaurantsTakeOut') == 'True' else 0

    return business['business_id'], tuple(features.values())

# Apply the feature extraction function and collect as a map
bus_ff = bus_RDD.map(extract_business_features).collectAsMap()
''' add sentiment manually'''
x_train = []
y_train = []

# train features and rating
for u,b,r in train_data.collect():
  # business in train also in bus feature data
  if b in bus_ff.keys():
    b_stars = bus_ff[b][0]
    b_r_cnt = bus_ff[b][1]
    b_open = bus_ff[b][2]
    b_city = bus_ff[b][3]
    b_state = bus_ff[b][4]
    b_bike_parking = bus_ff[b][5]
    b_credit_card = bus_ff[b][6]
    b_kids = bus_ff[b][7]
    b_tv = bus_ff[b][8]
    b_deli = bus_ff[b][9]
    b_reserve = bus_ff[b][10]
    b_takeout = bus_ff[b][11]

  # bus in train not in feature data, assign 0
  else:
    b_stars = 0
    b_r_count = 0
    b_open = 0
    b_city = -1
    b_state = -1
    b_bike_parking = 0
    b_credit_card = 0
    b_kids = 0
    b_tv = 0
    b_deli = 0
    b_reserve = 0
    b_takeout = 0

  # user in train also in user feature data
  if u in user_f.keys():
    u_r_cnt = user_f[u][0]
    u_stars = user_f[u][1]
  # user not in feature data, assign 0
  else:
    u_r_cnt = 0
    u_stars = 0
  if b in business_sentiment.keys():
    b_sentiment = business_sentiment[b]
  else:
    b_sentiment = 0
  if u in user_sentiment.keys():
    u_sentiment = user_sentiment[u]
  else:
    u_sentiment = 0
  if b in total_checkins.keys():
    b_checkin = total_checkins[b]
  else:
    b_checkin = 0
  # construct the feature for model
  x_train.append([u_r_cnt, u_stars,u_sentiment, b_stars, b_r_cnt,b_open,b_city,b_state,b_bike_parking,b_credit_card,b_kids,b_tv,b_deli,b_reserve,b_takeout,b_sentiment,b_checkin])
  # construct the target
  y_train.append(r)

# test features
x_test = []
for u,b in test_data.collect():
  # business in train also in bus feature data
  if b in bus_ff.keys():
    b_stars = bus_ff[b][0]
    b_r_cnt = bus_ff[b][1]
    b_open = bus_ff[b][2]
    b_city = bus_ff[b][3]
    b_state = bus_ff[b][4]
    b_bike_parking = bus_ff[b][5]
    b_credit_card = bus_ff[b][6]
    b_kids = bus_ff[b][7]
    b_tv = bus_ff[b][8]
    b_deli = bus_ff[b][9]
    b_reserve = bus_ff[b][10]
    b_takeout = bus_ff[b][11]

  # bus in train not in feature data, assign 0
  else:
    b_stars = 0
    b_r_count = 0
    b_open = 0
    b_city = -1
    b_state = -1
    b_bike_parking = 0
    b_credit_card = 0
    b_kids = 0
    b_tv = 0
    b_deli = 0
    b_reserve = 0
    b_takeout = 0

  # user in train also in user feature data
  if u in user_f.keys():
    u_r_cnt = user_f[u][0]
    u_stars = user_f[u][1]
  # user not in feature data, assign 0
  else:
    u_r_cnt = 0
    u_stars = 0
  if b in business_sentiment.keys():
    b_sentiment = business_sentiment[b]
  else:
    b_sentiment = 0
  if u in user_sentiment.keys():
    u_sentiment = user_sentiment[u]
  else:
    u_sentiment = 0
  if b in total_checkins.keys():
    b_checkin = total_checkins[b]
  else:
    b_checkin = 0
  # construct the feature for model
  x_test.append([u_r_cnt, u_stars,u_sentiment, b_stars, b_r_cnt,b_open,b_city,b_state,b_bike_parking,b_credit_card,b_kids,b_tv,b_deli,b_reserve,b_takeout,b_sentiment,b_checkin])

import numpy as np
import xgboost as xgb

train_features = np.array(x_train,dtype='float32')
train_ratings = np.array(y_train,dtype='float32')
test_features = np.array(x_test,dtype='float32')
# reshape train data into 2D matrix
#num_samples, num_features = train_features.shape
#x_train_reshaped = train_features.reshape(num_samples, num_features * 2)
# reshape test data into 2D matrix
#num_samples_test, num_features_test = test_features.shape
#x_test_reshaped = test_features.reshape(num_samples_test, num_features_test * 2)

# define XGBoost model
xg_reg = xgb.XGBRegressor(verbosity=0, learning_rate=0.3, n_estimators=100, random_state=20, max_depth=5)

# train the model
xg_reg.fit(train_features, train_ratings)

# make predictions
model_predictions = xg_reg.predict(test_features)

# From Case 1 and Case 2, the model based method gives smaller rmse, thus assign smaller weight to item based CF
# Tried 0.05, 0.1, 0.15, 0.2, 0.5, factor = 0.1 gave the smaller RMSE
factor = 0.1
final_ratings = []
for i in range(test_data.count()):
  final_r = factor * cf_predictions[i] + (1-factor) * model_predictions[i]
  final_ratings.append(final_r)
final_results = list(zip(test_data.collect(), final_ratings))

