import sys
import os

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scrapbook as sb

import lightfm
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm import cross_validation

# Import LightFM's evaluation metrics
from lightfm.evaluation import precision_at_k as lightfm_prec_at_k
from lightfm.evaluation import auc_score

# Import repo's evaluation metrics
from recommenders.evaluation.python_evaluation import precision_at_k, recall_at_k

from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.models.lightfm.lightfm_utils import (
    track_model_metrics, prepare_test_df, prepare_all_predictions,
    compare_metric, similar_users, similar_items)

# Select MovieLens data size
MOVIELENS_DATA_SIZE = '100k'

# default number of recommendations
K = 10
# percentage of data used for testing
TEST_PERCENTAGE = 0.25
# model learning rate
LEARNING_RATE = 0.25
# no of latent factors
NO_COMPONENTS = 20
# no of epochs to fit model
NO_EPOCHS = 20
# no of threads to fit model
NO_THREADS = 32
# regularisation for both user and item features
ITEM_ALPHA = 1e-6
USER_ALPHA = 1e-6

# seed for pseudonumber generations
SEED = 42

data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    genres_col='genre',
    year_col= 'year',
    header=["userID", "itemID", "rating"]
)

# converting year to int datatype
data['year'] = pd.to_numeric(data['year'], errors='coerce')
data['year'] = pd.to_numeric(data['year'], errors='coerce').convert_dtypes()

# replacing na with 0
data['year'] = data['year'].fillna(0)

# categorizing year
bins = [0,1930,1940,1950,1960, 1970,1980, 1990, 2000]
labels = ['<=1930','<=1940','<=1950','<=1960','<=1970','<=1980','<=1990','1990+']
data['year group'] = pd.cut(data['year'], bins=bins, labels=labels, right=True)
data['year group'] = data['year group'].fillna('<=1930')

# combining item features for passing to model
data['combined'] = data.apply(lambda x:'%s|%s' % (x['genre'],x['year group']),axis=1)
combined_item_features = [x.split('|') for x in data['combined']]

# data1 = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)
threshold = 3.0
data['rating'] = data['rating'].gt(threshold).astype(int)

# split the genre based on the separator
movie_genre = [x.split('|') for x in data['genre']]

# retrieve the all the unique genres in the data
all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

# list of year groups
all_year_groups = list(set(data['year group']))

all_year = list(set(data['year']))
print(len(all_year))

# merging genre and year list for fitting the dataset
item_features_merged = all_movie_genre + all_year_groups

user_feature_URL = 'http://files.grouplens.org/datasets/movielens/ml-100k/u.user'
columns = ['userID','age','gender','occupation','zipcode']
user_data = pd.read_table(user_feature_URL, sep='|', header=None, names=columns)

# merging user feature with existing data
new_data = data.merge(user_data[['userID','occupation', 'age', 'gender', 'zipcode']], left_on='userID', right_on='userID')
new_data["gender"] = "Gender " + new_data["gender"]
new_data["occupation"] = "Occupation " + new_data["occupation"]

# categorizing age
bins= [0,10,20,30,50,200]
labels = ['<=10','<=20','<=30','<=50','50+']
new_data['age group'] = pd.cut(new_data['age'], bins=bins, labels=labels, right=True)

# categorizing zipcode
new_data['zipcode group'] = new_data['zipcode'].apply(lambda x: x[0])
new_data["zipcode group"] = "Zipcode " + new_data["zipcode group"]

# merging all user features for passing into model
new_data['combined'] = new_data.apply(lambda x:'%s_%s_%s_%s' % (x['occupation'],x['age group'], x['gender'], x['zipcode group']),axis=1)
combined_user_features = [x.split('_') for x in new_data['combined']]

# sorted list of occupations
all_occupations = sorted(list(set(new_data['occupation'])))

# list of age groups
all_age_groups = list(set(new_data['age group']))
all_age = list(set(new_data['age']))
print(len(all_age))

# sorted list of genders
all_genders = sorted(list(set(new_data['gender'])))

# sorted list of zipcode groups
all_zipcode_groups = sorted(list(set(new_data['zipcode group'])))
all_zipcode = list(set(new_data['zipcode']))
print(len(all_zipcode))
# merging all user feature lists
user_features_merged = all_occupations + all_age_groups + all_genders + all_zipcode_groups


def without_features_WARP():
    dataset = Dataset()

    dataset.fit(users=data['userID'],
                items=data['itemID'])

    (interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)

    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED))
    # print(f"Shape of train interactions: {train_interactions.shape}")
    # print(f"Shape of test interactions: {test_interactions.shape}")

    # print(repr(train_interactions))
    # print(repr(test_interactions))

    model1 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     random_state=np.random.RandomState(SEED))

    model1.fit(interactions=train_interactions,
              epochs=NO_EPOCHS);

    eval_precision_lfm = lightfm_prec_at_k(model1, test_interactions,
                                               train_interactions, k=K).mean()
    eval_auc_lfm = auc_score(model1, test_interactions,
                                              train_interactions).mean()

    print(
        "\n------ Without any features - WARP ------",
        f"Precision@K:\t{eval_precision_lfm:.6f}",
        f"AUC:\t{eval_auc_lfm:.6f}",
        sep='\n')

def without_features_BPR():
    dataset = Dataset()

    dataset.fit(users=data['userID'],
                items=data['itemID'])

    (interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)

    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions, test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED))
    # print(f"Shape of train interactions: {train_interactions.shape}")
    # print(f"Shape of test interactions: {test_interactions.shape}")

    # print(repr(train_interactions))
    # print(repr(test_interactions))

    model1 = LightFM(loss='bpr', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     random_state=np.random.RandomState(SEED))

    model1.fit(interactions=train_interactions,
              epochs=NO_EPOCHS);

    eval_precision_lfm = lightfm_prec_at_k(model1, test_interactions,
                                               train_interactions, k=K).mean()
    eval_auc_lfm = auc_score(model1, test_interactions,
                                              train_interactions).mean()

    print(
        "\n------ Without any features - BPR ------",
        f"Precision@K:\t{eval_precision_lfm:.6f}",
        f"AUC:\t{eval_auc_lfm:.6f}",
        sep='\n')


def with_all_features():

    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                data['itemID'],
                item_features=item_features_merged,
                user_features=user_features_merged)


    item_features = dataset2.build_item_features((x, y) for x,y in zip(data.itemID, combined_item_features))
    print(item_features)

    user_features = dataset2.build_user_features((x, y) for x,y in zip(new_data.userID, combined_user_features))
    print(user_features)

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                    )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                    test_interactions2, train_interactions2, user_features=user_features, item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features, item_features=item_features).mean()

    print(
        "\n------ With all features ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')
    print()

def lasso():
    #print(combined_user_features)
    item_features_merged = ['<=1930', '<=1940', '<=1950', '<=1960', '<=1990',
       '1990+', 'Adventure', 'Animation', "Children's", 'Comedy', 'Fantasy',
       'Film-Noir', 'Musical', 'Mystery', 'War', 'Western', 'unknown']
    user_features_merged = ['<=10', '<=30', '50+', 'Zipcode 0', 'Zipcode 2', 'Zipcode 3',
       'Zipcode 5', 'Zipcode 9', 'Zipcode E', 'Zipcode K', 'Zipcode L',
       'Zipcode M', 'Zipcode N', 'Zipcode R', 'Zipcode V', 'Zipcode Y',
       'Occupation administrator', 'Occupation artist', 'Occupation doctor',
       'Occupation educator', 'Occupation engineer',
       'Occupation entertainment', 'Occupation executive',
       'Occupation healthcare', 'Occupation homemaker', 'Occupation lawyer',
       'Occupation marketing', 'Occupation none', 'Occupation programmer',
       'Occupation retired', 'Occupation salesman', 'Occupation scientist',
       'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=10', '<=30', '50+', 'Zipcode 0', 'Zipcode 2', 'Zipcode 3',
       'Zipcode 5', 'Zipcode 9', 'Zipcode E', 'Zipcode K', 'Zipcode L',
       'Zipcode M', 'Zipcode N', 'Zipcode R', 'Zipcode V', 'Zipcode Y',
       'Occupation administrator', 'Occupation artist', 'Occupation doctor',
       'Occupation educator', 'Occupation engineer',
       'Occupation entertainment', 'Occupation executive',
       'Occupation healthcare', 'Occupation homemaker', 'Occupation lawyer',
       'Occupation marketing', 'Occupation none', 'Occupation programmer',
       'Occupation retired', 'Occupation salesman', 'Occupation scientist',
       'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=1930', '<=1940', '<=1950', '<=1960', '<=1990',
       '1990+', 'Adventure', 'Animation', "Children's", 'Comedy', 'Fantasy',
       'Film-Noir', 'Musical', 'Mystery', 'War', 'Western', 'unknown']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With lasso feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')


def random_forest():
    #print(combined_user_features)
    item_features_merged = ["1990+", "<=1980", "<=1990", "<=1960", "<=1950", "<=1970", "<=1940", "Comedy", "Action", "Adventure"
                            "Drame", "Romance", "Thriler", "Musical", "War", "Horror", "Sci-Fi", "Mystery", "Children's", "Fantasy",
                            "Crime", "Animation", "Film-Noir"]
    user_features_merged = ["Occupation healthcare", "<=30", "Gender M", "Gender F", "Occupation writer", "Zipcode 2", "<=50", "Occupation executive",
                            "Zipcode 9", "50+", "Zipcode 6", "Occupation educator", "Zipcode 1", "Zipcode 0", "Zipcode 5", "Occupation other",
                            "Occupation student", "Zipcode 8", "Occupation administrator", "Zipcode 3", "Occupation programmer", "<20", "Zipcode 4",
                            "Occupation none", "Occupation engineer", "Zipcode 7", "Occupation lawyer"]
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ["Occupation healthcare", "<=30", "Gender M", "Gender F", "Occupation writer", "Zipcode 2", "<=50", "Occupation executive",
                            "Zipcode 9", "50+", "Zipcode 6", "Occupation educator", "Zipcode 1", "Zipcode 0", "Zipcode 5", "Occupation other",
                            "Occupation student", "Zipcode 8", "Occupation administrator", "Zipcode 3", "Occupation programmer", "<20", "Zipcode 4",
                            "Occupation none", "Occupation engineer", "Zipcode 7", "Occupation lawyer"]:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ["1990+", "<=1980", "<=1990", "<=1960", "<=1950", "<=1970", "<=1940", "Comedy", "Action", "Adventure"
                            "Drame", "Romance", "Thriler", "Musical", "War", "Horror", "Sci-Fi", "Mystery", "Children's", "Fantasy",
                            "Crime", "Animation", "Film-Noir"]:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With random forest feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')


def gbt():
    #print(combined_user_features)
    item_features_merged = ["1990+", "Thriller", "Romance", "Drama", "Mystery", "Action",
                            "Adventure", "Comedy", "Sci-Fi", "<=1990", "War", "Musical", "Horror", "Crime", "Children's", "<=1980", "Film-Noir",
                            "Animation", "<=1970", "<=1940", "<=1960"
                            ]
    user_features_merged = ["Occupation healthcare", "<=30", "Zipcode 0", "<=50", "Zipcode 9", "Occupation executive", "Zipcode 2", "Occupation writer", "Zipcode 8",
                            "Zipcode 1", "Zipcode 4", "Occupation other", "Zipcode 5", "Gender F", "Occupation student", "Occupation programmer",
                            "<=20", "Gender M", "50+", "zipcode 7", "Occupation engineer", "Zipcode 6", "Occupation administrator", "Occupation educator",
                            "Zipcode 3", "Occupation artist", "Occupation retired", "Occupation librarian", "Occupation marketing"]
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ["Occupation healthcare", "<=30", "Zipcode 0", "<=50", "Zipcode 9", "Occupation executive", "Zipcode 2", "Occupation writer", "Zipcode 8",
                            "Zipcode 1", "Zipcode 4", "Occupation other", "Zipcode 5", "Gender F", "Occupation student", "Occupation programmer",
                            "<=20", "Gender M", "50+", "zipcode 7", "Occupation engineer", "Zipcode 6", "Occupation administrator", "Occupation educator",
                            "Zipcode 3", "Occupation artist", "Occupation retired", "Occupation librarian", "Occupation marketing"]:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ["1990+", "Thriller", "Romance", "Drama", "Mystery", "Action",
                            "Adventure", "Comedy", "Sci-Fi", "<=1990", "War", "Musical", "Horror", "Crime", "Children's", "<=1980", "Film-Noir",
                            "Animation", "<=1970", "<=1940", "<=1960"
                            ]:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With gradient boosted trees feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')


def mutual_inf_filter():
    #print(combined_user_features)
    item_features_merged = ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980', '<=1990', '1990+',
       'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
       'unknown']
    user_features_merged = ['Gender F', 'Gender M', '<=30', '<=50', 'Zipcode 4', 'Zipcode 5',
       'Zipcode 6', 'Zipcode 7', 'Zipcode 9', 'Zipcode N', 'Zipcode R',
       'Zipcode T', 'Zipcode V', 'Zipcode Y', 'Occupation administrator',
       'Occupation executive', 'Occupation healthcare', 'Occupation lawyer',
       'Occupation librarian', 'Occupation marketing', 'Occupation salesman',
       'Occupation scientist', 'Occupation student', 'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['Gender F', 'Gender M', '<=30', '<=50', 'Zipcode 4', 'Zipcode 5',
       'Zipcode 6', 'Zipcode 7', 'Zipcode 9', 'Zipcode N', 'Zipcode R',
       'Zipcode T', 'Zipcode V', 'Zipcode Y', 'Occupation administrator',
       'Occupation executive', 'Occupation healthcare', 'Occupation lawyer',
       'Occupation librarian', 'Occupation marketing', 'Occupation salesman',
       'Occupation scientist', 'Occupation student', 'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980', '<=1990', '1990+',
       'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western',
       'unknown']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With mutual information filter feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')

def chi_square():
    #print(combined_user_features)
    item_features_merged = ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980', '<=1990', '1990+',
       'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Fantasy',
       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Thriller', 'War',
       'Western']
    user_features_merged = ['<=30', '<=50', '50+', 'Zipcode 0', 'Zipcode 1', 'Zipcode 2',
       'Zipcode 3', 'Zipcode 5', 'Zipcode 7', 'Zipcode 8', 'Zipcode 9',
       'Zipcode E', 'Zipcode M', 'Zipcode N', 'Zipcode R', 'Zipcode V',
       'Zipcode Y', 'Occupation administrator', 'Occupation artist',
       'Occupation doctor', 'Occupation educator', 'Occupation engineer',
       'Occupation entertainment', 'Occupation executive',
       'Occupation healthcare', 'Occupation lawyer', 'Occupation marketing',
       'Occupation none', 'Occupation programmer', 'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=30', '<=50', '50+', 'Zipcode 0', 'Zipcode 1', 'Zipcode 2',
       'Zipcode 3', 'Zipcode 5', 'Zipcode 7', 'Zipcode 8', 'Zipcode 9',
       'Zipcode E', 'Zipcode M', 'Zipcode N', 'Zipcode R', 'Zipcode V',
       'Zipcode Y', 'Occupation administrator', 'Occupation artist',
       'Occupation doctor', 'Occupation educator', 'Occupation engineer',
       'Occupation entertainment', 'Occupation executive',
       'Occupation healthcare', 'Occupation lawyer', 'Occupation marketing',
       'Occupation none', 'Occupation programmer', 'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980', '<=1990', '1990+',
       'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Fantasy',
       'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Thriller', 'War',
       'Western']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With chi-square feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')

def forward_selection():
    #print(combined_user_features)
    item_features_merged = ['1990+', 'Adventure', "Children's", 'Comedy', 'Crime', 'Drama',
       'Film-Noir', 'Horror', 'Musical', 'Romance', 'Sci-Fi', 'Thriller',
       'War']
    user_features_merged = ['Gender F', '<=30', '<=50', 'Zipcode 0', 'Zipcode 1', 'Zipcode 2',
       'Zipcode 3', 'Zipcode 4', 'Zipcode 5', 'Zipcode 6', 'Zipcode 7',
       'Zipcode 8', 'Zipcode 9', 'Zipcode E', 'Zipcode K', 'Zipcode L',
       'Zipcode N', 'Zipcode R', 'Occupation administrator',
       'Occupation artist', 'Occupation doctor', 'Occupation educator',
       'Occupation engineer', 'Occupation executive', 'Occupation healthcare',
       'Occupation homemaker', 'Occupation lawyer', 'Occupation librarian',
       'Occupation marketing', 'Occupation none', 'Occupation other',
       'Occupation programmer', 'Occupation salesman', 'Occupation scientist',
       'Occupation student', 'Occupation technician', 'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['Gender F', '<=30', '<=50', 'Zipcode 0', 'Zipcode 1', 'Zipcode 2',
       'Zipcode 3', 'Zipcode 4', 'Zipcode 5', 'Zipcode 6', 'Zipcode 7',
       'Zipcode 8', 'Zipcode 9', 'Zipcode E', 'Zipcode K', 'Zipcode L',
       'Zipcode N', 'Zipcode R', 'Occupation administrator',
       'Occupation artist', 'Occupation doctor', 'Occupation educator',
       'Occupation engineer', 'Occupation executive', 'Occupation healthcare',
       'Occupation homemaker', 'Occupation lawyer', 'Occupation librarian',
       'Occupation marketing', 'Occupation none', 'Occupation other',
       'Occupation programmer', 'Occupation salesman', 'Occupation scientist',
       'Occupation student', 'Occupation technician', 'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['1990+', 'Adventure', "Children's", 'Comedy', 'Crime', 'Drama',
       'Film-Noir', 'Horror', 'Musical', 'Romance', 'Sci-Fi', 'Thriller',
       'War']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With forward selection feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')

def backward_elimination():
    #print(combined_user_features)
    item_features_merged = ['<=1930', '1990+',
       'Action', 'Adventure', 'Comedy', 'Drama', 'Film-Noir', 'Musical',
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']
    user_features_merged = ['Gender F', 'Gender M', '<=30', '<=50', 'Zipcode 0', 'Zipcode 1',
       'Zipcode 2', 'Zipcode 3', 'Zipcode 4', 'Zipcode 5', 'Zipcode 6',
       'Zipcode 7', 'Zipcode 8', 'Zipcode 9', 'Zipcode E', 'Zipcode M',
       'Zipcode N', 'Zipcode Y', 'Occupation administrator',
       'Occupation doctor', 'Occupation engineer', 'Occupation executive',
       'Occupation healthcare', 'Occupation homemaker', 'Occupation lawyer',
       'Occupation librarian', 'Occupation marketing', 'Occupation none',
       'Occupation other', 'Occupation programmer', 'Occupation retired',
       'Occupation salesman', 'Occupation scientist', 'Occupation student',
       'Occupation technician', 'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['Gender F', 'Gender M', '<=30', '<=50', 'Zipcode 0', 'Zipcode 1',
       'Zipcode 2', 'Zipcode 3', 'Zipcode 4', 'Zipcode 5', 'Zipcode 6',
       'Zipcode 7', 'Zipcode 8', 'Zipcode 9', 'Zipcode E', 'Zipcode M',
       'Zipcode N', 'Zipcode Y', 'Occupation administrator',
       'Occupation doctor', 'Occupation engineer', 'Occupation executive',
       'Occupation healthcare', 'Occupation homemaker', 'Occupation lawyer',
       'Occupation librarian', 'Occupation marketing', 'Occupation none',
       'Occupation other', 'Occupation programmer', 'Occupation retired',
       'Occupation salesman', 'Occupation scientist', 'Occupation student',
       'Occupation technician', 'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=1930', '1990+',
       'Action', 'Adventure', 'Comedy', 'Drama', 'Film-Noir', 'Musical',
       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'unknown']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With backward elimination feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')

def recursive_feature_elimination():
    #print(combined_user_features)
    item_features_merged = ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980',
       '<=1990', '1990+', 'Action', 'Adventure', 'Animation', "Children's",
       'Comedy', 'Crime', 'Drama', 'Horror', 'Musical', 'Mystery', 'Romance',
       'Sci-Fi', 'Thriller', 'War']
    user_features_merged = ['Gender F', 'Gender M', '<=20', '<=30', '<=50', '50+', 'Zipcode 0',
       'Zipcode 1', 'Zipcode 2', 'Zipcode 3', 'Zipcode 4', 'Zipcode 5',
       'Zipcode 6', 'Zipcode 7', 'Zipcode 8', 'Zipcode 9',
       'Occupation administrator', 'Occupation artist', 'Occupation educator',
       'Occupation engineer', 'Occupation executive', 'Occupation healthcare',
       'Occupation librarian', 'Occupation other', 'Occupation programmer',
       'Occupation scientist', 'Occupation student', 'Occupation technician',
       'Occupation writer']
    dataset2 = Dataset()
    dataset2.fit(data['userID'],
                 data['itemID'],
                 item_features=item_features_merged,
                 user_features=user_features_merged)

    for a in combined_user_features:
        a1 = a
        for b in a1[:]:
            if b not in ['Gender F', 'Gender M', '<=20', '<=30', '<=50', '50+', 'Zipcode 0',
       'Zipcode 1', 'Zipcode 2', 'Zipcode 3', 'Zipcode 4', 'Zipcode 5',
       'Zipcode 6', 'Zipcode 7', 'Zipcode 8', 'Zipcode 9',
       'Occupation administrator', 'Occupation artist', 'Occupation educator',
       'Occupation engineer', 'Occupation executive', 'Occupation healthcare',
       'Occupation librarian', 'Occupation other', 'Occupation programmer',
       'Occupation scientist', 'Occupation student', 'Occupation technician',
       'Occupation writer']:
                a.remove(b)

    for a in combined_item_features:
        a1 = a
        for b in a1[:]:
            if b not in ['<=1940', '<=1950', '<=1960', '<=1970', '<=1980',
       '<=1990', '1990+', 'Action', 'Adventure', 'Animation', "Children's",
       'Comedy', 'Crime', 'Drama', 'Horror', 'Musical', 'Mystery', 'Romance',
       'Sci-Fi', 'Thriller', 'War']:
                a.remove(b)

    item_features = dataset2.build_item_features((x, y) for x, y in zip(data.itemID, combined_item_features))

    user_features = dataset2.build_user_features((x, y) for x, y in zip(new_data.userID, combined_user_features))

    interactions2, weights2 = dataset2.build_interactions(data.iloc[:, 0:3].values)

    train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
        interactions2,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEED)
    )

    model2 = LightFM(loss='warp', no_components=NO_COMPONENTS,
                     learning_rate=LEARNING_RATE,
                     item_alpha=ITEM_ALPHA,
                     user_alpha=USER_ALPHA,
                     random_state=np.random.RandomState(SEED)
                     )

    model2.fit(interactions=train_interactions2,
               user_features=user_features,
               item_features=item_features,
               epochs=NO_EPOCHS
               )

    eval_precision2 = lightfm_prec_at_k(model2,
                                        test_interactions2, train_interactions2, user_features=user_features,
                                        item_features=item_features, k=K).mean()
    eval_recall2 = auc_score(model2, test_interactions2, train_interactions2, user_features=user_features,
                             item_features=item_features).mean()

    print(
        "\n------ With recursive feature elimination feature selection technique ------",
        f"Precision@K:\t{eval_precision2:.6f}",
        f"AUC:\t{eval_recall2:.6f}",
        sep='\n')

# print(without_features_WARP())
# print(with_all_features())
print(lasso())
print(random_forest())
print(gbt())
print(mutual_inf_filter())
print(chi_square())
print(forward_selection())
print(backward_elimination())
print(recursive_feature_elimination())
