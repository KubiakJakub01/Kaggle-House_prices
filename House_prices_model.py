# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:51:05 2020

Kaggle competition:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

@author: Admin
"""

#Importing libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split

#Importing data
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')
test_ID = list(df_test['Id'].values)

'''--------------- Clean data ---------------'''
#Deal with missing data
#missing_values = df.isnull().sum().sort_values(ascending=False)  
def fill_missing_data(df):
    #Nan == 0 for float type value
    for i in df.columns[df.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        df[i].fillna(0,inplace=True)
        
    return df

def replace_categorical_to_numeric_data(df):
    #Replace
    dict_state = {
            'Ex': 5,
            'Gd': 4,
            'TA': 3,
            'Fa': 2,
            'Po': 1,
            np.nan: 0    
        }   
    list_of_features_to_replace = ['GarageQual', 'GarageCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                                  'KitchenQual', 'FireplaceQu', 'PoolQC']
                                
    for feature in list_of_features_to_replace:
        df = df.replace({feature: dict_state})
        
    return df
        

'''--------------- Make new features ---------------'''
def Total_Unfinish_Bsmt(x):
    if x['TotalBsmtSF'] != 0:
        return x['BsmtUnfSF']/x['TotalBsmtSF']
    else:
        return -1

def Is_garage_finish(x):
    if x['GarageFinish'] == 'Fin' or x['GarageFinish'] == 'RFn':
        return 1
    else:
        return 0
def Is_paved_drive(x):
    if x['PavedDrive'] == 'Y':
        return 1
    else:
        return 0

def Is_pool(x):
    if x['PoolArea'] == 0:
        return 0
    else:
        return 1
def make_new_features(df):
    df['BuildingAge'] = df.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)
    df['YearsSinceRenovation'] = df.apply(lambda x: x['YrSold']-x['YearRemodAdd'], axis=1)
    df['QuantityBath'] = df.apply(lambda x: x['FullBath'] + x['BsmtFullBath'], axis=1)
    df['TotalArea'] = df.apply(lambda x: x['TotalBsmtSF'] + x['GrLivArea'], axis=1)
    df['Total/UnfinishBsmt'] = df.apply(lambda x: Total_Unfinish_Bsmt(x), axis=1)
    df['GarageAge'] = df.apply(lambda x: x['YrSold'] - x['GarageYrBlt'], axis=1)
    df['IsGarageFinish'] = df.apply(lambda x: Is_garage_finish(x), axis=1)
    df['IsPavedDrive'] = df.apply(lambda x: Is_paved_drive(x), axis=1)
    df['IsPool'] = df.apply(lambda x: Is_pool(x), axis=1)
    
    return df



'''--------------- Select features ---------------'''
#Selecting the most valuable features
FEATURE_LIST = ['BuildingAge', 'YearsSinceRenovation','QuantityBath', 'TotalArea', 'Total/UnfinishBsmt', 'GarageAge','IsGarageFinish',
                'IsPavedDrive','IsPool','GarageQual', 'ExterQual', 'BsmtQual', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'PoolQC',
                'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Neighborhood', 'HouseStyle', 'RoofStyle', 'MasVnrArea',
                'Foundation', 'BsmtFinType1', 'CentralAir', 'TotRmsAbvGrd', 'Fireplaces', 'GarageType',
                'GarageArea', 'WoodDeckSF']

#List with index of outliners
DROP_LIST = [691, 1182, 1298, 523, 496, 934]    

'''--------------- Prepare data ---------------'''
def prepare_data(df):
    df = fill_missing_data(df)
    df = replace_categorical_to_numeric_data(df)
    df = make_new_features(df)
    df = df[FEATURE_LIST]
    df = pd.get_dummies(df)
    df = StandardScaler().fit_transform(df)
    return df

#Dealing outliners
df_train = df_train.drop(DROP_LIST)
#log transform
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#Spliting data 
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
#Preparing data
X = prepare_data(X)
X_test = df_test.copy()
X_test = prepare_data(df_test)

#Spliting for local test and train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)


'''---------------Make model---------------'''
'''TPOT'''
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
from sklearn.metrics import r2_score

# define evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTRegressor(generations=1, population_size=100, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(X_train, y_train)
#model.export('tpot_best_model.py')

pred = model.predict(X_val)
pred = np.exp(pred)
correct_results = np.exp(y_val)
print(r2_score(correct_results, pred))
          

final_result = np.exp(model.predict(X_test))

#Blended model
# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
# Setup cross validation folds
kf = KFold(n_splits=12, random_state=42, shuffle=True)

# Define error metrics
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y_val, scoring="neg_mean_squared_error", cv=kf))
    return (rmse)

# Light Gradient Boosting Regressor
lightgbm = LGBMRegressor(objective='regression', 
                       num_leaves=6,
                       learning_rate=0.01, 
                       n_estimators=7000,
                       max_bin=200, 
                       bagging_fraction=0.8,
                       bagging_freq=4, 
                       bagging_seed=8,
                       feature_fraction=0.2,
                       feature_fraction_seed=8,
                       min_sum_hessian_in_leaf = 11,
                       verbose=-1,
                       random_state=42)

# XGBoost Regressor
xgboost = XGBRegressor(learning_rate=0.01,
                       n_estimators=6000,
                       max_depth=4,
                       min_child_weight=0,
                       gamma=0.6,
                       subsample=0.7,
                       colsample_bytree=0.7,
                       objective='reg:linear',
                       nthread=-1,
                       scale_pos_weight=1,
                       seed=27,
                       reg_alpha=0.00006,
                       random_state=42)

# Ridge Regressor
ridge_alphas = [1e-15, 1e-10, 1e-8, 9e-4, 7e-4, 5e-4, 3e-4, 1e-4, 1e-3, 5e-2, 1e-2, 0.1, 0.3, 1, 3, 5, 10, 15, 18, 20, 30, 50, 75, 100]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=ridge_alphas, cv=kf))

# Support Vector Regressor
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003))

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=42)  

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=42)

# Stack up all the models above, optimized using xgboost
stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ridge, gbr, rf),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)

#Get cross validation scores for each model
scores = {}

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))
scores['lgb'] = (score.mean(), score.std())