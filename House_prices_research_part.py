# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:57:48 2020

@author: Admin
"""
#Importing libs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Importing data
df_train = pd.read_csv('Data/train.csv')
df_test = pd.read_csv('Data/test.csv')

df = pd.concat([df_train, df_test])

#Look at data
df_train.head()
desc_data = df_train.describe()

#Look at types of data
type_data = df_train.dtypes

#Show histogram of SalePrice
sns.distplot(df_train['SalePrice'])

#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(corrmat, square=True)

#the most correleted value with saleprice
k = 10
columns = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[columns].values.T)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=columns.values, xticklabels=columns.values)
plt.show()

'''---------------Plot Part---------------'''
#Focus on the most correleted values
#scatter plot var/saleprice
def scatter_plot(var):
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))

#box plot fullbath/saleprice
def box_plot(var):
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)

#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
scatter_plot(var)

#scatter plot garagearea/saleprice
var = 'GarageArea'
scatter_plot(var)

#scatter plot garagearea/saleprice
var = 'LotArea'
scatter_plot(var)

#scatter plot MasVnrArea/saleprice
var = 'MasVnrArea'
scatter_plot(var)

#scatter plot WoodDeckSF/saleprice
var = 'WoodDeckSF'
scatter_plot(var)

#box plot overallqual/overallcond
var = 'OverallCond'
scatter_plot(var)

#box plot overallqual/saleprice
var = 'OverallQual'
box_plot(var)

#box plot yearbuilt/saleprice
data = df_train[['YearBuilt', 'SalePrice']].groupby('YearBuilt').mean()
data.plot(ylim=(0,500000))

#box plot fullbath/saleprice
var = 'FullBath'
box_plot(var)

#box plot MSSubClass/saleprice
var = 'MSSubClass'
box_plot(var)

#box plot MSZoning/saleprice
var = 'MSZoning'
box_plot(var)

#box plot Street/saleprice
var = 'Street'
box_plot(var)

#box plot LotShape/saleprice
var = 'LotShape'
box_plot(var)

#box plot LandContour/saleprice
var = 'LandContour'
box_plot(var)

#box plot Utilities/saleprice
var = 'Utilities'
box_plot(var)

#box plot LotConfig/saleprice
var = 'LotConfig'
box_plot(var)

#box plot LandSlope/saleprice
var = 'LandSlope'
box_plot(var)

#box plot Neighborhood/saleprice
var = 'Neighborhood'
box_plot(var)

#box plot Neighborhood/saleprice
var = 'BldgType'
box_plot(var)

#box plot HouseStyle/saleprice
var = 'HouseStyle'
box_plot(var)

#box plot RoofStyle/saleprice
var = 'RoofStyle'
box_plot(var)

#box plot RoofMatl/saleprice
var = 'RoofMatl'
box_plot(var)

#box plot Exterior1st/saleprice
var = 'Exterior1st'
box_plot(var)

#box plot MasVnrType/saleprice
var = 'MasVnrType'
box_plot(var)

#box plot Foundation/saleprice
var = 'Foundation'
box_plot(var)

#box plot CentralAir/saleprice
var = 'CentralAir'
box_plot(var)

#box plot Electrical/saleprice
var = 'Electrical'
box_plot(var)

#box plot Functional/saleprice
var = 'Functional'
box_plot(var)

#box plot Fireplaces/saleprice
var = 'Fireplaces'
box_plot(var)

#box plot GarageType/saleprice
var = 'GarageType'
box_plot(var)

#box plot GarageFinish/saleprice
var = 'GarageFinish'
box_plot(var)

#box plot PavedDrive/saleprice
var = 'PavedDrive'
box_plot(var)

#box plot Fence/saleprice
var = 'Fence'
box_plot(var)

#box plot SaleCondition/saleprice
var = 'SaleCondition'
box_plot(var)

#Look on other values
#box plot overallqual/overallcond
var_1 = 'OverallQual'
var_2 = 'OverallCond'
data = pd.concat([df_train[var_1], df_train[var_2]], axis=1)
fig = sns.boxplot(x=var_1, y=var_2, data=data)
#fig.axis(ymin=0, ymax=800000)

'''--------------- Make new features ---------------'''        
#Deal with missing data
missing_values = df_train.isnull().sum().sort_values(ascending=False)


#Nan == 0
for i in df_train.columns[df_train.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    #df_test[i].fillna(0,inplace=True)
    #print(i, df_train[i].dtypes)
    if df_train[i].dtypes == 'float64':
        df_train[i].fillna(0,inplace=True)
        
#Data correction
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
    df_train = df_train.replace({feature: dict_state})

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


#check built 
#yearbuilt_min_max = df_train['YearBuilt'].max()-df_train['YearBuilt'].min() 
df_train['BuildingAge'] = df_train.apply(lambda x: x['YrSold'] - x['YearBuilt'], axis=1)
df_train['Years_Since_Renovation'] = df_train.apply(lambda x: x['YrSold']-x['YearRemodAdd'], axis=1)
df_train['QuantityBath'] = df_train.apply(lambda x: x['FullBath'] + x['BsmtFullBath'], axis=1)
df_train['TotalArea'] = df_train.apply(lambda x: x['TotalBsmtSF'] + x['GrLivArea'], axis=1)
df_train['QualxCond'] = df_train.apply(lambda x: x['OverallQual'] * x['OverallCond'], axis=1)
df_train['Total/Unfinish_Bsmt'] = df_train.apply(lambda x: Total_Unfinish_Bsmt(x), axis=1)
df_train['GarageAge'] = df_train.apply(lambda x: x['YrSold'] - x['GarageYrBlt'], axis=1)
df_train['IsGarageFinish'] = df_train.apply(lambda x: Is_garage_finish(x), axis=1)
df_train['IsPavedDrive'] = df_train.apply(lambda x: Is_paved_drive(x), axis=1)
df_train['IsPool'] = df_train.apply(lambda x: Is_pool(x), axis=1)


list_of_new_features = ['BuildingAge', 'Years_Since_Renovation','QuantityBath', 'TotalArea', 'QualxCond',
                        'Total/Unfinish_Bsmt', 'GarageAge']

'''Look at new features'''
#Correlation with new features
new_feature_corr_df = pd.concat([df_train['SalePrice'], df_train[list_of_features_to_replace], df_train[list_of_new_features]], axis=1)

corrmat_new_feature = corr_df.corr()
f, ax = plt.subplots(figsize=(20, 15))
sns.heatmap(corrmat_new_feature, square=True, cbar=True, annot=True, fmt='.2f', annot_kws={'size': 20})

#scatter plot LotArea/saleprice
var = 'LotArea'
scatter_plot(var)

#scatter plot LotFrontage/saleprice
var = 'LotFrontage'
scatter_plot(var)

#scatter plot totalarea/saleprice
var = 'TotalArea'
scatter_plot(var)

#box plot ExterQual/saleprice
var = 'ExterQual'
box_plot(var)

#box plot BsmtQual/saleprice
var = 'BsmtQual'
box_plot(var)

#box plot BsmtCond/saleprice
var = 'BsmtCond'
box_plot(var)

#box plot BsmtExposure/saleprice
var = 'BsmtExposure'
box_plot(var)

#box plot BsmtExposure/saleprice
var = 'BsmtFinType1'
box_plot(var)

#box plot quantitybath/saleprice
var = 'QuantityBath'
box_plot(var)

#box plot overallqual/saleprice
var = 'QualxCond'
box_plot(var)

#box plot HeatingQC/saleprice
var = 'HeatingQC'
box_plot(var)

#box plot KitchenQual/saleprice
var = 'KitchenQual'
box_plot(var)

#box plot FireplaceQu/saleprice
var = 'FireplaceQu'
box_plot(var)

#box plot IsGarageFinish/saleprice
var = 'IsGarageFinish'
box_plot(var)

#box plot IsPool/saleprice
var = 'IsPool'
box_plot(var)

def correlation_with_Price(var=''):
    print(df_train['SalePrice'].corr(df_train[var]))
    
    