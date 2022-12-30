#####################
#  VERİ SETİ HİKAYESİ
#####################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

df = train.append(test, ignore_index=False)
df.head()
df.shape

# Functions

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, cat_but_car, num_cols

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

def outlier_thresholds(dataframe, variable, low_quantile=0.5, up_quantile=0.95):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit

# Aykırı değer kontrolü
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# Corelation
def get_redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop
def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]
#



# MSSubClass
df['MSSubClass'].nunique()
df['MSSubClass'].isnull().sum()
mssubclass_index = df.groupby('MSSubClass').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)
mssubclass_index = mssubclass_index.reset_index()
df['MSSubClass'].replace({60: 15, 120: 14, 75: 13, 20: 12, 80: 11, 70: 10, 40: 9, 85: 8, 50: 7, 160: 6, 90: 5,
                             190: 4, 45: 3, 180: 2, 30: 1, 150: 15}, inplace=True)

# MSZoning
df['MSZoning'].nunique()
df['MSZoning'].isnull().sum()
df['MSZoning'].value_counts()
df['MSZoning'].fillna(3, inplace=True)
df.groupby('MSZoning').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)
df['MSZoning'].replace({'FV': 4, 'RL': 3, 'RH': 2, 'RM': 2, 'C (all)': 1}, inplace=True)
df['MSZoning'] = df['MSZoning'].astype('int64')

# Street
df['Street'].value_counts() #1454 pave 6 tane grvl var. Anlamlı bir değişken olmadığı için silinmeli
df.drop('Street', axis=1, inplace=True)

# Alley
df['Alley'].value_counts()
df['Alley'].isnull().sum() # 1370 tane boş değer var o yüzden gereksiz sütun
df.drop('Alley', axis=1, inplace=True)

# LotShape
df['LotShape'].value_counts()
df['LotShape'].isnull().sum()
df.groupby('LotShape').agg({'SalePrice': 'mean'})

# LandContour
df['LandContour'].value_counts()
df['LandContour'].isnull().sum()
df.groupby('LandContour').agg({'SalePrice': 'mean'})

# Utilities
df['Utilities'].value_counts() # 1459 - 1
df.drop('Utilities', axis=1, inplace=True)

# LotConfig
df['LotConfig'].value_counts()
df['LotConfig'].isnull().sum()
df.groupby('LotConfig').agg({'SalePrice': 'mean'})
df['LotConfig'].replace({'Inside': 1, 'FR2': 1, 'Corner': 1, 'CulDSac': 2, 'FR3': 2},
                           inplace=True)

# LandSlope
df.drop('LandSlope', axis=1, inplace=True)

# Neighborhood
df['Neighborhood'].value_counts()
df['Neighborhood'].isnull().sum()
len(df[df['Neighborhood'] == 'Blueste']) / df.shape[0]  #0.00136

df.groupby('Neighborhood').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)

# Condition1, Condition2
df['Condition1'].value_counts()
df['Condition2'].value_counts()

# BldgType
df['BldgType'].value_counts()
df.groupby('BldgType').agg({'SalePrice': 'mean'})

# HouseStyle
df['HouseStyle'].value_counts()
df.groupby('HouseStyle').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)

# OverallQual
df['OverallQual'].value_counts()
df.groupby('OverallQual').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)

# OverallCond
df['OverallCond'].value_counts()
df.groupby('OverallCond').agg({'SalePrice': 'mean'}).sort_values('SalePrice', ascending=False)

# RoofStyle
df['RoofStyle'].value_counts()
df['RoofStyle'].isnull().sum()

df['RoofMatl'].value_counts()
df['RoofMatl'].isnull().sum()

# Exter
df['Exterior1st'].value_counts()
df['Exterior1st'].isnull().sum()

df['Exterior2nd'].value_counts()
df['Exterior2nd'].isnull().sum()

# Mas
mas = ['MasVnrType', 'MasVnrArea']
[df[col].fillna(df[col].mode()[0], inplace=True) for col in mas]
[df[col].isnull().sum() for col in mas]

# Exter
df['ExterQual'].value_counts()
df['ExterQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

df['ExterCond'].value_counts()
df['ExterCond'].replace({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)

# Found
df['Foundation'].value_counts()
df['Foundation'].isnull().sum()

# Bsmt
bsmt_obj = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1']
bsmt_val = ['BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']

bsmt_mis_col = ['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
bsmt_mis_col2 = ['BsmtQual', 'BsmtCond']
[df[col].fillna('No', inplace=True) for col in bsmt_mis_col]
[df[col].fillna(0, inplace=True) for col in bsmt_mis_col2]

df['BsmtExposure'].replace({'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3}, inplace=True)
df['BsmtQual'].replace({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)
df['BsmtCond'].replace({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4}, inplace=True)

df.head()

# Heating
heat = ['Heating', 'HeatingQC']

df['HeatingQC'].replace({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4}, inplace=True)
df['HeatingQC'].value_counts()

# CentralAir
df['CentralAir'].value_counts()
df['CentralAir'].replace({'Y': 1, 'N': 0}, inplace=True)
df.groupby('CentralAir').agg({'SalePrice': 'mean'})

# Electrical
df['Electrical'].value_counts()

# Floor sq. feet
flrsqrfeet = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea']

# Bath
bath = ['BsmtHalfBath', 'FullBath', 'HalfBath', 'BsmtFullBath']

# Above Grade
abvgrd = ['BedroomAbvGr', 'KitchenAbvGr']

# KitchenQual
df['KitchenQual'].value_counts()
df['KitchenQual'].fillna(df['KitchenQual'].mode()[0], inplace=True)
df['KitchenQual'].isnull().sum()

df.groupby('KitchenQual').agg({'SalePrice': 'mean'})
df['KitchenQual'].replace({'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1}, inplace=True)

# Functional
df.drop('Functional', axis=1, inplace=True)

# Fireplace
df['Fireplaces'].isnull().sum()
df['Fireplaces'].value_counts()
df.groupby('Fireplaces').agg({'SalePrice': 'mean'})

df['FireplaceQu'].value_counts()
df['FireplaceQu'].isnull().sum()
df['FireplaceQu'].replace({np.NaN: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)

# Garage
garage_obj = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars']
garage_int = ['GarageArea']

[df[col].value_counts() for col in garage_obj]
[df[col].isnull().sum() for col in garage_obj]

[df[col].fillna('NoGarage', inplace=True) for col in garage_obj]
df['GarageYrBlt'].fillna(2005, inplace=True)

df['GarageQual'].replace({'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)
df['GarageCond'].replace({'NoGarage': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, inplace=True)

df['GarageCars'].fillna(df['GarageCars'].mode()[0], inplace=True)
df['GarageCars'].replace({'NoGarage': 0}, inplace=True)
df['GarageCars'] = df['GarageCars'].astype('int64')

# PavedDrive
df['PavedDrive'].unique()
df.groupby('PavedDrive').agg({'SalePrice': 'mean'})
df['PavedDrive'].replace({'N': 0, 'P': 1, 'Y': 2}, inplace=True)

# PoolQC
df.drop(['PoolArea', 'PoolQC'], axis=1, inplace=True)

# Fence, MiscFeature and MiscVal
df['Fence'].value_counts()
df['MiscFeature'].value_counts()

df.drop(['Fence', 'MiscFeature', 'MiscVal'], axis=1, inplace=True)
df.drop(['SaleType', 'SaleCondition'], axis=1, inplace=True)

# LotFrontage
num_summary(df, 'LotFrontage', plot=True)
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)

# Missing Value Analysis
df.isnull().sum()
miss_col = ['Exterior1st', 'Exterior2nd', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
            'Electrical', 'BsmtFullBath', 'BsmtHalfBath', 'GarageArea']

[df[col].fillna(df[col].mode()[0], inplace=True) for col in miss_col]

df.drop(['Heating', 'Electrical', '3SsnPorch', 'PavedDrive', 'BsmtHalfBath', 'LowQualFinSF', 'Condition1',
         'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
         'BsmtFinType1', 'BsmtFinType2'], axis=1, inplace=True)

df_copied = df.copy()


df.head()

###########################
#       FEATURE ENG.      #
###########################

# Outlier Detection
cat_cols, cat_but_car, num_cols = grab_col_names(df)

num_cols.remove('SalePrice')

for col in num_cols:
    print(check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)
    print(check_outlier(df, col))


# Create New Features
plt.scatter(x=df['LotFrontage'], y=df['SalePrice'], c='red')
plt.show()

df['New_MS_Sum'] = df['MSSubClass'] + df['MSZoning']
df['New_MS_Mul'] = df['MSSubClass'] * df['MSZoning']


total_qual = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual',
              'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual',
              'GarageCond']
i = 0
j = 0
for i in total_qual:
    for j in total_qual:
        df[f'New_{i}_{j}_sum'] = df[i] + df[j]
        df[f'New_{i}_{j}_mul'] = df[i] * df[j]

df['New_Constructor_Age'] = dt.datetime.today().year - df['YearBuilt']
df['New_LastRemod_Age'] = dt.datetime.today().year - df['YearRemodAdd']

df["New_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["New_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2

df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df.WoodDeckSF
df["NEW_TotalHouseArea"] = df.New_TotalFlrSF + df.TotalBsmtSF
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

# Ratios
area_cols = [col for col in df.columns if 'Area' in col]

df['MasVnrArea'].replace({0: 1}, inplace=True)

# Create ratio variables
j = 1
for i in area_cols:
    for j in area_cols:
        df[f'New_{i}_{j}_Ratio'] = df[i] / df[j]

df.drop(['New_LotArea_LotArea_Ratio', 'New_MasVnrArea_MasVnrArea_Ratio', 'New_GrLivArea_GrLivArea_Ratio',
         'New_GarageArea_GarageArea_Ratio', 'New_NEW_PorchArea_NEW_PorchArea_Ratio', 'New_NEW_TotalHouseArea_NEW_TotalHouseArea_Ratio'],
        axis=1, inplace=True)

df['New_GarageArea_NEW_PorchArea_Ratio'].fillna(df['New_GarageArea_NEW_PorchArea_Ratio'].mode()[0], inplace=True)
df['New_NEW_PorchArea_GarageArea_Ratio'].fillna(df['New_NEW_PorchArea_GarageArea_Ratio'].mode()[0], inplace=True)



# Create categoric variables with Target
cat_list = ['LotShape', 'LandContour', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'BsmtFinType1', 'GarageType', 'Heating']


float64 = []
[float64.append(col) for col in df.columns if df[col].dtypes == 'float64']

for i in float64:
    df[i] = df[i].astype('float32')

df.head()



## Corelation Analysis


print("Top Absolute Correlations !")
print(get_top_abs_correlations(train.select_dtypes(include=['int32','int64']), 30))

df.drop(['TotRmsAbvGrd', '1stFlrSF'], axis=1, inplace=True)



############################
#        ENCODING          #
############################

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


ohe_cols = ['LotShape', 'LandContour', 'Neighborhood', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure',
            'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'GarageType', 'GarageFinish',
            'Heating', 'Electrical']

ohe_cols2 = ['LotShape', 'LandContour', 'Neighborhood', 'RoofMatl', 'Foundation', 'Functional', 'GarageType',
             'GarageFinish', ]

df.head()

df = one_hot_encoder(df, ohe_cols2, drop_first=True)


###############
#   MODELING  #
###############

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice'] # np.log1p(df['SalePrice'])
X = train_df.drop("SalePrice", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

####################
#  PLOT IMPORTANCE #
####################

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(50,50))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = GradientBoostingRegressor()
model.fit(X, y)

plot_importance(model, X)

df.head()


















































































