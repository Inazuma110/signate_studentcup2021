import pandas as pd
import xgboost as xgb
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import model_selection
import optuna
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from pandas_profiling import ProfileReport
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
# from xfeat import SelectCategorical, LabelEncoder, Pipeline, ConcatCombination, SelectNumerical, \
#     ArithmeticCombinations, TargetEncoder, aggregation, GBDTFeatureSelector, GBDTFeatureExplorer
import seaborn as sns
import warnings
import featuretools as ft
import featuretools.variable_types as vtypes
import imblearn as imb
import collections
import pycm
import math
from sklearn.manifold import TSNE
warnings.simplefilter('ignore')

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(0)

def save_result(array, fname):
    res = pd.DataFrame([range(4046, 8092), array]).T
    res.to_csv(fname, index=False, header=False)
    print('saved!!')

test = pd.read_csv("./data/test.csv")
train = pd.read_csv("./data/train.csv")

y = train['genre']
train.drop(columns='genre', inplace=True)

# フォーラムを参考に特徴量重み付け
def weights_df(df):
    feature_weight = {}
    for i in df.columns:
        if 'region_' in i:
            feature_weight[i] = 100

    feature_weight["popularity"] = 8.0
    feature_weight["tempo"] = 0.001
    feature_weight["tempo_range"] = 0.001
#     feature_weight["svc"] = 10
    feature_weight['count_nan'] = 100
    for col in df.columns:
        if not col in feature_weight:
            feature_weight[col] = 1

    features = feature_weight.keys()
    features_weight = [feature_weight[col] for col in features]
    df = df.astype('float64')
    return df[features] * feature_weight

# 各音楽の特徴量の欠損値を数え，それを特徴量とする．
def count_nan(df):
    df["count_nan"] = 0
    for col in [
        "acousticness",
        "positiveness",
        "danceability",
        "energy",
        "liveness",
        "speechiness",
        "instrumentalness",
    ]:
        df["count_nan"] += df[col].isna()
    return df


# テンポを平均と差にする
def tempo2mean_and_range(df):
    # tempo カラムの各要素を '-' で区切ってリストにする
    df['tempo'] = df['tempo'].apply(lambda x: x.split('-'))
    df['tempo_min'] = df['tempo'].apply(lambda x: x[0])
    df['tempo_max'] = df['tempo'].apply(lambda x: x[1])
    df['tempo_min'] = df['tempo_min'].astype(float)
    df['tempo_max'] = df['tempo_max'].astype(float)
    df['tempo_range'] = df['tempo_max'] - df['tempo_min']
    df['tempo'] = (df['tempo_max']+df['tempo_min'])/2
    df['tempo'] = df['tempo'].astype(float)
    df = df.drop(columns='tempo_max')
    df = df.drop(columns='tempo_min')

    return df

# カテゴリカルデータをすべて非One-Hot表現に変換
def region2not_onehot(df):
    df['region'] = df['region'].astype('category').cat.codes
    return df

# カテゴリカルデータをすべてOne-Hot表現に変換
def region2onehot(df):
    tmp = df['tempo']
    df = df.drop(columns='tempo')
    df = pd.get_dummies(df, drop_first=True)
    df['tempo'] = tmp
    return df

# 各regionごとの統計量を特徴量とする
def aggregation(df):
    df['region'] = df['region'].astype('category')
    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id='df', dataframe=df, index='index')
    es = es.normalize_entity(base_entity_id='df',
                         new_entity_id='region',
                         index='region')
    feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity='df',
                                      trans_primitives=[],
#                                       agg_primitives=['count', 'sum', 'mean'],
                                      max_depth=2,)
    return feature_matrix


# 各特徴量の2つの組み合わせから新たな特徴量を作る
def comb_feature(df):
    df['region'] = df['region'].astype('category')
    es = ft.EntitySet()
    es = es.entity_from_dataframe(entity_id='df', dataframe=df, index='index')
    df, feature_defs = ft.dfs(entityset=es,
                                      target_entity='df',
                                      trans_primitives=['divide_numeric'],
                                      agg_primitives=[],
                                      max_depth=1)
    df=df.replace([np.inf, -np.inf], 1e6)
    return df

# 外れ値を抑える
def cliping(df):
    for i in df.columns:
        p01 = df[i].quantile(0.01)
        p99 = df[i].quantile(0.99)
        df[i] = df[i].clip(p01, p99)
    return df

# tsneで低次元に埋め込む
def tsne_embedded(df):
    emb = TSNE(n_components=2).fit_transform(df)
    df['emb0'] = emb[:, 0]
    df['emb1'] = emb[:, 1]
    return df


# 前処理
def preprocessing(train_df, test_df):
    df = pd.concat([train_df, test_df])
    df = region2onehot(df)
    df = tempo2mean_and_range(df)
#     df = comb_feature(df[features])
#     df2 = aggregation(df[features])
#     df = pd.merge(df, df2)

#     df['region'] = df['region'].astype('int64')
#     df = cliping(df)
    df = count_nan(df)
    df = df.fillna(-1)
#     df = tsne_embedded(df)
#     df['region'] = df['region'].astype('category')
#     df = df.drop(columns='region_region_M') # because importance is zero.
    df = df.drop(columns='index')
    df[:] = StandardScaler().fit_transform(df)
    df = weights_df(df)
    test_x = df[len(df)-len(test_df):]
    train_x = df[:len(df)-len(test_df)]

    return train_x, test_x

train_x, test_x = preprocessing(train, test)

# Optunaを用いて最適化する関数
def knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 100),
        'leaf_size': trial.suggest_int('leaf_size', 1, 100),
        'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
    }

    model = KNeighborsClassifier(**params)
    accuracy = cross_val_score(model, train_x, y, scoring='f1_macro', cv=KFold(shuffle=True, random_state=0)).mean()
    return (1-accuracy)

knn_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
knn_study.optimize(knn, n_trials=30)
knn_model = KNeighborsClassifier(**knn_study.best_params)
knn_model.fit(train_x, y)
pred_knn = knn_model.predict(test_x)

# KNNを用いた疑似ラベルの割り振り
tmp = list(map(max, knn_model.predict_proba(test_x)))
tmp = np.asarray(tmp)
# 疑似ラベル付きのデータ
new_train_x = pd.concat([train_x, test_x[tmp > 0.7]])
new_y = pd.concat([y, pd.DataFrame(pred_knn)[tmp > 0.7]])

def xg(trial):
    params = {
        'objective': 'multi:softmax',
        'num_class': 11,
        'eval_metric': lambda i, j: 1-f1_score(i, j, average='macro'),
        'max_depth': trial.suggest_int('max_depth', 7, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.5, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 0.9, 0.1),
        'n_estimators': 1000,
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-8, 1.0),
        'reg_lambda': trial.suggest_discrete_uniform('lambda', 0, 1, 0.1)
    }
    model = xgb.XGBClassifier(**params)
    accuracy = cross_val_score(model, new_train_x, new_y, scoring='f1_macro').mean()
    return (1-accuracy)

def rf(trial):
    params = {
        'n_estimators': 1000,
        'max_depth':trial.suggest_int('max_depth', 10, 100),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 2, 1000),
        'min_samples_split': trial.suggest_int('min_samples_split',2, 5),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf',1,10),
    }

    model = RandomForestClassifier(**params, random_state=0)
    accuracy = cross_val_score(model, new_train_x, new_y, scoring='f1_macro', cv=KFold(shuffle=True, random_state=0)).mean()

    return 1-accuracy

def svc(trial):
    params = {
        'kernel': 'rbf',
        'C': trial.suggest_loguniform('C', 1e+0, 1e+2/2),
        'gamma': trial.suggest_loguniform('gamma', 1e-3, 3.0),
        # None が良さそう
#         'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
    }
    model = SVC(**params)
    accuracy = cross_val_score(model, new_train_x, new_y, scoring='f1_macro', cv=KFold(shuffle=True, random_state=0, n_splits=4)).mean()
    return 1-accuracy


def knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 100),
        'leaf_size': trial.suggest_int('leaf_size', 1, 100),
        'weights': trial.suggest_categorical('weights', ['distance', 'uniform']),
    }

    model = KNeighborsClassifier(**params)
    accuracy = cross_val_score(model, new_train_x, new_y, scoring='f1_macro', cv=KFold(shuffle=True, random_state=0)).mean()
    return (1-accuracy)


# 疑似ラベル込みのデータの学習
rf_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
rf_study.optimize(rf, n_trials=50)
rf_model = RandomForestClassifier(**rf_study.best_params)
rf_model.fit(new_train_x, new_y.values)
pred_rf = rf_model.predict(test_x)

svc_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
svc_study.optimize(svc, n_trials=50)
svc_model = SVC(**svc_study.best_params, random_state=0)
svc_cv = np.mean(cross_val_score(svc_model, new_train_x, new_y, cv=KFold(shuffle=True, random_state=0), scoring='f1_macro'))
svc_model.fit(new_train_x, new_y.values)
pred_svc = svc_model.predict(test_x)

xg_study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
xg_study.optimize(xg, n_trials=50)
xg_model = xgb.XGBClassifier(**xg_study.best_params)
xg_model.fit(new_train_x, new_y.values)
pred_xg = xg_model.predict(test_x)

# アンサンブル SVCは重めに
voting = VotingClassifier([('svc', svc_model), ('knn', knn_model), ('xg', xg_model), ('rf', rf_model)], weights=[1.1, 1, 1, 1])
voting.fit(new_train_x, new_y)
print(cross_val_score(voting, new_train_x, new_y, scoring='f1_macro', cv=KFold(shuffle=True, random_state=0)))
