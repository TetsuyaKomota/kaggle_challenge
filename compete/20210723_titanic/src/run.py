import lightgbm as lgb
import numpy  as np
import optuna
import os
import pandas as pd
from collections             import Counter
from datetime                import datetime
from sklearn.metrics         import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold

np.random.seed(0)

# 事前にデータをダウンロード
# kaggle competitions download -c titanic

# データをロード
train = pd.read_csv("tmp/train.csv")
test  = pd.read_csv("tmp/test.csv")

# 文字列データ=カテゴリカルデータとして変換
# 多い順にcardinality個選び，0-9に変換する
# c_thresh個未満のデータは足切り
def preprocess(train, test, c_thresh=10, cardinality=10):
    cat_features = test.select_dtypes(include="object").columns
    for col in cat_features:
        c = Counter(list(train[col]) + list(test[col]))
        l = [x for x in sorted(c.keys(), key=c.get, reverse=True) if c[x] >= c_thresh]
        l = l[:cardinality]
        print(col, l, sep="\t")
        train[col] = train[col].apply(lambda x:l.index(x) if x in l else len(l)).astype("category")
        test[ col] = test[ col].apply(lambda x:l.index(x) if x in l else len(l)).astype("category")
    return train, test, sorted(list(cat_features))
        
train, test, cat_features = preprocess(train, test)

# データとラベルを分離
y_train = train["Survived"]
X_train = train.drop("Survived", axis=1)
X_test  = test

# 学習
def objective(trial=None, params=None):
    params       = {
            "objective": "binary", 
            "max_bin": params["max_bin"] if trial is None else trial.suggest_int("max_bin", 255, 400), 
            "learning_rate": 0.01, 
            "num_leaves": params["num_leaves"] if trial is None else trial.suggest_int("num_leaves", 32, 256), 
            "verbose": -1, 
            }

    models    = []
    oof_train = np.zeros((len(X_train), ))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for t_index, v_index in cv.split(X_train, y_train):
        X_t = X_train.loc[t_index, :]
        X_v = X_train.loc[v_index, :]
        y_t = y_train[t_index]
        y_v = y_train[v_index]
        lgb_train = lgb.Dataset(X_t, y_t, categorical_feature=cat_features)
        lgb_valid = lgb.Dataset(X_v, y_v, categorical_feature=cat_features, reference=lgb_train)
        clf       = lgb.train(
                params, 
                lgb_train, 
                valid_sets=[lgb_train, lgb_valid], 
                verbose_eval=10 if trial is None else -1, 
                num_boost_round=10000, 
                early_stopping_rounds=100, 
                categorical_feature=cat_features, 
                )
        models.append(clf)
        oof_train[v_index] = clf.predict(X_v, num_iteration=clf.best_iteration)

    if trial is not None:
        return log_loss(y_train, oof_train)
    else:
        clf    = sorted(models, key=lambda x:x.best_score["valid_1"]["binary_logloss"])[0]
        y_eval = (oof_train > 0.5).astype(int)
        acc    = len([v for v in zip(y_train, y_eval) if int(v[0]) == int(v[1])]) / len(X_train)
        print(f"正答率: {acc:.5f}")
        return clf, acc

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=100)

clf, acc = objective(params=study.best_params)

# 予測
y_pred = (clf.predict(X_test, num_iteration=clf.best_iteration) > 0.5).astype(int)

# 提出用ファイル作成
sub             = pd.read_csv("tmp/gender_submission.csv")
sub["Survived"] = list(map(int, y_pred))
result_ts       = int(datetime.now().timestamp())
if not os.path.exists("tmp/results"):
    os.makedirs("tmp/results")
sub.to_csv(f"tmp/results/submission_acc={acc:.5f}_{result_ts}.csv", index=False)

