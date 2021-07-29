import lightgbm as lgb
import numpy  as np
import optuna
import os
import pandas as pd
from collections             import Counter
from datetime                import datetime
from sklearn.metrics         import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from nltk import pos_tag
from tqdm import tqdm 
tqdm.pandas()

np.random.seed(0)

# 事前にデータをダウンロード
# kaggle competitions download -c titanic

# 適宜実行
#  nltk.download('averaged_perceptron_tagger')

# データをロード
train = pd.read_csv("tmp/train.csv")
test  = pd.read_csv("tmp/test.csv")
tid   = test["id"]

# 文字列データ=カテゴリカルデータとして変換
# 多い順にcardinality個選び，0-9に変換する
# c_thresh個未満のデータは足切り
def preprocess(train, test, c_thresh=10, cardinality=10):
    # テキストの統計情報を付与
    train = add_text_info(train, "excerpt", f=True)
    test  = add_text_info(test,  "excerpt")

    # 不要なカラムを削除
    non_use_col = set([
                "id", 
                "excerpt", 
                "standard_error", 
            ])
    train = train.drop(non_use_col & set(train.columns), axis=1)
    test  = test.drop( non_use_col & set(test.columns),  axis=1)

    # 文字列データをカテゴリカルデータに変換
    cat_features = test.select_dtypes(include="object").columns
    for col in cat_features:
        c = Counter(list(train[col]) + list(test[col]))
        l = [x for x in sorted(c.keys(), key=c.get, reverse=True) if c[x] >= c_thresh]
        l = l[:cardinality]
        print(col, l, sep="\t")
        train[col] = train[col].apply(lambda x:l.index(x) if x in l else len(l)).astype("category")
        test[ col] = test[ col].apply(lambda x:l.index(x) if x in l else len(l)).astype("category")
    return train, test, sorted(list(cat_features))

# テキストデータの統計情報を付与する
def add_text_info(df, col, f=False):
    # 単語数
    df["ti_wc"] = df[col].apply(lambda x:len(x.split()))
    # 行数
    df["ti_lc"] = df[col].apply(lambda x:len(x.split("\n")))

    # 各品詞の単語数
    df["plist"] = df[col].progress_apply(lambda x:pos_tag(x.split()))
    pos_counter = []
    for pos_list in df["plist"]:
        pos_counter = pos_counter + [p[1] for p in pos_list]
    pos_counter = Counter(pos_counter)
    pos_set     = set(sorted(pos_counter.keys(), key=pos_counter.get, reverse=True)[:20])

    for i, pos in enumerate(pos_set):
        df[f"ti_c{i}"] = df["plist"].apply(lambda x:len([n for n in x if n[1] == pos]))
    df = df.drop("plist", axis=1)
        
    if f:
        df.to_csv("tmp/gomi.tsv", sep="\t", index=None)

    return df

train, test, cat_features = preprocess(train, test)

# データとラベルを分離
y_train = train["target"]
X_train = train.drop("target", axis=1)
X_test  = test

# 学習
def objective(trial=None, params=None):
    params       = {
            "objective": "regression", 
            "metric": "rmse", 
            "boosting_type": "gbdt", 
            "max_bin": params["max_bin"] if trial is None else trial.suggest_int("max_bin", 255, 800), 
            "learning_rate": 0.01, 
            "num_leaves": params["num_leaves"] if trial is None else trial.suggest_int("num_leaves", 32, 512), 
            "verbose": -1, 
            }

    models    = []
    oof_train = np.zeros((len(X_train), ))
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    for t_index, v_index in cv.split(X_train):
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
        return mean_squared_error(y_train, oof_train)
    else:
        clf  = sorted(models, key=lambda x:x.best_score["valid_1"]["rmse"])[0]
        rmse = np.sqrt(mean_squared_error(y_train, oof_train))
        print(f"rmse: {rmse:.5f}")
        return clf, rmse

study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
study.optimize(objective, n_trials=100)

clf, acc = objective(params=study.best_params)

# 予測
y_pred = clf.predict(X_test, num_iteration=clf.best_iteration)

# 提出用ファイル作成
sub           = pd.concat([tid, test], axis=1)
sub["target"] = list(y_pred)
result_ts     = int(datetime.now().timestamp())
if not os.path.exists("tmp/results"):
    os.makedirs("tmp/results")
# sub[["id", "target"]].to_csv(f"tmp/results/submission_lightgbm.csv", index=False)
sub[["id", "target"] + [f"target_{v}" for v in val_loss]].to_csv(f"tmp/results/submission_lightgbm.csv", index=False)

