import pandas as pd
import nltk
import re
import os
import numpy as np

from nltk.corpus import stopwords
from nltk import pos_tag
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.utils.rnn as rnn

from tqdm import tqdm
tqdm.pandas()

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
model     = BertModel.from_pretrained('bert-base-uncased').eval()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics         import mean_squared_error

# データをロード
def load():
    if os.path.exists("tmp/converted_train.pkl"):
        train = pd.read_pickle("tmp/converted_train.pkl")
    else:
        train = pd.read_csv("tmp/train.csv")
    if os.path.exists("tmp/converted_test.pkl"):
        test  = pd.read_pickle("tmp/converted_test.pkl")
    else:
        test  = pd.read_csv("tmp/test.csv")
    return train, test

# 前処理
def preprocess(text):
    # find alphabets
    e = re.sub("[^a-zA-Z]", " ", text)
    
    # convert to lower case
    e = e.lower()
    
    # tokenize words
    # e = nltk.word_tokenize(e)
    
    # remove stopwords
    # e = [word for word in e if not word in set(stopwords.words("english"))]
    
    # lemmatization
    # lemma = nltk.WordNetLemmatizer()
    # e = [lemma.lemmatize(word) for word in e]
    # e=" ".join(e)
    
    # encode
    i = tokenizer.encode(e)

    # o = model(torch.tensor([i])).pooler_output
    # return o[0]
    return i

if __name__ == "__main__":
    train_df, test_df = load()

    if "token" not in train_df.columns:
        train_df["token"] = train_df.excerpt.progress_apply(preprocess)
        test_df["token"]  = test_df.excerpt.progress_apply(preprocess)
        train_df.to_pickle("tmp/converted_train.pkl")
        test_df.to_pickle("tmp/converted_test.pkl")

    if "vec" not in train_df.columns:
        with open("tmp/vec_train.csv", "w") as f:
            for i, r in tqdm(train_df.iterrows()):
                vec = model(torch.tensor([r.token])).pooler_output[0]
                f.write(f"{r['id']}," + "\t".join([str(x.detach().numpy()) for x in vec]) + "\n")
        with open("tmp/vec_test.csv", "w") as f:
            for i, r in tqdm(test_df.iterrows()):
                vec = model(torch.tensor([r.token])).pooler_output[0]
                f.write(f"{r['id']}," + "\t".join([str(x.detach().numpy()) for x in vec]) + "\n")
        vec_train = pd.read_csv("tmp/vec_train.csv", header=None, names=["id", "vec"])
        vec_test  = pd.read_csv("tmp/vec_test.csv",  header=None, names=["id", "vec"])
        vec_train["vec"] = vec_train.vec.apply(lambda x:np.array([float(v) for v in x.split("\t")]))
        vec_test["vec"]  = vec_test.vec.apply( lambda x:np.array([float(v) for v in x.split("\t")]))
        train_df = pd.merge(train_df, vec_train, on="id")
        test_df  = pd.merge(test_df,  vec_test,  on="id")
        train_df.to_pickle("tmp/converted_train.pkl")
        test_df.to_pickle("tmp/converted_test.pkl")


    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    oof_train = np.zeros((len(train_df), ))
    for t_index, v_index in cv.split(train_df["vec"]):
        X_t = np.array(list(train_df.loc[t_index, :].vec))
        y_t = np.array(list(train_df.loc[t_index, :].target))
        X_v = np.array(list(train_df.loc[v_index, :].vec))
        y_v = np.array(list(train_df.loc[v_index, :].target))
        clf = LinearRegression(normalize=True)
        clf.fit(X_t, y_t)
        oof_train[v_index] = clf.predict(X_v)

    print(np.sqrt(mean_squared_error(train_df.target, oof_train)))
