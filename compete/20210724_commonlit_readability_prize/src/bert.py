import nltk
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import random
import shutil
import torch
import torch.nn as nn

from collections      import Counter
from glob             import glob
from nltk             import pos_tag
from torch.utils.data import DataLoader
from tqdm             import tqdm
from transformers     import BertTokenizer, BertModel

tqdm.pandas()
# nltk.download("averaged_perceptron_tagger")

DATA_DIR      = "tmp"
MODEL_PATH    = "bert-base-uncased"
MAX_LENGTH    = 128
RANDOM_STATE  =   2 # 1
RATE_TEST     =   0.1
RATE_VAL      =   0.1
BATCH_SIZE    =  16
EPOCHS        = 200
LEARNING_RATE =   1e-5
DROPOUT_RATE  =   0.8
MAX_W         = 200
MAX_S         =  25
MAX_P         =  10
NUM_P         =  20
NUM_L         =   4
NUM_D         =   5
DIM_A         =   2
TOP_K         =  10

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True

class ScoreModule(nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.h1 = nn.Linear(dim_h, dim_h)
        self.h2 = nn.Linear(dim_h, dim_h)
        self.o  = nn.Linear(dim_h, 1)
        self.d  = nn.Dropout(DROPOUT_RATE)
        for h in [self.o, self.h1, self.h2]:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.normal_(h.bias, 0)

    def forward(self, x):
        x = torch.tanh(self.h1(x))
        x = self.d(x)
        x = torch.tanh(self.h2(x))
        x = self.d(x)
        x = self.o(x)
        return x

class AppendModule(nn.Module):
    def __init__(self, dim_h):
        super().__init__()
        self.h1 = nn.Linear(dim_h, dim_h)
        self.o  = nn.Linear(dim_h, DIM_A)
        self.d  = nn.Dropout(DROPOUT_RATE)
        for h in [self.o, self.h1]:
            nn.init.normal_(h.weight, std=0.02)
            nn.init.normal_(h.bias, 0)

    def forward(self, x):
        x = torch.tanh(self.h1(x))
        x = self.d(x)
        x = self.o(x)
        return x

class BertModelPL(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert      = BertModel.from_pretrained(MODEL_PATH)
        # self.app       = AppendModule(1 + 1 + NUM_P + NUM_L + 1 + NUM_D + 1)
        self.app       = AppendModule(1 + 1 + NUM_P)
        self.score     = ScoreModule(768 + DIM_A)
        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        tar  = batch.pop("target")
        info = batch.pop("info")
        vec  = self.bert(**batch).pooler_output
        app  = self.app(info)
        out  = self.score(torch.cat((vec, app), 1))
        loss = self.criterion(out, tar)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tar  = batch.pop("target")
        info = batch.pop("info")
        vec  = self.bert(**batch).pooler_output
        app  = self.app(info)
        out  = self.score(torch.cat((vec, app), 1))
        loss = self.criterion(out, tar)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        tar  = batch.pop("target")
        info = batch.pop("info")
        vec  = self.bert(**batch).pooler_output
        app  = self.app(info)
        out  = self.score(torch.cat((vec, app), 1))
        loss = self.criterion(out, tar)
        self.log("test_rmse", torch.sqrt(loss))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# 素性として利用するセットを作成
def get_list(df):
    plist = df.excerpt.progress_apply(lambda x:pos_tag(x.split()))
    p_c = []
    for pos_list in plist:
        p_c = p_c + [p[1] for p in pos_list]
    p_c = Counter(p_c)
    p   = sorted(p_c, key=p_c.get, reverse=True)[:NUM_P]

    c = Counter(df.license)
    l = [k for k in sorted(c.keys(), key=c.get, reverse=True)][:NUM_L]

    d = df.url_legal.apply(lambda x: x.split("://")[1].split("/")[0])
    c = Counter(d)
    d = [k for k in sorted(c.keys(), key=c.get, reverse=True)][:NUM_D]

    return p, ["nan"] + l, ["nan"] + d


# サブモデルに入力する素性を作成
def get_sub_info(r, p_list, l_list, d_list):
    output = []

    text = r["excerpt"]

    # 単語数
    output.append(min(1, len(text.split())/MAX_W))

    # 行数
    output.append(min(1, len(text.split("."))/MAX_S))

    # 各品詞の単語数
    p_c    = Counter([p[1] for p in pos_tag(text.split())])
    output = output + [min(MAX_P, p_c.get(p, 0))/MAX_P for p in sorted(p_list)]

    """
    # ライセンス情報
    for l in l_list:
        output.append(int(r.license == l))

    # ドメイン情報
    for d in d_list:
        output.append(int(d in r.url_legal))
    """

    return output


def load_traindata():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

    train = train.fillna({"url_legal": "http://nan/", "license": "nan"})

    p_list, l_list, d_list = get_list(train)

    tokenizer = BertTokenizer.from_pretrained(
            MODEL_PATH, 
            do_lower_case=True
            )
    dataset_for_loader = [] 
    for i, r in train.iterrows():
        encoded = tokenizer(
                r.excerpt, 
                max_length = MAX_LENGTH, 
                padding    = "max_length", 
                truncation = True, 
                )
        encoded["target"] = [r.target]
        encoded["info"]   = get_sub_info(r, p_list, l_list, d_list)
        encoded           = {k: torch.tensor(v) for k, v in encoded.items()}
        dataset_for_loader.append(encoded)

    random.shuffle(dataset_for_loader)
    n       = len(dataset_for_loader)
    n_val   = int(n * RATE_VAL)
    n_test  = int(n * RATE_TEST)
    d_val   = dataset_for_loader[:n_val]
    d_test  = dataset_for_loader[n_val:n_val+n_test]
    d_train = dataset_for_loader[n_val+n_test:]
    
    d_train = DataLoader(d_train, batch_size=BATCH_SIZE, shuffle=True)
    d_test  = DataLoader(d_test,  batch_size=BATCH_SIZE)
    d_val   = DataLoader(d_val,   batch_size=BATCH_SIZE)

    return d_train, d_val, d_test


def train(model_pl, d_train, d_val, modeldir="tmp/model"):
    if os.path.exists(modeldir):
        shutil.rmtree(modeldir)
    os.makedirs(modeldir)
    checkpoint = pl.callbacks.ModelCheckpoint(
            auto_insert_metric_name = True, 
            dirpath                 = modeldir, 
            filename                = "checkpoint_{epoch:03d}-{val_loss:.5f}",
            mode                    = "min", 
            monitor                 = "val_loss", 
            save_top_k              = TOP_K,
            save_weights_only       = True,
            )

    trainer = pl.Trainer(
            callbacks  = [checkpoint], 
            gpus       = 1, 
            max_epochs = EPOCHS, 
            )

    trainer.fit(model_pl, d_train, d_val)    

    t = trainer.test(test_dataloaders=d_test)
    print(t)

    model_pl.app.eval()
    model_pl.score.eval()
    return model_pl

def make_submission(modeldir="tmp/model"):
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    train = train.fillna({"url_legal": "http://nan/", "license": "nan"})
    p_list, l_list, d_list = get_list(train)

    sub   = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    sub   = sub.fillna(  {"url_legal": "http://nan/", "license": "nan"})
    tokenizer = BertTokenizer.from_pretrained(
            MODEL_PATH, 
            do_lower_case = True, 
            )

    dataset_for_loader = [] 
    for i, r in sub.iterrows():
        encoded = tokenizer(
                r.excerpt, 
                max_length = MAX_LENGTH, 
                padding    = "max_length", 
                truncation = True, 
                )
        encoded["info"] = get_sub_info(r, p_list, l_list, d_list)
        encoded         = {k: torch.tensor(v) for k, v in encoded.items()}
        dataset_for_loader.append(encoded)
    d_sub = DataLoader(dataset_for_loader, batch_size=len(sub))

    val_loss = []
    for model_path in glob(os.path.join(modeldir, "*")):
        model_pl = BertModelPL.load_from_checkpoint(model_path)
        val_loss.append(float(model_path[:-5].split("val_loss=")[1]))

        with torch.no_grad():
            for batch in d_sub:
                info = batch.pop("info")
                vec  = model_pl.bert(**batch).pooler_output
                app  = model_pl.app(info)
                out  = model_pl.score(torch.cat((vec, app), 1))

        sub[f"target_{val_loss[-1]}"] = list(out.detach().numpy().reshape(-1))

    e = sum([np.exp(-1 * v) for v in val_loss])
    sub["target"] = sum([(np.exp(-1 * v)/e) * sub[f"target_{v}"] for v in val_loss])

    sub[["id", "target"]].to_csv("submission.csv", index=None)


if __name__ == "__main__":
    d_train, d_val, d_test = load_traindata()

    model_pl = BertModelPL(lr=LEARNING_RATE)

    model_pl = train(model_pl, d_train, d_val)
    
    make_submission()
