import nltk
import string
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from fasttext import load_model
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

def word_tokenize(text,max_len):
    alltoken = []
    for t in text:
        tokens = nltk.word_tokenize(t)
        tokens = [token for token in tokens if token not in string.punctuation+'。，！？']
        if len(tokens) > max_len:
            alltoken.append(tokens[:max_len])
        else:
            alltoken.append(tokens)
    return alltoken

def vocab_to_id(X,max_len,vocab_id):
    Xn = np.zeros((len(X),max_len))
    for i in range(len(X)):
        for j in range(len(X[i])):
            word = X[i][j]
            if word in vocab_id.keys():
                Xn[i][j] = vocab_id[X[i][j]]
    return Xn

def pre_process(filepath,max_len,vocab_id):
    df = pd.read_csv(filepath,encoding='utf8')
    X = vocab_to_id(word_tokenize(df["review"],max_len),max_len,vocab_id)
    X = torch.from_numpy(X).type(dtype=torch.long)
    y = np.array(df["label"])
    y = torch.from_numpy(y).type(dtype=torch.long)
    return X,y


def load_w2v(filepath):

     model = load_model(filepath)
     vocab_size = len(model.words)
     embed_size = len(model[model.words[0]]),
     embedding_matrix = np.zeros((vocab_size,embed_size),dtype=np.float)
     vocab_id = {},
     for i in range(len(model.words)):
         embedding_matrix[i] = model[model.words[i]]
         vocab_id[model.words[i]] = i,
         embedding_matrix=torch.Tensor(embedding_matrix)
     return vocab_size,embed_size,vocab_id,embedding_matrix

class LSTM(nn.Module):
    '''LSTM'''
    def __init__(self,vocab_size,embedding_size,embedding_matrix,hidden_size,num_layers,n_class,dropout):
        super(LSTM,self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size,_weight=embedding_matrix)
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size//2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.out = nn.Linear(hidden_size,hidden_size)
        self.drop = nn.Dropout(dropout)
        self.out_final = nn.Linear(hidden_size,n_class)

    def forward(self,x):
        embed = self.embed(x)
        out, _ = self.lstm(embed,None)
        #out = F.relu(self.drop(out[:,-1,:]))
        out = self.out_final(out[:,-1,:])
        out = F.softmax(out,dim=1)
        return out

def run(filename,w2vpath):
    '''running models'''

    vocab_size,embedding_size,vocab_id,embedding_matrix = load_w2v(w2vpath) # Word embedding model

    # Parameters
    max_len = 50
    epoch_n = 10
    batch_size = 32
    hidden_size = embedding_size
    num_layers = 1
    n_class = 2
    dropout = 0
    lr = 0.001

    rnn = LSTM(vocab_size=vocab_size,embedding_matrix=embedding_matrix,embedding_size=embedding_size,hidden_size=hidden_size,num_layers=num_layers,n_class=n_class,dropout=dropout) # 选择LSTM模型

    optimizer = torch.optim.Adam(rnn.parameters(),lr=lr) # Optimizer
    loss_function = nn.CrossEntropyLoss() # Loss function

    
    X_train, y_train = pre_process(filename+"-train.csv",max_len,vocab_id)  # Training dataset
    X_test, y_test = pre_process(filename+"-test.csv",max_len,vocab_id)  # Test dataset

    train_set = torch.utils.data.TensorDataset(X_train,y_train)
    train_iter = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=False)
    test_set = torch.utils.data.TensorDataset(X_test,y_test)
    test_iter = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

    print('--------begin-----------')
    for epoch in range(epoch_n):
        start = time.time()
        print('epoch:{}'.format(epoch+1))

        rnn.train()
        pred_train = torch.randn(1,n_class)
        for step,(train_input, train_labels) in enumerate(train_iter):
            pred_train_step = rnn(train_input)
            pred_train = torch.cat((pred_train, pred_train_step), 0)
            loss = loss_function(pred_train_step, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        pred_train = pred_train[1:]
        loss_train = loss_function(pred_train, y_train)
        pred_train = pred_train.argmax(dim=1)
        acc_train = accuracy_score(y_train, pred_train)
        f1_train = f1_score(y_train, pred_train, average='macro')
        print('train loss:', float(loss_train.data))
        print('acc_train:', acc_train)
        print('f1_train:', f1_train)

        rnn.eval()
        pred_test = torch.randn(1, n_class)
        for step,(test_input,test_labels) in enumerate(test_iter):
            pred_test_step = rnn(test_input)
            pred_test = torch.cat((pred_test, pred_test_step), 0)
        pred_test = pred_test[1:]
        loss_test = loss_function(pred_test, y_test)
        pred_test = pred_test.argmax(dim=1)
        acc_test = accuracy_score(y_test, pred_test)
        f1_test = f1_score(y_test, pred_test, average='macro')
        print('test loss:', float(loss_test.data))
        print('acc_test:', acc_test)
        print('f1_test:', f1_test)

        end = time.time()
        print('Time  : {} 秒'.format(end - start))
        print('__' * 40)

    pred_train = torch.randn(1,n_class)
    for step,(train_input,train_labels) in enumerate(train_iter):
        pred_train_step = rnn(train_input)
        pred_train = torch.cat((pred_train, pred_train_step), 0)
    pred_train = pred_train[1:].argmax(dim=1)
    acc_train = accuracy_score(y_train, pred_train)
    f1_train = f1_score(y_train, pred_train,average='macro')
    print('acc_train:', acc_train)
    print('f1_train:', f1_train)
    print(classification_report(y_train, pred_train))

    pred_test = torch.randn(1,n_class)
    for step, (test_input,test_labels) in enumerate(test_iter):
        pred_test_step = rnn(test_input)
        pred_test = torch.cat((pred_test, pred_test_step), 0)
    pred_test = pred_test[1:].argmax(dim=1)
    acc_test = accuracy_score(y_test, pred_test)
    f1_test = f1_score(y_test, pred_test,average='macro')
    print('acc_test:', acc_test)
    print('f1_test:', f1_test)
    print(classification_report(y_test,pred_test))

filename = r'D:\coding\onlinereview.csv'
w2vpath = r'C:\Users\Administrator\Desktop\amazon_review_polarity.ftz'
w2vpath2 = r'C:\Users\Administrator\Desktop\amazon_review_polarity.ftz'
run(filename,w2vpath2)


