import numpy as np
import pandas as pd
import joblib

def generate_hawkes(mu, alpha, beta):
    n_train, n_dev, n_test = 60000, 20000, 20000
    train_dict = {'dim_process': 1, 'devtest': [], 'args': None, 'train': [[]] , 'test': [], 'dev': []}
    train_times = simulate_hawkes(n_train, mu, alpha, beta)
    train_diffs = np.diff(train_times)
    train_diffs = np.insert(train_diffs,0, 0)
    train_type = np.zeros((n_train,), dtype=np.int)
    for i,j,k in zip(train_times, train_diffs, train_type):
        train_dict['train'][0].append({'time_since_start': i, 'time_since_last_event': j, 'type_event': k})
    
    dev_dict = {'dim_process': 1, 'devtest': [], 'args': None, 'train': [] , 'test': [], 'dev': [[]]}
    dev_times = simulate_hawkes(n_dev, mu, alpha, beta)
    dev_diffs = np.diff(dev_times)
    dev_diffs = np.insert(dev_diffs,0, 0)
    dev_type = np.zeros((n_dev,), dtype=np.int)
    for i,j,k in zip(dev_times, dev_diffs, dev_type):
        dev_dict['dev'][0].append({'time_since_start': i, 'time_since_last_event': j, 'type_event': k})

    test_dict = {'dim_process': 1, 'devtest': [], 'args': None, 'train': [] , 'test': [[]], 'dev': []}
    test_times = simulate_hawkes(n_test, mu, alpha, beta)
    test_diffs = np.diff(test_times)
    test_diffs = np.insert(test_diffs,0, 0)
    test_type = np.zeros((n_test,), dtype=np.int)
    for i,j,k in zip(test_times, test_diffs, test_type):
        test_dict['test'][0].append({'time_since_start': i, 'time_since_last_event': j, 'type_event': k})

    return train_dict, dev_dict, test_dict

def simulate_hawkes(n,mu,alpha,beta):
    T = []
    LL = []
    
    x = 0
    l_trg1 = 0
    l_trg2 = 0
    l_trg_Int1 = 0
    l_trg_Int2 = 0
    mu_Int = 0
    count = 0
    
    while 1:
        l = mu + l_trg1 + l_trg2
        step = np.random.exponential()/l
        x = x + step
        
        l_trg_Int1 += l_trg1 * ( 1 - np.exp(-beta[0]*step) ) / beta[0]
        l_trg_Int2 += l_trg2 * ( 1 - np.exp(-beta[1]*step) ) / beta[1]
        mu_Int += mu * step
        l_trg1 *= np.exp(-beta[0]*step)
        l_trg2 *= np.exp(-beta[1]*step)
        l_next = mu + l_trg1 + l_trg2
        
        if np.random.rand() < l_next/l: #accept
            T.append(x)
            LL.append( np.log(l_next) - l_trg_Int1 - l_trg_Int2 - mu_Int )
            l_trg1 += alpha[0]*beta[0]
            l_trg2 += alpha[1]*beta[1]
            l_trg_Int1 = 0
            l_trg_Int2 = 0
            mu_Int = 0
            count += 1
            
            if count == n:
                break
        
    return np.array(T)

def pickle_dict(dict, file_name):
    with open(file_name, 'wb') as f:
        joblib.dump(dict, f)


train1, dev1, test1 = generate_hawkes(0.05,[0.4,0.4],[1.0,20.0])
train2, dev2, test2 = generate_hawkes(0.2 ,[1, 10],[50, 0.5])
train3, dev3, test3 = generate_hawkes(1,[10, 50],[1, 1])
train4, dev4, test4 = generate_hawkes(20,[100, 0.1],[80, 0.8])
train5, dev5, test5 = generate_hawkes(100,[1, 100],[0.4, 500])

pickle_dict(train1, './Task1/train.pkl')
pickle_dict(dev1, './Task1/dev.pkl')
pickle_dict(test1, './Task1/test.pkl')

pickle_dict(train2, './Task2/train.pkl')
pickle_dict(dev2, './Task2/dev.pkl')
pickle_dict(test2, './Task2/test.pkl')

pickle_dict(train3, './Task3/train.pkl')
pickle_dict(dev3, './Task3/dev.pkl')
pickle_dict(test3, './Task3/test.pkl')

pickle_dict(train4, './Task4/train.pkl')
pickle_dict(dev4, './Task4/dev.pkl')
pickle_dict(test4, './Task4/test.pkl')

pickle_dict(train5, './Task5/train.pkl')
pickle_dict(dev5, './Task5/dev.pkl')
pickle_dict(test5, './Task5/test.pkl')
