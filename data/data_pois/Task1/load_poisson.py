import joblib

df = joblib.load('train.pickle')
print(df['train'][1])