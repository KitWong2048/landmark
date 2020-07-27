import pandas as pd
from sklearn.model_selection import train_test_split
import os

seed = 77777
test_size = 0.1
csv_path = '/kaggle/input/landmark-retrieval-2020'

out_path = '/kaggle/working/'

df = pd.read_csv(csv_path)


train_split, val_split = train_test_split(df, test_size=test_size, random_state=seed)

train_split.to_csv(out_path+os.sep+'train.csv')
val_split.to_csv(out_path+os.sep+'val.csv')