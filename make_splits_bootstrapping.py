#!/usr/bin/python

import pandas as pd
import numpy as np

df = pd.read_csv('split.txt')

print(df.describe())
VAL_LEN = 6
TEST_LEN = 23
for col in df.columns:
   if col != 'CASES':
       indices = np.random.choice(df.index,VAL_LEN+TEST_LEN,replace=False)
       df[col].iloc[indices[:VAL_LEN]] = 'val'
       df[col].iloc[indices[VAL_LEN:VAL_LEN+TEST_LEN]] = 'test'
