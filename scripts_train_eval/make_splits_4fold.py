
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5)
df = pd.read_csv('split.txt')

print(df.describe())
TRAIN_LEN=65
VAL_LEN = 6
TEST_LEN = 23
print(df['fold_1'])
print(df['CASES'])
fold_1 = df['fold_1']
train_val_indices = [ i for i, f in enumerate(df['fold_1']) if f == 'train' or f == 'val' ]
test_indices = [ i for i, f in enumerate(df['fold_1']) if f == 'test']

k_val2 = 23 # whish is the int( 94/3)
fold_1_indices = np.array(test_indices)
fold_2_indices = np.random.choice(train_val_indices, 23, replace=False)
train_val_indices = [i for i in train_val_indices if i not in fold_2_indices]
fold_3_indices = np.random.choice(train_val_indices, 24, replace=False)
train_val_indices = [i for i in train_val_indices if i not in fold_3_indices]
fold_4_indices = np.random.choice(train_val_indices, 24, replace=False)
train_val_indices = [i for i in train_val_indices if i not in fold_4_indices]


df['fold_2'].iloc[fold_2_indices] = 'test'
indices = np.random.choice(np.concatenate([fold_1_indices, fold_3_indices, fold_4_indices]), VAL_LEN)
df['fold_2'].iloc[indices] = 'val'

df['fold_3'].iloc[fold_3_indices] = 'test'
indices = np.random.choice(np.concatenate([fold_1_indices, fold_2_indices, fold_4_indices]), VAL_LEN-1)
df['fold_3'].iloc[indices] = 'val'

df['fold_4'].iloc[fold_4_indices] = 'test'
indices = np.random.choice(np.concatenate([fold_1_indices, fold_2_indices, fold_3_indices]), VAL_LEN-1)
df['fold_4'].iloc[indices] = 'val'


print('fold 1:', df['fold_1'].describe())
print('fold 2:', df['fold_2'].describe())
print('fold 3:', df['fold_3'].describe())
print('fold 4:', df['fold_4'].describe())
df.to_csv('split.txt')

# df['fold_2'].iloc[fold_2_indices] = 'test'
# indices = np.random.choice(np.concatenate([fold_1_indices, fold_3_indices, fold_4_indices]), VAL_LEN)
# df['fold_2'].iloc[indices] = 'val'

# for col in df.columns:
#    if col != 'CASES' or col != 'fold_1':
#        indices = np.random.choice(df.index,VAL_LEN+TEST_LEN,replace=False)
#        df[col].iloc[indices[:VAL_LEN]] = 'val'
#        df[col].iloc[indices[VAL_LEN:VAL_LEN+TEST_LEN]] = 'test'
