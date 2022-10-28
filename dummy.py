import pandas as pd

train_data = pd.read_csv('Data/train_final.csv')
test_data = pd.read_csv('Data/test_final.csv')

test_data['pred'] = (test_data['sex'] == 'Male').astype(int)
test_data[['ID', 'pred']].to_csv('Data/dummy.csv', header=['ID', 'Prediction'],
                                 index=False)
print()
