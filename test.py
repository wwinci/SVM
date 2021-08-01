import pandas as pd
df = pd.read_csv('svm_training_set.csv', encoding='utf-8')
df1 = df.loc[:,['index','label']]
df1.to_csv("test.csv",index= False)