import pandas as pd

train_ds = pd.read_csv("classify-leaves/train.csv", names=['image','label'], header=1)
count_label = train_ds.groupby('label').count()
labels = train_ds.label.unique().tolist()
print(train_ds.info(verbose=True))
print(train_ds.label.unique().shape)
print(count_label.describe())