# Imports
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

# Create Model
class Model(nn.Module):
    #Input layer (in this case flowers)
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

torch.manual_seed(41)
model = Model()

url = 'https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv'
my_df = pd.read_csv(url)

#print(my_df) my_df.tail will show last 5 and .head will show first 5


#Change last col from string to int
my_df['variety'] = my_df['variety'].replace('Setosa', 0.0)
my_df['variety'] = my_df['variety'].replace('Versicolor', 1.0)
my_df['variety'] = my_df['variety'].replace('Virginica', 2.0)

# Train Test & Split
X = my_df.drop('variety', axis=1)
y = my_df['variety']

# Convert nmp arrays
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
X_train = torch.FloatTensor(X_train)
x_test = torch.FloatTensor(X_test)


y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Loss
criterion = nn.CrossEntropyLoss()
# Choose optim
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train Model
epoch = 100
losses = []
for i in range(epoch):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)
    losses.append(loss.detach().numpy())
    
    #print every 10 epo
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')

    #back prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epoch), losses)
plt.ylabel("loss error")
plt.xlabel("Epoch")
plt.show()
