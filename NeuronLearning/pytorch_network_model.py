import os

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import gym
import neurogym as ngym
import warnings

# need this to run the data in the second part of the article, try taking from a neurogym
'''
train = pd.read_csv(r"/home/emily/Downloads/.csv",dtype = np.float32)

# split data into features(pixels) and labels(numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values/255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy, targets_numpy, test_size = 0.2, random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long
'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
task = 'MotorTiming-v0'
kwargs = {'dt': 100}
seq_len = 100

dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16, seq_len=seq_len)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

EPOCHS = 2000

class Net(nn.Module):
    def __init__(self, num_h):
        super(Net, self).__init__()
        self.gru = nn.GRU(ob_size, num_h)
        self.linear = nn.Linear(num_h, act_size)

    def forward(self, x):
        out, hidden = self.gru(x)
        x = self.linear(out)
        return x

lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = Net(num_h=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr)


running_loss = 0.0
for i in range (EPOCHS):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)

    optimizer.zero_grad()

    outputs = net(inputs)

    loss = criterion(outputs.view(-1, act_size), labels) # X object has no atribute Y
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    if (i+1) % 100 == 0: #every 100
        print('{:d} loss: {:0.5f}'.format(i+1, running_loss / 100))
        running_loss = 0.0

print('Finished Training')



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

def infer_test_timing(env):
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing

#Net for training analysis --> taking file made by test and analizing it
perf = 0
num_trial = 200

activity = list()
info = pd.DataFrame()

for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    ob = ob[:, np.newaxis, :]
    inputs = torch.from_numpy(ob).type(torch.float).to(device)

    action_pred = net(inputs)
    action_pred = action_pred.detach().numpy()
    action_pred = np.argmax(action_pred, axis=-1)
    perf += gt[-1] == action_pred[-1, 0]
    choice = np.argmax(action_pred[-1, 0])
    correct = choice == gt[-1]

    trial_info = env.trial
    trial_info.update({'correct': correct, 'choice': choice})
    info = info_append(trial_info, ignore_index=True)

    # Log stimulus period activity
    activity.append(np.array(hidden)[:, 0, :])

perf /= num_trial
print('Average score in {:d} trials'.format(num_trial))
print(perf)

#General analysis with graphs !!!
plt.figure(figsize=(1.2, 0.8))
t_plot = np.arange(ob_size.shape[1]) * kwargs
plt.plot(t_plot, (axis=0)


         


