import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import gym
import neurogym as ngym
import torch
import torch.nn as nn
import warnings
from collections import OrderedDict
from IPython.display import clear_output
clear_output()
warnings.filterwarnings('ignore')

task = 'MotorTiming-v0'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_modelpath(task):
    path = Path('/home/emily/motornproj/NeuronLearning') / 'files'
    os.makedirs(path, exist_ok=True)
    path = path / task
    os.makedirs(path, exist_ok=True)
    return path

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_layer, hl_dim, output_dim):
        super(RNNModel, self).__init__()
        self.hl_dim = hl_dim
        # LSTM & RNN compare for paper
        self.lstm = nn.LSTM(input_dim, hidden_layer, hl_dim, batch_first=True)
        self.linear = nn.Linear(hidden_layer, output_dim)
    
    def forward(self, x):
        out, hn = self.lstm(x)
        v = self.linear(out) 
        return v, out

modelpath = get_modelpath(task)
config = {
    'dt': 100,
    'hidden_layer':64,
    'lr': 1e-3,
    'batch_size': 16,
    'seq_len': 100,
    'EPOCHS': 2000,
    'hl_dim' : 5, #hidden layer dimention
}

hidden_layer = config['hidden_layer']
env_kwargs = {'dt': config['dt']}
config['env_kwargs'] = env_kwargs
with open(modelpath / 'config.json', 'w') as f:
    json.dump(config, f)

dataset = ngym.Dataset(task, env_kwargs=env_kwargs, batch_size=config['batch_size'], seq_len=100)
env = dataset.env
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

model = RNNModel(input_dim=input_dim, 
                 hidden_layer=config['hidden_layer'],
                 hl_dim=config['hl_dim'],
                 output_dim=output_dim,
                 )
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

print('Training Task ', task)

loss_list = []
iteration_list = []
count = 0
running_loss = 0.0
for i in range (config['EPOCHS']):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        
    optimizer.zero_grad()
    
    outputs, _ = model(inputs)
    
    loss = criterion(outputs.view(-1, output_dim), labels)
    loss.backward()
    optimizer.step()
    
    
    count += 1
    loss_list.append(loss.data)
    iteration_list.append(count)
    running_loss += loss.item()
    if count % 100 == 0:     
        #print('{:d} loss: {:0.5f}'.format(i+1, running_loss / 100))
        running_loss = 0.0
        torch.save(model.state_dict(), modelpath / 'model.pth')

print('Finished Training')

# visualization loss 

plt.plot(iteration_list,loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss")
plt.savefig('network_model_graph.png')
plt.show()

def infer_test_timing(env):
    timing = {}
    for period in env.timing.keys():
        period_times = [env.sample_time(period) for _ in range(100)]
        timing[period] = np.median(period_times)
    return timing

# Run network for analysis

modelpath = get_modelpath(task)
with open(modelpath / 'config.json') as f:
    config = json.load(f)

env_kwargs = config['env_kwargs']

#Get info
#Env
env = ngym.make(task, **env_kwargs)
env.timing = infer_test_timing(env)
env.reset()

#Finding average of trials and collecting data 

with torch.no_grad():
    model = RNNModel(input_dim=input_dim,
                    hidden_layer=config['hidden_layer'],
                    hl_dim=config['hl_dim'],
                    output_dim=env.action_space.n).to(device)
    model.load_state_dict(torch.load(modelpath / 'model.pth'))

    perf = 0
    num_trial = 100
    
    activity = list()
    info = pd.DataFrame({
        'correct': [],
        'choice' : []
    })
    
    for i in range(num_trial):
        env.new_trial()
        ob, gt = env.ob, env.gt
        inputs = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
        action_pred, hidden = model(inputs)

        action_pred = action_pred.detach().numpy()
        choice = np.argmax(action_pred[-1, 0, :])
        correct = choice == gt[-1]    

        trial_info = env.trial
        trial_info.update({'correct': correct, 'choice': choice})
        info.loc[len(info)] = trial_info
        
        activity.append(np.array(hidden)[:, 0, :])
 
    print('Average score : ', np.mean(info['correct']))
tensors = [torch.tensor(seq) for seq in activity]
# pad because of sequence lengths
padded_tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
activity = np.array((torch.tensor(padded_tensors)).numpy())

#General analysis

def analysis_average_activity(activity, info, config):
    #Load and preprocess results
    plt.savefig('loaded_results')
    plt.title('Activity of RNN')
    t_plot = np.arange(activity.shape[1]) * config['dt']
    plt.plot(t_plot, activity.mean(axis=0).mean(axis=-1))

analysis_average_activity(activity, info, config)

plt.show()

def analysis_of_weights():
    od = model.state_dict()
    wanted_tensor = od.pop('linear.weight')
    wanted_tensor = wanted_tensor.numpy()
    plt.imshow(wanted_tensor)
    wanted_tensor = wanted_tensor.reshape((2,8,8))
    eigvals, eigvecs = np.linalg.eig(wanted_tensor)
    plt.title('Learned Weight Matrix Eigenvalues')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.scatter(eigvals.real, eigvals.imag)
    
    plt.savefig('eigenvalues_weights')

analysis_of_weights()

plt.show()


        






