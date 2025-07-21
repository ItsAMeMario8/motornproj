import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurogym as ngym
import torch
import torch.nn as nn
import warnings
from IPython.display import clear_output
clear_output()
warnings.filterwarnings('ignore')

task = 'MotorTiming-v0'
kwargs = {'dt': 100}
seq_len = 100

dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16, seq_len=seq_len)
env = dataset.env
ob_size = env.observation_space.shape[0]
act_size = env.action_space.n

EPOCHS = 2000

class RNNModel(nn.Module):
    def __init__(self, hidden_dim):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(ob_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, act_size)
    
    def forward(self, x):
        out, hidden = self.rnn(x)
        out = self.linear(out) 
        return out

lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = RNNModel(hidden_dim=64).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr)

loss_list = []
iteration_list = []
count = 0
running_loss = 0.0
for i in range (EPOCHS):
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
        
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    loss = criterion(outputs.view(-1, act_size), labels)
    loss.backward()
    optimizer.step()
    
    
    running_loss += loss.item()

    count += 1

    loss_list.append(loss.data)
    iteration_list.append(count)
    if (i+1) % 100 == 0:     
        print('{:d} loss: {:0.5f}'.format(i+1, running_loss / 100))

print('Finished Training')

# visualization loss 
plt.plot(iteration_list,loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss")
plt.savefig('network_model_graph.png')
plt.show()