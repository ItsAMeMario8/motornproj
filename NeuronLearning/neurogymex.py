import matplotlib.pyplot as plt

import neurogym as ngym
import numpy as np
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

loss_list = []
iteration_list = []
count = 0
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

    count += 1
    
    running_loss += loss.item()
    if count % 100 == 0: #every 100
        print('{:d} loss: {:0.5f}'.format(i+1, running_loss / 100))
        running_loss = 0.0
        loss_list.append(loss.data)
        iteration_list.append(count)

print('Finished Training')
plt.plot(iteration_list, loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

perf = 0
num_trial = 200
for i in range(num_trial):
    env.new_trial()
    ob, gt = env.ob, env.gt
    ob = ob[:, np.newaxis, :]
    inputs = torch.from_numpy(ob).type(torch.float).to(device)

    action_pred = net(inputs)
    action_pred = action_pred.detach().numpy()
    action_pred = np.argmax(action_pred, axis=-1)
    perf += gt[-1] == action_pred[-1, 0]

perf /= num_trial
print('Average performance in {:d} trials'.format(num_trial))
print(perf)
