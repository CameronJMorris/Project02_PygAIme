import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Linear_Qnet(nn.Module):
    # initialization for hte model using the inputted parameters
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_Qnet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    #gets the value of the model at one spos
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    # sames the model to the path given or just to model.pth
    def save(self, file_name='model.pth'):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):#making file
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)#saving model
        #print(f"Model saved to {file_name}")


class QTrainer:

    #makes the trainer with optimization of loss
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    # this is the actual method that tarins the model
    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert list of states to a tensor and flatten each state to size 64
        #print(torch.tensor(states[0]).flatten())
        states = [torch.tensor(s, dtype=torch.float).flatten() for s in states]  # Convert and flatten each state
        states = torch.stack(states)  # Combine list of tensors into a single batch tensor


        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = [torch.tensor(ns, dtype=torch.float).flatten() for ns in next_states] # flattneing them so that torch can handle it
        next_states = torch.stack(next_states)

        if len(states.shape) == 1: # this is for if there is short term training though I did not need this
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            dones = (dones,)

        pred = self.model(states) # gets the prediction

        targ = pred.clone() # makes a copy to be turned into target values


        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new += self.gamma * torch.max(self.model(next_states[idx]))

            # creates the target values for each of the indecies
            targ[idx][actions[idx][0] * 8 + actions[idx][1]] = Q_new

        # optimizes the losses
        self.optimizer.zero_grad()
        loss = self.criterion(targ, pred)# optimizes the losses
        loss.backward() # optimizes the losses
        self.optimizer.step() # updates the optimizer