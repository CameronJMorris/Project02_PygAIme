import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Linear_Qnet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_Qnet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = "./model"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        #print(f"Model saved to {file_name}")


class QTrainer:

    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        # Convert list of states to a tensor and flatten each state to size 64
        #print(torch.tensor(states[0]).flatten())
        states = [torch.tensor(s, dtype=torch.float).flatten() for s in states]  # Convert and flatten each state
        states = torch.stack(states)  # Combine list of tensors into a single batch tensor


        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = [torch.tensor(ns, dtype=torch.float).flatten() for ns in next_states]
        next_states = torch.stack(next_states)

        if len(states.shape) == 1:
            # In case it's a single sample, unsqueeze to add a batch dimension
            states = torch.unsqueeze(states, 0)
            actions = torch.unsqueeze(actions, 0)
            rewards = torch.unsqueeze(rewards, 0)
            next_states = torch.unsqueeze(next_states, 0)
            dones = (dones,)

        # Predict Q-values with the current states (batch prediction)
        pred = self.model(states)  # shape: [batch_size, output_size]

        # Clone predictions to create the target tensor
        targ = pred.clone()

        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new += self.gamma * torch.max(self.model(next_states[idx]))

            # Update the target for the action taken
            #print(targ[idx])
            #print(int(actions[idx][0]))
            targ[idx][actions[idx][0] * 8 + actions[idx][1]] = Q_new

        # Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(targ, pred)
        loss.backward()
        self.optimizer.step()