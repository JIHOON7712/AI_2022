from torch.optim import SGD #Default optimizer. you may use other optimizers
from torch.utils.data import DataLoader
import argparse
import torch
import torch.nn as nn
import gym

from utils import get_explore_rate, select_action
from model import Q_net

import random
import copy

from collections import deque, namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def __getitem__(self, idx):
        transition = self.memory[idx]
        state = torch.tensor(transition.state, dtype=torch.float)
        action = torch.tensor(transition.action, dtype=torch.float)
        next_state = torch.tensor(transition.next_state, dtype=torch.float)
        reward = torch.tensor(transition.reward, dtype=torch.float)
        return state, action, next_state, reward


def simulate(model, args): #model: the neural network
    #optimizer and loss for the neural network
    model = model.to(device=args.device)
    optimizer = SGD(model.parameters(), lr = 0.001)
    criterion = nn.MSELoss()
    ## Instantiating the learning related parameters
    explore_rate = get_explore_rate(0, args.decay_constant, args.min_explore_rate)

    memory = ReplayMemory(args.max_memory)
    num_streaks = 0
    for episode in range(args.num_episodes):
        # Reset the environment
        state_0 = env.reset()
        
        for t in range(args.max_timestep):
            env.render()#you may want to comment this line out, to run code silently without rendering
            
            # Selecting an action. the action MUST be choosed from the neural network's output.
            with torch.no_grad():
                actiontable = model(torch.Tensor(state_0).unsqueeze(0).to(args.device))
                action = select_action(actiontable.squeeze(0), explore_rate, env)

            # Execute the action then collect outputs
            state, reward, done, _ = env.step(action)

            #Update the memory
            
            ####################################################
            ## TODO:Implement your memory
            memory.push(state_0, action, state, reward)
            ####################################################
            
            
            # Update the Q-net parameters
            replay(model, memory, args, criterion, optimizer, iteration = 1)
            
            state_0 = state

            #done: the cart failed to maintain balance
            if done == True:
                break

        # Update parameters
        explore_rate = get_explore_rate(episode, args.decay_constant, args.min_explore_rate)
        
        result_msg = "Episode %d finished after %f time steps. Streak: %d" % (episode, t, num_streaks)
        print(result_msg)
        with open("result_msg.txt", 'a') as f:
            f.write(f"{result_msg}\n")
        #The episode is considered as a success if timestep >SOLVED_TIMESTEP 
        if (t >= args.solved_timestep):
            num_streaks += 1
        else:
            num_streaks = 0
            
        #  when the agent 'solves' the environment: steak over 120 times consecutively
        if num_streaks > args.streak_to_end:
            print("The Environment is solved")
            torch.save(model.state_dict(), 'modelparam.pt')
            break

    env.close()#closes window

def replay(model, memory, args, criterion, optimizer, iteration = 1):
    d_loader = DataLoader(memory, args.batchsize ,shuffle = True, drop_last= True)

    device = args.device

    policy_net = model
    target_net = copy.deepcopy(model).to(device)
    for i, batch in enumerate(d_loader):
        ####################################################
        ## TODO: Implement your replay function.
        if iteration == i:
            break
        state, action, next_state, reward = batch
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_state)), 
                                      device=device, dtype=torch.bool)
        non_final_next_states = next_state[non_final_mask].to(device)
        state_batch = state.to(device)
        action_batch = action.to(device)
        reward_batch = reward.to(device)

        q_value = policy_net(state_batch)
        state_action_values = [q_value[jdx, act.type(torch.long)] for jdx, act in enumerate(action_batch)]
        state_action_values = torch.stack(state_action_values)

        next_state_values = torch.zeros(args.batchsize, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * 0.8) + reward_batch

        # Compute Huber Loss
        loss = criterion(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()
        ####################################################
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model!')
    parser.add_argument('--num_episodes', type = int, default= 10000)
    parser.add_argument('--max_timestep', type = int, default= 250)
    parser.add_argument('--solved_timestep', type = int, default= 199)
    parser.add_argument('--streak_to_end', type = int, default= 120)
    parser.add_argument('--batchsize', type = int, default= 128)
    parser.add_argument('--min_explore_rate', type = float, default= 0.01)
    parser.add_argument('--discount_factor', type = float, default= 0.99)
    parser.add_argument('--decay_constant', type = int, default= 25)
    parser.add_argument('--max_memory', type = int, default = 1000)
    parser.add_argument('--device', type = str, default = "cuda")
    train_args = parser.parse_args()

    env = gym.make('CartPole-v1')
    qnet = Q_net()
    simulate(qnet, train_args)