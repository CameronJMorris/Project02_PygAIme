import torch
import random
import numpy as np
from numpy.f2py.auxfuncs import throw_error

from board import Board
from config import WHITE, BLACK, EMPTY, NOTHING, LOSE, WIN, TIE
from collections import deque

from model import Linear_Qnet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self, color):

        self.color = color
        self.number_games = 0
        self.epsilon = 0 # rand
        self.gamma = 0.9 # disc r/.8-.9
        self.memory = deque(maxlen=MAX_MEMORY) # gets rid of elements on the left if it runs out of memory
        self.model = Linear_Qnet(64, 256, 64)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)




    def get_board(self, game):
        return game.board

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if reward != 0:
            pass
            #print("asdfasdfasdfgdfg")
            #print(self.memory)
            #print(reward)
        pass

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        #print(rewards, self.color)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, game, color):
        if len(game.get_valid_moves(color)) == 0:
            return [-2,-2]

        #print(game.get_as_list())
        self.epsilon = 80 - self.number_games
        final_move = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]]
        movesl = game.get_valid_moves(color)
        moves2 = game.get_valid_moves2(color) #array with 1 as the legal moves and 0 as the illegal moves
        moves = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]]
        AI = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]]
        #print("\n\n\n\n\n")
        #print(game.get_valid_moves(color))
        if random.randint(0,200) < self.epsilon:
            #print("rand -")
            mov = random.choice(movesl)
            AI[mov[0]][mov[1]] = 1
            return [mov[0], mov[1]]
        else:
            state0 = torch.tensor(game.get_as_list(), dtype=torch.float)
            prediction = self.model(state0)
            #print("pred - ")
            #print(prediction)
            _, sorted_indices = torch.sort(prediction, descending=True)
            #print(moves2)
            a = sorted_indices.numpy()
            for i in range(len(sorted_indices)):
                n = a[i]
                #print(n)
                row = n % 8
                col = n // 8
                #print("sdfgsdfg")
                #print(len(moves2))
                #print(str(int(row)) + " " + str(int(col)))

                #print(str(i) + " " + str(int(row)) + " " + str(int(col)) + " " + str(moves2[int(row)][int(col)]))
                if moves2[int(row)][int(col)] == 1:
                    return [int(row), int(col)]
            #print((max(prediction)))
            #print(prediction[(max(prediction))])

        for i in range(8):
            for j in range(8):
                moves2[i][j] = AI[i][j] * moves[i][j]
        #print(str(moves2).replace("], [","]\n[").replace("]]", "]").replace("[[","["))
        num = max(moves2)
        #print("\n\n\n")
        #print(str(num).replace("], [","]\n[").replace("]]", "]").replace("[[","["))
        num = max(num)
        #print(num)
        for i in range(8):
            for j in range(8):
                if moves2[i][j] == num:
                    if moves[i][j] == 0:
                        #print(str(i) + " " + str(j))
                        quit()

                    return [i, j]

def format_nums(num):
    n = len(num)
    for i in range(8 - n):
        num = "0" + num
    return num

def get_letter(white, black, blank):
    if white > black:
        return "w"
    if black > white:
        return "b"
    else:
        return "t"

def train():
    alpha = "ABCDEFGHIJKLMONPQRSTUVWXYZ"
    plot_scores = []
    plot_mean_scores = []
    total_games = 0
    record = 0
    agentw = Agent(WHITE)
    agentb = Agent(BLACK)
    game = Board()
    w = 0
    b = 0
    t = 0
    while True:
        state_old = agentb.get_board(game)
        final_move = agentb.get_action(game, BLACK)
        #print(final_move)
        #print(str(alpha[final_move[0]])+str(int(final_move[1])+1))
        rewardb, done = game.play_step(BLACK, final_move[0], final_move[1])
        state_new = agentb.get_board(game)
        #print(rewardb, done)
        #agentb.train_short_memory(state_old, final_move, rewardb, state_new, done)
        agentb.remember(state_old, final_move, rewardb, state_new, done)
        if done:
            white, black, empty = game.count_stones()
            if ((b + w + t) % 100 == 0):
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                b = 0
                t = 0
                w = 0
            if black > white:
                b += 1
            elif white > black:
                w += 1
            else:
                t += 1
            #print("black")
            line = agentw.memory[len(agentw.memory) - 1]
            line2 = (line[0], line[1], -rewardb, line[3], line[4])
            agentw.memory[len(agentw.memory) - 1] = line2
            #game.print_board()
            agentb.number_games += 1
            agentb.train_long_memory()
            agentw.number_games += 1
            agentw.train_long_memory()
            agentb.model.save()
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank))
            game.reset()

        state_old = agentw.get_board(game)
        final_move = agentw.get_action(game, WHITE)
        #print(final_move)
        #print(str(alpha[final_move[0]])+str(int(final_move[1])+1))
        rewardw, done = game.play_step(WHITE, final_move[0], final_move[1])
        state_new = agentw.get_board(game)
        #print(rewardw, done, "WHITE")
        # agentw.train_short_memory(state_old, final_move, rewardw, state_new, done)
        agentw.remember(state_old, final_move, rewardw, state_new, done)
        if done:
            if ((agentw.number_games) % 100 == 0):
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                b = 0
                t = 0
                w = 0
            if game.game_state(BLACK) == WIN:
                b += 1
            elif game.game_state(WHITE) == WIN:
                w += 1
            else:
                t += 1
            #print("white")
            line = agentb.memory[len(agentb.memory) - 1]
            line2 = (line[0], line[1], -rewardw, line[3], line[4])
            agentb.memory[len(agentb.memory) - 1] = line2

            #game.print_board()
            agentb.number_games += 1
            agentb.train_long_memory()
            agentw.number_games += 1
            agentw.train_long_memory()
            agentw.model.save()
            #print(game.count_stones())
            white, black, blank = game.count_stones()
            #print(white, black, blank)
            #print(game.get_as_list())
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank))
            game.reset()



if __name__ == '__main__':
    """game = Board()
    agent = Agent(BLACK)


    for i in range(10):
        act = agent.get_action(game, BLACK)
        print(act)
        game.play_step(BLACK, act[0], act[1])
        act = agent.get_action(game, WHITE)
        print(act)
        game.play_step(WHITE, act[0], act[1])
        print("\n\n")
        game.print_board()"""
    train()