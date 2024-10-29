import torch
import random
import time

from board import Board
from config import WHITE, BLACK
from collections import deque

from model import Linear_Qnet, QTrainer

import pygame
import sys

pygame.init()


SCREEN_SIZE = 600
CELL_SIZE = SCREEN_SIZE // 8
BL = (0, 0, 0)
WH = (255, 255, 255)
GREEN = (0, 128, 0)
GRAY = (169, 169, 169)

screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Othello")

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self, color, file=None):

        self.color = color
        self.number_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        if file != None:
            self.model = Linear_Qnet(64, 256, 64)
            self.model.load_state_dict(torch.load(file))
            self.model.eval()
        else:
            self.model = Linear_Qnet(64, 256, 64)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def get_board(self, game):
        return game.board

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if reward != 0:
            pass
        pass

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, game, color):
        if len(game.get_valid_moves(color)) == 0:
            return [-2,-2]

        self.epsilon = 4000 - self.number_games
        movesl = game.get_valid_moves(color)
        moves2 = game.get_valid_moves2(color)
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
        if random.randint(0,4000) < self.epsilon:
            mov = random.choice(movesl)
            AI[mov[0]][mov[1]] = 1
            return [mov[0], mov[1]]
        else:
            state0 = torch.tensor(game.get_as_list(), dtype=torch.float)
            prediction = self.model(state0)
            _, sorted_indices = torch.sort(prediction, descending=True)
            a = sorted_indices.numpy()
            for i in range(len(sorted_indices)):
                n = a[i]
                row = n % 8
                col = n // 8
                if moves2[int(row)][int(col)] == 1:
                    return [int(row), int(col)]

        for i in range(8):
            for j in range(8):
                moves2[i][j] = AI[i][j] * moves[i][j]
        num = max(moves2)
        num = max(num)
        for i in range(8):
            for j in range(8):
                if moves2[i][j] == num:
                    if moves[i][j] == 0:
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

def get_number(white, black, blank, color1, color2):
    if white == black:
        return "0"
    if black > white:
        if color1 == BLACK:
            return "1"
        else:
            return "2"
    if white > black:
        if color1 == WHITE:
            return "1"
        else:
            return "2"

def train():
    pygame.quit()
    alpha = "ABCDEFGHIJKLMONPQRSTUVWXYZ"
    agentw = Agent(WHITE, "model/main_model.pth")
    agentb = Agent(BLACK, "model/main_model.pth")
    game = Board()
    w = 0
    b = 0
    t = 0
    s = []
    while True:
        state_old = agentb.get_board(game)
        final_move = agentb.get_action(game, BLACK)
        rewardb, done = game.play_step(BLACK, final_move[0], final_move[1])
        state_new = agentb.get_board(game)

        agentb.remember(state_old, final_move, rewardb, state_new, done)
        white, black, empty = game.count_stones()

        if done:
            s.append(get_letter(white, black, empty))
            if (agentw.number_games % 100 == 0):
                b = s.count("b")
                t = s.count("t")
                w = s.count("w")
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                s = []

            #print("black")
            line = agentw.memory[len(agentw.memory) - 1]
            line2 = (line[0], line[1], -rewardb, line[3], line[4])
            agentw.memory[len(agentw.memory) - 1] = line2
            agentb.number_games += 1
            agentb.train_long_memory()
            agentw.number_games += 1
            agentw.train_long_memory()
            agentb.model.save(file_name="modelb.pth")
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank) + "\t |  ")
            game.reset()

        state_old = agentw.get_board(game)
        final_move = agentw.get_action(game, WHITE)
        rewardw, done = game.play_step(WHITE, final_move[0], final_move[1])
        state_new = agentw.get_board(game)
        agentw.remember(state_old, final_move, rewardw, state_new, done)
        if done:
            white, black, empty = game.count_stones()
            s.append(get_letter(white, black, empty))
            if ((agentw.number_games) % 100 == 0):
                b = s.count("b")
                t = s.count("t")
                w = s.count("w")
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                s = []

            #print("white")
            line = agentb.memory[len(agentb.memory) - 1]
            line2 = (line[0], line[1], -rewardw, line[3], line[4])
            agentb.memory[len(agentb.memory) - 1] = line2

            agentb.number_games += 1
            agentb.train_long_memory()
            agentw.number_games += 1
            agentw.train_long_memory()
            agentw.model.save(file_name="modelw.pth")
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank) + "\t |  ")
            game.reset()

def draw_board(board):
    screen.fill(GREEN)

    for row in range(8):
        for col in range(8):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GRAY, rect, 1)

            if board[row][col] == 1:
                pygame.draw.circle(screen, BL, rect.center, CELL_SIZE // 2 - 10)
            elif board[row][col] == 2:
                pygame.draw.circle(screen, WH, rect.center, CELL_SIZE // 2 - 10)

def get_board_position_from_click(pos):
    x, y = pos
    col = x // CELL_SIZE
    row = y // CELL_SIZE
    return row, col

def play_game(st=None):
    alpha = "ABCDEFGHIJKLMONPQRSTUVWXYZ"
    if st == "none":
        agentw = Agent(WHITE, "model/main_model.pth")
        agentb = Agent(BLACK, "model/main_model.pth")
    else:
        agentw = Agent(WHITE, "model/"+st+".pth")
        agentb = Agent(BLACK, "model/"+st+".pth")
    game = Board()
    while True:
        final_move = agentb.get_action(game, BLACK)
        rewardb, done = game.play_step(BLACK, final_move[0], final_move[1])

        if done:
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(
                black) + "\t" + get_letter(white, black, blank) + "\t |  ")
            time.sleep(15)
            game.reset()

        draw_board(game.board)
        pygame.display.flip()

        final_move = agentw.get_action(game, WHITE)
        val = game.get_valid_moves(WHITE)
        row = 10
        col = 10
        stuff = True
        if not val:
            print("skipping your turn")
        else:
            while stuff:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        pygame.quit()
                        sys.exit()
                    elif event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        row, col = get_board_position_from_click(pos)
                    if (row, col) in val:
                        reward, done = game.play_step(WHITE, row, col)
                        stuff = False
                        row = 10
                        col = 10
            stuff = True
        rewardw, done = game.play_step(WHITE, row, col)



        if done:
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(
                black) + "\t" + get_letter(white, black, blank) + "\t |  ")
            time.sleep(15)
            game.reset()
        draw_board(game.board)
        pygame.display.flip()

        time.sleep(1.5)

def compare(a1, a2, j=100):
    color1 = BLACK
    color2 = WHITE
    Agent1 = Agent(color1, "model/"+a1+".pth")
    Agent2 = Agent(color2, "model/"+a2+".pth")
    ag1 = 0
    ag2 = 0
    game = Board()
    w = 0
    b = 0
    t = 0
    s = []
    i = 0
    Agent1.number_games = 1
    Agent2.number_games = 1
    while Agent1.number_games <= j:
        final_move = Agent1.get_action(game, color1)
        reward, done = game.play_step(color1, final_move[0], final_move[1])
        if done:
            white, black, empty = game.count_stones()
            s.append(get_number(white, black, empty, color1, color2))
            if ((Agent1.number_games) % 100 == 0 and Agent1.number_games != 0):
                a1 = s.count("1")
                t = s.count("0")
                a2 = s.count("2")
                print("a1 - " + str(a1) + "\ta2 - " + str(a2) + "\tt - " + str(t))
                s = []
            Agent1.number_games += 1
            Agent2.number_games += 1
            i = color1
            color1 = color2
            color2 = i

            game.reset()
        final_move = Agent2.get_action(game, color2)
        reward, done = game.play_step(color2, final_move[0], final_move[1])
        if done:
            white, black, empty = game.count_stones()
            s.append(get_number(white, black, empty, color1, color2))
            if ((Agent1.number_games) % 100 == 0 and Agent1.number_games != 0):
                a1 = s.count("1")
                t = s.count("0")
                a2 = s.count("2")
                print("a1 - " + str(a1) + "\ta2 - " + str(a2) + "\tt - " + str(t))
                s = []
            i = color1
            color1 = color2
            color2 = i
            Agent1.number_games += 1
            Agent2.number_games += 1
            game.reset()

if __name__ == '__main__':
    s = input("Would you like to train or play: ")
    if s == "train":
        train()
    if s == "play":
        s = input("Which model would you like to play: ")
        play_game(s)
    if s == "compare":
        a1= input("Which agent would you like to compare: ")
        a2= input("Which agent would you like to compare: ")
        i = int(input("How many games would you like to simulate: "))
        if i == 0:
            compare(a1, a2)
        else:
            compare(a1, a2, i)