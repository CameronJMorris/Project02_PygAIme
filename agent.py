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

# general pygame configs
SCREEN_SIZE = 600
CELL_SIZE = SCREEN_SIZE // 8
BL = (0, 0, 0)
WH = (255, 255, 255)
GREEN = (0, 128, 0)
GRAY = (169, 169, 169)

# sets up the start of pygame
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Othello")

# sets up configs for the model
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:

    def __init__(self, color, file=None):

        self.color = color
        self.number_games = 0
        self.epsilon = 0
        self.gamma = 0.9 # how important it values the future
        self.memory = deque(maxlen=MAX_MEMORY)
        if file != None: # if their is an input, this is the file from which the model is loaded
            self.model = Linear_Qnet(64, 256, 64)
            self.model.load_state_dict(torch.load(file)) # loading model
            self.model.eval()
        else:
            self.model = Linear_Qnet(64, 256, 64) # sets the size of the model
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma) # sets up the trainer

    #returns the board from the game given
    def get_board(self, game):
        return game.board

    # simply stores the state, action, reward, next_state and whether the game is done in the deque
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if reward != 0:
            pass
        pass

    # after the game is over, it will train using every single action to associate different actions with rewards
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # was planning on using this but decided not to
    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    # makes the move and is random based on subtracting random epsilon
    def get_action(self, game, color):
        if len(game.get_valid_moves(color)) == 0:
            return [-2,-2]

        self.epsilon = 4000 - self.number_games # using subtracting epsilon so it decreases slower at the start and reaches zero at the end so I can see how the model is doing
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
        if random.randint(0,4000) < self.epsilon: # if the move should be random
            mov = random.choice(movesl)
            AI[mov[0]][mov[1]] = 1
            return [mov[0], mov[1]]
        else:
            state0 = torch.tensor(game.get_as_list(), dtype=torch.float)
            prediction = self.model(state0)
            _, sorted_indices = torch.sort(prediction, descending=True)
            a = sorted_indices.numpy() # is a sorted list from which index had the highest value from the model
            for i in range(len(sorted_indices)):
                n = a[i]
                row = n % 8
                col = n // 8
                if moves2[int(row)][int(col)] == 1: # when it goes through the order, the first one that is a valid move is the move of choice
                    return [int(row), int(col)]

        for i in range(8):
            for j in range(8):
                moves2[i][j] = AI[i][j] * moves[i][j] # multiplies the AI values with the valid moves to only include tha AI moves that are valid
        num = max(moves2)
        num = max(num)
        for i in range(8):
            for j in range(8):
                if moves2[i][j] == num:
                    if moves[i][j] == 0:
                        quit() # if there are no possible moves since the max is zero, it will quit
                    return [i, j] # returns the move to play

# simply a formatting method for printing what game it is
def format_nums(num):
    n = len(num)
    for i in range(8 - n):
        num = "0" + num
    return num

#simply a helper method what prints which color or agent won to help with telling how the trianing is going
def get_letter(white, black, blank):
    if white > black:
        return "w"
    if black > white:
        return "b"
    else:
        return "t"

#returns whether agent1 or agent2 won based on their colors
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

# this is the training method
def train():
    pygame.quit()
    alpha = "ABCDEFGHIJKLMONPQRSTUVWXYZ"
    agentw = Agent(WHITE, "model/main_model.pth")#loading from the main model
    agentb = Agent(BLACK, "model/main_model.pth")#loading from the main model
    game = Board()
    w = 0
    b = 0
    t = 0
    s = []
    while True:
        state_old = agentb.get_board(game)
        final_move = agentb.get_action(game, BLACK) # getting what the move is
        rewardb, done = game.play_step(BLACK, final_move[0], final_move[1]) # plays the move
        state_new = agentb.get_board(game)

        agentb.remember(state_old, final_move, rewardb, state_new, done)
        white, black, empty = game.count_stones()

        if done: # if the game is over
            s.append(get_letter(white, black, empty))# for the purpose of seeing which agent is doing better
            if (agentw.number_games % 100 == 0):
                b = s.count("b")
                t = s.count("t")
                w = s.count("w")
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                s = []

            #print("black")
            line = agentw.memory[len(agentw.memory) - 1] # setting the memory for white
            line2 = (line[0], line[1], -rewardb, line[3], line[4])# setting the memory for white
            agentw.memory[len(agentw.memory) - 1] = line2# setting the memory for white
            agentb.number_games += 1
            agentb.train_long_memory() # training the model
            agentw.number_games += 1
            agentw.train_long_memory() # training the model
            agentb.model.save(file_name="modelb.pth") # saves the model
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank) + "\t |  ") # prints what happened in the game
            game.reset()

        state_old = agentw.get_board(game)
        final_move = agentw.get_action(game, WHITE)
        rewardw, done = game.play_step(WHITE, final_move[0], final_move[1])
        state_new = agentw.get_board(game)
        agentw.remember(state_old, final_move, rewardw, state_new, done)
        if done: # if the game is over
            white, black, empty = game.count_stones()
            s.append(get_letter(white, black, empty)) # for the purpose of seeing which agent is doing better
            if ((agentw.number_games) % 100 == 0):
                b = s.count("b")
                t = s.count("t")
                w = s.count("w")
                print("w - " + str(w) + "\tb - " + str(b) + "\tt - " + str(t))
                s = []

            #print("white")
            line = agentb.memory[len(agentb.memory) - 1]# setting the memory for black
            line2 = (line[0], line[1], -rewardw, line[3], line[4])# setting the memory for black
            agentb.memory[len(agentb.memory) - 1] = line2# setting the memory for black

            agentb.number_games += 1
            agentb.train_long_memory() # training the model
            agentw.number_games += 1
            agentw.train_long_memory() # training the model
            agentw.model.save(file_name="modelw.pth") # saves the model
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(black) + "\t" + get_letter(white, black, blank) + "\t |  ") # prints what happened in the game
            game.reset()

# sets up the pygame based on the imported board
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

# finding the position of the mouse upon clicking, and returning its cell
def get_board_position_from_click(pos):
    x, y = pos
    col = x // CELL_SIZE
    row = y // CELL_SIZE
    return row, col

# allows the player to play against the computer with the imported model as the parameter
def play_game(st=None):
    alpha = "ABCDEFGHIJKLMONPQRSTUVWXYZ"
    if st == "none":
        agentw = Agent(WHITE, "model/main_model.pth") # loads the main model
        agentb = Agent(BLACK, "model/main_model.pth")# loads the main model
    else:
        agentw = Agent(WHITE, "model/"+st+".pth") # loads the inputted model
        agentb = Agent(BLACK, "model/"+st+".pth") # loads the inputted model
    game = Board()
    while True: # runs forever
        final_move = agentb.get_action(game, BLACK) # AI move
        rewardb, done = game.play_step(BLACK, final_move[0], final_move[1]) # AI move

        if done:
            white, black, blank = game.count_stones()
            print("Game: " + format_nums(str(agentb.number_games)) + "\t | w - " + str(white) + "\t | b - " + str(
                black) + "\t" + get_letter(white, black, blank) + "\t |  ")
            time.sleep(15) # 15 second delay after the game finishes to look at the boards
            game.reset()

        draw_board(game.board)
        pygame.display.flip()

        final_move = agentw.get_action(game, WHITE)
        val = game.get_valid_moves(WHITE)
        row = 10
        col = 10
        stuff = True
        if not val: # skips the players turn if they cannot make a move
            print("skipping your turn")
        else: # gets the players input of move from clicking on a cell
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
            draw_board(game.board) # draws the board
            pygame.display.flip()
            time.sleep(15)
            game.reset()
        draw_board(game.board)# draws the board
        pygame.display.flip()

        time.sleep(1.5) # waiting 1.5 seconds after the player moves to then go to the AI to allow the user to see more easily

# gets the two models the user would like to compare and optional how many games they would like to simulate default: 100
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
    while Agent1.number_games <= j: # while they haven't reached the max amount of games played
        final_move = Agent1.get_action(game, color1)
        reward, done = game.play_step(color1, final_move[0], final_move[1])
        if done:
            white, black, empty = game.count_stones()
            s.append(get_number(white, black, empty, color1, color2)) # getting which agent won
            if ((Agent1.number_games) % 100 == 0 and Agent1.number_games != 0): # sowing the recap of 100 games
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
            s.append(get_number(white, black, empty, color1, color2)) # getting which agent won
            if ((Agent1.number_games) % 100 == 0 and Agent1.number_games != 0): # sowing the recap of 100 games
                a1 = s.count("1")
                t = s.count("0")
                a2 = s.count("2")
                print("a1 - " + str(a1) + "\ta2 - " + str(a2) + "\tt - " + str(t))
                s = []
            i = color1 # flips colors
            color1 = color2  # flips colors
            color2 = i  # flips colors
            Agent1.number_games += 1
            Agent2.number_games += 1
            game.reset()

#gets whether the user wants to train, play, or compare and then prompts the user with certain inputs then selects their choice
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