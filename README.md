# Othello AI Project
## Credits
- Chatgpt - I mainly used chatgpt for debugging by giving it error messages, though I also used it a bit for learning different syntaxes related to each library
- Humberto Henrique Campos Pinheiro - This was the base Othello game that I used though I ended up only using his config.py file and some parts of his board file denoted by what is above the dashed line, though even some of the parts that were not the same as his took some of th e same inspiration
## How to run the program
- To start the program you should start the agent file which will prompt you to enter whether you would like to train or play, I would recommend play, so that you can play against the AI. You can enter the model you would like to play against, or you can leave the field blank by entering "none".
## What I would do next
- My next actions with this project would be to try different models because I think that there are many other models which would be much better than the one I am using since the model is still not very good. Beyond this, I may train it longer and change the epsilon value to make it better in general.
## How it works
### Config File
- The config file contains simple constants that will be used throughout the program, most notably "BLACK" and "WHITE", which are 1 and 2 respectively.
### Board File
- The board file is the file that contains the game and stores the board as an 8 by 8 array. Additionally, it contains many different methods with valid moves to help in the future class.
### Model File
#### Model Class
- The model class is a simple class that just contains the linear_Qnet and the matrices that accompany it. The only functions in theis class are the save function which saves the model to a file and the forward function which allows for a prediction.
#### Trainer Class
- The trainer class takes in the different values including reward, next_state, old_state, action, and whether the game si done. From this it creates the model and trains it according to Bellman's equation, which optimizes the model.
### Agent File          
#### Agent Class
- The agent class contains multiple functions related to training the model. The first of which is remember, which commits the past actions results and the other values that are passed into train into memory(a deque). After it there is a train_long_memory which takes all of those from throughout the game and passes them into the trainer class. Next is train_short_memory which is intended to train one move at a time, and I have not implemented it here though that may be one of the changes that I make in the future.
- The last component of the class is the method that gets the action of the agent. At the start the moves are partially random and partially based on the model though as you get further into training more of the moves are decided by the model. This function also makes sure that the returned location is a possible move for the agent.
#### Outside The Class
- Outside the class there are many small functions that help with the processing of information to present to the person.
- Beyond these, there is the train function which calls upon all the previously described methods of memory, saving, getting the action to create a coherent action and training. The file also allows the user to decide if they want to load in different models that were created earlier to look into them. The method prints out the result of every game in the form of how many tiles each color had at the end of the game. Additionally, it will print out what the overall score was over the past 100 matches.
- Additionally, the file has a play method that allows the user to play one of the models of their choice. This will be in the form of a pygame. There is an artificial delay after each user move, so that the user can see the moves more clearly, though this could be removed which would allow the game to be played very quickly. The user can play their moves by clicking on the box that they wish to place a piece on. The user will be the white color though which color goes first can change to allow for the user to go first or second.
- Finally, there is a callable main which will allow the user to choose between the training method or the play method from above. This is also where the user would enter which model they would wish to play against.
