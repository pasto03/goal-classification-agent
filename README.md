# Goal Classification Agent
Goal classification agent is the combination of text classification model and Reinforcement Learning algorithm(PPO), making the agent available to execute command from user natural language input.

## Features
Simply run _game.py_ and an agent will be initialized as a Pygame GUI.

https://github.com/pasto03/goal-classification-agent/assets/101184462/d0e619d1-e0db-4875-bb2c-70fdcdc8613c

## Architecture
The agent combines two models:

### 1. Objective Classifier(_obj_classifier.py_):
This model classifies user text input into goal index, combination of seq2seq encoder and output layer

### 2. RL Agent(_components.py_):
This model utilize PPO algorithm to train and eval policy given observation and goal state.

## Example Usage
An example of training custom RL policy:
```python
from components import Walk2D_interact

game = Walk2D_interact(size=7, load_ckpt=True)

# pretrain model
# game.pretrain_policy(N_EPISODES=1000, random_player_pos=True)
```

Fine-tune objective classifier for one specific goal command:
```python
from obj_classifier import ObjectiveClassifier

cls = ObjectiveClassifier(load_ckpt=True)
train_data = [GOAL_COMMAND, GOAL_IDX]
cls.fine_tune(train_data, n_epochs=n_epochs)
```


