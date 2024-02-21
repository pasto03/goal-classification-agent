import gym
from gym import spaces
import numpy as np
import random
from copy import deepcopy


class Walk2D(gym.Env):
    def __init__(self, size=10, max_steps=50, player_position=None, goal_position=None):
        self.size = size
        self.grid = np.zeros((self.size, self.size))
        self.max_steps = max_steps
        self.steps = 0

        # Set default positions if not provided
        self.default_player_position = (self.size // 2, self.size // 2)
        self.default_goal_position = (0, self.size-1)  # Top right corner
        self.player_position = player_position if player_position else self.default_player_position
        self.goal_position = goal_position if goal_position else self.default_goal_position
        
        self.goal_states = [(0, 0), (0, self.size - 1), (self.size - 1, 0), (self.size - 1, self.size - 1)]
        self.available_coors = [[i, j] for i in range(self.size) for j in range(self.size)]
        for goal in self.goal_states:
            self.available_coors.remove(list(goal))

        # Define the action space: 0 = up, 1 = down, 2 = left, 3 = right
        self.actions = {0: "left", 1: "right", 2: "up", 3: "down"}
        self.action_space = spaces.Discrete(4)
        # Define the observation space
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.size, self.size), dtype=np.float32)

    def generate_player_position(self):
        return tuple(random.choice(self.available_coors))

    def reset(self, fixed_goal=None):
        self.steps = 0
        self.grid = np.zeros((self.size, self.size))

        # Set a random goal from the four specified states if not provided
        self.goal_position = fixed_goal if fixed_goal else random.choice(self.goal_states)

        self.grid[self.goal_position] = 2
        self.player_position = self.default_player_position
        self.grid[self.player_position] = 1
        return self.grid

    def render(self, mode='human'):
        print(self.grid.astype(np.int8))

    def switch_goal(self, new_goal):
        if new_goal in self.goal_states:
            self.goal_position = new_goal
            self.grid = np.zeros((self.size, self.size))
            self.grid[self.goal_position] = 2
            self.grid[self.player_position] = 1

    def step(self, action):
        # Update player position based on action
        if action == 0 and self.player_position[0] > 0:  # Left
            self.player_position = (self.player_position[0] - 1, self.player_position[1])
        elif action == 1 and self.player_position[0] < self.size - 1:  # Right
            self.player_position = (self.player_position[0] + 1, self.player_position[1])
        elif action == 2 and self.player_position[1] > 0:  # Up
            self.player_position = (self.player_position[0], self.player_position[1] - 1)
        elif action == 3 and self.player_position[1] < self.size - 1:  # Down
            self.player_position = (self.player_position[0], self.player_position[1] + 1)

        # Update grid
        self.grid = np.zeros((self.size, self.size))
        self.grid[self.goal_position] = 2
        self.grid[self.player_position] = 1

        # Calculate reward and check if goal is reached
        done = False
        reward = -0.1  # Default step cost
        if self.player_position == self.goal_position:
            reward = 1  # Reward for reaching the goal
            done = True

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self.grid, reward, done, {}
    
if __name__ == "__main__":
    env = Walk2D(size=7)
    env.default_player_position = env.generate_player_position()
    print(env.default_player_position)
    state = env.reset()
    env.render()