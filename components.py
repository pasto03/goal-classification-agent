"""
Components required by objective classifier:
1. Environment -- walk_2D.py
2. Policy algorithm -- ppo.py
3. Components -- components.py
"""
import numpy as np
import random
import time
import os

from tqdm import tqdm
import matplotlib.pyplot as plt

from walk_2D import Walk2D
from ppo import PPO

import keyboard


# hyperparameters
lr_actor = 1e-4
lr_critic = 1e-3
gamma = 0.97
K_epochs = 8
eps_clip = 0.2
has_continuous_action_space = False


class Walk2D_interact:
    def __init__(self, size=5, ckpt_path=None, load_ckpt=True):
        self.env = Walk2D(size=size)
        self.state_dim = self.env.size ** 2 + 2
        self.action_dim = self.env.action_space.n
        self.policy = PPO(self.state_dim, self.action_dim, lr_actor, lr_critic, gamma, 
                          K_epochs, eps_clip, has_continuous_action_space)
        self.ckpt_path = f"ppo_ckpt/ckpt1-size-{size}.pt" if not ckpt_path else ckpt_path
        # load previous checkpoint if exists
        if load_ckpt:
            if os.path.exists(self.ckpt_path):
                self.policy.load(self.ckpt_path)
                print("---policy checkpoint loaded---")
        
        
    # collect experience buffer
    def collect_experience(self, goal=(0, 0), random_player_pos=False, max_step=50, render=False, duration=0.3):
        i = 0
        if random_player_pos:
            self.env.default_player_position = self.env.generate_player_position()
        state = self.env.reset(fixed_goal=goal)
        done = False
        self.policy.buffer.clear()
        while not done:
    #         ec_state = encoder(state, goal).flatten()
            ec_state = np.concatenate((state.flatten(), np.array(goal)))
            action = self.policy.select_action(ec_state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.policy.buffer.rewards.append(reward)
            self.policy.buffer.is_terminals.append(done)
            state = next_state
            
            if render:
                time.sleep(duration)
                # clear_output(wait=True)
                os.system('cls' if os.name == 'nt' else 'clear')
                print("Step {}: Goal -- {} Action -- {}".format(i+1, goal, self.env.actions[action]))
                self.env.render()

                # set new goal
                # print(keyboard.is_pressed('c'))
                if keyboard.is_pressed("c"):
                    goals = game.env.goal_states
                    print("Setting new goal:\n1. {}\n2. {}\n3. {}\n4. {}".format(*goals))
                    goal_idx = int(input("New goal(index):"))
                    goal = goals[goal_idx-1]
                    print(f"New goal: {goal}. Will be applied in next iteration.")
                    self.env.switch_goal(goal)
                    time.sleep(1.5)
            
            i += 1
            if i >= max_step:
                break
                
        return sum(self.policy.buffer.rewards)
                  
    # train model
    def train_goal(self, init_goal=None, random_player_pos=False, N_EPISODES=200, print_every=5):
        bar = tqdm(range(N_EPISODES))
        ep_rewards = []
        losses = []
        goals = self.env.goal_states
    #     goal = goals[0]
        for i in bar:
            if not init_goal:
                goal = goals[i % len(goals)]
            else:
                goal = init_goal
            ep_reward = self.collect_experience(goal=goal, random_player_pos=random_player_pos)
            loss = self.policy.update()

            ep_rewards.append(ep_reward)
            losses.append(loss)

            if i % print_every == 0:
                bar.set_description("Ep {}: Goal -- {} | Ep Reward -- {:.2f} | Loss -- {:.2f}".format(i+1, goal, 
                                                                                                    np.mean(ep_rewards), loss))
        return ep_rewards, losses

    # inspect reward and loss
    @staticmethod
    def inspect(ep_rewards, losses):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(ep_rewards)
        plt.title("Episode Reward")

        plt.subplot(2, 1, 2)
        plt.plot(losses)
        plt.title("Actor Loss")

        plt.show()

    # pretrain policy
    def pretrain_policy(self, N_EPISODES=500, random_player_pos=False, save_ckpt=True, print_every=5):
        self.train_goal(N_EPISODES=N_EPISODES, random_player_pos=random_player_pos, print_every=print_every)
        if save_ckpt:
            self.policy.save(self.ckpt_path)
            print("---checkpoint saved---")

    # user interact with RL agent
    def interact(self):
        raise NotImplementedError
    
    # generate positions for pygame render
    def yield_positions(self, goal=None, random_player_pos=True, max_step=50):
        i = 0
        if not goal:
            goal = random.choice(self.env.goal_states)
        if random_player_pos:
            self.env.default_player_position = self.env.generate_player_position()
        state = self.env.reset(fixed_goal=goal)
        yield goal, self.env.player_position, None
        done = False
        self.policy.buffer.clear()
        while not done:
            ec_state = np.concatenate((state.flatten(), np.array(goal)))
            action = self.policy.select_action(ec_state)
            next_state, reward, done, _ = self.env.step(action)
            
            self.policy.buffer.rewards.append(reward)
            self.policy.buffer.is_terminals.append(done)
            state = next_state

            yield goal, self.env.player_position, action
            
            i += 1
            if i >= max_step:
                break
                        
if __name__ == "__main__":
    game = Walk2D_interact(size=7, load_ckpt=True)
    # pretrain model
    # game.pretrain_policy(N_EPISODES=1000, random_player_pos=True)

    # render interaction of RL agent
    ### debug
    # while True:
    #     goals = game.env.goal_states
    #     goal = goals[-1]
    #     game.collect_experience(goal=goal, random_player_pos=True, render=True, duration=0.66)
    #     cont = input("Episode completed. Do you want to continue?(Y/n):")
    #     if cont.lower() != "y":
    #         break
    gen = list(game.yield_positions())
    for idx, (flag_pos, player_pos, action) in enumerate(gen):
        print(idx, flag_pos, player_pos, action)
