import pygame
import time
import sys
pygame.init()

# from env
grid_size = 7

tile_width = 64
tile_height = 64
margin = 8  # space between the boxes

WIDTH = (tile_width + margin) * grid_size + margin
interact_space = 200
HEIGHT = (tile_height + margin) * grid_size + margin + interact_space
# print(WIDTH, HEIGHT)
win = pygame.display.set_mode((WIDTH, HEIGHT))

white = (255, 255, 255)
FONT = pygame.font.SysFont("Sans", 20)
TEXT_COLOR = white

flag = pygame.image.load('images/finish flag.png').convert_alpha()
flag = pygame.transform.scale(flag, (tile_width, tile_height))
# flag_pos = (0, 0)

player = pygame.image.load('images/robot icon.png').convert_alpha()
player = pygame.transform.scale(player, (tile_width, tile_height))
# player_pos = (grid_size // 2, grid_size // 2)

# action = None

clock = pygame.time.Clock()

from button import Button, PauseButton
btn_size = (100, 30)
pause_btn = PauseButton(button_pos=(WIDTH-(20+100+20), HEIGHT-interact_space+20), button_size=btn_size, font_size=26)
start_btn = Button(button_pos=(WIDTH-(20+100+20+100+20), HEIGHT-interact_space+20), button_size=btn_size, font_size=26, label='start')

from textinputbox import TextInputBox
text_input_box = TextInputBox(20, HEIGHT-100, WIDTH-(20+100+20+100+20)+80, 30, font_size=26)
text_prompt = text_input_box.font.render("Enter objective command:", True, text_input_box.text_color)
clear_btn = Button(button_pos=(WIDTH-(20+100+20), HEIGHT-100), button_size=btn_size, font_size=26, label='clear')

path_history = []
def render_grid(flag_pos, player_pos):
    x = margin
    for column in range(grid_size):
        y = margin
        for row in range(grid_size):
            current_pos = (column, row)
            rect = pygame.Rect(x, y, tile_width, tile_height)
            if current_pos in path_history:
                pygame.draw.rect(win, (211, 211, 211), rect)
            else:   
                pygame.draw.rect(win, white, rect)
            y = y + tile_height + margin
            # draw flag
            if current_pos == flag_pos and (player_pos != flag_pos):
                win.blit(flag, rect)
            # draw player
            if current_pos == player_pos:
                win.blit(player, rect)
            
        x = x + tile_width + margin


run = True
step = 0

from components import Walk2D_interact
game = Walk2D_interact(size=7, load_ckpt=True)

from obj_classifier import ObjectiveClassifier
cls = ObjectiveClassifier(load_ckpt=True)

import random
def simp(cmd):
    size=grid_size
    return random.choice([(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)])

# goal_idx = 0
# goal = game.env.goal_position[goal_idx]
# interact_gen = list(game.yield_positions(goal))
delay = 750
pygame.time.set_timer(pygame.USEREVENT, delay)

flag_pos, player_pos, action = (0, 0), (grid_size // 2, grid_size // 2), None
action_name = "None"
cmd = None
start = False
while run:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if pause_btn.is_clicked(event):
            pause_btn.switch()
        
        cmd = text_input_box.text
        if cmd:
            if start_btn.is_clicked(event):
                goal_idx, _ = cls.translate_command(cmd, debug=True)
                goal = game.env.goal_states[goal_idx]
                interact_gen = list(game.yield_positions(goal))
                start = True
                step = 0
                # flag_pos = simp(cmd)
                pygame.time.set_timer(pygame.USEREVENT, delay)
                path_history = []
            if start and not pause_btn.is_paused and (event.type == pygame.USEREVENT or step == 0):
                if step == len(interact_gen):
                    continue
                flag_pos, player_pos, action = interact_gen[step]
                step += 1
        if clear_btn.is_clicked(event):
            text_input_box.text = ""
        text_input_box.handle_event(event)
    
    win.fill((0, 0, 0))

    path_history.append(player_pos)
    render_grid(flag_pos, player_pos)

    action_name = game.env.actions[action] if action != None else "-"
    message = f"Step {step} | Action -- {action_name}"
    win.blit(FONT.render(message, True, TEXT_COLOR), (20, HEIGHT-interact_space+20))

    # add button
    pause_btn.draw(win)
    start_btn.draw(win)

    # add textbox
    text_input_box.draw(win)
    clear_btn.draw(win)
    win.blit(text_prompt, (20, HEIGHT-130))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
