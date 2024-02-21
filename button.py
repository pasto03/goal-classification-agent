import pygame
import sys

class Button:
    def __init__(self, button_pos: tuple, button_size=(200, 100), font_size=36, label: str = "button"):
        pygame.font.init()
        self.label = label
        self.font = pygame.font.Font(None, font_size)
        self.button_color = (255, 0, 0)  # red button
        self.button_size = button_size
        self.button_rect = pygame.Rect(button_pos[0], button_pos[1], *self.button_size)
    
    def draw(self, screen):
        # draw button
        pygame.draw.rect(screen, self.button_color, self.button_rect)

        # show text at the middle of button
        text = self.font.render(self.label, True, (255, 255, 255))
        text_rect = text.get_rect(center=self.button_rect.center)
        screen.blit(text, text_rect)

    def is_clicked(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # check if button clicked
            if self.button_rect.collidepoint(event.pos):
                return True
            
class PauseButton(Button):
    def __init__(self, button_pos: tuple, button_size=(200, 100), font_size=36, label: str = "pause"):
        super().__init__(button_pos, button_size, font_size, label)
        self.is_paused = False

    def is_clicked(self, event):
        return super().is_clicked(event)
    
    def switch(self):
        if not self.is_paused:
            self.label = 'continue'
            self.is_paused = True
            self.button_color = (0, 255, 0)
        else:
            self.label = 'pause'
            self.is_paused = False
            self.button_color = (255, 0, 0)
    

if __name__ == "__main__":
    screen_size = (800, 600)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Pygame Button Demo")

    # instance of button
    button_position = (100, 100)
    button = Button(button_position, label="Click Me")

    pause_btn = PauseButton(button_pos=(100, 300))

    # define font
    font = pygame.font.Font(None, 36)

    # main loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            elif button.is_clicked(event):
                print("Button clicked!")
            
            elif pause_btn.is_clicked(event):
                pause_btn.switch()

        screen.fill((255, 255, 255))

        # draw button
        button.draw(screen)
        pause_btn.draw(screen)

        pygame.display.flip()
