import pygame
import sys
from button import Button

class TextInputBox:
    def __init__(self, x, y, width, height, font_size=32, text_color=(255, 255, 255), background_color=(0, 0, 0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.Font(None, font_size)
        self.text_color = text_color
        self.background_color = background_color
        self.text = ""
        self.is_active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.is_active = not self.is_active
            else:
                self.is_active = False
        elif event.type == pygame.KEYDOWN:
            if self.is_active:
                if event.key == pygame.K_RETURN:
                #     self.is_active = False
                #     # print(self.text)
                #     text = self.text
                #     self.text = ""
                    pass
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode

    def draw(self, screen):
        pygame.draw.rect(screen, self.background_color, self.rect, 0)
        pygame.draw.rect(screen, (255, 255, 255), self.rect, 2)
        text_surface = self.font.render(self.text, True, self.text_color)
        screen.blit(text_surface, (self.rect.x + 5, self.rect.y + 5))

if __name__ == "__main__":
    # 示例使用：
    pygame.init()

    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("TextInputBox Example")

    text_input_box = TextInputBox(100, 100, 200, 40)
    clear_btn = Button(button_pos=(350, 100), button_size=(100, 40), label='clear')
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if clear_btn.is_clicked(event):
                text_input_box.text = ""
            text_input_box.handle_event(event)

        screen.fill((0, 0, 0))

        text_input_box.draw(screen)
        clear_btn.draw(screen)

        pygame.display.flip()
        clock.tick(30)
