import pygame
import sys
from snake_logic import SnakeGame

class SnakeGameUI:
    # Couleurs
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    
    def __init__(self, width=600, height=400, cell_size=20):
        # Initialisation de Pygame
        pygame.init()
        self.width = width
        self.height = height
        self.cell_size = cell_size
        
        # Création de la fenêtre
        self.window = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake Minimal')
        
        # Initialisation du jeu
        self.game = SnakeGame(width, height, cell_size)
        
        # Autres paramètres
        self.clock = pygame.time.Clock()
        self.fps = 10
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.game.change_direction([0, -self.cell_size])
                elif event.key == pygame.K_DOWN:
                    self.game.change_direction([0, self.cell_size])
                elif event.key == pygame.K_LEFT:
                    self.game.change_direction([-self.cell_size, 0])
                elif event.key == pygame.K_RIGHT:
                    self.game.change_direction([self.cell_size, 0])
                elif event.key == pygame.K_r and self.game.is_game_over():
                    self.game.reset()
        return True
    
    def draw(self):
        # Efface l'écran
        self.window.fill(self.BLACK)
        
        # Dessine la nourriture
        food = self.game.get_food()
        pygame.draw.rect(self.window, self.RED, 
                         (food[0], food[1], self.cell_size, self.cell_size))
        
        # Dessine le serpent
        snake = self.game.get_snake()
        for segment in snake:
            pygame.draw.rect(self.window, self.GREEN, 
                            (segment[0], segment[1], self.cell_size, self.cell_size))
        
        # Affiche le score
        score_text = f"Score: {self.game.get_score()}"
        font = pygame.font.Font(None, 36)
        text = font.render(score_text, True, self.WHITE)
        self.window.blit(text, (10, 10))
        
        # Affiche un message de fin de jeu si nécessaire
        if self.game.is_game_over():
            game_over_text = "Game Over! Appuyez sur R pour rejouer"
            go_text = font.render(game_over_text, True, self.WHITE)
            text_rect = go_text.get_rect(center=(self.width/2, self.height/2))
            self.window.blit(go_text, text_rect)
        
        # Met à jour l'affichage
        pygame.display.update()
    
    def run(self):
        running = True
        while running:
            # Gestion des événements
            running = self.handle_events()
            
            # Mise à jour de la logique
            if not self.game.is_game_over():
                self.game.update()
            
            # Dessin
            self.draw()
            
            # Contrôle de la vitesse
            self.clock.tick(self.fps)
        
        pygame.quit()
        sys.exit()

