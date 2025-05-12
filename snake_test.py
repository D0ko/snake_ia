import pygame
import numpy as np
import time
from snake_logic import SnakeGame
from snake_dqn_agent import DQNAgent

def test_agent(model_path="snake_model.h5", episodes=10, render=True, delay=0.1):
    # Initialisation
    env = SnakeGame(width=400, height=400, cell_size=20)
    state_size = 12
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0  # Pas d'exploration en test
    
    # Pour le rendu Pygame
    if render:
        pygame.init()
        screen = pygame.display.set_mode((env.width, env.height))
        pygame.display.set_caption('Snake AI')
        clock = pygame.time.Clock()
        colors = {
            'bg': (0, 0, 0),
            'snake': (0, 255, 0),
            'food': (255, 0, 0)
        }
    
    total_scores = []
    
    for e in range(episodes):
        state = env.reset()
        score = 0
        done = False
        steps = 0
        
        while not done:
            # Gestion des événements
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
            
            # L'agent choisit une action
            action = agent.act(state, training=False)
            
            # Exécution de l'action
            next_state, reward, done = env.step(action)
            state = next_state
            steps += 1
            
            # Rendu graphique
            if render:
                screen.fill(colors['bg'])
                
                # Dessin de la nourriture
                food_x, food_y = env.food
                pygame.draw.rect(screen, colors['food'], 
                                (food_x, food_y, env.cell_size, env.cell_size))
                
                # Dessin du serpent
                for segment in env.snake:
                    pygame.draw.rect(screen, colors['snake'], 
                                    (segment[0], segment[1], env.cell_size, env.cell_size))
                
                # Affichage du score
                font = pygame.font.Font(None, 36)
                score_text = font.render(f"Score: {env.score}", True, (255, 255, 255))
                screen.blit(score_text, (10, 10))
                
                pygame.display.update()
                time.sleep(delay)  # Ralentir pour mieux voir
        
        score = env.score
        total_scores.append(score)
        print(f"Episode: {e+1}/{episodes}, Score: {score}, Steps: {steps}")
    
    if render:
        pygame.quit()
    
    avg_score = np.mean(total_scores)
    print(f"Score moyen sur {episodes} épisodes: {avg_score:.2f}")
    print(f"Score maximum: {max(total_scores)}")
    
    return total_scores

if __name__ == "__main__":
    scores = test_agent(model_path="snake_model_best.h5", episodes=5, render=True, delay=0.1)