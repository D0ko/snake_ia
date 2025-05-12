import numpy as np
import pygame
import matplotlib.pyplot as plt
import time
import os
from snake_logic import SnakeGame
from snake_dqn_agent import DQNAgent

# Réduire les messages de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_dqn_agent_with_visualization(episodes=1000, batch_size=64, update_target_every=5, 
                                       render_every=1, save_every=100, fps=30):
    """
    Entraîne l'agent DQN avec visualisation en temps réel
    
    Args:
        episodes: Nombre total d'épisodes d'entraînement
        batch_size: Taille du batch pour l'apprentissage
        update_target_every: Fréquence de mise à jour du modèle cible
        render_every: Afficher le rendu visuel tous les N épisodes
        save_every: Sauvegarder le modèle tous les N épisodes
        fps: Images par seconde pour le rendu
    """
    # Initialisation du jeu et de l'agent
    env = SnakeGame(width=400, height=400, cell_size=20)
    state_size = 12
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    
    # Essayer de charger un modèle existant
    try:
        agent.load("snake_model.h5")
        print("Modèle existant chargé.")
    except:
        print("Aucun modèle existant trouvé. Démarrage avec un nouveau modèle.")
    
    # Initialisation de pygame pour la visualisation
    pygame.init()
    screen = pygame.display.set_mode((800, 600))  # Taille augmentée pour inclure les statistiques
    pygame.display.set_caption("Entraînement Snake AI")
    clock = pygame.time.Clock()
    
    # Couleurs
    colors = {
        'bg': (10, 10, 40),
        'snake': (0, 255, 0),
        'food': (255, 0, 0),
        'text': (255, 255, 255),
        'grid': (30, 30, 70),
        'chart_bg': (20, 20, 50),
        'chart_line': (0, 200, 255),
        'danger': (255, 100, 100)
    }
    
    # Police pour le texte
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 32)
    
    # Variables pour le suivi des performances
    scores = []
    episodes_x = []
    avg_scores = []
    rewards_history = []
    max_score = 0
    episode_rewards = []
    
    # Zone de jeu dans la fenêtre
    game_rect = pygame.Rect(20, 20, env.width, env.height)
    
    # Boucle d'entraînement
    running = True
    episode = 0
    
    while running and episode < episodes:
        episode += 1
        
        # Réinitialisation de l'environnement
        state = env.reset()
        
        # Variables pour l'épisode actuel
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Pour le rendu visuel des dangers
        dangers = [0, 0, 0]  # Danger devant, à droite, à gauche
        
        # Boucle d'un épisode
        while not done and running:
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Pause
                        paused = True
                        while paused:
                            for evt in pygame.event.get():
                                if evt.type == pygame.QUIT:
                                    running = False
                                    paused = False
                                elif evt.type == pygame.KEYDOWN:
                                    if evt.key == pygame.K_SPACE:
                                        paused = False
                            pygame.display.update()
                            clock.tick(10)
            
            # L'agent choisit une action
            action = agent.act(state)
            
            # Extrait les informations de danger pour la visualisation
            dangers = [state[4], state[5], state[6]]
            
            # Exécution de l'action
            next_state, reward, done = env.step(action)
            
            # Stockage de l'expérience
            agent.remember(state, action, reward, next_state, done)
            
            # Passage à l'état suivant
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            # Entraînement de l'agent
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            
            # Rendu visuel (si actif pour cet épisode)
            if episode % render_every == 0:
                # Effacer l'écran
                screen.fill(colors['bg'])
                
                # Dessiner la grille de jeu
                pygame.draw.rect(screen, colors['grid'], game_rect, 1)
                
                # Dessiner la nourriture
                food_pos = (
                    game_rect.x + env.food[0],
                    game_rect.y + env.food[1],
                    env.cell_size,
                    env.cell_size
                )
                pygame.draw.rect(screen, colors['food'], food_pos)
                
                # Dessiner le serpent
                for segment in env.snake:
                    segment_pos = (
                        game_rect.x + segment[0],
                        game_rect.y + segment[1],
                        env.cell_size,
                        env.cell_size
                    )
                    pygame.draw.rect(screen, colors['snake'], segment_pos)
                
                # Afficher les infos de l'épisode
                episode_text = f"Episode: {episode}/{episodes}"
                text_surface = font_large.render(episode_text, True, colors['text'])
                screen.blit(text_surface, (500, 30))
                
                score_text = f"Score actuel: {env.score}"
                text_surface = font.render(score_text, True, colors['text'])
                screen.blit(text_surface, (500, 70))
                
                steps_text = f"Étapes: {episode_steps}"
                text_surface = font.render(steps_text, True, colors['text'])
                screen.blit(text_surface, (500, 100))
                
                reward_text = f"Récompense cumulée: {episode_reward:.1f}"
                text_surface = font.render(reward_text, True, colors['text'])
                screen.blit(text_surface, (500, 130))
                
                epsilon_text = f"Epsilon: {agent.epsilon:.4f}"
                text_surface = font.render(epsilon_text, True, colors['text'])
                screen.blit(text_surface, (500, 160))
                
                # Afficher le meilleur score
                if scores:
                    max_score_text = f"Meilleur score: {max(scores)}"
                    text_surface = font.render(max_score_text, True, colors['text'])
                    screen.blit(text_surface, (500, 190))
                
                # Afficher l'action actuelle
                actions = ["↑ HAUT", "→ DROITE", "↓ BAS", "← GAUCHE"]
                action_text = f"Action: {actions[action]}"
                text_surface = font.render(action_text, True, colors['text'])
                screen.blit(text_surface, (500, 220))
                
                # Afficher les dangers détectés
                danger_titles = ["Devant", "Droite", "Gauche"]
                for i, (danger, title) in enumerate(zip(dangers, danger_titles)):
                    color = colors['danger'] if danger > 0.5 else colors['text']
                    danger_text = f"Danger {title}: {'OUI' if danger > 0.5 else 'NON'}"
                    text_surface = font.render(danger_text, True, color)
                    screen.blit(text_surface, (500, 250 + i*30))
                
                # Afficher un mini graphique d'historique des scores
                if len(scores) > 1:
                    chart_rect = pygame.Rect(500, 350, 280, 200)
                    pygame.draw.rect(screen, colors['chart_bg'], chart_rect)
                    pygame.draw.rect(screen, colors['text'], chart_rect, 1)
                    
                    # Titre du graphique
                    chart_title = "Historique des scores"
                    title_surface = font.render(chart_title, True, colors['text'])
                    screen.blit(title_surface, (chart_rect.centerx - title_surface.get_width()//2, chart_rect.y + 5))
                    
                    # Dessine la courbe des scores
                    last_scores = scores[-100:] if len(scores) > 100 else scores
                    max_displayable = 100
                    display_scores = last_scores[-max_displayable:]
                    
                    if len(display_scores) > 1:
                        max_score_chart = max(display_scores) if max(display_scores) > 0 else 1
                        
                        for i in range(len(display_scores) - 1):
                            # Normaliser les valeurs pour la hauteur du graphique
                            x1 = chart_rect.x + i * (chart_rect.width / (len(display_scores) - 1))
                            y1 = chart_rect.bottom - (display_scores[i] / max_score_chart) * (chart_rect.height - 30)
                            x2 = chart_rect.x + (i + 1) * (chart_rect.width / (len(display_scores) - 1))
                            y2 = chart_rect.bottom - (display_scores[i + 1] / max_score_chart) * (chart_rect.height - 30)
                            
                            pygame.draw.line(screen, colors['chart_line'], (x1, y1), (x2, y2), 2)
                
                # Mise à jour de l'affichage
                pygame.display.update()
                clock.tick(fps if episode_steps > 1 else 1)  # Au premier pas, attend une seconde
        
        # Fin de l'épisode
        score = env.score
        scores.append(score)
        episodes_x.append(episode)
        episode_rewards.append(episode_reward)
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        avg_scores.append(avg_score)
        
        # Mise à jour du modèle cible périodiquement
        if episode % update_target_every == 0:
            agent.update_target_model()
        
        # Sauvegarde du meilleur modèle
        if score > max_score:
            max_score = score
            agent.save("snake_model_best.h5")
            print(f"Nouveau meilleur score: {score} à l'épisode {episode}! Modèle sauvegardé.")
        
        # Affichage des progrès dans la console
        if episode % 10 == 0 or score > 3:
            print(f"Episode: {episode}/{episodes}, Score: {score}, Récompense: {episode_reward:.1f}, Avg Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Sauvegarde périodique du modèle
        if episode % save_every == 0:
            agent.save(f"snake_model_checkpoint_{episode}.h5")
            
            # Sauvegarde des graphiques
            plt.figure(figsize=(15, 10))
            
            # Score par épisode
            plt.subplot(2, 2, 1)
            plt.plot(episodes_x, scores)
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.title('Score par épisode')
            
            # Score moyen
            plt.subplot(2, 2, 2)
            plt.plot(episodes_x, avg_scores)
            plt.xlabel('Episode')
            plt.ylabel('Score moyen (100 épisodes)')
            plt.title('Score moyen')
            
            # Récompense par épisode
            plt.subplot(2, 2, 3)
            plt.plot(episodes_x, episode_rewards)
            plt.xlabel('Episode')
            plt.ylabel('Récompense totale')
            plt.title('Récompense par épisode')
            
            # Distribution des scores
            plt.subplot(2, 2, 4)
            plt.hist(scores, bins=10)
            plt.xlabel('Score')
            plt.ylabel('Fréquence')
            plt.title('Distribution des scores')
            
            plt.tight_layout()
            plt.savefig(f"snake_training_progress_{episode}.png")
            plt.close()
    
    # Fin de l'entraînement
    pygame.quit()
    
    # Sauvegarde du modèle final
    agent.save("snake_model.h5")
    
    # Sauvegarde du graphique final
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(episodes_x, scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Evolution du score')
    
    plt.subplot(1, 2, 2)
    plt.plot(episodes_x, avg_scores)
    plt.xlabel('Episode')
    plt.ylabel('Score moyen')
    plt.title('Evolution du score moyen')
    
    plt.tight_layout()
    plt.savefig("snake_final_training_curve.png")
    plt.close()
    
    return agent, scores

if __name__ == "__main__":
    # Adaptez ces paramètres selon vos besoins
    agent, scores = train_dqn_agent_with_visualization(
        episodes=1000,
        batch_size=64,
        update_target_every=5,
        render_every=1,  # Afficher chaque épisode
        save_every=100,
        fps=30  # 30 images par seconde (diminuez pour ralentir)
    )