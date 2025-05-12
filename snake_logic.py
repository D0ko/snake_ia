import random
import numpy as np

class SnakeGame:
    # Actions possibles
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    # Récompenses
    REWARD_FOOD = 100
    REWARD_DEATH = -100
    REWARD_STEP = -0.1  # Petite pénalité pour chaque pas pour encourager l'efficacité
    
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.reset()
    
    def reset(self):
        # Position initiale du serpent (au centre)
        center_x = (self.width // self.cell_size) // 2 * self.cell_size
        center_y = (self.height // self.cell_size) // 2 * self.cell_size
        self.snake = [[center_x, center_y]]
        self.direction = [self.cell_size, 0]  # Direction initiale: droite
        self.generate_food()
        self.game_over = False
        self.score = 0
        self.steps_without_food = 0
        self.max_steps_without_food = 100  # Pour éviter les boucles infinies
        
        # Retourner l'état initial
        return self.get_state()
    
    def generate_food(self):
        # Génère de la nourriture à une position aléatoire (sur la grille)
        max_x = (self.width // self.cell_size) - 1
        max_y = (self.height // self.cell_size) - 1
        
        while True:
            food_x = random.randint(0, max_x) * self.cell_size
            food_y = random.randint(0, max_y) * self.cell_size
            food_pos = [food_x, food_y]
            
            # S'assure que la nourriture n'apparaît pas sur le serpent
            if food_pos not in self.snake:
                self.food = food_pos
                break
    
    def get_state(self):
        """
        Convertit l'état du jeu en une représentation adaptée à l'IA.
        Retourne un tableau numpy avec les informations pertinentes.
        """
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        
        # Normalisation des coordonnées pour les rendre indépendantes de la taille du jeu
        grid_width = self.width // self.cell_size
        grid_height = self.height // self.cell_size
        
        # État vectoriel simple
        state = np.zeros(12)
        
        # Direction du serpent (one-hot encoding)
        if self.direction == [0, -self.cell_size]:  # Haut
            state[0] = 1
        elif self.direction == [self.cell_size, 0]:  # Droite
            state[1] = 1
        elif self.direction == [0, self.cell_size]:  # Bas
            state[2] = 1
        elif self.direction == [-self.cell_size, 0]:  # Gauche
            state[3] = 1
        
        # Danger immédiat (1 si danger, 0 sinon)
        # Vérifier danger devant
        check_x = head_x + self.direction[0]
        check_y = head_y + self.direction[1]
        state[4] = 1 if (check_x < 0 or check_x >= self.width or 
                         check_y < 0 or check_y >= self.height or
                         [check_x, check_y] in self.snake) else 0
        
        # Vérifier danger à droite (relatif à la direction actuelle)
        if self.direction == [0, -self.cell_size]:  # Si va vers le haut, droite = droite
            check_x = head_x + self.cell_size
            check_y = head_y
        elif self.direction == [self.cell_size, 0]:  # Si va vers la droite, droite = bas
            check_x = head_x
            check_y = head_y + self.cell_size
        elif self.direction == [0, self.cell_size]:  # Si va vers le bas, droite = gauche
            check_x = head_x - self.cell_size
            check_y = head_y
        else:  # Si va vers la gauche, droite = haut
            check_x = head_x
            check_y = head_y - self.cell_size
        
        state[5] = 1 if (check_x < 0 or check_x >= self.width or 
                         check_y < 0 or check_y >= self.height or
                         [check_x, check_y] in self.snake) else 0
        
        # Vérifier danger à gauche (relatif à la direction actuelle)
        if self.direction == [0, -self.cell_size]:  # Si va vers le haut, gauche = gauche
            check_x = head_x - self.cell_size
            check_y = head_y
        elif self.direction == [self.cell_size, 0]:  # Si va vers la droite, gauche = haut
            check_x = head_x
            check_y = head_y - self.cell_size
        elif self.direction == [0, self.cell_size]:  # Si va vers le bas, gauche = droite
            check_x = head_x + self.cell_size
            check_y = head_y
        else:  # Si va vers la gauche, gauche = bas
            check_x = head_x
            check_y = head_y + self.cell_size
        
        state[6] = 1 if (check_x < 0 or check_x >= self.width or 
                         check_y < 0 or check_y >= self.height or
                         [check_x, check_y] in self.snake) else 0
        
        # Direction de la nourriture par rapport à la tête
        state[7] = 1 if food_x < head_x else 0  # Nourriture à gauche
        state[8] = 1 if food_x > head_x else 0  # Nourriture à droite
        state[9] = 1 if food_y < head_y else 0  # Nourriture en haut
        state[10] = 1 if food_y > head_y else 0  # Nourriture en bas
        
        # Longueur du serpent (normalisée)
        state[11] = len(self.snake) / (grid_width * grid_height)
        
        return state
    
    def step(self, action):
        """
        Exécute une action et retourne l'état suivant, la récompense,
        et un booléen indiquant si l'épisode est terminé.
        
        action: 0 (haut), 1 (droite), 2 (bas), 3 (gauche)
        """
        # Convertir l'action en direction
        if action == self.UP:
            new_direction = [0, -self.cell_size]
        elif action == self.RIGHT:
            new_direction = [self.cell_size, 0]
        elif action == self.DOWN:
            new_direction = [0, self.cell_size]
        elif action == self.LEFT:
            new_direction = [-self.cell_size, 0]
        else:
            raise ValueError(f"Action invalide: {action}")
        
        # Empêcher le serpent de faire demi-tour
        if (new_direction[0] != -self.direction[0] or 
            new_direction[1] != -self.direction[1]):
            self.direction = new_direction
        
        # Récompense par défaut (légèrement négative pour encourager l'efficacité)
        reward = self.REWARD_STEP
        self.steps_without_food += 1
        
        # Calcule la nouvelle position de la tête
        head_x, head_y = self.snake[0]
        new_x = head_x + self.direction[0]
        new_y = head_y + self.direction[1]
        new_head = [new_x, new_y]
        
        # Vérifie collision avec les bords ou avec soi-même
        if (new_x < 0 or new_x >= self.width or 
            new_y < 0 or new_y >= self.height or
            new_head in self.snake or
            self.steps_without_food >= self.max_steps_without_food):
            self.game_over = True
            return self.get_state(), self.REWARD_DEATH, True
        
        # Ajoute la nouvelle tête
        self.snake.insert(0, new_head)
        
        # Vérifie si le serpent a mangé
        if new_head[0] == self.food[0] and new_head[1] == self.food[1]:
            self.score += 1
            self.steps_without_food = 0
            reward = self.REWARD_FOOD
            self.generate_food()
        else:
            # Retire la queue si pas mangé
            self.snake.pop()
        
        return self.get_state(), reward, self.game_over
    
    def render(self):
        """
        Méthode optionnelle pour afficher le jeu (à implémenter si besoin de visualisation)
        Pourrait utiliser matplotlib ou autre bibliothèque de visualisation
        """
        pass