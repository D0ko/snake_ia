import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import os

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Taille de l'état (12 dans notre cas)
        self.action_size = action_size  # Nombre d'actions possibles (4: haut, droite, bas, gauche)
        self.memory = deque(maxlen=10000)  # Mémoire pour stocker les expériences
        self.gamma = 0.95  # Facteur d'actualisation
        self.epsilon = 1.0  # Taux d'exploration initial
        self.epsilon_min = 0.01  # Taux d'exploration minimum
        self.epsilon_decay = 0.995  # Décroissance du taux d'exploration
        self.learning_rate = 0.001  # Taux d'apprentissage
        self.model = self._build_model()  # Modèle de réseau de neurones
        self.target_model = self._build_model()  # Modèle cible (pour stabilité)
        self.update_target_model()  # Copie des poids initiaux
        
    def _build_model(self):
        """Construit le réseau de neurones pour approximer la fonction Q"""
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """Copie les poids du modèle principal vers le modèle cible"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Stocke une expérience dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Choisit une action à partir de l'état actuel (avec exploration possible)"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)  # Action aléatoire (exploration)
        
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])  # Action avec la plus grande valeur Q (exploitation)
    
    def replay(self, batch_size):
        """Entraîne le modèle sur un batch d'expériences"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Calcul des valeurs cibles
        target = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.gamma * np.amax(target_next[i])
        
        # Entraînement du modèle
        self.model.fit(states, target, epochs=1, verbose=0)
        
        # Diminution du taux d'exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        """Charge les poids d'un modèle sauvegardé"""
        if os.path.exists(name):
            self.model.load_weights(name)
            self.update_target_model()
    
    def save(self, name):
        """Sauvegarde les poids du modèle"""
        self.model.save_weights(name)