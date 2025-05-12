# 🐍 Snake AI - Apprentissage par Renforcement

Un jeu de Snake classique avec une IA entraînée par apprentissage par renforcement profond (Deep Q-Learning).

## 📋 Description

Ce projet implémente le jeu classique Snake avec deux modes :
- **Mode manuel** : jouable avec les touches directionnelles
- **Mode IA** : le serpent est contrôlé par un modèle de Deep Q-Learning

L'agent d'IA apprend à jouer au Snake en maximisant les récompenses (manger de la nourriture) tout en évitant les obstacles (murs et son propre corps).

## 🚀 Fonctionnalités

- Interface graphique Pygame
- Système de score
- Agent DQN (Deep Q-Network) pour l'apprentissage par renforcement
- Visualisation de l'entraînement en temps réel avec graphiques
- Sauvegarde des meilleurs modèles

## 🔧 Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/snake-ai.git
cd snake-ai

# Installer les dépendances
pip install -r requirements.txt
```

## 📦 Dépendances

- Python 3.8+
- TensorFlow 2.x
- Pygame
- NumPy
- Matplotlib

## 💻 Utilisation

### Jouer manuellement

```bash
python main.py
```

### Entraîner l'IA

```bash
python snake_train.py
```

### Tester l'IA entraînée

```bash
python snake_test.py
```

## 🧠 Architecture

- `main.py` : Point d'entrée principal
- `snake_logic.py` : Moteur du jeu Snake
- `snake_graphic.py` : Interface graphique Pygame
- `snake_dqn_agent.py` : Agent DQN pour l'apprentissage par renforcement
- `snake_train.py` : Script d'entraînement avec visualisation
- `snake_test.py` : Script de test pour l'agent entraîné

## 🔄 Processus d'apprentissage

L'agent apprend grâce à un réseau de neurones qui prédit les valeurs Q pour chaque action possible. Le processus comprend :

1. Observation de l'état actuel (12 caractéristiques)
2. Choix d'une action (haut, droite, bas, gauche)
3. Réception d'une récompense (+100 pour manger, -100 pour mourir)
4. Mise à jour du modèle via l'algorithme Q-learning

## 🎮 Contrôles

- **Flèches directionnelles** : Déplacer le serpent
- **R** : Recommencer après game over
- **Espace** : Pause (pendant l'entraînement)
- **Échap** : Quitter

## 📊 Performances

Après 1000 épisodes d'entraînement, l'agent atteint généralement un score moyen de 10-15, avec des pics à 25+.

## 🛠️ Améliorations possibles

- Implémenter un algorithme PPO ou A3C
- Ajouter des obstacles dynamiques
- Créer différents niveaux de difficulté
- Optimiser la représentation d'état pour de meilleures performances

## Author

Doko

## 📜 Licence

MIT