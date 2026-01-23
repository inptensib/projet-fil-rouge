# projet-fil-rouge
Projet fil rouge - Classification hiérarchique

## 📖 Description
Ce projet implémente un système de classification hiérarchique.

## 📁 Structure du projet
```
├── notebooks/          # Expériences reproductibles (Jupyter notebooks)
├── src/                # Code de production
├── models/             # Checkpoints des modèles entraînés
├── experiments/        # Résultats des expériences (JSON/CSV)
├── docs/               # Diagrammes d'architecture, documentation API
└── README.md           # Documentation principale
```

## 🚀 Quick Start

### Prérequis
- Python 3.8+
- pip

### Installation
```bash
# Cloner le repository
git clone https://github.com/<username>/projet-fil-rouge.git
cd projet-fil-rouge

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Reproduction des expériences
```bash
# Lancer les notebooks
jupyter notebook notebooks/
```

## 📊 Expériences
Les résultats des expériences sont stockés dans le dossier `experiments/` au format JSON et CSV.

## 📚 Documentation
La documentation complète est disponible dans le dossier `docs/`.

## 📝 License
MIT License

