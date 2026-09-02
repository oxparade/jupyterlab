# Kubeflow — ajustements proposés

## État actuel

Le dépôt est déjà bien structuré pour MLflow/DVC, mais il n’est pas encore Kubeflow-native :

- les workflows s’appuient sur des chemins locaux et implicites dans [dvc.yaml](../dvc.yaml) ;
- l’orchestration actuelle passe par des scripts Python lancés localement dans [MLproject](../MLproject), [main.py](../main.py) et [retrain_validate.py](../retrain_validate.py) ;
- le serving est encore pensé comme un processus local avec host/port fixes dans [main.py](../main.py) et [MLproject](../MLproject) ;
- le code dépend d’un dataset détecté via le filesystem dans [config.py](../config.py).

## Ajustements recommandés pour Kubeflow

### 1. Ajouter un pipeline KFP dédié

Créer un fichier de pipeline Kubeflow v2, par exemple `kubeflow_pipeline.py`, qui enchaîne :

1. préparation des données ;
2. entraînement ;
3. évaluation ;
4. enregistrement / promotion si la qualité est suffisante.

Cela évite de faire porter à MLflow Projects ou à DVC le rôle d’orchestrateur principal.

### 2. Passer à des artefacts explicites

Chaque étape devrait lire/écrire via des paramètres et des chemins d’artefacts fournis par le pipeline, au lieu de compter sur :

- `data/processed/...` ;
- `shared/dataset/...` ;
- le répertoire courant.

C’est le point le plus important pour exécuter correctement sur la VM Kubeflow avec PVC ou stockage objet.

### 3. Séparer entraînement et serving

Le serving local ne doit pas faire partie du pipeline Kubeflow.

À garder dans Kubeflow :
- entraînement ;
- validation ;
- registry ;
- promotion.

À déployer à part :
- le service FastAPI / MLflow model server.

### 4. Éviter les sous-processus locaux dans les composants

`retrain_validate.py` et `main.py` lancent encore des commandes système et des appels `mlflow.run(...)` en mode local. Pour Kubeflow, il vaut mieux :

- transformer les étapes en fonctions pures ou en composants conteneurisés ;
- injecter les paramètres par arguments de composant ;
- écrire les résultats dans des artefacts KFP.

### 5. Rendre les chemins configurables

Ajouter des paramètres explicites pour :

- le chemin du dataset brut ;
- le chemin de sortie des splits ;
- l’URI MLflow tracking ;
- le nom du modèle dans le registry ;
- le bucket / PVC d’artefacts si nécessaire.

### 6. Encoder la gouvernance dans le registry

Pour la partie validation / promotion, il faut expliciter dans MLflow :

- la description métier du modèle ;
- ses limites connues ;
- la stratégie de features choisie ;
- le statut de validation (`passed` / `failed`) ;
- le motif de la décision ;
- les métriques de comparaison contre le champion ;
- les alias cibles (`challenger`, puis `champion` si la gate passe).

L’idée est de pouvoir récupérer tous les modèles validés, les comparer, puis décider
de manière tracée lesquels deviennent `champion`.

### 6. Standardiser l’image de conteneur

Le pipeline Kubeflow devrait utiliser une image unique contenant au minimum :

- `mlflow` ;
- `kfp` ;
- `pandas` ;
- `scikit-learn` ;
- `pyarrow` ;
- `dvc` si vous gardez DVC dans la chaîne.

## Priorité pratique

Je ferais dans cet ordre :

1. créer le pipeline Kubeflow minimal ;
2. rendre `prep.py`, `train_mlflow.py` et `predict_mlflow.py` totalement paramétrables ;
3. externaliser les chemins de données ;
4. séparer le serving du pipeline ;
5. seulement ensuite brancher `register.py` et `promote.py` dans une étape de décision.

## Fichiers les plus concernés

- [dvc.yaml](../dvc.yaml)
- [MLproject](../MLproject)
- [config.py](../config.py)
- [prep.py](../prep.py)
- [train_mlflow.py](../train_mlflow.py)
- [predict_mlflow.py](../predict_mlflow.py)
- [retrain_validate.py](../retrain_validate.py)
- [main.py](../main.py)
- [register.py](../register.py)
- [promote.py](../promote.py)
