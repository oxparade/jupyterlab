# TP03 — Déploiement et gestion des modèles ML avec MLOps

## Partie 1 — Sélection et enregistrement du modèle

### 1.1 — Que contiennent les artefacts du modèle sauvegardé par MLflow, pourquoi sont-ils utiles ?

Les artefacts MLflow contiennent le modèle sérialisé, sa signature, des exemples d’entrée, les paramètres, les métriques, et éventuellement des fichiers complémentaires comme des coefficients, des courbes ou des rapports HTML.

Ils sont utiles car ils permettent de rejouer, inspecter, comparer et déployer le modèle sans devoir relancer tout l’entraînement.

### 1.2 — Quel mécanisme vous permet de promouvoir un modèle ?

Le mécanisme est le MLflow Model Registry, en particulier l’usage des versions de modèle et des alias. Dans ce projet, l’alias `champion` sert à pointer vers la version promue.

### 1.3 — Pour plusieurs environnements/régions, quel mécanisme utiliser ?

On utilise des alias ou des tags pour distinguer les modèles selon le contexte, par exemple `champion-eu`, `champion-us`, ou des tags comme `region=eu-west`.

## Partie 2 — Création du service de prédiction

### 2.1 — Quel endpoint pour vérifier le bon fonctionnement ? Quel code HTTP ?

Le endpoint de santé est `GET /health`.
Le code HTTP attendu est `200 OK` si le service et le modèle sont disponibles.

### 2.2 — À quoi sert un endpoint de santé ?

Il sert à vérifier rapidement que l’application est démarrée, que le service répond, et éventuellement que ses dépendances critiques sont accessibles.

### 2.3 — Quelles informations le client doit-il fournir ? Pourquoi ?

Le client doit fournir l’identifiant du client et l’instant de prédiction.
Ces informations permettent de retrouver les features associées au bon individu et au bon moment.

### 2.4 — Quelles informations le client ne peut-il pas forcément fournir ? Comment les récupérer ?

Le client ne fournit pas forcément les features dérivées comme les lags ou rolling means.
Le service de prédiction peut les récupérer depuis une feature store, une base de données, un cache, ou un dictionnaire simulé dans le cadre du TP.

### 2.5 — Quelle méthode HTTP utiliser pour le endpoint de prédiction ? Pourquoi ?

La méthode `POST` est la plus adaptée, car la requête transmet un payload de données à traiter sans modifier une ressource côté serveur.

## Partie 3 — Récupération des features

### 2.6 — Rappelez les features nécessaires à la prédiction dans votre modèle.

Les features utilisées sont :

- `lag_1d`
- `lag_7d`
- `lag_30d`
- `lag_365d`
- `rolling_mean_7d`
- `rolling_mean_30d`

### 2.7 — Dans un système réel, d’où proviennent ces features ?

Elles proviennent généralement d’une feature store, d’un pipeline de calcul, d’un data warehouse, ou d’un cache de features calculées à l’avance.

### 2.8 — Pourquoi séparer la récupération de feature et le calcul de la prédiction ?

Pour découpler la logique métier, simplifier les tests, faciliter la maintenance, et permettre de réutiliser les features pour plusieurs modèles ou services.

### 2.9 — Bonus : que se passe-t-il si les features ne peuvent pas être récupérées ?

L’API doit renvoyer une erreur explicite, par exemple `404 Not Found` si l’individu ou l’instant est inconnu, ou `503 Service Unavailable` si la source de features est indisponible.

## Partie 4 — Chargement du modèle depuis le Registry

### 2.10 — Pourquoi utiliser un alias plutôt qu’un numéro de version ?

Parce qu’un alias représente un rôle métier stable, comme `champion`, alors qu’un numéro de version change à chaque nouvel enregistrement.

### 2.11 — Quels avantages apporte le Registry par rapport à un simple fichier modèle ?

Le Registry apporte :

- versioning centralisé ;
- promotion contrôlée ;
- alias ou tags ;
- traçabilité ;
- comparaison entre modèles ;
- gestion du cycle de vie.

### 2.12 — Bonus : que se passe-t-il si le fichier requirements du modèle n’est pas cohérent avec l’API ?

Le chargement ou l’exécution peut échouer à cause d’incompatibilités de dépendances. Une bonne réponse consiste à isoler l’inférence dans un environnement dédié, un conteneur, ou un service séparé par modèle.

## Partie 5 — Création du endpoint de prédiction

### 2.13 — Quelles étapes réaliser à l’arrivée d’une requête ?

1. Valider le payload.
2. Récupérer les features.
3. Charger le modèle promu si nécessaire.
4. Construire le tableau d’entrée.
5. Calculer la prédiction.
6. Retourner la réponse JSON.

### 2.14 — Quel format de réponse et quel statut HTTP ?

Le service doit renvoyer un JSON avec l’identifiant, l’horodatage, les features utilisées et la prédiction.
Le statut attendu est `200 OK`.

## Partie 6 — Bonus gestion des erreurs

### 3.1 — L’application doit-elle échouer ou intercepter l’erreur ?

Elle doit intercepter l’erreur et renvoyer une réponse claire au client, sans faire tomber tout le service.

### 3.2 — Quel code HTTP est adapté ?

`404 Not Found` est adapté si l’identifiant demandé n’existe pas.

## Partie 7 — Bonus prédictions en batch

### 4.1 — Un modèle doit-il forcément être exposé par API ?

Non. Il peut aussi être utilisé en batch, dans un job planifié, un pipeline d’orchestration, ou un traitement offline.

### 4.2 — Quelles briques restent identiques entre batch et real time ?

La récupération des features, le chargement du modèle, la préparation des données, et l’appel à `predict()` restent identiques.

### 4.3 — Comment récupérer les informations de prédiction ?

Via une requête contenant plusieurs couples `individual` / `timestamp`, puis en boucle ou par lot sur la source de features.

## Ce qui est déjà implémenté dans le projet

- promotion du modèle dans le Model Registry avec l’alias `champion` ;
- API FastAPI dédiée dans [model_serving/app.py](model_serving/app.py) ;
- endpoint de santé `GET /health` ;
- endpoint de prédiction `POST /predict` ;
- endpoint batch `POST /predict/batch` ;
- lancement local via [model_serving/cli.py](model_serving/cli.py) ;
- documentation dans [README.md](README.md).
