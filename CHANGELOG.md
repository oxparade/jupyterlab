# Changelog

Toutes les modifications importantes du travail MLOps sont documentées ici.

Le format suit l'esprit de **Keep a Changelog**.

## [Unreleased]

### Added

- Notebook d'exploration interactive du dataset `ElectricityLoadDiagrams20112014`.
- Analyse de la structure, des valeurs manquantes, des zéros et de la fréquence temporelle.
- Visualisations globales et individuelles avec Plotly.
- Analyse de saisonnalité journalière et hebdomadaire.
- Introduction des lags et moyennes glissantes.
- Première comparaison de distributions temporelles pour observer le drift.
- Chargement des datasets de modelling `features.parquet` et `target.parquet`.
- Fusion features / target sur l'index `(individual, timestamp)`.
- Nettoyage des lignes non exploitables avec `dropna()`.
- Split temporel explicite basé sur l'année via `np.isin(...)`.
- Séparation train / validation / test conforme à la logique des séries temporelles.
- Comparaison de plusieurs stratégies de features.
- Baseline `DummyRegressor`.
- Entraînement de régressions linéaires.
- Tracking des expériences dans MLflow.
- Logging des paramètres, métriques, coefficients et modèles.
- Comparaison RMSE / MAE / R².
- Deuxième stratégie de split orientée données plus récentes.
- Expériences Ridge avec `alpha = 1`, `1e3`, `1e9`.
- Choix des hyperparamètres sur validation avant utilisation du test final.
- Analyse de la MAE par client.
- Bonus `HistGradientBoostingRegressor` désactivé par défaut.
- Mode `SMOKE_TEST` pour limiter la consommation mémoire pendant le développement.
- Option de matérialisation des splits en Parquet pour versionnement DVC.
- Documentation `README.md`.
- Documentation `CHANGELOG.md`.

### Changed

- Remplacement du simple split train/test par une séparation temporelle train/validation/test.
- Clarification de la distinction entre validation et test final.
- Réduction de la consommation mémoire en privilégiant `float32` pour les données numériques.
- Limitation des modèles lourds durant les smoke tests.
- Le test 2014 n'est plus utilisé pour sélectionner directement les hyperparamètres.

### Fixed

- Prédiction sur un échantillon unique conservée en tableau 2D pour respecter l'API scikit-learn.
- Suppression des sorties d'erreur obsolètes dans les notebooks corrigés.
- Correction du risque de confusion entre split aléatoire et split chronologique.
- Alignement des notebooks sur les colonnes réellement présentes dans `data/modelling/features.parquet`.

### Known Issues

- La définition exacte de « début 2014 / fin 2014 » n'étant pas fournie dans l'énoncé,
  le notebook utilise le 1er juillet 2014 comme hypothèse de coupure.
- `kubectl top` n'est pas disponible sur la VM car la Metrics API Kubernetes n'est pas configurée.

## Infrastructure locale

### Changed

- Console Debian configurée en AZERTY.
- VM utilisée en réseau VMware bridgé.
- Reconfiguration post-start exécutée après changement d'adresse IP.
- Swap additionnel de 16 GiB ajouté et rendu persistant.

### Current memory configuration

- environ 26 GiB de RAM alloués à la VM ;
- environ 21 GiB de swap total.

Cette configuration réduit le risque d'un nouveau `Out Of Memory`, sans remplacer une gestion raisonnable
de la taille des trainings et du parallélisme.
