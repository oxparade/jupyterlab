# ElectricityLoadDiagrams20112014 — MLOps Lab

<img width="1686" height="901" alt="image" src="https://github.com/user-attachments/assets/dd993bef-4f82-4d3f-8709-3ee04789358e" />
<img width="1700" height="740" alt="image" src="https://github.com/user-attachments/assets/0791c56d-8b33-4800-9d12-100ae5cb0a44" />

Projet pédagogique autour du dataset **ElectricityLoadDiagrams20112014** :
exploration de séries temporelles de consommation électrique, préparation des données,
entraînement de modèles de régression, traçabilité MLflow et versionnement DVC.

## Objectifs

Le projet permet de parcourir progressivement un pipeline ML/MLOps complet :

```text
données brutes
    ↓
analyse / contrôle qualité
    ↓
feature engineering
    ↓
split temporel
    ↓
training
    ↓
validation
    ↓
comparaison d'expériences
    ↓
test final
    ↓
déploiement / monitoring
```

## Dataset

Le dataset UCI contient des mesures de consommation électrique de clients portugais,
à une granularité de 15 minutes, sur plusieurs années.

Le dataset est centralisé dans un dossier partagé, en amont du dossier TP :

```text
shared/dataset/
    LD2011_2014.txt
    LD2011_2014_kwh.parquet
```

Initialisation recommandée (scripts formateur) :

```bash
uv run dataset.py download
uv run dataset.py convert
uv run python prep.py
```

Le script `prep.py` lit par défaut le parquet partagé en kWh et génère :

```text
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
```

Les données préparées pour le modelling sont attendues dans :

```text
data/modelling/features.parquet
data/modelling/target.parquet
```

La cible utilisée est :

```text
consumption_kwh
```

## Notebooks

### `electricity_load_data_analysis.ipynb`

Exploration interactive des données :

- structure du dataset ;
- qualité des données ;
- profils globaux et individuels ;
- saisonnalité journalière / hebdomadaire ;
- lags ;
- rolling means ;
- premières observations de drift ;
- visualisations Plotly.

### `model_training_tp02.ipynb`

Notebook d'expérimentation aligné avec le TP02 :

- `dropna()` ;
- split temporel train / validation / test ;
- plusieurs stratégies de features ;
- régression linéaire ;
- tracking MLflow ;
- comparaison RMSE / MAE ;
- comparaison de deux stratégies de split ;
- Ridge avec `alpha = [1, 1e3, 1e9]` ;
- comparaison Ridge avec et sans standardisation ;
- logging des coefficients ;
- évaluation finale sur le test ;
- analyse des erreurs par client ;
- modèle bonus optionnel.

### Service FastAPI de prédiction

Le TP03 ajoute un service REST pour exposer le modèle promu dans MLflow.

Lancement local :

```bash
uv run python -m model_serving.cli
```

Si vous lancez le modèle via MLflow Projects, utilisez bien :

```bash
mlflow run . -e serve --env-manager local -P run_id=<run_id> -P host=127.0.0.1 -P port=5001
```

Endpoints principaux :

- `GET /health` : vérifie que le service et le modèle sont disponibles ;
- `POST /predict` : retourne une prédiction pour un couple `individual` / `timestamp`.
- `POST /predict/batch` : retourne plusieurs prédictions dans une seule requête.

L’interface Swagger est disponible sur `/docs`.

## Parquets générés dans `data/modelling/`

Les notebooks de training et de test s'appuient sur les fichiers générés par le pipeline de préparation :

- `data/modelling/features.parquet`
- `data/modelling/target.parquet`

Le parquet de features contient les colonnes suivantes :

```text
lag_1d
lag_7d
lag_30d
lag_365d
rolling_mean_7d
rolling_mean_30d
```

Le notebook `model_training_tp02.ipynb` est aligné sur ces colonnes et ajuste ses stratégies de features en conséquence.

## Split temporel

Pour le labo, les commandes suivantes sont disponibles :

```bash
uv run python -m dataset_split.cli
uv run python -m model_training.cli
```

Le module `dataset_split` génère un dossier `dataset/` avec :

- `train.parquet`
- `validation.parquet`
- `test.parquet`
- `train_head.csv`
- `validation_head.csv`
- `test_head.csv`

Le fichier `dataset_split/constants.py` contient `RANDOM_SEED`, qui pilote les fichiers `*_head.csv` pour le test de reproductibilité du labo.

### Split v1

| Dataset | Période |
|---|---|
| Training | 2011–2012 |
| Validation | 2013 |
| Test | 2014 |

La sélection temporelle suit notamment le principe :

```python
df[np.isin(df.index.get_level_values("timestamp").year, [2011, 2012])]
```

### Split v2

| Dataset | Période |
|---|---|
| Training | 2013 |
| Validation | début 2014 |
| Test | fin 2014 |

Le TP ne donnant pas de date de coupure exacte, le notebook documente l'hypothèse
du **1er juillet 2014** pour séparer début et fin 2014.

## Stratégies de features

À partir des features réellement générées, les stratégies proposées sont :

### `short_term`

```text
lag_1d, lag_7d
```

Hypothèse : la consommation dépend surtout des informations les plus récentes.

### `weekly`

```text
lag_1d, lag_7d
```

Hypothèse : la consommation dépend surtout des informations les plus récentes et du cycle hebdomadaire.

### `rolling`

```text
lag_1d, lag_7d, rolling_mean_7d, rolling_mean_30d
```

Hypothèse : les moyennes glissantes apportent une vue lissée de la tendance récente.

### `full`

```text
lag_1d, lag_7d, lag_30d, lag_365d,
rolling_mean_7d, rolling_mean_30d
```

Hypothèse : combiner mémoire courte, tendance et lissage améliore la prédiction.

## Métriques

Les métriques principales du TP sont :

- **RMSE** : pénalise fortement les grosses erreurs ;
- **MAE** : erreur absolue moyenne.

Le notebook calcule également `R²` comme information complémentaire.

## MLflow

L'expérience utilisée par défaut par les scripts TP est :

```text
electricity_consumption_forecasting
```

Chaque run logge au minimum :

- type de modèle ;
- stratégie de features ;
- liste des features ;
- stratégie de split ;
- taille train / validation ;
- RMSE ;
- MAE ;
- R² ;
- hyperparamètres ;
- coefficients pour les modèles linéaires ;
- artifact du modèle.

Le training logge aussi des tags de reproductibilité quand ils sont disponibles :

- `git_commit` ;
- `git_branch` ;
- `git_dirty` ;
- `dvc_version` ;
- `dvc_lock_sha256`.

Les runs d’évaluation produisent également des diagnostics plus détaillés :

- résidus globaux ;
- résidus par heure, jour, mois et client ;
- rapport de drift Evidently si la dépendance est installée.

Le projet utilise MLflow comme source de vérité pour les modèles entraînés et promus.
Les modèles destinés au serving sont rechargés depuis le Model Registry / les artifacts MLflow
(`runs:/...` et `models:/...@alias`), plutôt que depuis des dumps locaux.

Les scripts actifs du pipeline TP n'utilisent pas de sérialisation manuelle `pickle` / `joblib`
pour livrer le modèle : on logge et sert au format MLflow Models (signature + input example).

L'URI MLflow est récupérée depuis la configuration de la VM :

```python
mlflow.get_tracking_uri()
```

Pour éviter les confusions entre plusieurs URLs nip.io, fixer explicitement l'URI avant chaque session :

```bash
export MLFLOW_TRACKING_URI="https://mlflow.10-53-101-61.nip.io"
```

## DVC

Les datasets volumineux ne doivent pas être commit directement dans Git.

Le projet expose un pipeline DVC déclaratif dans [dvc.yaml](dvc.yaml) avec trois stages :

- `prepare_modelling_data` : génère `data/modelling/features.parquet` et `data/modelling/target.parquet`.
- `materialize_temporal_splits` : génère les splits parquet dans `data/splits/`.
- `prep_mlflow_data` : exécute `prep.py` et génère `data/processed/train.parquet`,
  `data/processed/validation.parquet`, `data/processed/test.parquet`.

La stage `prep_mlflow_data` est paramétrée via [params.yaml](params.yaml) :

- `prep_mlflow_data.strategy`
- `prep_mlflow_data.split_strategy`

Exécuter le pipeline complet :

```bash
dvc repro
```

Exécuter uniquement la préparation :

```bash
dvc repro prepare_modelling_data
```

Exécuter uniquement la matérialisation des splits :

```bash
dvc repro materialize_temporal_splits
```

Exécuter uniquement la préparation MLflow :

```bash
dvc repro prep_mlflow_data
```

Changer les paramètres sans modifier le code :

```bash
dvc exp run prep_mlflow_data --set-param prep_mlflow_data.strategy=seasonality
```

Les `.parquet` volumineux restent hors Git (ignorés), tandis que `dvc.yaml`, `dvc.lock`
et les paramètres sont versionnés.

## Model Registry (TP02)

Le cycle de vie modèle est scripté avec aliases :

- `register.py` : entraîne 2 stratégies, enregistre 2 versions, pose `champion` / `challenger`.
- `promote.py` : applique une quality gate (`validated`, `passed_validation`, gain minimal),
    trace acceptation/rejet, puis démontre promotion/rollback.
- `govern.py` : compare tous les modèles validés, les classe par métriques, puis décide
    s’ils peuvent devenir `champion`.
- aliases multi-environnements : `prod-eu`, `prod-us`, `shadow`.

Le retraining est aussi tracé comme un vrai cycle automatisé :

- `train_mlflow.py` logge des runs imbriqués pour chaque candidat et compare les résultats avec
    `mlflow.search_runs` avant de sélectionner automatiquement le meilleur modèle ;
- les jeux de données d'entraînement et de validation sont versionnés via `mlflow.data` et leur
    digest est loggé pour garantir la traçabilité exacte des données utilisées ;
- `retrain_validate.py` orchestre le cycle complet déclencheur → challenger → comparaison → décision
    et conserve la décision dans MLflow ; le déclencheur est tracé explicitement comme
    `manual`, `scheduled` ou `ci` ;
- `promote.py` trace la quality gate jusqu'au run du challenger et aux tags de la version registry ;
- les versions portent des tags de gouvernance : scope, limites, digest des données, état de validation,
    motif de décision et gain vs champion ;
- les runs et versions portent maintenant aussi des tags de provenance pour filtrer facilement
    dans l’UI MLflow.

Oui, les alias sont bien en place aussi :

- `champion` / `challenger` dans [register.py](register.py) ;
- alias de déploiement `prod-eu`, `prod-us`, `shadow` dans [promote.py](promote.py).

Commandes :

```bash
mlflow run . -e register --env-manager local
mlflow run . -e promote --env-manager local -P min_gain=0.02
mlflow run . -e govern --env-manager local -P min_gain=0.02 -P dry_run=true
```

Visualiser le graphe des stages :

```bash
dvc dag
```

Pour versionner un changement de pipeline ou de données recalculées :

```bash
git add dvc.yaml dvc.lock .gitignore
git commit -m "dvc: add modelling and split stages"
```

## Précaution RAM

Le dataset de modelling contient plusieurs millions de lignes.
Une Random Forest très large (`n_estimators=200`, `n_jobs=-1`) peut consommer énormément de mémoire.

Le notebook possède donc un mode :

```python
SMOKE_TEST = True
```

Il permet de valider rapidement le pipeline sur un sous-échantillon.

Pour les expériences finales du TP :

```python
SMOKE_TEST = False
```

La VM dispose également d'un swap supplémentaire configuré pour limiter les risques d'OOM,
mais le swap ne remplace pas la RAM et peut fortement ralentir un training.

## Contrôles supplémentaires recommandés

### 1. Baseline

Toujours comparer le modèle à une référence naïve (`DummyRegressor`, ou une prédiction basée sur `lag_1d`).

### 2. Data leakage

Pour une moyenne glissante utilisée comme feature, vérifier qu'elle ne contient pas la cible courante.

Exemple sûr :

```python
series.shift(1).rolling(window).mean()
```

### 3. Performance par client

Une bonne RMSE globale peut masquer certains clients très mal prédits.
Le pipeline calcule maintenant aussi des résumés de résidus par client, heure, jour et mois.

### 4. Validation temporelle

Ne pas utiliser un split aléatoire classique sur cette time series :
le futur ne doit jamais servir à prédire le passé.

### 5. Test final

Le jeu de test ne sert pas à choisir les features ou `alpha`.
Il est utilisé après le choix du modèle pour obtenir une estimation finale de généralisation.

### 6. Drift

Le drift est désormais mesuré automatiquement via Evidently lorsque le package est disponible.

## Ce qui a été ajouté

Les pistes ci-dessous ont finalement été implémentées dans le projet :

- `TimeSeriesSplit` / backtesting temporel ;
- comparaison avec `HistGradientBoostingRegressor` ;
- comparaison Ridge avec/sans standardisation ;
- comparaison automatique des runs imbriqués avec `mlflow.search_runs` ;
- traçabilité des données d'entraînement et de validation via `mlflow.data.digest` ;
- résidus par heure, jour, mois et client ;
- suivi du drift avec Evidently ;
- logging du hash Git et de la version DVC dans MLflow ;
- enregistrement du meilleur modèle dans un Model Registry ;
- pipeline automatisé de retraining et validation.

## Kubeflow sur la VM

Le dépôt contient maintenant un squelette Kubeflow v2 dans [kubeflow_pipeline.py](kubeflow_pipeline.py).

Ce pipeline enchaîne les scripts existants sans réécrire toute la logique métier :

- préparation des splits via `prep.py` ;
- enregistrement des candidats via `register.py` ;
- décision de gouvernance via `govern.py` ;
- évaluation du champion via `predict_mlflow.py`.

Hypothèse de déploiement : sur la VM, il existe déjà un container Kubeflow / un runtime partagé.
Dans ce cas, on réutilise ce runtime existant puis on monte le dépôt et les artefacts.
Le dépôt ne conserve donc pas d’image Docker custom dédiée à Kubeflow.

Compilation du pipeline :

```bash
python kubeflow_pipeline.py --output kubeflow_pipeline.yaml
```

Soumettre un run au cluster Kubeflow (runtime déjà présent sur la VM) :

```bash
export KFP_HOST="https://<kfp-endpoint>"
export KFP_TOKEN="<token-si-necessaire>"
python submit_kubeflow_run.py --compile-if-missing --namespace kubeflow --insecure
```

Alternative via MLflow Projects :

```bash
mlflow run . -e submit_kubeflow --env-manager local -P host=https://<kfp-endpoint> -P namespace=kubeflow
```

Variables utiles lors du déploiement :

- `KUBEFLOW_COMPONENT_IMAGE` : image Kubeflow à utiliser pour les composants ;
- `KUBEFLOW_SOURCE_DIR` : chemin du dépôt monté dans le conteneur ;
- `KUBEFLOW_RAW_DATASET` : chemin du parquet source dans l’environnement Kubeflow.

Paramètre pipeline utile :

- `governance_dry_run` : `false` pour autoriser le changement d’alias `champion`, `true` pour audit sans promotion.

En pratique, le prochain ajustement utile sera de brancher ce squelette sur le stockage réellement exposé par la VM
(PVC ou bucket objet) pour éviter les chemins locaux implicites.

Concrètement :

- comparaison Ridge standardisé / non standardisé dans `train_mlflow.py` ;
- artefacts de résidus détaillés dans `predict_mlflow.py` ;
- tags Git/DVC dans les runs MLflow ;
- rapport Evidently de drift ;
- orchestrateur [retrain_validate.py](retrain_validate.py) ;
- workflow GitHub Actions [retrain-validation.yml](.github/workflows/retrain-validation.yml).
