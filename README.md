# ElectricityLoadDiagrams20112014 — MLOps Lab

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

Dans la VM ENI, le fichier brut est placé dans :

```text
data/raw/raw.txt
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
- logging des coefficients ;
- évaluation finale sur le test ;
- analyse des erreurs par client ;
- modèle bonus optionnel.

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

### `medium_term`

```text
lag_7d, lag_30d
```

Hypothèse : la mémoire hebdomadaire et mensuelle capte une partie importante de la dynamique.

### `trend_and_seasonality`

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

L'expérience utilisée par défaut est :

```text
electricity-load-tp02
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

L'URI MLflow est récupérée depuis la configuration de la VM :

```python
mlflow.get_tracking_uri()
```

## DVC

Les datasets volumineux ne doivent pas être commit directement dans Git.

Pour matérialiser un split, passer dans le notebook :

```python
WRITE_DVC_SPLITS = True
```

Puis versionner par exemple :

```bash
dvc add data/splits/split_v1_2011-2012_2013_2014
git add data/splits/split_v1_2011-2012_2013_2014.dvc data/splits/.gitignore
git commit -m "data: version temporal split v1"
```

Même principe pour le split v2.

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
Le notebook calcule donc aussi la MAE par `individual`.

### 4. Validation temporelle

Ne pas utiliser un split aléatoire classique sur cette time series :
le futur ne doit jamais servir à prédire le passé.

### 5. Test final

Le jeu de test ne sert pas à choisir les features ou `alpha`.
Il est utilisé après le choix du modèle pour obtenir une estimation finale de généralisation.

## Pistes pour aller plus loin

- `TimeSeriesSplit` / backtesting temporel ;
- comparaison avec `HistGradientBoostingRegressor` ;
- comparaison Ridge avec/sans standardisation ;
- résidus par heure, jour, mois et client ;
- suivi du drift avec Evidently ;
- logging du hash Git et de la version DVC dans MLflow ;
- enregistrement du meilleur modèle dans un Model Registry ;
- pipeline automatisé de retraining et validation.
