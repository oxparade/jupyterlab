# TP02 — Comparer et tracer des expérimentations Machine Learning

## Résumé court

Le TP02 est globalement réalisé dans le projet :

- versionnement des données avec DVC ;
- split temporel v1 et v2 ;
- comparaison de plusieurs stratégies de features ;
- entraînement de régressions linéaires ;
- comparaison dans MLflow ;
- test de Ridge avec plusieurs valeurs de `alpha` ;
- logging des coefficients ;
- synthèse finale dans le notebook.

## Partie 1 — Comparaison de stratégies de features

### Stratégies testées

Les stratégies visibles dans le notebook sont notamment :

- `short_term`
- `medium_term`
- `trend_and_seasonality`

La stratégie `trend_and_seasonality` utilise les features les plus informatives du projet :

- `lag_1d`
- `lag_7d`
- `lag_30d`
- `lag_365d`
- `rolling_mean_7d`
- `rolling_mean_30d`

### 1.1 — Quelle stratégie obtient les meilleures performances ?

Sur le notebook, la meilleure stratégie pour les deux splits est `trend_and_seasonality`.

- Split v1 : RMSE validation = 7.4411, MAE = 4.0653
- Split v2 : RMSE validation = 6.8276, MAE = 3.8078

### 1.2 — Ajouter davantage de features améliore-t-il systématiquement le modèle ?

Non. Ajouter des features peut aider, mais ce n’est pas automatique.
Certaines features peuvent être redondantes, peu informatives, ou introduire du bruit.

### 1.3 — Quelles hypothèses semblent validées ?

L’hypothèse la plus crédible est que la consommation dépend à la fois :

- de la mémoire courte ;
- du cycle hebdomadaire ;
- de la tendance plus longue ;
- du lissage via les moyennes glissantes.

## Partie 2 — Comparaison des stratégies d’entraînement

### 2.1 — Quel split obtient les meilleures performances ?

Le split v2 obtient de meilleures performances que le split v1 dans le notebook.

### 2.2 — Est-il préférable d’utiliser davantage d’historique ou des données plus récentes ?

Pour ce cas, les données plus récentes semblent plus utiles.
Le split v2, basé sur 2013 puis début/fin 2014, donne de meilleurs résultats que le split v1.

### 2.3 — Avantages et inconvénients de chaque approche

**Split v1**

- avantage : plus d’historique pour l’entraînement ;
- inconvénient : données potentiellement moins proches du contexte test.

**Split v2**

- avantage : entraînement plus proche des données réelles à prédire ;
- inconvénient : moins d’historique disponible.

### Bonus coefficients

Les coefficients de la régression sont bien loggés dans MLflow.
Ils permettent d’identifier les features les plus influentes et celles qui contribuent peu au modèle.

## Partie 3 — Influence des hyperparamètres avec Ridge

### 3.1 — Quelle valeur de `alpha` obtient les meilleures performances ?

La meilleure valeur observée est `alpha = 1.0`.

Résultat noté dans le notebook :

- RMSE test = 7.0268
- MAE test = 3.8824

### 3.2 — Que se passe-t-il sur les coefficients lorsque `alpha` augmente ?

Quand `alpha` augmente, les coefficients sont davantage contraints vers zéro.
Le modèle devient plus régularisé et moins sensible aux variations des données.

### 3.3 — Pourquoi est-il indispensable de logger les hyperparamètres ?

Parce que cela permet :

- de reproduire exactement un run ;
- de comparer les essais entre eux ;
- d’expliquer les résultats ;
- de sélectionner le meilleur modèle de façon traçable.

### 3.4 — Rôle du bruit dans les données

Le bruit correspond aux variations non pertinentes ou aléatoires dans les données.
Il n’est pas souhaitable que le modèle le mémorise trop fortement.
Exemple : une consommation ponctuellement très élevée à cause d’un événement exceptionnel ne doit pas devenir une règle générale.

### 3.5 — Effet de l’ordre de grandeur des coefficients

Des coefficients trop grands peuvent rendre le modèle sensible au bruit et à l’overfitting.
Avec du bruit, Ridge aide à limiter cette sensibilité.

### 3.6 — Rôle de `alpha`

`alpha` contrôle la force de la régularisation.
Plus `alpha` est élevé, plus le modèle est contraint, ce qui réduit la variance mais peut augmenter le biais.

### 3.7 — Utilisation du dataset de test pour le choix d’hyperparamètre

Le dataset de test ne doit pas servir à choisir `alpha`.
Il doit être conservé pour l’évaluation finale uniquement, après sélection sur validation.

## Partie 4 — Réflexion en production

### 4.1 — Sur quelles données entraîneriez-vous le modèle ?

Sur l’historique le plus pertinent disponible, typiquement un mélange des données récentes et suffisamment de passé pour capter la saisonnalité.
Le split v2 est un bon candidat si l’on veut favoriser la proximité temporelle.

### 4.2 — À quelle fréquence réentraîner ?

Par période régulière, par exemple mensuellement ou trimestriellement, selon la vitesse de dérive observée.

### 4.3 — Quels indicateurs signaleraient un réentraînement ?

- dégradation de RMSE / MAE ;
- drift des features ;
- dérive de la distribution cible ;
- baisse de performance par client ;
- rupture de saisonnalité ou changement de comportement.

## Livrable attendu — Synthèse courte

### Stratégies testées

- `short_term`
- `medium_term`
- `trend_and_seasonality`

### Résultats obtenus

- meilleure stratégie v1 : `trend_and_seasonality` ;
- meilleure stratégie v2 : `trend_and_seasonality` ;
- meilleur Ridge : `alpha = 1.0`.

### Meilleur run MLflow identifié

Le meilleur run correspond à la stratégie `trend_and_seasonality` sur le split v2.

### Impact du changement de split

Le split v2 améliore les performances, ce qui suggère que des données plus récentes sont plus représentatives du comportement à prédire.

### Recommandation pour la mise en production

Je recommanderais la stratégie `trend_and_seasonality` avec le split v2 comme base de production, puis `Ridge(alpha=1.0)` si l’on veut une version régularisée robuste.

### Raisons du choix

- meilleures métriques validation ;
- logique temporelle cohérente ;
- bon compromis entre historique et actualité ;
- traçabilité correcte avec MLflow et DVC.
