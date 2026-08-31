# Pipeline DAG and notebook cell classification

Pipeline DAG (simple linear):

- load_data (LD2011_2014.txt) --> build_features (lags, rolling means)
- build_features --> split_chronological (train/test by timestamp)
- split_chronological --> train_ridge (train model)
- train_ridge --> evaluate (RMSE / MAE)
- train_ridge --> save_model (model.pkl)

Notebook cell classification (high level):

- Data loading: PRODUCTION (to industrialize) -> `load_data`
- Data cleaning / clock / DST analysis: EXPLORATION (jetable)
- Visualisations (plotly): EXPLORATION
- Feature engineering (lags, rolling means): PRODUCTION -> `build_features`
- Correlation/autocorr analysis: EXPLORATION
- Model experiments (LinearRegression / Ridge / Tree): PRODUCTION (select one) / EXPLORATION (hyperparam search)
- Validation split logic: PRODUCTION -> `split_chronological`
- Evaluation (RMSE/MAE, residuals): PRODUCTION -> `evaluate`
- Model save: PRODUCTION -> `save_model`

Notes:
- For a reproducible chronological split we split by unique timestamps so that all client rows
  at a timestamp are on the same side of the cut.
- Chosen reference model: `Ridge(alpha=1.0)` with features `lag_1d, lag_7d, lag_30d, rolling_mean_30d`.
