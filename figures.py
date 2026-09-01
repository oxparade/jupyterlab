"""Figures de diagnostic pour runs MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.base import BaseEstimator
from sklearn.metrics import root_mean_squared_error

PALETTE: dict[str, str] = {
    "primary": "#3b6ea5",
    "positive": "#1a7f47",
    "highlight": "#e08a1e",
    "neutral": "#7f8c9a",
}
LAYOUT: dict[str, object] = {"template": "plotly_white"}
WEEKDAY_NAMES: list[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

DENSITY_COLORSCALE: list[tuple[float, str]] = [
    (0.0, "#cde2fb"),
    (0.25, "#6da7ec"),
    (0.5, "#2a78d6"),
    (0.75, "#256abf"),
    (1.0, "#0d366b"),
]


def coefficient_weights(model: BaseEstimator, features: list[str]) -> go.Figure:
    weights = pd.Series(np.asarray(model.coef_).ravel(), index=features).sort_values()
    figure = go.Figure(
        go.Bar(
            x=weights.values,
            y=weights.index,
            orientation="h",
            marker_color=[
                PALETTE["positive"] if weight >= 0 else PALETTE["highlight"]
                for weight in weights.values
            ],
        )
    )
    figure.add_vline(x=0, line={"color": PALETTE["neutral"], "width": 1})
    figure.update_xaxes(title_text="coefficient")
    figure.update_layout(
        **LAYOUT, title="Ridge coefficients", height=90 + 40 * weights.shape[0]
    )
    return figure


def residual_diagnostics(
    y_true: pd.Series,
    predicted: pd.Series | np.ndarray,
    bins: int = 120,
    distribution_bins: int = 80,
) -> go.Figure:
    predicted_series = pd.Series(np.asarray(predicted), index=y_true.index)
    residuals = y_true - predicted_series
    counts, prediction_edges, residual_edges = np.histogram2d(
        predicted_series.values, residuals.values, bins=bins
    )
    distribution, residual_bins = np.histogram(residuals.values, bins=distribution_bins)
    by_weekday = residuals.groupby(residuals.index.dayofweek).apply(
        lambda day_residuals: root_mean_squared_error(
            np.zeros(day_residuals.shape[0]), day_residuals
        )
    )

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            f"Residual vs predicted ({y_true.shape[0]:,} rows)",
            "Residual distribution",
            "RMSE by day of week",
        ],
    )
    figure.add_heatmap(
        x=prediction_edges[:-1],
        y=residual_edges[:-1],
        z=np.log10(counts.T + 1),
        colorscale=DENSITY_COLORSCALE,
        colorbar={"title": "log10 rows", "thickness": 12, "x": 0.28},
        row=1,
        col=1,
    )
    figure.add_bar(
        x=residual_bins[:-1],
        y=distribution,
        marker_color=PALETTE["primary"],
        row=1,
        col=2,
    )
    figure.add_bar(
        x=WEEKDAY_NAMES,
        y=by_weekday.values,
        marker_color=PALETTE["primary"],
        row=1,
        col=3,
    )
    figure.add_hline(y=0, line={"color": "black", "width": 1}, row=1, col=1)
    figure.update_xaxes(title_text="predicted (kWh)", row=1, col=1)
    figure.update_yaxes(title_text="residual (kWh)", row=1, col=1)
    figure.update_xaxes(title_text="residual (kWh)", row=1, col=2)
    figure.update_yaxes(title_text="RMSE (kWh)", row=1, col=3)
    figure.update_layout(**LAYOUT, height=400, showlegend=False)
    return figure


def residual_breakdown(
    y_true: pd.Series,
    predicted: pd.Series | np.ndarray,
    *,
    client: pd.Series | None = None,
    top_clients: int = 25,
) -> dict[str, go.Figure]:
    predicted_series = pd.Series(np.asarray(predicted), index=y_true.index)
    residuals = y_true - predicted_series
    abs_error = residuals.abs()

    frame = pd.DataFrame({"abs_error": abs_error})
    if isinstance(frame.index, pd.MultiIndex) and "timestamp" in frame.index.names:
        timestamps = pd.to_datetime(frame.index.get_level_values("timestamp"))
    else:
        timestamps = pd.to_datetime(frame.index)
    frame["timestamp"] = timestamps

    by_hour = frame.groupby(frame["timestamp"].dt.hour, observed=True)["abs_error"].mean().reset_index(name="mae")
    by_hour.columns = ["hour", "mae"]
    hour_fig = px.bar(
        by_hour,
        x="hour",
        y="mae",
        title="Residual MAE by hour",
        labels={"hour": "Hour", "mae": "MAE (kWh)"},
    )
    hour_fig.update_layout(**LAYOUT)

    by_day = frame.groupby(frame["timestamp"].dt.dayofweek, observed=True)["abs_error"].mean().reindex(range(7)).reset_index(name="mae")
    by_day.columns = ["weekday", "mae"]
    by_day["weekday_name"] = [WEEKDAY_NAMES[i] for i in by_day["weekday"].fillna(0).astype(int)]
    day_fig = px.bar(
        by_day,
        x="weekday_name",
        y="mae",
        title="Residual MAE by weekday",
        labels={"weekday_name": "Weekday", "mae": "MAE (kWh)"},
    )
    day_fig.update_layout(**LAYOUT)

    by_month = frame.groupby(frame["timestamp"].dt.month, observed=True)["abs_error"].mean().reset_index(name="mae")
    by_month.columns = ["month", "mae"]
    month_fig = px.bar(
        by_month,
        x="month",
        y="mae",
        title="Residual MAE by month",
        labels={"month": "Month", "mae": "MAE (kWh)"},
    )
    month_fig.update_layout(**LAYOUT)

    result: dict[str, go.Figure] = {
        "residuals_by_hour": hour_fig,
        "residuals_by_weekday": day_fig,
        "residuals_by_month": month_fig,
    }

    if client is not None:
        client_values = client.astype(str)
        client_mae = (
            pd.DataFrame({"client": client_values, "abs_error": abs_error.to_numpy()})
            .groupby("client", observed=True)["abs_error"]
            .mean()
            .sort_values(ascending=False)
            .head(top_clients)
            .reset_index(name="mae")
        )
        client_fig = px.bar(
            client_mae,
            x="mae",
            y="client",
            orientation="h",
            title=f"Residual MAE by client (top {top_clients})",
            labels={"client": "Client", "mae": "MAE (kWh)"},
        )
        client_fig.update_layout(**LAYOUT)
        result["residuals_by_client"] = client_fig

    return result
