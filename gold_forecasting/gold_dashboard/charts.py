"""Pure Plotly builders from validated API rows. No data access or model metrics.

Forecast points are sparse: markers do not imply interpolated daily predictions.
Missing observations remain gaps; residuals are supplied by notebook reports.
"""

from collections.abc import Sequence

import plotly.graph_objects as go

from gold_contracts.schemas import EvaluationRow, ForecastPoint, HistoryRow


def evaluation_chart(rows: Sequence[EvaluationRow], target_unit: str) -> go.Figure:
    """Plot actual and predicted prices by target date for one selected horizon."""
    figure = go.Figure()
    dates = [row.target_date for row in rows]
    figure.add_trace(
        go.Scatter(
            x=dates, y=[row.actual for row in rows], name="Observed", mode="lines"
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[row.prediction for row in rows],
            name="Prediction",
            mode="lines+markers",
        )
    )
    if any(row.baseline is not None for row in rows):
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=[row.baseline for row in rows],
                name="Persistence",
                mode="lines",
            )
        )
    figure.update_layout(
        title="Test evaluation: actual vs predicted",
        xaxis_title="Target date",
        yaxis_title=target_unit,
    )
    return figure


def residual_chart(rows: Sequence[EvaluationRow], target_unit: str) -> go.Figure:
    """Plot saved prediction-minus-actual residuals; no evaluation is performed here."""
    figure = go.Figure(
        go.Scatter(
            x=[row.target_date for row in rows],
            y=[row.residual for row in rows],
            name="Prediction − actual",
            mode="markers",
        )
    )
    figure.add_hline(y=0, line_dash="dash")
    figure.update_layout(
        title="Test residuals", xaxis_title="Target date", yaxis_title=target_unit
    )
    return figure


def future_chart(
    history: Sequence[HistoryRow], points: Sequence[ForecastPoint], target_unit: str
) -> go.Figure:
    """Plot observed history and configured-horizon forecast markers with optional bands."""
    figure = go.Figure(
        go.Scatter(
            x=[row.date for row in history],
            y=[row.price if row.observed else None for row in history],
            name="Observed",
            connectgaps=False,
            mode="lines",
        )
    )
    ordered = sorted(points, key=lambda row: row.target_date)
    figure.add_trace(
        go.Scatter(
            x=[row.target_date for row in ordered],
            y=[row.prediction for row in ordered],
            name="Configured-horizon forecasts",
            mode="markers",
            marker={"size": 10},
        )
    )
    interval = [
        row for row in ordered if row.lower is not None and row.upper is not None
    ]
    if interval:
        # Error bars show uncertainty only at supplied horizon points, not between them.
        figure.add_trace(
            go.Scatter(
                x=[row.target_date for row in interval],
                y=[
                    (row.lower + row.upper) / 2
                    for row in interval
                    if row.lower is not None and row.upper is not None
                ],
                mode="markers",
                name="Published intervals",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [
                        (row.upper - row.lower) / 2
                        for row in interval
                        if row.upper is not None and row.lower is not None
                    ],
                    "arrayminus": [
                        (row.upper - row.lower) / 2
                        for row in interval
                        if row.upper is not None and row.lower is not None
                    ],
                },
            )
        )
    figure.update_layout(
        title="History and future forecasts",
        xaxis_title="Date",
        yaxis_title=target_unit,
    )
    return figure
