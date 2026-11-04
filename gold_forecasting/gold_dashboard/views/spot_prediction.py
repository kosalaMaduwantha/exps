"""Metadata-driven form for one-horizon frozen-model scenario inference."""

import math

import streamlit as st

from gold_contracts.schemas import ForecastResponse, ModelMetadata, SpotRequest
from gold_dashboard.api_client import ApiClient


def render(client: ApiClient, model: ModelMetadata) -> None:
    """Collect exact finite regressor values and submit only on explicit form action."""
    origins = client.origins(model.model_id, limit=2000)
    if origins.version != model.version:
        st.warning("Active model changed. Refresh the dashboard before predicting.")
        return
    if not origins.items:
        st.info("No eligible observed origins for this model.")
        return
    if origins.total > len(origins.items):
        st.caption(
            f"Showing the latest {len(origins.items)} of {origins.total} eligible origins."
        )
    key = f"{model.model_id}:{model.version}"
    with st.form(f"spot:{key}"):
        origin = st.selectbox("Observed origin date", origins.items)
        horizon = st.selectbox("Horizon (weekday-grid steps)", model.horizons)
        values = {}
        for regressor in model.external_regressors:
            value = st.number_input(
                regressor.name,
                value=None,
                help=regressor.description,
                key=f"{key}:{regressor.name}",
            )
            values[regressor.name] = value
        if not model.external_regressors:
            st.caption(
                "This model requires no external regressors; historical gold/context features are server-derived."
            )
        submitted = st.form_submit_button("Predict spot")
    if submitted:
        st.session_state.pop("spot_result", None)
        if any(value is None or not math.isfinite(value) for value in values.values()):
            st.error("Provide a finite numeric value for every required regressor.")
            return
        request = SpotRequest(
            model_id=model.model_id,
            expected_version=model.version,
            origin_date=origin,
            horizon=horizon,
            regressors={
                name: value for name, value in values.items() if value is not None
            },
        )
        result = client.spot(request)
        st.session_state["spot_result"] = result.model_dump(mode="json")
    if "spot_result" in st.session_state:
        result = ForecastResponse.model_validate(st.session_state["spot_result"])
        if result.model_id != model.model_id or result.version != model.version:
            return
        st.caption(
            "Last submitted scenario below; edits in the form are not applied until you submit again."
        )
        st.metric("Predicted gold price", f"{result.points[0].prediction:,.2f}")
        st.write(
            f"Origin {result.origin_date} → target {result.points[0].target_date}; "
            f"{result.points[0].horizon} weekday-grid steps; origin price {result.origin_price:,.2f}"
        )
        st.caption(
            f"{result.target_unit} | version {result.version} | regressors: {result.regressors}"
        )
        if result.retrospective:
            st.warning(
                "Retrospective scenario: this model was trained after the origin. This is not an out-of-sample evaluation."
            )
