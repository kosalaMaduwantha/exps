"""Read-only published forecast view with observed history supplied by the API."""

from datetime import timedelta

import streamlit as st

from gold_contracts.schemas import ModelMetadata
from gold_dashboard.api_client import ApiClient
from gold_dashboard.charts import future_chart


def render(client: ApiClient, model: ModelMetadata) -> None:
    """Plot saved forecast markers and context without requesting model fitting."""
    summaries = client.snapshots(model.model_id).items
    if not summaries:
        st.info(
            "No future forecast snapshots published. Publish a notebook forecast first."
        )
        return
    identity = st.selectbox(
        "Forecast snapshot", [item.snapshot_id for item in summaries]
    )
    snapshot = client.snapshot(model.model_id, identity)
    page = client.history(
        snapshot.origin_date - timedelta(days=365), snapshot.origin_date
    )
    history = list(page.items)
    while len(history) < page.total:
        following = client.history(
            snapshot.origin_date - timedelta(days=365),
            snapshot.origin_date,
            offset=len(history),
        )
        if not following.items:
            break
        history.extend(following.items)
    st.caption(
        f"Origin: {snapshot.origin_date} | training cutoff: {snapshot.provenance.training_cutoff} | "
        f"version: {snapshot.provenance.version} | created: {snapshot.provenance.created_at.isoformat()}"
    )
    st.caption(snapshot.provenance.assumptions)
    if page.data_hash != snapshot.provenance.data_hash:
        st.warning(
            "Current history has a different data identity from this snapshot. Historical observations may differ."
        )
    st.info(
        "Forecast markers represent only configured weekday-grid horizons. Snapshot dates are historical; data is not refreshed automatically."
    )
    st.plotly_chart(
        future_chart(history, snapshot.points, snapshot.target_unit),
        use_container_width=True,
    )
    st.dataframe(
        [point.model_dump(exclude={"schema_version"}) for point in snapshot.points],
        use_container_width=True,
    )
