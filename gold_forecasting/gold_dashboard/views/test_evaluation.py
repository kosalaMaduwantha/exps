"""Published test evaluation view; absent reports never become accuracy claims."""

import streamlit as st

from gold_contracts.schemas import ModelMetadata
from gold_dashboard.api_client import ApiClient
from gold_dashboard.charts import evaluation_chart, residual_chart


def render(client: ApiClient, model: ModelMetadata) -> None:
    """Display a selected immutable report and all paginated rows for one horizon."""
    reports = client.evaluations(model.model_id).items
    if not reports:
        st.info(
            "No published test evaluation. Run and publish a compatible holdout evaluation from a notebook."
        )
        return
    report_id = st.selectbox("Test report", [item.report_id for item in reports])
    summary = next(item for item in reports if item.report_id == report_id)
    horizon = st.selectbox("Evaluation horizon (weekday-grid steps)", summary.horizons)
    page = client.evaluation(model.model_id, report_id, horizon)
    rows = list(page.items)
    while len(rows) < page.total:
        following = client.evaluation(
            model.model_id, report_id, horizon, offset=len(rows)
        )
        if not following.items:
            break
        rows.extend(following.items)
    st.caption(
        f"{summary.split} | {summary.start_date} to {summary.end_date} | "
        f"version {page.provenance.version} | recipe {page.provenance.recipe}"
    )
    st.caption(page.provenance.assumptions)
    st.caption(
        "Metrics below describe the full published report for this horizon, independent of pagination."
    )
    st.dataframe(
        [metric.model_dump(exclude={"schema_version"}) for metric in page.metrics],
        use_container_width=True,
    )
    st.plotly_chart(evaluation_chart(rows, page.target_unit), use_container_width=True)
    st.plotly_chart(residual_chart(rows, page.target_unit), use_container_width=True)
