"""Streamlit entry point: ``streamlit run gold_dashboard/app.py``.

Configure GOLD_API_URL (default http://127.0.0.1:8000). Read-only GET responses
are cached for 60 seconds in this process; POST inference is never cached.
"""

import os

import streamlit as st

from gold_contracts.schemas import ModelList
from gold_dashboard.api_client import ApiClient, ApiError, clear_read_cache
from gold_dashboard.views import future_forecast, spot_prediction, test_evaluation


@st.cache_data(ttl=60, max_entries=16)
def model_catalog(base_url: str) -> ModelList:
    """Cache active metadata briefly; explicit refresh clears it after promotion."""
    with ApiClient(base_url) as client:
        return client.models()


def main() -> None:
    """Render API-only dashboard navigation and actionable service error states."""
    st.set_page_config(page_title="Gold Forecasting", layout="wide")
    st.title("Gold forecasting dashboard")
    base_url = os.environ.get("GOLD_API_URL", "http://127.0.0.1:8000")
    st.sidebar.caption(f"API: {base_url}")
    if st.sidebar.button("Refresh API data"):
        st.cache_data.clear()
        clear_read_cache()
        st.session_state.pop("spot_result", None)
    try:
        catalog = model_catalog(base_url)
        if not catalog.items:
            st.info("No models promoted. Publish and promote a model from a notebook.")
            return
        identity = st.sidebar.selectbox(
            "Model", [item.model_id for item in catalog.items]
        )
        metadata = next(item for item in catalog.items if item.model_id == identity)
        st.sidebar.caption(f"Version: {metadata.version} | recipe: {metadata.recipe}")
        view = st.sidebar.radio(
            "View", ["Test evaluation", "Future forecasts", "Spot prediction"]
        )
        with ApiClient(base_url, cache_version=metadata.version) as client:
            with st.spinner("Loading API data…"):
                if view == "Test evaluation":
                    test_evaluation.render(client, metadata)
                elif view == "Future forecasts":
                    future_forecast.render(client, metadata)
                else:
                    spot_prediction.render(client, metadata)
    except ApiError as exc:
        st.error(str(exc))
        if exc.status_code == 409:
            st.cache_data.clear()
            clear_read_cache()
            st.session_state.pop("spot_result", None)
            st.info("Refresh API data to load the current model inputs.")


if __name__ == "__main__":
    main()
