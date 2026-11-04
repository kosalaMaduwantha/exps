"""No-training integration tests across publication, API, typed client, and charts."""

from datetime import date, datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pydantic import ValidationError

from gold_api.main import create_app
from gold_api.service import ForecastService
from gold_contracts.schemas import (
    EvaluationReport,
    EvaluationRow,
    ForecastPoint,
    ForecastSnapshot,
    HorizonMetrics,
    ModelMetadata,
    Provenance,
    Regressor,
    SpotRequest,
)
from gold_dashboard.api_client import ApiClient, ApiError
from gold_dashboard.charts import evaluation_chart, future_chart, residual_chart
from model_lib.artifacts.registry import ArtifactStore, config_digest, prepared_digest


class ApplicationTests(unittest.TestCase):
    """Small fixture bundle with injected frozen predictions; never fit a model."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        processed = self.root / "data/processed"
        processed.mkdir(parents=True)
        for filename in (
            "panel.csv",
            "features.csv",
            "config_snapshot.json",
            "source_manifest.csv",
        ):
            (processed / filename).write_text("fixture")
        self.store = ArtifactStore(self.root / "serving")
        self.cfg = {
            "horizons": [5, 21],
            "target_unit": "USD; quotation unit unverified",
        }
        self.native = {
            "kind": "xgboost",
            "columns": ["gold_latest", "macro"],
            "params": {},
            "training_config": self.cfg,
            "training_cutoff": "2024-02-01",
        }
        source = self.root / "native"
        source.mkdir()
        (source / "metadata.json").write_text(json.dumps(self.native))
        (source / "fake-model.json").write_text("native fixture")
        self.metadata = ModelMetadata(
            model_id="xgboost",
            version="v1",
            recipe="validated_daily",
            training_cutoff=date(2024, 2, 1),
            horizons=[5, 21],
            feature_columns=["gold_latest", "macro"],
            external_regressors=[
                Regressor(name="macro", description="Origin-time external input")
            ],
            target_unit=self.cfg["target_unit"],
            config_hash=config_digest(self.cfg),
            data_hash=prepared_digest(self.root),
        )
        self.store.publish_bundle(source, self.metadata, promote=True)
        dates = pd.bdate_range("2024-01-01", periods=100)
        self.panel = pd.DataFrame({"y": 2000.0, "target_observed": 1}, index=dates)
        self.features = pd.DataFrame({"gold_latest": 2000.0, "macro": 1.0}, index=dates)
        self.origin = dates[80].date()
        self.seen = []

        def predictor(bundle, panel, features, origins, config):
            self.seen.append(features.copy())
            return np.array([[float(features.macro.iloc[-1]) + 2000, 2100]])

        self.service = ForecastService(
            self.store,
            self.root,
            prepared_loader=lambda: (self.panel, self.features, pd.DataFrame()),
            bundle_loader=lambda path: dict(self.native),
            predictor=predictor,
        )
        self.api = TestClient(create_app(self.service))
        self.addCleanup(self.api.close)
        self.request = {
            "model_id": "xgboost",
            "expected_version": "v1",
            "origin_date": self.origin.isoformat(),
            "horizon": 5,
            "regressors": {"macro": 7.0},
        }
        self.provenance = Provenance(
            model_id="xgboost",
            version="holdout-v0",
            recipe="validated_daily",
            config_hash=self.metadata.config_hash,
            data_hash=self.metadata.data_hash,
            training_cutoff=date(2024, 2, 1),
            assumptions="Origin-known inputs only.",
            created_at=datetime(2024, 6, 1, tzinfo=timezone.utc),
        )

    def publish_reports(self) -> None:
        """Publish two test rows and a future snapshot with independent versions."""
        rows = [
            EvaluationRow(
                origin_date=date(2024, 3, day),
                target_date=date(2024, 4, day),
                horizon=5,
                prediction=2010,
                actual=2000,
                baseline=1990,
                residual=10,
            )
            for day in (1, 4)
        ]
        self.store.publish_evaluation(
            EvaluationReport(
                report_id="test-1",
                provenance=self.provenance,
                split="holdout",
                target_unit="USD",
                rows=rows,
                metrics=[
                    HorizonMetrics(
                        horizon=5,
                        n=2,
                        mae=10,
                        rmse=10,
                        baseline_mae=10,
                        skill_vs_persistence=0,
                    )
                ],
            )
        )
        self.store.publish_snapshot(
            ForecastSnapshot(
                snapshot_id="future-1",
                provenance=self.provenance,
                origin_date=self.origin,
                origin_price=2000,
                target_unit="USD",
                points=[
                    ForecastPoint(
                        target_date=date(2024, 6, 3), horizon=5, prediction=2050
                    )
                ],
            )
        )

    def test_spot_override_is_request_local_and_target_is_weekday_grid(self) -> None:
        """Override only origin features, preserve saved history and return resolved identity."""
        response = self.api.post("/forecasts/spot", json=self.request)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["points"][0]["prediction"], 2007)
        self.assertEqual(
            result["points"][0]["target_date"],
            (pd.Timestamp(self.origin) + pd.offsets.BDay(5)).date().isoformat(),
        )
        self.assertEqual(result["version"], "v1")
        self.assertEqual(len(self.seen[0]), 81)
        self.assertEqual(self.seen[0].macro.iloc[-2], 1)
        self.assertTrue(self.features.macro.eq(1).all())
        self.assertEqual(self.api.get("/ready").status_code, 200)

    def test_input_validation_and_stale_version(self) -> None:
        """Every invalid request fails before invoking the predictor."""
        invalid = [
            ({"regressors": {}}, 422),
            ({"regressors": {"macro": 2, "extra": 3}}, 422),
            ({"horizon": 7}, 422),
            ({"expected_version": "v0"}, 409),
            ({"origin_date": "2024-01-06"}, 422),
            ({"origin_date": "2030-01-01"}, 422),
        ]
        for update, code in invalid:
            with self.subTest(update=update):
                response = self.api.post("/forecasts/spot", json=self.request | update)
                self.assertEqual(response.status_code, code, response.text)
                self.assertIn("code", response.json())
        self.assertEqual(len(self.seen), 0)
        with self.assertRaises(ValidationError):
            SpotRequest.model_validate(
                self.request | {"regressors": {"macro": float("nan")}}
            )
        response = self.api.post(
            "/forecasts/spot",
            content=json.dumps(self.request | {"regressors": {"macro": float("inf")}}),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(response.status_code, 422)

    def test_unobserved_and_recurrent_gap_origins(self) -> None:
        """Unobserved origin prices and long recurrent context gaps are not eligible."""
        self.panel.loc[pd.Timestamp(self.origin), ["y", "target_observed"]] = [
            np.nan,
            0,
        ]
        self.assertEqual(
            self.api.post("/forecasts/spot", json=self.request).status_code, 422
        )
        recurrent = self.metadata.model_copy(update={"model_id": "gru", "lookback": 20})
        self.panel.iloc[70:76, self.panel.columns.get_loc("y")] = np.nan
        self.assertNotIn(
            self.origin, self.service.eligible(recurrent, self.panel, self.features)
        )

    def test_report_pagination_metrics_and_null_intervals(self) -> None:
        """Metrics describe the full saved population even when only one row is served."""
        self.publish_reports()
        response = self.api.get(
            "/models/xgboost/evaluations/test-1", params={"limit": 1}
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["total"], 2)
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["metrics"][0]["n"], 2)
        self.assertEqual(result["provenance"]["version"], "holdout-v0")
        self.assertIsNone(result["items"][0]["lower"])
        self.assertNotIn("NaN", response.text)
        self.assertEqual(
            self.api.get("/models/xgboost/evaluations/missing").status_code, 404
        )
        self.assertEqual(
            self.api.get(
                "/models/xgboost/evaluations/test-1", params={"horizon": 252}
            ).status_code,
            422,
        )
        self.assertEqual(
            self.api.get(
                "/history",
                params={"start_date": "2025-01-01", "end_date": "2024-01-01"},
            ).status_code,
            422,
        )
        self.assertEqual(
            self.api.get("/history", params={"limit": 2001}).status_code, 422
        )
        self.assertEqual(len(self.seen), 0)

    def test_empty_reports_and_corrupt_bundle(self) -> None:
        """Empty research outputs are valid; broken promoted native files are unavailable."""
        self.assertEqual(
            self.api.get("/models/xgboost/evaluations").json()["items"], []
        )
        native = self.store.bundle_directory("xgboost", "v1") / "native/fake-model.json"
        native.write_text("corrupt")
        self.assertEqual(
            self.api.post("/forecasts/spot", json=self.request).status_code, 503
        )
        self.assertEqual(self.api.get("/health").status_code, 200)
        self.assertEqual(self.api.get("/ready").status_code, 503)

    def test_missing_registry_and_data_identity(self) -> None:
        """Data changes invalidate a promoted bundle instead of serving stale features."""
        (self.root / "data/processed/features.csv").write_text("changed")
        self.assertEqual(
            self.api.post("/forecasts/spot", json=self.request).status_code, 503
        )
        (self.store.root / "registry.json").unlink()
        self.assertEqual(self.api.get("/models").status_code, 503)

    def test_immutable_publication_and_rollback(self) -> None:
        """Versions and reports cannot be overwritten, and promotion supports old versions."""
        v2 = self.metadata.model_copy(update={"version": "v2"})
        self.store.publish_bundle(self.root / "native", v2, promote=True)
        self.assertEqual(self.store.metadata("xgboost").version, "v2")
        self.store.promote("xgboost", "v1")
        self.assertEqual(self.store.metadata("xgboost").version, "v1")
        with self.assertRaises(FileExistsError):
            self.store.publish_bundle(self.root / "native", self.metadata)
        self.publish_reports()
        with self.assertRaises(FileExistsError):
            self.store.publish_evaluation(self.store.evaluation("xgboost", "test-1"))

    def test_typed_client_and_plotly_data(self) -> None:
        """Actual HTTP schemas reach chart traces and inference unchanged."""
        self.publish_reports()

        def transport(request: httpx.Request) -> httpx.Response:
            response = self.api.request(
                request.method,
                request.url.path,
                params=request.url.params,
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, content=response.content)

        with ApiClient(
            "http://test", transport=httpx.MockTransport(transport)
        ) as client:
            self.assertEqual(client.models().items[0].version, "v1")
            page = client.evaluation("xgboost", "test-1", 5)
            figure = evaluation_chart(page.items, page.target_unit)
            self.assertEqual(list(figure.data[1].y), [2010, 2010])
            self.assertEqual(
                list(residual_chart(page.items, page.target_unit).data[0].y), [10, 10]
            )
            snapshot = client.snapshot("xgboost", "future-1")
            history = client.history(date(2024, 1, 1), self.origin)
            chart = future_chart(history.items, snapshot.points, "USD")
            self.assertEqual(list(chart.data[1].y), [2050])
            result = client.spot(SpotRequest.model_validate(self.request))
            self.assertEqual(result.regressors, {"macro": 7})
            with self.assertRaises(ApiError) as failure:
                client.snapshot("xgboost", "unknown")
            self.assertEqual(failure.exception.status_code, 404)

    def test_client_rejects_bad_responses_and_connection_errors(self) -> None:
        """Schema mismatches and transport failures become actionable dashboard errors."""
        with ApiClient(
            "http://test",
            transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
        ) as client:
            with self.assertRaises(ApiError):
                client.models()

        def offline(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("offline")

        with ApiClient("http://test", transport=httpx.MockTransport(offline)) as client:
            with self.assertRaisesRegex(ApiError, "Cannot reach"):
                client.models()

    def test_streamlit_views_and_form_submission(self) -> None:
        """Exercise actual Streamlit widgets and Plotly rendering through the API client."""
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        self.publish_reports()

        def transport(request: httpx.Request) -> httpx.Response:
            response = self.api.request(
                request.method,
                request.url.path,
                params=request.url.params,
                content=request.content,
                headers=dict(request.headers),
            )
            return httpx.Response(response.status_code, content=response.content)

        def client(base_url: str, **_kwargs: object) -> ApiClient:
            return ApiClient(base_url, transport=httpx.MockTransport(transport))

        st.cache_data.clear()
        with patch("gold_dashboard.api_client.ApiClient", side_effect=client):
            app = AppTest.from_file(
                str(Path(__file__).resolve().parents[1] / "gold_dashboard/app.py")
            ).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 2)
            app.radio[0].set_value("Future forecasts").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            app.radio[0].set_value("Spot prediction").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(self.seen), 0)
            app.number_input[0].set_value(9.0)
            next(
                button for button in app.button if button.label == "Predict spot"
            ).click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "2,009.00")
            self.assertEqual(len(self.seen), 1)
            app.run()
            self.assertEqual(
                len(self.seen), 1, "Widget reruns must not repeat inference."
            )
        st.cache_data.clear()


if __name__ == "__main__":
    unittest.main()
