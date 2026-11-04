"""Named diagnostic-only notebook recipes, separate from validated daily serving.

Import helpers from legacy_monthly_prophet, daily_prophet_experiment, and
daily_xgboost_challenger. Their explicit keyword-only context documents all data
dependencies; no notebook globals or mutable module-level model state are used.
"""

from datetime import datetime, timezone
import json
from pathlib import Path


def record_diagnostic_recipe(output_directory: Path, recipe_id: str) -> Path:
    """Write provenance beside every legacy notebook output collection.

    Args:
        output_directory: Notebook-owned directory containing diagnostic exports.
        recipe_id: Explicit name of the extracted experimental recipe.
    Returns:
        Path to recipe_provenance.json, refreshed on each notebook execution.
    Side effects:
        Creates the directory and writes a diagnostic-only provenance sidecar.
        Existing numeric outputs are not rewritten or promoted for serving.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "recipe_provenance.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recipe_id": recipe_id,
                "diagnostic_only": True,
                "availability": "oracle_or_conditional_snapshot_diagnostic",
                "selection": "legacy_full_series_screening_or_holdout_tuning_not_for_deployment",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "notes": "Published legacy metrics must not rank new serving artifacts. See notebook assumptions.",
            },
            indent=2,
        )
    )
    return path
