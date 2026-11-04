"""Local immutable artifact store, explicit notebook publication, atomic promotion.

Publishers run in notebooks. Readers never train or mutate the store. A model
publication contains native files, typed serving metadata, and file checksums.
JSON reports/snapshots are immutable and independently versioned. One local
writer is supported; use external coordination for concurrent publishers.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from gold_contracts.schemas import (
    EvaluationReport,
    ForecastSnapshot,
    ModelId,
    ModelMetadata,
)

T = TypeVar("T", bound=BaseModel)
ID = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}$")


class ArtifactUnavailable(RuntimeError):
    """Required store data is absent, corrupt, or incompatible."""


class ArtifactNotFound(LookupError):
    """Requested immutable report or snapshot does not exist."""


def safe_id(value: str) -> str:
    """Validate a single path component; raise ValueError for traversal or bad IDs."""
    if not ID.fullmatch(value):
        raise ValueError("Invalid artifact identifier.")
    return value


def digest_file(path: Path) -> str:
    """Return a SHA-256 content digest; missing files raise OSError."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_digest(config: dict[str, object]) -> str:
    """Return a deterministic digest independent of JSON whitespace/key order."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def prepared_digest(root: Path) -> str:
    """Hash panel, features, config snapshot, and source manifest as one data identity."""
    names = ("panel.csv", "features.csv", "config_snapshot.json", "source_manifest.csv")
    return hashlib.sha256(
        "".join(digest_file(root / "data/processed" / name) for name in names).encode()
    ).hexdigest()


def read_contract(path: Path, schema: type[T]) -> T:
    """Read validated JSON; map malformed bytes/schema failures to store errors."""
    try:
        return schema.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise ArtifactUnavailable("Published artifact is missing or invalid.") from exc


def write_immutable(path: Path, value: BaseModel) -> None:
    """Publish complete JSON once using an atomic no-overwrite hard link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".publish-")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class ArtifactStore:
    """Typed readers and explicit publication for a local artifact root.

    Args:
        root: Directory containing registry.json, models/, evaluations/, forecasts/.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def active_versions(self) -> dict[str, str]:
        """Read the atomic active-version map; missing store is unavailable, not empty."""
        try:
            raw = json.loads((self.root / "registry.json").read_text())
            if raw["schema_version"] != 1 or not isinstance(raw["active"], dict):
                raise ValueError("Invalid registry.")
            result = {}
            for model_id, version in raw["active"].items():
                if model_id not in (
                    "prophet",
                    "xgboost",
                    "gru",
                    "lstm",
                ) or not isinstance(version, str):
                    raise ValueError("Invalid registry model.")
                result[model_id] = safe_id(version)
            return result
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ArtifactUnavailable(
                "Registry is missing or invalid; publish from a notebook."
            ) from exc

    def bundle_directory(self, model_id: str, version: str) -> Path:
        """Resolve a validated immutable bundle path; never accept caller file paths."""
        return self.root / "models" / safe_id(model_id) / safe_id(version)

    def metadata(self, model_id: str, version: str | None = None) -> ModelMetadata:
        """Read metadata for an active or explicit version, checking embedded identity."""
        version = version or self.active_versions().get(model_id)
        if version is None:
            raise ArtifactNotFound("Model has no active publication.")
        result = read_contract(
            self.bundle_directory(model_id, version) / "serving.json", ModelMetadata
        )
        if result.model_id != model_id or result.version != version:
            raise ArtifactUnavailable("Bundle identity mismatch.")
        return result

    def verify_bundle(self, metadata: ModelMetadata) -> Path:
        """Verify every native model file before deserialization; return native path."""
        directory = self.bundle_directory(metadata.model_id, metadata.version)
        try:
            checksums = json.loads((directory / "checksums.json").read_text())
            actual_files = {
                p.relative_to(directory / "native").as_posix()
                for p in (directory / "native").rglob("*")
                if p.is_file()
            }
            if (
                not isinstance(checksums, dict)
                or set(checksums) != actual_files
                or not checksums
            ):
                raise ValueError("Invalid checksum inventory.")
            for name, digest in checksums.items():
                path = directory / "native" / name
                if (
                    not path.resolve().is_relative_to((directory / "native").resolve())
                    or digest_file(path) != digest
                ):
                    raise ValueError("Checksum mismatch.")
        except (OSError, ValueError, TypeError) as exc:
            raise ArtifactUnavailable("Bundle files are missing or corrupt.") from exc
        return directory / "native"

    def publish_bundle(
        self, source: Path, metadata: ModelMetadata, *, promote: bool = False
    ) -> Path:
        """Copy a trusted saved bundle without training; optionally promote after validation.

        Raises FileExistsError for an existing version and ValueError for metadata
        that does not describe the native bundle. Source data is never modified.
        """
        native = json.loads((source / "metadata.json").read_text())
        checks = (
            native["kind"] == metadata.model_id,
            native["columns"] == metadata.feature_columns,
            native["training_cutoff"] == metadata.training_cutoff.isoformat(),
            native["training_config"]["horizons"] == metadata.horizons,
            config_digest(native["training_config"]) == metadata.config_hash,
        )
        if not all(checks):
            raise ValueError(
                "Serving metadata does not describe the saved native bundle."
            )
        destination = self.bundle_directory(metadata.model_id, metadata.version)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".publish-"))
        try:
            shutil.copytree(source, temporary / "native")
            (temporary / "serving.json").write_text(metadata.model_dump_json(indent=2))
            checksums = {
                p.relative_to(temporary / "native").as_posix(): digest_file(p)
                for p in (temporary / "native").rglob("*")
                if p.is_file()
            }
            (temporary / "checksums.json").write_text(
                json.dumps(checksums, sort_keys=True)
            )
            if destination.exists():
                raise FileExistsError(destination)
            temporary.rename(destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if promote:
            self.promote(metadata.model_id, metadata.version)
        return destination

    def promote(self, model_id: ModelId, version: str) -> None:
        """Atomically activate an existing verified version; also supports rollback."""
        self.verify_bundle(self.metadata(model_id, version))
        active = (
            self.active_versions() if (self.root / "registry.json").exists() else {}
        )
        active[model_id] = version
        fd, temporary = tempfile.mkstemp(dir=self.root, prefix=".registry-")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump({"schema_version": 1, "active": active}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.root / "registry.json")
        finally:
            Path(temporary).unlink(missing_ok=True)

    def publish_evaluation(self, report: EvaluationReport) -> None:
        """Publish a frozen test report produced in a notebook; never run evaluation."""
        if not report.rows:
            raise ValueError("Cannot publish an empty evaluation.")
        if any(
            row.origin_date <= report.provenance.training_cutoff for row in report.rows
        ):
            raise ValueError(
                "Test origins must follow the evaluation model training cutoff."
            )
        write_immutable(
            self._report_path(
                "evaluations", report.provenance.model_id, report.report_id
            ),
            report,
        )

    def publish_snapshot(self, snapshot: ForecastSnapshot) -> None:
        """Publish a notebook forecast with explicit provenance, without inference."""
        if not snapshot.points:
            raise ValueError("Cannot publish an empty forecast.")
        write_immutable(
            self._report_path(
                "forecasts", snapshot.provenance.model_id, snapshot.snapshot_id
            ),
            snapshot,
        )

    def _report_path(self, kind: str, model_id: str, identity: str) -> Path:
        return self.root / kind / safe_id(model_id) / f"{safe_id(identity)}.json"

    def evaluation(self, model_id: str, report_id: str) -> EvaluationReport:
        """Read one immutable report or raise ArtifactNotFound."""
        path = self._report_path("evaluations", model_id, report_id)
        if not path.exists():
            raise ArtifactNotFound("Evaluation report not found.")
        value = read_contract(path, EvaluationReport)
        if value.report_id != report_id or value.provenance.model_id != model_id:
            raise ArtifactUnavailable("Evaluation identity mismatch.")
        return value

    def snapshot(self, model_id: str, snapshot_id: str) -> ForecastSnapshot:
        """Read one immutable forecast without fitting or prediction."""
        path = self._report_path("forecasts", model_id, snapshot_id)
        if not path.exists():
            raise ArtifactNotFound("Forecast snapshot not found.")
        value = read_contract(path, ForecastSnapshot)
        if value.snapshot_id != snapshot_id or value.provenance.model_id != model_id:
            raise ArtifactUnavailable("Snapshot identity mismatch.")
        return value

    def evaluations(self, model_id: str) -> list[EvaluationReport]:
        """List available reports; absence is a valid empty collection."""
        directory = self.root / "evaluations" / safe_id(model_id)
        return [
            self.evaluation(model_id, path.stem)
            for path in sorted(directory.glob("*.json"))
        ]

    def snapshots(self, model_id: str) -> list[ForecastSnapshot]:
        """List snapshots in descending origin order; preserve their original version."""
        directory = self.root / "forecasts" / safe_id(model_id)
        return sorted(
            [self.snapshot(model_id, path.stem) for path in directory.glob("*.json")],
            key=lambda item: item.origin_date,
            reverse=True,
        )
