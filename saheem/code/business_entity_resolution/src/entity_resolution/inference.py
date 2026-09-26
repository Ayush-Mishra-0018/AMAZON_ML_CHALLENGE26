"""Streaming model inference and competition output generation."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .calibration import _require_scoring_dependencies, iter_scored_entities
from .contracts import CANDIDATE_COLUMNS, MATCHING_COLUMNS
from .tracking import stage_timer, write_json_atomic


def predict_submission(
    pairs_path: str | Path,
    selection_path: str | Path,
    output_directory: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Score test candidates and write exact model-input and match sets."""

    joblib, _, _ = _require_scoring_dependencies()
    pairs_path = Path(pairs_path)
    selection_path = Path(selection_path)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    matching_path = output_directory / "matching_results.tsv"
    candidate_path = output_directory / "candidate_pairs.tsv"
    existing = [path for path in (matching_path, candidate_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "refusing to overwrite prediction files: "
            + ", ".join(str(path) for path in existing)
        )

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    model_path = selection_path.parent / selection["model_file"]
    bundle = joblib.load(model_path)
    threshold = float(selection["threshold"])
    temporary_matching = matching_path.with_name(f".{matching_path.name}.tmp")
    temporary_candidates = candidate_path.with_name(f".{candidate_path.name}.tmp")
    report: dict[str, object] = {
        "stage": "predict",
        "pairs_path": str(pairs_path.resolve()),
        "selection_path": str(selection_path.resolve()),
        "model_name": selection["model_name"],
        "threshold": threshold,
    }
    rows = candidate_links = matched_links = empty_predictions = 0
    success = False
    with stage_timer(report):
        try:
            with (
                temporary_matching.open("w", encoding="utf-8", newline="") as matching_handle,
                temporary_candidates.open("w", encoding="utf-8", newline="") as candidate_handle,
            ):
                matching_writer = csv.writer(
                    matching_handle,
                    delimiter="\t",
                    lineterminator="\n",
                )
                candidate_writer = csv.writer(
                    candidate_handle,
                    delimiter="\t",
                    lineterminator="\n",
                )
                matching_writer.writerow(MATCHING_COLUMNS)
                candidate_writer.writerow(CANDIDATE_COLUMNS)
                for entity in iter_scored_entities(
                    pairs_path,
                    bundle,
                    split_name="test",
                ):
                    ordered = sorted(
                        entity.candidates,
                        key=lambda candidate: (-candidate.score, candidate.entity_id),
                    )
                    candidate_ids = [candidate.entity_id for candidate in ordered]
                    matched_ids = [
                        candidate.entity_id
                        for candidate in ordered
                        if candidate.score >= threshold
                    ]
                    matching_writer.writerow(
                        (entity.source1_entity_id, ",".join(matched_ids))
                    )
                    candidate_writer.writerow(
                        (entity.source1_entity_id, ",".join(candidate_ids))
                    )
                    rows += 1
                    candidate_links += len(candidate_ids)
                    matched_links += len(matched_ids)
                    empty_predictions += int(not matched_ids)
                for handle in (matching_handle, candidate_handle):
                    handle.flush()
                    os.fsync(handle.fileno())
            temporary_matching.replace(matching_path)
            temporary_candidates.replace(candidate_path)
            success = True
        finally:
            if not success:
                temporary_matching.unlink(missing_ok=True)
                temporary_candidates.unlink(missing_ok=True)
    report.update(
        {
            "rows": rows,
            "candidate_links": candidate_links,
            "matched_links": matched_links,
            "empty_predictions": empty_predictions,
            "matching_path": str(matching_path),
            "candidate_path": str(candidate_path),
        }
    )
    write_json_atomic(output_directory / "prediction_report.json", report)
    return report
