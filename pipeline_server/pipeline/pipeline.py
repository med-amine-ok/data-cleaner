from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from pipeline.cleaners import (
    clean_commune,
    clean_description,
    clean_dob,
    clean_email,
    clean_gender,
    clean_last,
    clean_user_type,
    clean_school_type,
    clean_name,
    clean_phone_number,
    clean_preferences,
    clean_profile_picture,
    clean_username,
    clean_wilaya,
)
from pipeline.dedup import SmartDeduplicator
from pipeline.inferer import MissingValueInferer
from pipeline.mapper import ColumnAutoMapper, MappingResult
from pipeline.reader import FileIngester
from pipeline.schema_builder import SchemaBuilder


class StudentProfilePipeline:
    """
    Orchestrate the full stateless student profile cleaning pipeline.

    The pipeline reads a raw uploaded file, maps column names, cleans values,
    performs deduplication, infers safe missing fields, detects anomalies,
    builds schema-compatible records, and returns outputs ready for ZIP export.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        mapping_threshold: float = 85.0,
        duplicate_threshold: float = 85.0,
        anomaly_contamination: float = 0.08,
    ) -> None:
        """
        Initialize the pipeline and load lookup data.

        Args:
            data_dir: Path to the data directory containing JSON lookup files.
            mapping_threshold: Minimum fuzzy score for column auto-mapping.
            duplicate_threshold: Minimum fuzzy score for duplicate detection.
            anomaly_contamination: Expected fraction of anomalies for IsolationForest.
        """
        self._base_dir = Path(__file__).resolve().parent
        self._data_dir = Path(data_dir) if data_dir is not None else self._base_dir.parent / "data"
        self._mapping_threshold = mapping_threshold
        self._duplicate_threshold = duplicate_threshold
        self._anomaly_contamination = anomaly_contamination

        self._reader = FileIngester()

        self._field_aliases = self._load_field_aliases(self._data_dir / "aliases.json")
        self._wilayas_lookup = self._load_flat_lookup(self._data_dir / "wilayas.json")
        self._subjects_lookup = self._load_flat_lookup(self._data_dir / "subjects.json")

        self._mapper = ColumnAutoMapper(field_aliases=self._field_aliases, threshold=self._mapping_threshold)
        self._deduplicator = SmartDeduplicator(threshold=self._duplicate_threshold)
        self._inferer = MissingValueInferer()
        self._schema_builder = SchemaBuilder()

    def run(self, file_path: str) -> dict[str, Any]:
        """
        Run the full pipeline on one uploaded file.

        Args:
            file_path: Path to the uploaded file on disk.

        Returns:
            A dictionary containing clean records, quarantine records,
            duplicate records, summary text, and mapping metadata.
        """
        raw_dataframe = self._reader.read(file_path)
        mapped_dataframe, mapping_results = self._mapper.map_columns(raw_dataframe)
        cleaned_dataframe = self._clean_dataframe(mapped_dataframe)
        cleaned_dataframe = self._compose_display_name(cleaned_dataframe)

        deduped_dataframe, duplicate_records = self._deduplicator.find_duplicates(cleaned_dataframe)
        inferred_dataframe = self._inferer.infer_dataframe(deduped_dataframe)

        anomaly_quarantine, stable_dataframe = self._detect_anomalies(inferred_dataframe)

        clean_records, schema_rejected_records = self._schema_builder.build_dataframe(stable_dataframe)

        quarantine_records = self._combine_quarantine_frames(
            schema_rejected_records,
            anomaly_quarantine,
        )

        pipeline_summary = self._build_summary(
            input_rows=len(raw_dataframe),
            mapped_rows=len(mapped_dataframe),
            cleaned_rows=len(cleaned_dataframe),
            deduped_rows=len(deduped_dataframe),
            clean_rows=len(clean_records),
            rejected_rows=len(quarantine_records),
            duplicate_rows=len(duplicate_records),
            mapping_results=mapping_results,
            anomaly_rows=len(anomaly_quarantine),
        )

        return {
            "clean_records": clean_records,
            "quarantine_records": quarantine_records,
            "duplicates_records": duplicate_records,
            "pipeline_summary": pipeline_summary,
            "column_mappings": [asdict(result) for result in mapping_results],
        }

    def _clean_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Clean all supported columns in a dataframe.

        Args:
            dataframe: DataFrame after column mapping.

        Returns:
            A new dataframe with normalized field values.
        """
        working = dataframe.copy()

        if "name" in working.columns:
            working["name"] = working["name"].map(clean_name)

        if "last" in working.columns:
            working["last"] = working["last"].map(clean_last)

        if "email" in working.columns:
            working["email"] = working["email"].map(clean_email)

        if "username" in working.columns:
            working["username"] = working["username"].map(clean_username)

        if "phoneNumber" in working.columns:
            working["phoneNumber"] = working["phoneNumber"].map(clean_phone_number)

        if "DOB" in working.columns:
            working["DOB"] = working["DOB"].map(clean_dob)

        if "gender" in working.columns:
            working["gender"] = working["gender"].map(clean_gender)

        if "description" in working.columns:
            working["description"] = working["description"].map(clean_description)

        if "profilePicture" in working.columns:
            working["profilePicture"] = working["profilePicture"].map(clean_profile_picture)

        if "preferences" in working.columns:
            working["preferences"] = working["preferences"].map(
                lambda value: clean_preferences(value, self._subjects_lookup)
            )

        if "wilaya" in working.columns:
            working["wilaya"] = working["wilaya"].map(
                lambda value: clean_wilaya(value, self._wilayas_lookup)
            )

        if "commune" in working.columns:
            working["commune"] = working["commune"].map(clean_commune)

        if "location" in working.columns or "wilaya" in working.columns or "commune" in working.columns:
            working["location"] = working.apply(self._build_location_object, axis=1)

        if "userType" in working.columns:
            working["userType"] = working["userType"].map(clean_user_type)

        if "schoolType" in working.columns:
            working["schoolType"] = working["schoolType"].map(clean_school_type)

        if "verificationType" in working.columns:
            working["verificationType"] = working["verificationType"].map(self._normalize_text_lower)

        if "blurhash" in working.columns:
            working["blurhash"] = working["blurhash"].map(self._normalize_optional_text)

        if "_id" in working.columns:
            working["_id"] = working["_id"].map(self._normalize_optional_text)

        if "socialLinks" in working.columns:
            working["socialLinks"] = working["socialLinks"].map(self._normalize_social_links)

        return working

    def _compose_display_name(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Merge split name parts into the full display name when possible.

        Args:
            dataframe: Cleaned dataframe with name-related columns.

        Returns:
            A dataframe where the name column contains the full display name.
        """
        if dataframe.empty or "name" not in dataframe.columns or "last" not in dataframe.columns:
            return dataframe.copy()

        working = dataframe.copy()
        working["name"] = working.apply(self._merge_name_and_last, axis=1)
        return working

    def _merge_name_and_last(self, row: pd.Series) -> str | None:
        """
        Build a full display name from the available name parts.

        Args:
            row: One dataframe row.

        Returns:
            A full display name when a first name is present, otherwise the
            original name value.
        """
        name_value = self._normalize_optional_text(row.get("name"))
        last_value = self._normalize_optional_text(row.get("last"))

        if not name_value:
            return None

        if not last_value:
            return name_value

        name_parts = name_value.split()
        if name_parts and name_parts[-1].casefold() == last_value.casefold():
            return name_value

        return f"{name_value} {last_value}"

    def _detect_anomalies(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Detect outliers using a small feature set and quarantine strong anomalies.

        Uses adaptive contamination based on dataset size to avoid over-quarantining
        small datasets. Requires at least 20 rows to run anomaly detection.

        Args:
            dataframe: Cleaned dataframe after inference.

        Returns:
            A tuple of:
                - quarantine dataframe for anomalous rows
                - stable dataframe with anomalous rows removed
        """
        if dataframe.empty:
            return pd.DataFrame(), dataframe.copy()

        features = self._build_anomaly_features(dataframe)
        if features.empty or len(features) < 20:
            return pd.DataFrame(), dataframe.copy()

        if features.shape[1] == 0:
            return pd.DataFrame(), dataframe.copy()

        if np.all(np.nanstd(features.to_numpy(dtype=float), axis=0) == 0):
            return pd.DataFrame(), dataframe.copy()

        adaptive_contamination = max(0.01, min(0.05, 3.0 / len(features)))

        model = IsolationForest(
            random_state=42,
            contamination=adaptive_contamination,
        )
        predictions = model.fit_predict(features)
        anomaly_scores = model.decision_function(features)

        stable_rows: list[dict[str, Any]] = []
        quarantine_rows: list[dict[str, Any]] = []

        for position, (index, row) in enumerate(dataframe.iterrows()):
            row_dict = row.to_dict()
            row_dict["anomaly_score"] = float(anomaly_scores[position])
            row_dict["anomaly_flag"] = bool(predictions[position] == -1)

            if row_dict["anomaly_flag"]:
                row_dict["failure_reason"] = "anomaly_detected"
                row_dict["row_index"] = int(index)
                quarantine_rows.append(row_dict)
            else:
                stable_rows.append(row_dict)

        return pd.DataFrame(quarantine_rows), pd.DataFrame(stable_rows)

    def _build_anomaly_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Build numeric features for anomaly detection.

        Args:
            dataframe: Cleaned dataframe.

        Returns:
            A dataframe of numeric anomaly features.
        """
        features = pd.DataFrame(index=dataframe.index)

        if "age" in dataframe.columns:
            features["age"] = pd.to_numeric(dataframe["age"], errors="coerce")
        elif "DOB" in dataframe.columns:
            features["age"] = dataframe["DOB"].map(self._estimate_age_from_timestamp)
        else:
            features["age"] = np.nan

        if "activity_score" in dataframe.columns:
            features["activity_score"] = pd.to_numeric(dataframe["activity_score"], errors="coerce")
        else:
            features["activity_score"] = 0.0

        features = features.dropna(axis=1, how="all")
        return features.fillna(features.median(numeric_only=True)).fillna(0.0)

    def _combine_quarantine_frames(
        self,
        schema_rejected_records: pd.DataFrame,
        anomaly_quarantine: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine rejected and quarantined rows into one output dataframe.

        Args:
            schema_rejected_records: Rows rejected by schema validation.
            anomaly_quarantine: Rows rejected by anomaly detection.

        Returns:
            A combined quarantine dataframe with a failure_reason column.
        """
        frames: list[pd.DataFrame] = []

        if not schema_rejected_records.empty:
            schema_frame = schema_rejected_records.copy()
            if "failure_reason" not in schema_frame.columns:
                schema_frame["failure_reason"] = "schema_rejected"
            frames.append(schema_frame)

        if not anomaly_quarantine.empty:
            anomaly_frame = anomaly_quarantine.copy()
            if "failure_reason" not in anomaly_frame.columns:
                anomaly_frame["failure_reason"] = "anomaly_detected"
            frames.append(anomaly_frame)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True, sort=False)

    def _build_summary(
        self,
        input_rows: int,
        mapped_rows: int,
        cleaned_rows: int,
        deduped_rows: int,
        clean_rows: int,
        rejected_rows: int,
        duplicate_rows: int,
        mapping_results: list[MappingResult],
        anomaly_rows: int,
    ) -> str:
        """
        Build a human-readable pipeline summary.

        Args:
            input_rows: Number of rows read from the uploaded file.
            mapped_rows: Number of rows after column mapping.
            cleaned_rows: Number of rows after field cleaning.
            deduped_rows: Number of rows after deduplication.
            clean_rows: Number of rows written to clean output.
            rejected_rows: Number of rows rejected or quarantined.
            duplicate_rows: Number of duplicate-review rows.
            mapping_results: Mapping decisions from the column mapper.
            anomaly_rows: Number of anomalous rows detected.

        Returns:
            A multiline summary string.
        """
        mapping_lines = [
            f"- {result.original_column} -> {result.mapped_column} ({result.score:.1f}%, {result.matched_alias})"
            for result in mapping_results
        ]

        lines = [
            "Student Profile Pipeline Summary",
            f"Run time: {datetime.now(timezone.utc).isoformat()}",
            "",
            f"Input rows: {input_rows}",
            f"Mapped rows: {mapped_rows}",
            f"Cleaned rows: {cleaned_rows}",
            f"Deduped rows: {deduped_rows}",
            f"Clean records: {clean_rows}",
            f"Quarantined/rejected rows: {rejected_rows}",
            f"Duplicate review rows: {duplicate_rows}",
            f"Anomaly rows: {anomaly_rows}",
            "",
            "Column mappings:",
            *(mapping_lines if mapping_lines else ["- none"]),
        ]

        return "\n".join(lines)

    def _build_location_object(self, row: pd.Series) -> dict[str, Any] | None:
        """
        Build a schema-compatible location object from available columns.

        Args:
            row: One dataframe row.

        Returns:
            A location dictionary or None.
        """
        raw_location = row.get("location")
        if isinstance(raw_location, Mapping):
            return {
                "wilaya": self._normalize_optional_text(raw_location.get("wilaya")),
                "commune": self._normalize_optional_text(raw_location.get("commune")),
                "coordinates": self._normalize_coordinates(raw_location.get("coordinates")),
                "fullLocation": self._normalize_optional_text(raw_location.get("fullLocation")),
            }

        wilaya = self._normalize_optional_text(row.get("wilaya"))
        commune = self._normalize_optional_text(row.get("commune"))
        full_location = self._normalize_optional_text(raw_location)

        if not full_location:
            full_location = ", ".join(part for part in [wilaya, commune] if part)

        if not any([wilaya, commune, full_location]):
            return None

        return {
            "wilaya": wilaya,
            "commune": commune,
            "coordinates": {"lang": None, "lat": None},
            "fullLocation": full_location,
        }

    def _normalize_coordinates(self, value: Any) -> dict[str, float | None]:
        """
        Normalize a coordinate mapping.

        Args:
            value: Raw coordinate value.

        Returns:
            A dictionary containing lang and lat.
        """
        if not isinstance(value, Mapping):
            return {"lang": None, "lat": None}

        return {
            "lang": self._to_float(value.get("lang")),
            "lat": self._to_float(value.get("lat")),
        }

    def _normalize_social_links(self, value: Any) -> dict[str, str] | None:
        """
        Normalize socialLinks into a string-only mapping.

        Args:
            value: Raw socialLinks value.

        Returns:
            A cleaned mapping of social links or None.
        """
        if not isinstance(value, Mapping):
            return None

        normalized: dict[str, str] = {}
        for key, raw_value in value.items():
            text = self._normalize_optional_text(raw_value)
            if text:
                normalized[str(key)] = text

        return normalized or None

    def _normalize_user_type(self, value: Any) -> str | None:
        """
        Normalize userType to a lowercase canonical value.

        Args:
            value: Raw userType value.

        Returns:
            A valid userType or None.
        """
        text = self._normalize_text_lower(value)
        if text in {"school", "student", "teacher", "parent", "admin"}:
            return text
        return None

    def _normalize_text_lower(self, value: Any) -> str | None:
        """
        Normalize text to lowercase and strip it.

        Args:
            value: Raw text value.

        Returns:
            Cleaned lowercase text or None.
        """
        text = self._normalize_optional_text(value)
        if not text:
            return None
        return text.lower()

    def _normalize_optional_text(self, value: Any) -> str:
        """
        Normalize an arbitrary value into a trimmed string.

        Args:
            value: Raw value.

        Returns:
            A cleaned string, or an empty string if the value is missing.
        """
        if value is None:
            return ""

        text = str(value).strip()
        if text.lower() in {"", "nan", "none", "null"}:
            return ""

        return text

    def _to_float(self, value: Any) -> float | None:
        """
        Safely convert a value to float.

        Args:
            value: Raw numeric value.

        Returns:
            Float value or None.
        """
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _estimate_age_from_timestamp(self, value: Any) -> float | None:
        """
        Estimate age in years from a Unix timestamp.

        Args:
            value: Timestamp value in seconds.

        Returns:
            Approximate age in years or None.
        """
        timestamp = self._to_float(value)
        if timestamp is None or timestamp <= 0:
            return None

        try:
            birth_date = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

        now = datetime.now(timezone.utc)
        age_years = (now - birth_date).days / 365.25
        if age_years < 0:
            return None

        return float(age_years)

    def _load_field_aliases(self, file_path: Path) -> dict[str, list[str]]:
        """
        Load the field alias dictionary for column auto-mapping.

        Args:
            file_path: Path to aliases.json.

        Returns:
            A canonical field -> aliases mapping.
        """
        raw = self._load_json(file_path)

        if "FIELD_ALIASES" in raw and isinstance(raw["FIELD_ALIASES"], Mapping):
            raw = raw["FIELD_ALIASES"]

        aliases: dict[str, list[str]] = {}
        for canonical_field, value in raw.items():
            canonical_name = self._normalize_optional_text(canonical_field)
            if not canonical_name:
                continue

            if isinstance(value, list):
                aliases[canonical_name] = [
                    self._normalize_optional_text(item)
                    for item in value
                    if self._normalize_optional_text(item)
                ]
            elif isinstance(value, Mapping):
                nested_aliases = value.get("aliases")
                if isinstance(nested_aliases, list):
                    aliases[canonical_name] = [
                        self._normalize_optional_text(item)
                        for item in nested_aliases
                        if self._normalize_optional_text(item)
                    ]
                else:
                    aliases[canonical_name] = []
            else:
                aliases[canonical_name] = []

        return aliases

    def _load_flat_lookup(self, file_path: Path) -> dict[str, str]:
        """
        Load a flexible alias-to-canonical lookup from JSON.

        Args:
            file_path: Path to a JSON lookup file.

        Returns:
            A normalized alias -> canonical mapping.
        """
        raw = self._load_json(file_path)
        lookup: dict[str, str] = {}

        for key, value in raw.items():
            canonical = self._normalize_optional_text(key)
            if not canonical:
                continue

            if isinstance(value, str):
                value_text = self._normalize_optional_text(value)
                if value_text:
                    lookup[self._normalize_optional_text(key).lower()] = value_text
                    lookup[value_text.lower()] = value_text
            elif isinstance(value, list):
                lookup[canonical.lower()] = canonical
                for alias in value:
                    alias_text = self._normalize_optional_text(alias)
                    if alias_text:
                        lookup[alias_text.lower()] = canonical
            elif isinstance(value, Mapping):
                lookup[canonical.lower()] = canonical
                aliases = value.get("aliases", [])
                if isinstance(aliases, list):
                    for alias in aliases:
                        alias_text = self._normalize_optional_text(alias)
                        if alias_text:
                            lookup[alias_text.lower()] = canonical
            else:
                lookup[canonical.lower()] = canonical

        return lookup

    def _load_json(self, file_path: Path) -> dict[str, Any]:
        """
        Load a JSON file as a dictionary.

        Args:
            file_path: JSON file path.

        Returns:
            Parsed JSON dictionary, or an empty dict if the file is missing.
        """
        if not file_path.exists():
            return {}

        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

        if isinstance(data, dict):
            return data

        return {}