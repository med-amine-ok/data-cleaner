from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from pipeline.cleaners import clean_dob


@dataclass(frozen=True)
class BuildResult:
    """
    Represent the result of building a schema-ready record.

    Attributes:
        record: The final schema-compatible record, or None if invalid.
        is_valid: True when the record satisfies the schema shape.
        reason: Explanation for rejection or warnings.
    """
    record: dict[str, Any] | None
    is_valid: bool
    reason: str


class SchemaBuilder:
    """
    Build final schema-compatible user records from cleaned student rows.

    The builder ensures the output matches the TypeScript UserType shape:
    - SchoolType for school records
    - NonSchoolType for student/teacher/parent/admin records
    """

    ALLOWED_USER_TYPES: frozenset[str] = frozenset({"school", "student", "teacher", "parent", "admin"})
    ALLOWED_GENDERS: frozenset[str] = frozenset({"male", "female"})
    ALLOWED_SCHOOL_TYPES: frozenset[str] = frozenset(
        {
            "private",
            "language",
            "university",
            "formation",
            "public",
            "support",
            "private-university",
            "preschool",
        }
    )

    def build(self, row: Mapping[str, Any]) -> BuildResult:
        """
        Build a schema-valid record from one cleaned row.

        Args:
            row: Cleaned and inferred row data.

        Returns:
            BuildResult containing the final record or a rejection reason.
        """
        user_type = self._resolve_user_type(row)

        if user_type == "school":
            return self._build_school_record(row)

        if user_type in {"student", "teacher", "parent", "admin"}:
            return self._build_non_school_record(row, user_type)

        return BuildResult(
            record=None,
            is_valid=False,
            reason="Invalid or missing userType; expected school, student, teacher, parent, or admin.",
        )

    def build_dataframe(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Build schema records for an entire dataframe.

        Args:
            dataframe: Cleaned dataframe.

        Returns:
            A tuple of:
                - valid schema dataframe
                - rejected rows dataframe with failure reasons
        """
        if dataframe.empty:
            return pd.DataFrame(), pd.DataFrame()

        valid_records: list[dict[str, Any]] = []
        rejected_records: list[dict[str, Any]] = []

        for index, row in dataframe.iterrows():
            result = self.build(row.to_dict())
            if result.is_valid and result.record is not None:
                valid_records.append(result.record)
            else:
                rejected_row = row.to_dict()
                rejected_row["failure_reason"] = result.reason
                rejected_row["row_index"] = int(index)
                rejected_records.append(rejected_row)

        return pd.DataFrame(valid_records), pd.DataFrame(rejected_records)

    def _build_school_record(self, row: Mapping[str, Any]) -> BuildResult:
        """
        Build a SchoolType-compatible record.

        Args:
            row: Cleaned input row.

        Returns:
            BuildResult with a school record or rejection.
        """
        school_type = self._normalize_school_type(row.get("schoolType"))
        display_name = self._normalize_display_name(row)

        if not school_type:
            return BuildResult(
                record=None,
                is_valid=False,
                reason="Missing or invalid schoolType for school userType.",
            )

        record = {
            "_id": self._optional_string(row.get("_id")),
            "email": self._required_string(row.get("email"), "email"),
            "name": display_name,
            "username": self._required_string(row.get("username"), "username"),
            "phoneNumber": self._normalize_phone_number(row.get("phoneNumber")),
            "location": self._normalize_location(row.get("location")),
            "profilePicture": self._normalize_profile_picture(row.get("profilePicture")),
            "socialLinks": self._normalize_social_links(row.get("socialLinks")),
            "verificationType": self._normalize_verification_type(row.get("verificationType")),
            "description": self._optional_string(row.get("description")),
            "preferences": self._normalize_preferences(row.get("preferences")),
            "blurhash": self._optional_string(row.get("blurhash")),
            "userType": "school",
            "last": None,
            "schoolType": school_type,
            "DOB": None,
            "gender": None,
        }

        missing_fields = self._collect_missing_required_fields(record, ["email", "name", "username", "profilePicture"])
        if missing_fields:
            return BuildResult(
                record=None,
                is_valid=False,
                reason=f"Missing required field(s): {', '.join(missing_fields)}",
            )

        return BuildResult(record=record, is_valid=True, reason="")

    def _build_non_school_record(self, row: Mapping[str, Any], user_type: str) -> BuildResult:
        """
        Build a NonSchoolType-compatible record.

        Args:
            row: Cleaned input row.
            user_type: One of student, teacher, parent, or admin.

        Returns:
            BuildResult with a non-school record or rejection.
        """
        display_name = self._normalize_display_name(row)
        last_name = self._required_string(row.get("last"), "last")
        if not last_name:
            last_name = self._derive_last_name(display_name)

        record = {
            "_id": self._optional_string(row.get("_id")),
            "email": self._required_string(row.get("email"), "email"),
            "name": display_name,
            "username": self._required_string(row.get("username"), "username"),
            "phoneNumber": self._normalize_phone_number(row.get("phoneNumber")),
            "location": self._normalize_location(row.get("location")),
            "profilePicture": self._normalize_profile_picture(row.get("profilePicture")),
            "socialLinks": self._normalize_social_links(row.get("socialLinks")),
            "verificationType": self._normalize_verification_type(row.get("verificationType")),
            "description": self._optional_string(row.get("description")),
            "preferences": self._normalize_preferences(row.get("preferences")),
            "blurhash": self._optional_string(row.get("blurhash")),
            "userType": user_type,
            "last": last_name,
            "schoolType": None,
            "DOB": self._normalize_dob(row.get("DOB")),
            "gender": self._normalize_gender(row.get("gender")),
        }

        missing_fields = self._collect_missing_required_fields(record, ["email", "name", "username", "profilePicture", "last"])
        if missing_fields:
            return BuildResult(
                record=None,
                is_valid=False,
                reason=f"Missing required field(s): {', '.join(missing_fields)}",
            )

        return BuildResult(record=record, is_valid=True, reason="")

    def _normalize_display_name(self, row: Mapping[str, Any]) -> str | None:
        """
        Build a full display name from the available name fields.

        Args:
            row: Input row.

        Returns:
            A full display name when a first-name signal exists.
        """
        name = self._optional_string(row.get("name"))
        last = self._optional_string(row.get("last"))

        if not name:
            return last or None

        if not last:
            return name

        name_parts = name.split()
        if name_parts and name_parts[-1].casefold() == last.casefold():
            return name

        return f"{name} {last}"

    def _derive_last_name(self, value: Any) -> str | None:
        """
        Derive a last name from a full name value.

        Args:
            value: Full name value from the source row.

        Returns:
            The final token of the name when present, otherwise None.
        """
        text = self._optional_string(value)
        if not text:
            return None

        parts = [part for part in text.split() if part]
        if len(parts) < 2:
            return None

        return parts[-1]

    def _resolve_user_type(self, row: Mapping[str, Any]) -> str | None:
        """
        Determine the final userType for a record.

        Args:
            row: Input row.

        Returns:
            A valid userType string or None.
        """
        raw_user_type = self._optional_string(row.get("userType"))
        if raw_user_type in self.ALLOWED_USER_TYPES:
            return raw_user_type

        if self._optional_string(row.get("schoolType")) in self.ALLOWED_SCHOOL_TYPES:
            return "school"

        if self._optional_string(row.get("last")):
            return "student"

        if self._optional_string(row.get("name")):
            return "student"

        return None

    def _normalize_school_type(self, value: Any) -> str | None:
        """
        Normalize schoolType to an allowed value.

        Args:
            value: Raw schoolType value.

        Returns:
            A valid school type or None.
        """
        text = self._optional_string(value)
        if text in self.ALLOWED_SCHOOL_TYPES:
            return text
        return None

    def _normalize_gender(self, value: Any) -> str | None:
        """
        Normalize gender to the allowed schema values.

        Args:
            value: Raw gender value.

        Returns:
            male, female, or None.
        """
        text = self._optional_string(value)
        if text in self.ALLOWED_GENDERS:
            return text
        return None

    def _normalize_dob(self, value: Any) -> int | None:
        """
        Normalize DOB to an integer Unix timestamp.

        Args:
            value: Raw DOB value.

        Returns:
            Unix timestamp or None.
        """
        return clean_dob(value)

    def _normalize_profile_picture(self, value: Any) -> str:
        """
        Normalize profilePicture and ensure a usable default.

        Args:
            value: Raw profile picture value.

        Returns:
            A non-empty URL string.
        """
        text = self._optional_string(value)
        if text:
            return text
        return "https://example.com/default-avatar.png"

    def _normalize_phone_number(self, value: Any) -> str | None:
        """
        Normalize phone number as a string.

        Args:
            value: Raw phone number value.

        Returns:
            Normalized phone number string or None.
        """
        text = self._optional_string(value)
        return text or None

    def _normalize_location(self, value: Any) -> dict[str, Any] | None:
        """
        Normalize a location object.

        Args:
            value: Raw location value.

        Returns:
            A normalized location dictionary or None.
        """
        if not isinstance(value, Mapping):
            return None

        wilaya = self._optional_string(value.get("wilaya"))
        commune = self._optional_string(value.get("commune"))
        full_location = self._optional_string(value.get("fullLocation"))

        coordinates = value.get("coordinates")
        if isinstance(coordinates, Mapping):
            lang_value = coordinates.get("lang")
            lat_value = coordinates.get("lat")
            coordinates_normalized = {
                "lang": self._to_float(lang_value),
                "lat": self._to_float(lat_value),
            }
        else:
            coordinates_normalized = {"lang": None, "lat": None}

        if not any([wilaya, commune, full_location]):
            return None

        return {
            "wilaya": wilaya,
            "commune": commune,
            "coordinates": coordinates_normalized,
            "fullLocation": full_location,
        }

    def _normalize_social_links(self, value: Any) -> dict[str, str] | None:
        """
        Normalize social links into a plain dictionary of strings.

        Args:
            value: Raw social links value.

        Returns:
            A dictionary of valid string links, or None.
        """
        if not isinstance(value, Mapping):
            return None

        normalized: dict[str, str] = {}
        for key, raw_value in value.items():
            text = self._optional_string(raw_value)
            if text:
                normalized[str(key)] = text

        return normalized or None

    def _normalize_verification_type(self, value: Any) -> str | None:
        """
        Normalize verificationType to the schema-supported values.

        Args:
            value: Raw verificationType value.

        Returns:
            'email', 'phone', or None.
        """
        text = self._optional_string(value)
        if text in {"email", "phone"}:
            return text
        return None

    def _normalize_preferences(self, value: Any) -> list[str] | None:
        """
        Normalize preferences into a list of strings.

        Args:
            value: Raw preferences value.

        Returns:
            A list of preference strings or None.
        """
        if value is None or value == "":
            return None

        if isinstance(value, list):
            cleaned = [self._optional_string(item) for item in value]
            filtered = [item for item in cleaned if item]
            return filtered or None

        text = self._optional_string(value)
        if not text:
            return None

        parts = [part.strip() for part in text.split(",")]
        filtered = [part for part in parts if part]
        return filtered or None

    def _required_string(self, value: Any, field_name: str) -> str | None:
        """
        Convert a required field to a stripped string.

        Args:
            value: Raw value.
            field_name: Field name used only for clarity.

        Returns:
            The cleaned string, or None if invalid.
        """
        text = self._optional_string(value)
        return text or None

    def _optional_string(self, value: Any) -> str:
        """
        Convert any value to a stripped string.

        Args:
            value: Raw input.

        Returns:
            A stripped string, or empty string if missing.
        """
        if value is None:
            return ""

        text = str(value).strip()
        if text.lower() in {"nan", "none", "null"}:
            return ""

        return text

    def _to_float(self, value: Any) -> float | None:
        """
        Convert a value to float safely.

        Args:
            value: Raw numeric value.

        Returns:
            Float or None.
        """
        try:
            if value is None or value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def _collect_missing_required_fields(self, record: Mapping[str, Any], required_fields: list[str]) -> list[str]:
        """
        Check which required fields are missing or empty.

        Args:
            record: Output record.
            required_fields: Required field names.

        Returns:
            A list of missing field names.
        """
        missing: list[str] = []
        for field_name in required_fields:
            value = record.get(field_name)
            if value is None:
                missing.append(field_name)
            elif isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing