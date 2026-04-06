from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from rapidfuzz import fuzz


@dataclass(frozen=True)
class DuplicateMatch:
    """
    Represent a possible duplicate pair.

    Attributes:
        left_index: Index of the first row.
        right_index: Index of the second row.
        score: Composite duplicate score between 0 and 100.
        reason: Text explanation for why the rows were linked.
    """
    left_index: int
    right_index: int
    score: float
    reason: str


class SmartDeduplicator:
    """
    Detect duplicate or near-duplicate student records.

    The deduplicator first looks for exact email matches, then computes a
    composite fuzzy score using name, phone, DOB, and email. Blocking by wilaya
    is used to reduce unnecessary comparisons.
    """

    def __init__(
        self,
        exact_email_weight: float = 1.0,
        name_weight: float = 0.4,
        phone_weight: float = 0.3,
        dob_weight: float = 0.2,
        email_weight: float = 0.1,
        threshold: float = 85.0,
    ) -> None:
        """
        Initialize the deduplicator.

        Args:
            exact_email_weight: Weight used when email matches exactly.
            name_weight: Weight for name similarity in the composite score.
            phone_weight: Weight for phone similarity in the composite score.
            dob_weight: Weight for DOB similarity in the composite score.
            email_weight: Weight for email similarity in the composite score.
            threshold: Minimum score required to treat two rows as duplicates.
        """
        self._exact_email_weight = exact_email_weight
        self._name_weight = name_weight
        self._phone_weight = phone_weight
        self._dob_weight = dob_weight
        self._email_weight = email_weight
        self._threshold = threshold

    def find_duplicates(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Find likely duplicates in a dataframe.

        Args:
            dataframe: Cleaned or partially cleaned student dataframe.

        Returns:
            A tuple of:
                - dataframe containing canonical records with duplicate metadata
                - dataframe containing duplicate candidates for review
        """
        if dataframe.empty:
            return dataframe.copy(), pd.DataFrame()

        working = dataframe.copy()
        working["_duplicate_group_id"] = ""
        working["_duplicate_score"] = 0.0
        working["_duplicate_reason"] = ""

        duplicate_rows: list[dict[str, Any]] = []
        group_counter = 1

        if "email" in working.columns:
            email_groups = working[working["email"].notna() & (working["email"] != "")].groupby("email")

            for _, group in email_groups:
                if len(group) <= 1:
                    continue

                group_id = f"dup_{group_counter:06d}"
                group_counter += 1

                canonical_index = int(group.index[0])
                for row_index in group.index:
                    working.at[row_index, "_duplicate_group_id"] = group_id
                    working.at[row_index, "_duplicate_score"] = 100.0
                    working.at[row_index, "_duplicate_reason"] = "exact_email_match"

                for row_index in group.index[1:]:
                    duplicate_rows.append(self._row_to_dict(working.loc[row_index]))
                    duplicate_rows[-1]["duplicate_group_id"] = group_id
                    duplicate_rows[-1]["duplicate_reason"] = "exact_email_match"
                    duplicate_rows[-1]["duplicate_score"] = 100.0
                    duplicate_rows[-1]["canonical_row_index"] = canonical_index

            working = working.drop_duplicates(subset=["email"], keep="first")

        duplicates = self._fuzzy_duplicate_pass(working, group_counter, duplicate_rows)

        if duplicates:
            duplicate_frame = pd.DataFrame(duplicates)
        else:
            duplicate_frame = pd.DataFrame()

        canonical_columns = list(working.columns)
        if "_duplicate_group_id" not in canonical_columns:
            canonical_columns.extend(["_duplicate_group_id", "_duplicate_score", "_duplicate_reason"])

        return working[canonical_columns], duplicate_frame

    def _fuzzy_duplicate_pass(
        self,
        dataframe: pd.DataFrame,
        start_group_counter: int,
        duplicate_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Run fuzzy duplicate detection on the remaining dataframe rows.

        Args:
            dataframe: Dataframe after exact-email deduplication.
            start_group_counter: Starting counter for duplicate group IDs.
            duplicate_rows: Existing list of duplicate row dictionaries.

        Returns:
            The updated list of duplicate row dictionaries.
        """
        duplicates = duplicate_rows[:]
        group_counter = start_group_counter

        candidate_rows = list(dataframe.index)
        for left_pos, left_index in enumerate(candidate_rows):
            left_row = dataframe.loc[left_index]
            left_block = self._blocking_key(left_row)

            for right_index in candidate_rows[left_pos + 1 :]:
                right_row = dataframe.loc[right_index]

                if left_block and self._blocking_key(right_row) and left_block != self._blocking_key(right_row):
                    continue

                score = self._composite_score(left_row, right_row)
                if score < self._threshold:
                    continue

                group_id = f"dup_{group_counter:06d}"
                group_counter += 1

                reason = self._build_reason(left_row, right_row, score)

                left_record = self._row_to_dict(left_row)
                left_record["duplicate_group_id"] = group_id
                left_record["duplicate_reason"] = reason
                left_record["duplicate_score"] = score
                left_record["canonical_row_index"] = int(left_index)
                duplicates.append(left_record)

                right_record = self._row_to_dict(right_row)
                right_record["duplicate_group_id"] = group_id
                right_record["duplicate_reason"] = reason
                right_record["duplicate_score"] = score
                right_record["canonical_row_index"] = int(left_index)
                duplicates.append(right_record)

        return duplicates

    def _blocking_key(self, row: pd.Series) -> str:
        """
        Create a blocking key to limit duplicate comparisons.

        Args:
            row: A dataframe row.

        Returns:
            A coarse key based on wilaya or location.
        """
        wilaya_value = str(row.get("wilaya", "") or row.get("location", "") or "").strip().lower()
        if not wilaya_value:
            return ""
        return wilaya_value[:8]

    def _composite_score(self, left_row: pd.Series, right_row: pd.Series) -> float:
        """
        Compute a weighted duplicate similarity score.

        Args:
            left_row: First student row.
            right_row: Second student row.

        Returns:
            Composite similarity score from 0 to 100.
        """
        name_score = self._string_similarity(left_row.get("name"), right_row.get("name"))
        phone_score = self._string_similarity(left_row.get("phoneNumber"), right_row.get("phoneNumber"))
        dob_score = self._dob_similarity(left_row.get("DOB"), right_row.get("DOB"))
        email_score = self._string_similarity(left_row.get("email"), right_row.get("email"))

        weighted = (
            name_score * self._name_weight
            + phone_score * self._phone_weight
            + dob_score * self._dob_weight
            + email_score * self._email_weight
        )

        return float(weighted)

    def _string_similarity(self, left_value: Any, right_value: Any) -> float:
        """
        Compute similarity between two string-like values.

        Args:
            left_value: First value.
            right_value: Second value.

        Returns:
            Similarity score from 0 to 100.
        """
        left_text = "" if left_value is None else str(left_value).strip().lower()
        right_text = "" if right_value is None else str(right_value).strip().lower()

        if not left_text or not right_text:
            return 0.0

        if left_text == right_text:
            return 100.0

        return float(fuzz.ratio(left_text, right_text))

    def _dob_similarity(self, left_value: Any, right_value: Any) -> float:
        """
        Compare DOB values for duplicate detection.

        Args:
            left_value: First DOB value.
            right_value: Second DOB value.

        Returns:
            Similarity score from 0 to 100.
        """
        if left_value is None or right_value is None:
            return 0.0

        left_text = str(left_value).strip()
        right_text = str(right_value).strip()

        if not left_text or not right_text:
            return 0.0

        return 100.0 if left_text == right_text else 0.0

    def _build_reason(self, left_row: pd.Series, right_row: pd.Series, score: float) -> str:
        """
        Build a human-readable duplicate reason.

        Args:
            left_row: First row.
            right_row: Second row.
            score: Duplicate score.

        Returns:
            Explanation string for logging or quarantine output.
        """
        reasons: list[str] = []

        if str(left_row.get("email", "")).strip().lower() == str(right_row.get("email", "")).strip().lower():
            reasons.append("email_match")

        if self._string_similarity(left_row.get("name"), right_row.get("name")) >= 90.0:
            reasons.append("name_match")

        if self._string_similarity(left_row.get("phoneNumber"), right_row.get("phoneNumber")) >= 90.0:
            reasons.append("phone_match")

        if self._dob_similarity(left_row.get("DOB"), right_row.get("DOB")) >= 100.0:
            reasons.append("dob_match")

        reasons.append(f"score={score:.2f}")
        return ",".join(reasons)

    def _row_to_dict(self, row: pd.Series) -> dict[str, Any]:
        """
        Convert a pandas row into a regular dictionary.

        Args:
            row: Pandas Series row.

        Returns:
            Dictionary representation of the row.
        """
        return {key: row.get(key) for key in row.index}