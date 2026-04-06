from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

import pandas as pd
from gender_guesser.detector import Detector


class MissingValueInferer:
    """
    Infer safe missing values for student records.

    This class only fills values that are low-risk to infer, such as usernames,
    placeholder profile pictures, and optionally gender from a name signal.
    It does not overwrite valid existing values.

    Methods:
        infer_username: Generate a username if one is missing.
        infer_profile_picture: Return a default placeholder profile picture URL.
        infer_gender: Infer gender from a name when confidence is strong enough.
        infer_row: Apply inference logic to one row.
        infer_dataframe: Apply inference logic to an entire dataframe.
    """

    def __init__(
        self,
        placeholder_profile_picture_url: str = "https://example.com/default-avatar.png",
        username_suffix_length: int = 4,
    ) -> None:
        """
        Initialize the inferer.

        Args:
            placeholder_profile_picture_url: Default profile picture URL used when missing.
            username_suffix_length: Number of digits used in the generated username suffix.
        """
        self._placeholder_profile_picture_url = placeholder_profile_picture_url
        self._username_suffix_length = username_suffix_length
        self._gender_detector = Detector(case_sensitive=False)

    def infer_username(
        self,
        name: Any,
        existing_username: Any = None,
        row_index: int | None = None,
        existing_usernames: set[str] | None = None,
    ) -> str | None:
        """
        Generate a safe username from a student's name.

        Args:
            name: Student name used as the base for the username.
            existing_username: Existing username value, if already present.
            row_index: Optional row index used as a stable fallback suffix.
            existing_usernames: Set of usernames already taken in the batch.

        Returns:
            A unique slug-like username, or the existing username if present.
        """
        normalized_existing = self._normalize_text(existing_username)
        if normalized_existing:
            return normalized_existing

        base_name = self._normalize_text(name)
        if not base_name:
            return None

        base_slug = self._slugify(base_name)
        if not base_slug:
            return None

        suffix = self._build_suffix(row_index)
        candidate = f"{base_slug}_{suffix}" if suffix else base_slug

        if existing_usernames is None:
            return candidate

        unique_candidate = candidate
        counter = 1
        while unique_candidate in existing_usernames:
            unique_candidate = f"{candidate}_{counter:02d}"
            counter += 1

        existing_usernames.add(unique_candidate)
        return unique_candidate

    def infer_profile_picture(self) -> str:
        """
        Return a default profile picture URL.

        Returns:
            A fallback profile picture URL.
        """
        return self._placeholder_profile_picture_url

    def infer_gender(self, name: Any) -> str | None:
        """
        Infer gender from a name using the gender-guesser library.

        Args:
            name: Student name used as the inference signal.

        Returns:
            'male', 'female', or None if the signal is weak or unknown.
        """
        normalized_name = self._normalize_text(name)
        if not normalized_name:
            return None

        first_token = normalized_name.split(" ")[0]
        if not first_token:
            return None

        guessed = self._gender_detector.get_gender(first_token)

        if guessed in {"male", "mostly_male"}:
            return "male"

        if guessed in {"female", "mostly_female"}:
            return "female"

        return None

    def infer_row(
        self,
        row: Mapping[str, Any],
        row_index: int | None = None,
        existing_usernames: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Apply safe inference rules to a single row.

        Args:
            row: Input row as a mapping.
            row_index: Optional row index used for username generation.
            existing_usernames: Existing usernames already used in the batch.

        Returns:
            A new dictionary with inferred values filled in where appropriate.
        """
        inferred = dict(row)

        inferred["username"] = self.infer_username(
            name=inferred.get("name"),
            existing_username=inferred.get("username"),
            row_index=row_index,
            existing_usernames=existing_usernames,
        )

        if not inferred.get("profilePicture"):
            inferred["profilePicture"] = self.infer_profile_picture()

        if not inferred.get("gender"):
            inferred["gender"] = self.infer_gender(inferred.get("name"))

        return inferred

    def infer_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Apply inference to an entire dataframe.

        Args:
            dataframe: Input dataframe after cleaning.

        Returns:
            A new dataframe with safe inferred values filled in.
        """
        if dataframe.empty:
            return dataframe.copy()

        working = dataframe.copy()
        existing_usernames: set[str] = set()

        if "username" in working.columns:
            for value in working["username"].dropna().astype(str).tolist():
                normalized = self._normalize_text(value)
                if normalized:
                    existing_usernames.add(normalized)

        for row_index, row in working.iterrows():
            inferred_row = self.infer_row(
                row=row.to_dict(),
                row_index=int(row_index),
                existing_usernames=existing_usernames,
            )

            for column, value in inferred_row.items():
                if column not in working.columns:
                    working[column] = None
                if pd.isna(working.at[row_index, column]) or working.at[row_index, column] in ("", None):
                    working.at[row_index, column] = value

        return working

    def _build_suffix(self, row_index: int | None) -> str:
        """
        Build a numeric suffix for generated usernames.

        Args:
            row_index: Optional row index.

        Returns:
            A zero-padded suffix string.
        """
        if row_index is None:
            return "0000"

        modulus = 10 ** self._username_suffix_length
        suffix_number = abs(int(row_index)) % modulus
        return str(suffix_number).zfill(self._username_suffix_length)

    def _slugify(self, value: str) -> str:
        """
        Convert text into a safe slug for usernames.

        Args:
            value: Raw text value.

        Returns:
            A lowercase ASCII-only slug with separators collapsed.
        """
        normalized = unicodedata.normalize("NFKD", value)
        ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
        ascii_text = ascii_text.lower()
        ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
        ascii_text = re.sub(r"_+", "_", ascii_text).strip("_")
        return ascii_text

    def _normalize_text(self, value: Any) -> str:
        """
        Normalize an input value into a stripped string.

        Args:
            value: Raw input value.

        Returns:
            A normalized string, or an empty string if the input is missing.
        """
        if value is None:
            return ""

        text = str(value).strip()
        if not text:
            return ""

        return text