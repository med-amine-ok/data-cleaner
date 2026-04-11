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
        default_user_type: str = "student",
        default_gender: str = "male",
        placeholder_phone_number: str | None = None,
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
        self._default_user_type = default_user_type
        self._default_gender = default_gender
        self._placeholder_phone_number = placeholder_phone_number

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

        parts = [part for part in base_name.split() if part]
        if not parts:
            return None

        if len(parts) == 1:
            username_source = parts[0]
        else:
            username_source = f"{parts[0]} {parts[-1]}"

        base_slug = self._slugify(username_source)
        if not base_slug:
            base_slug = "user"

        candidate = base_slug
        if candidate == "user":
            suffix = self._build_suffix(row_index)
            candidate = f"{candidate}_{suffix}" if suffix else candidate

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

    def infer_email(
        self,
        name: Any,
        existing_email: Any = None,
        row_index: int | None = None,
        existing_emails: set[str] | None = None,
    ) -> str | None:
        """
        Generate a placeholder email from a student's name when missing.

        Generated emails use the domain 'placeholder.local' so they are
        clearly identifiable as inferred values.

        Args:
            name: Student name used as the base for the email.
            existing_email: Existing email value, if already present.
            row_index: Optional row index used for uniqueness.
            existing_emails: Set of emails already used in the batch.

        Returns:
            A unique placeholder email, or the existing email if present.
        """
        normalized_existing = self._normalize_text(existing_email)
        if normalized_existing:
            return normalized_existing

        base_name = self._normalize_text(name)
        if not base_name:
            return None

        slug = self._slugify(base_name)
        if not slug:
            slug = "user"

        slug = slug.replace("_", ".")
        suffix = self._build_suffix(row_index)
        candidate = f"{slug}.{suffix}@gmail.com"

        if existing_emails is None:
            return candidate

        unique_candidate = candidate
        counter = 1
        while unique_candidate in existing_emails:
            unique_candidate = f"{slug}.{suffix}.{counter:02d}@gmail.com"
            counter += 1

        existing_emails.add(unique_candidate)
        return unique_candidate

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
        existing_emails: set[str] | None = None,
    ) -> dict[str, Any]:
        """
        Apply safe inference rules to a single row.

        Args:
            row: Input row as a mapping.
            row_index: Optional row index used for username generation.
            existing_usernames: Existing usernames already used in the batch.
            existing_emails: Existing emails already used in the batch.

        Returns:
            A new dictionary with inferred values filled in where appropriate.
        """
        inferred = dict(row)

        if self._is_missing_value(inferred.get("name")):
            inferred_name = self._infer_name_from_email(inferred.get("email"))
            if self._is_missing_value(inferred_name):
                inferred_name = self._infer_name_from_username(inferred.get("username"))
            if not self._is_missing_value(inferred_name):
                inferred["name"] = inferred_name

        composed_name = self._compose_full_name(inferred.get("name"), inferred.get("last"))
        if not self._is_missing_value(composed_name):
            inferred["name"] = composed_name

        inferred["username"] = self.infer_username(
            name=inferred.get("name"),
            existing_username=inferred.get("username"),
            row_index=row_index,
            existing_usernames=existing_usernames,
        )

        if self._is_missing_value(inferred.get("email")):
            fallback_name = inferred.get("name") or inferred.get("username")
            inferred["email"] = self.infer_email(
                name=fallback_name,
                existing_email=inferred.get("email"),
                row_index=row_index,
                existing_emails=existing_emails,
            )

        if self._is_missing_value(inferred.get("last")):
            inferred["last"] = self._infer_last_name(inferred.get("name"))

        if self._is_missing_value(inferred.get("profilePicture")):
            inferred["profilePicture"] = self.infer_profile_picture()

        if self._is_missing_value(inferred.get("gender")):
            inferred["gender"] = self.infer_gender(inferred.get("name"))
        if self._is_missing_value(inferred.get("gender")):
            inferred["gender"] = self._default_gender

        if self._is_missing_value(inferred.get("userType")):
            inferred["userType"] = self._default_user_type

        if self._is_missing_value(inferred.get("DOB")):
            inferred["DOB"] = self._infer_dob_from_age(inferred.get("age"))

        if self._is_missing_value(inferred.get("phoneNumber")) and self._placeholder_phone_number:
            inferred["phoneNumber"] = self._placeholder_phone_number

        if self._is_missing_value(inferred.get("schoolType")) and inferred.get("userType") == "school":
            inferred["schoolType"] = "public"

        return inferred

    def _compose_full_name(self, name: Any, last: Any) -> str | None:
        """
        Combine first-name and last-name signals into a full display name.

        Args:
            name: Existing display name or first-name value.
            last: Existing last-name value.

        Returns:
            A full display name when a first-name signal exists.
        """
        normalized_name = self._normalize_text(name)
        if not normalized_name:
            return None

        normalized_last = self._normalize_text(last)
        if not normalized_last:
            return normalized_name

        name_parts = normalized_name.split()
        if name_parts and name_parts[-1].casefold() == normalized_last.casefold():
            return normalized_name

        return f"{normalized_name} {normalized_last}"

    def _infer_last_name(self, name: Any) -> str | None:
        """
        Infer a last name from a full display name.

        Args:
            name: Full name value.

        Returns:
            The last token of the name when available. If there is only one
            word, returns that same word. Returns None if completely empty.
        """
        normalized_name = self._normalize_text(name)
        if not normalized_name:
            return None

        parts = normalized_name.split()
        if len(parts) == 0:
            return None
        
        # If it's a single word name, duplicate the name for last name,
        # otherwise we'd get a schema rejection for missing last name.
        if len(parts) == 1:
            return parts[0]

        return parts[-1]

    def _infer_name_from_email(self, value: Any) -> str | None:
        text = self._normalize_text(value)
        if not text or "@" not in text:
            return None
        local_part = text.split("@", 1)[0]
        local_part = local_part.replace(".", " ").replace("_", " ")
        local_part = re.sub(r"\d+", "", local_part).strip()
        return local_part or None

    def _infer_name_from_username(self, value: Any) -> str | None:
        text = self._normalize_text(value)
        if not text:
            return None
        candidate = text.replace(".", " ").replace("_", " ")
        candidate = re.sub(r"\d+", "", candidate).strip()
        return candidate or None

    def _infer_dob_from_age(self, value: Any) -> int | None:
        if value is None:
            return None
        try:
            age = int(float(value))
        except (TypeError, ValueError):
            return None
        if age < 3 or age > 120:
            return None
        now = pd.Timestamp.utcnow()
        estimated_year = now.year - age
        estimated = pd.Timestamp(year=estimated_year, month=1, day=1, tz="UTC")
        return int(estimated.timestamp())

    def _is_missing_value(self, value: Any) -> bool:
        """
        Check whether a value should be treated as missing.

        Args:
            value: Input value.

        Returns:
            True if the value is missing, empty, or otherwise unusable.
        """
        if value is None:
            return True

        if isinstance(value, pd.Series):
            return value.empty or value.isna().all() or (value.astype(str).str.strip() == "").all()

        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass

        if isinstance(value, str):
            return not value.strip() or value.strip().lower() in {"nan", "none", "null"}

        return False

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
        
        # Convert all columns to object to prevent pandas from silently coercing 
        # inferred strings to NaN when inserting into a float64 (all NaN) column.
        for col in working.columns:
            working[col] = working[col].astype(object)

        existing_usernames: set[str] = set()
        existing_emails: set[str] = set()

        if "username" in working.columns:
            for value in working["username"].dropna().astype(str).tolist():
                normalized = self._normalize_text(value)
                if normalized:
                    existing_usernames.add(normalized)

        if "email" in working.columns:
            for value in working["email"].dropna().astype(str).tolist():
                normalized = self._normalize_text(value)
                if normalized:
                    existing_emails.add(normalized)

        for row_index, row in working.iterrows():
            inferred_row = self.infer_row(
                row=row.to_dict(),
                row_index=int(row_index),
                existing_usernames=existing_usernames,
                existing_emails=existing_emails,
            )

            for column, value in inferred_row.items():
                if column not in working.columns:
                    working[column] = None
                current_value = working.at[row_index, column]
                if self._is_missing_value(current_value):
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

    def _normalize_text(self, value: Any) -> str | None:
        """
        Normalize incoming string data.
        """
        if pd.isna(value) or value is None:
            return None
        text = str(value).strip()
        if text.lower() == "nan":
            return None
        if not text:
            return None
        return text