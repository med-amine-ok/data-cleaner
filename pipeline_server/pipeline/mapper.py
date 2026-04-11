from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process


@dataclass(frozen=True)
class MappingResult:
    """
    Store the result of a column mapping decision.

    Attributes:
        original_column: The raw incoming column name.
        mapped_column: The canonical field name chosen for the pipeline.
        score: Similarity score used to make the decision.
        matched_alias: The alias or candidate that matched best.
    """
    original_column: str
    mapped_column: str
    score: float
    matched_alias: str


class ColumnAutoMapper:
    """
    Map messy incoming column names to canonical schema field names.

    The mapper uses a combination of direct alias lookup and fuzzy matching
    to support Arabic, French, and English column headers.
    """

    def __init__(
        self,
        field_aliases: Mapping[str, list[str]] | None = None,
        threshold: float = 85.0,
    ) -> None:
        """
        Initialize the mapper.

        Args:
            field_aliases: Mapping of canonical field names to alias lists.
            threshold: Minimum fuzzy score required to accept a mapping.
        """
        self._field_aliases: dict[str, list[str]] = {
            key: [self._normalize_text(alias) for alias in value]
            for key, value in (field_aliases or {}).items()
        }
        self._threshold = threshold
        self._alias_to_field = self._build_alias_lookup(self._field_aliases)

    @classmethod
    def from_json(cls, file_path: str | Path, threshold: float = 85.0) -> ColumnAutoMapper:
        """
        Create a mapper from a JSON alias file.

        Args:
            file_path: Path to the JSON file containing aliases.
            threshold: Minimum fuzzy score required to accept a mapping.

        Returns:
            A configured ColumnAutoMapper instance.
        """
        path = Path(file_path)
        data = json.loads(path.read_text(encoding="utf-8"))

        aliases = data.get("FIELD_ALIASES", data)
        if not isinstance(aliases, dict):
            raise ValueError("Invalid aliases JSON format. Expected a FIELD_ALIASES dictionary.")

        return cls(field_aliases=aliases, threshold=threshold)

    def map_columns(self, dataframe: pd.DataFrame) -> tuple[pd.DataFrame, list[MappingResult]]:
        """
        Rename dataframe columns to canonical field names.

        Args:
            dataframe: Input dataframe with messy column headers.

        Returns:
            A tuple containing:
                - The dataframe with renamed columns.
                - A list of mapping decisions for audit/debugging.
        """
        renamed = dataframe.copy()
        results: list[MappingResult] = []
        new_columns: dict[str, str] = {}
        normalized_columns = {self._normalize_text(str(column)) for column in renamed.columns}

        first_name_markers = {
            "prenom",
            "prenom",
            "firstname",
            "first_name",
            "first name",
            "fname",
            "forename",
            "givenname",
            "given_name",
            "given name",
            "fstname",
            "fristname",
        }
        has_first_name_signal = any(marker in normalized_columns for marker in first_name_markers)

        for original_column in renamed.columns:
            canonical, score, alias = self._map_single_column(
                str(original_column),
                has_first_name_signal=has_first_name_signal,
            )
            if canonical is not None:
                new_columns[str(original_column)] = canonical
                results.append(
                    MappingResult(
                        original_column=str(original_column),
                        mapped_column=canonical,
                        score=score,
                        matched_alias=alias,
                    )
                )

        renamed = renamed.rename(columns=new_columns)
        renamed = self._collapse_duplicate_columns(renamed)
        return renamed, results

    def _map_single_column(
        self,
        column_name: str,
        has_first_name_signal: bool = False,
    ) -> tuple[str | None, float, str]:
        """
        Map one raw column name to a canonical field.

        Args:
            column_name: Raw input column name.

        Returns:
            A tuple of (canonical field or None, score, matched alias).
        """
        normalized = self._normalize_text(column_name)

        # In French-style sheets, `nom` is usually the family name when
        # a separate first-name column (e.g. `prenom`) is present.
        if has_first_name_signal and normalized in {"nom", "nom de famille"}:
            return "last", 100.0, "french_name_rule"

        heuristic_field = self._heuristic_name_field(normalized)
        if heuristic_field is not None:
            return heuristic_field, 99.0, "name_header_heuristic"

        if normalized in self._alias_to_field:
            canonical = self._alias_to_field[normalized]
            return canonical, 100.0, normalized

        # Very short headers are often ambiguous (e.g., 'lat' vs 'last').
        # Only accept them when they match an alias exactly.
        if len(normalized) <= 3:
            return None, 0.0, ""

        all_aliases = list(self._alias_to_field.keys())
        best_match = process.extractOne(
            normalized,
            all_aliases,
            scorer=fuzz.ratio,
        )

        if best_match is None:
            return None, 0.0, ""

        matched_alias, score, _ = best_match
        if float(score) < self._threshold:
            return None, float(score), matched_alias

        canonical = self._alias_to_field[matched_alias]
        return canonical, float(score), matched_alias

    def _heuristic_name_field(self, normalized: str) -> str | None:
        """
        Heuristically classify common first/last-name header variants.

        This catches frequent typos such as 'fstname' and 'laname' that may
        not be present in aliases but should still map deterministically.
        """
        compact = "".join(ch for ch in normalized if ch.isalnum())
        if not compact:
            return None

        first_name_tokens = {
            "fname",
            "firstname",
            "firstn",
            "fristname",
            "fstname",
            "forename",
            "givenname",
            "prenom",
            "prenomnom",
        }
        last_name_tokens = {
            "lname",
            "lastname",
            "lastnamee",
            "lastn",
            "laname",
            "lasname",
            "sirname",
            "surname",
            "familyname",
            "nomdefamille",
        }

        if compact in first_name_tokens:
            return "name"

        if compact in last_name_tokens:
            return "last"

        return None

    def _build_alias_lookup(self, field_aliases: Mapping[str, list[str]]) -> dict[str, str]:
        """
        Build a reverse lookup from alias to canonical field.

        Args:
            field_aliases: Canonical field mapping.

        Returns:
            A dictionary mapping alias -> canonical field.
        """
        alias_lookup: dict[str, str] = {}
        for canonical_field, aliases in field_aliases.items():
            alias_lookup[self._normalize_text(canonical_field)] = canonical_field
            for alias in aliases:
                alias_lookup[self._normalize_text(alias)] = canonical_field
        return alias_lookup

    def _normalize_text(self, value: Any) -> str:
        """
        Normalize a text value for comparison.

        Args:
            value: Input value to normalize.

        Returns:
            A lowercase, trimmed, whitespace-collapsed string.
        """
        text = "" if value is None else str(value)
        text = text.replace("\ufffd", " ")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.strip().lower()
        text = " ".join(text.split())
        return text

    def _collapse_duplicate_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Merge duplicate canonical columns after renaming.

        Args:
            dataframe: Renamed dataframe that may contain duplicate column labels.

        Returns:
            A dataframe with unique column names, keeping the first non-empty value
            across duplicate columns.
        """
        if dataframe.columns.is_unique:
            return dataframe

        collapsed = pd.DataFrame(index=dataframe.index)
        seen: set[str] = set()

        for column_name in dataframe.columns:
            column_key = str(column_name)
            if column_key in seen:
                continue
            seen.add(column_key)

            subset = dataframe.loc[:, dataframe.columns == column_name]
            if isinstance(subset, pd.Series):
                collapsed[column_key] = subset
                continue

            cleaned = subset.replace("", pd.NA).bfill(axis=1).iloc[:, 0]
            collapsed[column_key] = cleaned.fillna("")

        return collapsed