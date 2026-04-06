from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chardet
import pandas as pd


class FileIngester:
    """
    Read uploaded spreadsheet or data files into a Pandas DataFrame.

    This class supports CSV, JSON, XLSX, and ODS files. It normalizes the
    input into a DataFrame and forces all values to string form initially so
    later cleaning steps can safely handle messy data.

    Methods:
        read(file_path): Load the file at the given path and return a DataFrame.
    """

    def read(self, file_path: str) -> pd.DataFrame:
        """
        Read a file from disk and convert it into a DataFrame.

        Args:
            file_path: Path to the uploaded file on disk.

        Returns:
            A Pandas DataFrame containing the file contents with string columns.

        Raises:
            ValueError: If the file extension is not supported or the file cannot
                be read successfully.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return self._read_csv(path)

        if suffix == ".json":
            return self._read_json(path)

        if suffix in {".xlsx", ".ods"}:
            return self._read_excel(path)

        raise ValueError(
            f"Unsupported file format: {suffix}. Supported formats are .csv, .json, .xlsx, and .ods."
        )

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """
        Read a CSV file with encoding detection and string-safe loading.

        Args:
            path: Path to the CSV file.

        Returns:
            A DataFrame with all values loaded as strings.
        """
        raw_bytes = path.read_bytes()
        encoding = self._detect_encoding(raw_bytes)

        dataframe = pd.read_csv(
            path,
            encoding=encoding,
            dtype=str,
            keep_default_na=False,
            na_values=[],
        )
        return self._normalize_dataframe(dataframe)

    def _read_json(self, path: Path) -> pd.DataFrame:
        """
        Read a JSON file and convert it into a DataFrame.

        Args:
            path: Path to the JSON file.

        Returns:
            A DataFrame with all values loaded as strings.
        """
        raw_text = path.read_text(encoding=self._guess_text_encoding(path))
        parsed: Any = json.loads(raw_text)

        dataframe = pd.DataFrame(parsed)
        return self._normalize_dataframe(dataframe)

    def _read_excel(self, path: Path) -> pd.DataFrame:
        """
        Read an Excel or ODS file and convert it into a DataFrame.

        Args:
            path: Path to the spreadsheet file.

        Returns:
            A DataFrame with all values loaded as strings.
        """
        dataframe = pd.read_excel(path, dtype=str)
        return self._normalize_dataframe(dataframe)

    def _normalize_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize a DataFrame so all columns and values are string-based.

        Args:
            dataframe: Input DataFrame.

        Returns:
            Cleaned DataFrame with stripped column names and string values.
        """
        normalized = dataframe.copy()

        normalized.columns = [str(column).strip() for column in normalized.columns]

        for column in normalized.columns:
            normalized[column] = normalized[column].astype(str).replace({"nan": "", "None": ""}).str.strip()

        return normalized

    def _detect_encoding(self, raw_bytes: bytes) -> str:
        """
        Detect the likely text encoding of raw file bytes.

        Args:
            raw_bytes: Raw bytes from the uploaded file.

        Returns:
            A best-effort encoding name, defaulting to UTF-8 if detection fails.
        """
        detection = chardet.detect(raw_bytes)
        encoding = detection.get("encoding")
        return encoding or "utf-8"

    def _guess_text_encoding(self, path: Path) -> str:
        """
        Guess the encoding of a text-based file using a small sample.

        Args:
            path: Path to the text file.

        Returns:
            A best-effort encoding name.
        """
        raw_bytes = path.read_bytes()
        return self._detect_encoding(raw_bytes)