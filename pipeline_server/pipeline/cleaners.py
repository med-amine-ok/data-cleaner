from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
import arabic_reshaper
import phonenumbers
from bidi.algorithm import get_display
from dateutil import parser as date_parser
from gender_guesser.detector import Detector
from rapidfuzz import fuzz, process

EMAIL_DOMAIN_TYPOS: dict[str, str] = {
    "gmal.com": "gmail.com",
    "gmial.com": "gmail.com",
    "gmail.con": "gmail.com",
    "gmai.com": "gmail.com",
    "hotmial.com": "hotmail.com",
    "hotmai.com": "hotmail.com",
    "yaho.com": "yahoo.com",
    "outlok.com": "outlook.com",
    "iclod.com": "icloud.com",
    "protonmal.com": "protonmail.com",
}

GENDER_ALIASES: dict[str, str] = {
    "male": "male",
    "m": "male",
    "man": "male",
    "boy": "male",
    "ذكر": "male",
    "malee": "male",
    "female": "female",
    "f": "female",
    "woman": "female",
    "girl": "female",
    "أنثى": "female",
    "femelle": "female",
    "femme": "female",
}

ARABIC_INDIC_DIGITS: dict[int, str] = {
    ord("٠"): "0",
    ord("١"): "1",
    ord("٢"): "2",
    ord("٣"): "3",
    ord("٤"): "4",
    ord("٥"): "5",
    ord("٦"): "6",
    ord("٧"): "7",
    ord("٨"): "8",
    ord("٩"): "9",
    ord("۰"): "0",
    ord("۱"): "1",
    ord("۲"): "2",
    ord("۳"): "3",
    ord("۴"): "4",
    ord("۵"): "5",
    ord("۶"): "6",
    ord("۷"): "7",
    ord("۸"): "8",
    ord("۹"): "9",
}

USER_TYPE_ALIASES: dict[str, str] = {
    # Students
    "student": "student", "eleve": "student", "élève": "student", "etudiant": "student",
    "étudiant": "student", "طالب": "student", "تلميذ": "student",
    # Teachers
    "teacher": "teacher", "prof": "teacher", "professeur": "teacher",
    "enseignant": "teacher", "أستاذ": "teacher", "معلم": "teacher",
    # Parents
    "parent": "parent", "father": "parent", "mother": "parent", "pere": "parent",
    "père": "parent", "mere": "parent", "mère": "parent", "ولي": "parent", "أب": "parent", "أم": "parent",
    # Schools
    "school": "school", "ecole": "school", "école": "school", "etablissement": "school",
    "مدرسة": "school", "مؤسسة": "school",
    # Admins
    "admin": "admin", "administrateur": "admin", "مدير": "admin", "مشرف": "admin"
}

SCHOOL_TYPE_ALIASES: dict[str, str] = {
    "private": "private", "prive": "private", "privé": "private", "خاص": "private", "خاصة": "private",
    "language": "language", "langue": "language", "ecole de langue": "language", "لغات": "language", "مدرسة لغات": "language",
    "university": "university", "universite": "university", "université": "university", "جامعة": "university",
    "formation": "formation", "centre de formation": "formation", "تكوين": "formation", "مركز تكوين": "formation",
    "public": "public", "etatique": "public", "عام": "public", "حكومي": "public",
    "support": "support", "soutien": "support", "cours de soutien": "support", "دعم": "support", "دروس دعم": "support",
    "private-university": "private-university", "universite prive": "private-university", "جامعة خاصة": "private-university",
    "preschool": "preschool", "creche": "preschool", "maternelle": "preschool", "روضة": "preschool", "تحضيري": "preschool"
}

EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+$"
)

ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
WHITESPACE_RE = re.compile(r"\s+")
NON_USERNAME_RE = re.compile(r"[^a-z0-9_]+")


def normalize_text(value: Any) -> str:
    """
    Normalize an arbitrary value into a clean text string.

    Args:
        value: Raw input value from the spreadsheet.

    Returns:
        A normalized string with Unicode normalized, Arabic-Indic digits converted,
        and whitespace collapsed.
    """
    if value is None:
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_INDIC_DIGITS)
    text = text.replace("\u200f", " ").replace("\u200e", " ")
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def has_arabic_text(value: Any) -> bool:
    """
    Check whether a value contains Arabic script characters.

    Args:
        value: Raw text value.

    Returns:
        True if the text contains Arabic characters, otherwise False.
    """
    return bool(ARABIC_RE.search(normalize_text(value)))


def smart_title(value: Any) -> str:
    """
    Title-case a Latin-script name in a safer way than str.title().

    Args:
        value: Raw name value.

    Returns:
        A cleaned title-cased string.
    """
    text = normalize_text(value).lower()
    if not text:
        return ""

    words = text.split(" ")
    titled_words: list[str] = []

    for word in words:
        if not word:
            continue
        hyphen_parts = word.split("-")
        cleaned_hyphen_parts: list[str] = []

        for hyphen_part in hyphen_parts:
            apostrophe_parts = hyphen_part.split("'")
            cleaned_apostrophe_parts: list[str] = []

            for segment in apostrophe_parts:
                if not segment:
                    cleaned_apostrophe_parts.append(segment)
                else:
                    cleaned_apostrophe_parts.append(segment[:1].upper() + segment[1:].lower())

            cleaned_hyphen_parts.append("'".join(cleaned_apostrophe_parts))

        titled_words.append("-".join(cleaned_hyphen_parts))

    return " ".join(titled_words)


def _clean_display_text(value: Any) -> str:
    """
    Clean a human-readable text field while preserving script-specific rendering.

    Args:
        value: Raw input text.

    Returns:
        Cleaned display text.
    """
    text = normalize_text(value)
    if not text:
        return ""

    if has_arabic_text(text):
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)

    return smart_title(text)


def clean_name(value: Any) -> str | None:
    """
    Clean a person's display name.

    Args:
        value: Raw name value.

    Returns:
        A cleaned display name, or None if the input is empty.
    """
    cleaned = _clean_display_text(value)
    return cleaned or None


def clean_last(value: Any) -> str | None:
    """
    Clean a person's last (family) name.

    Args:
        value: Raw last name value.

    Returns:
        A cleaned last name string, or None if the input is empty.
    """
    cleaned = _clean_display_text(value)
    return cleaned or None


def clean_username(value: Any) -> str | None:
    """
    Normalize a username without generating a new one.

    Args:
        value: Raw username value.

    Returns:
        A lowercase, slug-friendly username string, or None if invalid.
    """
    text = normalize_text(value).lower()
    if not text:
        return None

    text = text.replace(" ", "_")
    text = NON_USERNAME_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or None


def clean_email(value: Any) -> str | None:
    """
    Clean and validate an email address.

    Args:
        value: Raw email value.

    Returns:
        A normalized email address if valid, otherwise None.
    """
    text = normalize_text(value).lower()
    if not text:
        return None

    if "@" not in text:
        return None

    local_part, domain_part = text.rsplit("@", 1)
    domain_part = domain_part.strip(" .")
    domain_part = EMAIL_DOMAIN_TYPOS.get(domain_part, domain_part)
    candidate = f"{local_part}@{domain_part}"

    if EMAIL_PATTERN.fullmatch(candidate):
        return candidate

    return None


def clean_phone_number(value: Any, region: str = "DZ") -> str | None:
    """
    Clean and validate a phone number using the phonenumbers library.

    Args:
        value: Raw phone number value.
        region: Default country/region code used for parsing.

    Returns:
        A phone number in E.164 format if valid, otherwise None.
    """
    text = normalize_text(value)
    if not text:
        return None

    # Some sheets store multiple phone numbers in one cell (e.g. "a / b").
    # Keep the first number only.
    text = re.split(r"\s*(?:/|\||;|,|،)+\s*", text, maxsplit=1)[0].strip()
    if not text:
        return None

    try:
        parsed = phonenumbers.parse(text, region)
    except phonenumbers.NumberParseException:
        parsed = None

    if parsed is None or not phonenumbers.is_valid_number(parsed):
        digits = re.sub(r"\D+", "", text)
        if not digits:
            return None

        candidate = None
        if digits.startswith("213") and len(digits) in {11, 12}:
            candidate = f"+{digits}"
        elif len(digits) == 10 and digits.startswith("0"):
            candidate = f"+213{digits[1:]}"
        elif len(digits) == 9:
            candidate = f"+213{digits}"

        if candidate:
            try:
                parsed = phonenumbers.parse(candidate, None)
            except phonenumbers.NumberParseException:
                parsed = None

        if parsed is None or not phonenumbers.is_valid_number(parsed):
            return None

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def clean_dob(value: Any) -> int | None:
    """
    Clean a date of birth and return it as a Unix timestamp in seconds.

    Args:
        value: Raw DOB value in string, date, datetime, or numeric form.

    Returns:
        Unix timestamp in seconds, or None if the input cannot be parsed.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, datetime):
        parsed_datetime = value
    elif isinstance(value, date):
        parsed_datetime = datetime.combine(value, datetime.min.time())
    elif isinstance(value, (int, float)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        numeric_value = float(value)
        if numeric_value > 10_000_000_000:
            parsed_datetime = datetime.fromtimestamp(numeric_value / 1000.0, tz=timezone.utc)
        elif numeric_value > 10_000_000:
            parsed_datetime = datetime.fromtimestamp(numeric_value, tz=timezone.utc)
        elif 1900 <= numeric_value <= datetime.now().year:
            parsed_datetime = datetime(int(numeric_value), 1, 1)
        else:
            return None
    else:
        text = normalize_text(value)
        if not text:
            return None

        text = text.translate(ARABIC_INDIC_DIGITS)

        candidate_texts = [text]
        year_fix_match = re.search(r"(?P<year>\d{5,})(?=[-/])", text)
        if year_fix_match:
            fixed_text = text[: year_fix_match.start("year")] + year_fix_match.group("year")[:4] + text[year_fix_match.end("year") :]
            candidate_texts.append(fixed_text)

        parsed_datetime = None
        for candidate_text in candidate_texts:
            normalized_match = re.fullmatch(r"(?P<year>\d{4,})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})", candidate_text)
            if normalized_match:
                year_text = normalized_match.group("year")
                month_text = normalized_match.group("month")
                day_text = normalized_match.group("day")

                candidate_years: list[int] = []
                if len(year_text) > 4:
                    candidate_years.append(int(year_text[:4]))
                candidate_years.append(int(year_text))

                for candidate_year in candidate_years:
                    try:
                        parsed_datetime = datetime(candidate_year, int(month_text), int(day_text))
                        break
                    except ValueError:
                        continue
                if parsed_datetime is not None:
                    break

            if re.fullmatch(r"\d{4}", candidate_text):
                year = int(candidate_text)
                if 1900 <= year <= datetime.now().year:
                    parsed_datetime = datetime(year, 1, 1)
                    break
                continue

            try:
                parsed_datetime = date_parser.parse(candidate_text, dayfirst=True, fuzzy=True)
                break
            except (ValueError, OverflowError):
                continue

        if parsed_datetime is None:
            return None

    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    if parsed_datetime > now:
        return None

    if parsed_datetime.year < 1900:
        return None

    return int(parsed_datetime.timestamp())


def clean_gender(value: Any) -> str | None:
    """
    Normalize a gender label to the schema-supported values.

    Args:
        value: Raw gender value.

    Returns:
        'male', 'female', or None if the input is unknown.
    """
    text = normalize_text(value).lower()
    if not text:
        return None

    if text in GENDER_ALIASES:
        return GENDER_ALIASES[text]

    detector = Detector(case_sensitive=False)
    guessed = detector.get_gender(text)

    if guessed in {"male", "mostly_male"}:
        return "male"

    if guessed in {"female", "mostly_female"}:
        return "female"

    return None

def clean_user_type(value: Any) -> str | None:
    """Normalize user type using French/Arabic aliases."""
    text = normalize_text(value).lower()
    if not text: return None
    return USER_TYPE_ALIASES.get(text)

def clean_school_type(value: Any) -> str | None:
    """Normalize school type using French/Arabic aliases."""
    text = normalize_text(value).lower()
    if not text: return None
    return SCHOOL_TYPE_ALIASES.get(text)


def _build_normalized_lookup(lookup: Mapping[str, str]) -> dict[str, str]:
    """
    Build a normalized reverse lookup for fuzzy-matched categorical values.

    Args:
        lookup: Mapping of alias text to canonical text.

    Returns:
        A dictionary mapping normalized aliases to canonical values.
    """
    normalized_lookup: dict[str, str] = {}
    for alias, canonical in lookup.items():
        alias_text = normalize_text(alias).lower()
        canonical_text = normalize_text(canonical)
        if alias_text:
            normalized_lookup[alias_text] = canonical_text
    return normalized_lookup


def _fuzzy_lookup(
    value: Any,
    lookup: Mapping[str, str],
    threshold: float = 85.0,
) -> str | None:
    """
    Match a raw value against a lookup table using exact and fuzzy matching.

    Args:
        value: Raw input value.
        lookup: Mapping of alias text to canonical text.
        threshold: Minimum fuzzy score required to accept a match.

    Returns:
        The canonical matched value, or None if no acceptable match exists.
    """
    normalized_lookup = _build_normalized_lookup(lookup)
    text = normalize_text(value).lower()
    if not text or not normalized_lookup:
        return None

    if text in normalized_lookup:
        return normalized_lookup[text]

    match = process.extractOne(
        text,
        list(normalized_lookup.keys()),
        scorer=fuzz.ratio,
    )

    if match is None:
        return None

    matched_alias, score, _ = match
    if float(score) < threshold:
        return None

    return normalized_lookup[matched_alias]


def clean_wilaya(
    value: Any,
    wilayas_lookup: Mapping[str, str],
    threshold: float = 85.0,
) -> str | None:
    """
    Clean a wilaya value using exact and fuzzy matching.

    Args:
        value: Raw wilaya text.
        wilayas_lookup: Mapping of alias names to canonical wilaya names.
        threshold: Minimum fuzzy score required for acceptance.

    Returns:
        The canonical wilaya name, or None if no good match is found.
    """
    return _fuzzy_lookup(value, wilayas_lookup, threshold=threshold)


def clean_commune(value: Any) -> str | None:
    """
    Clean a commune value into a display-friendly text string.

    Args:
        value: Raw commune text.

    Returns:
        A cleaned commune string, or None if the input is empty.
    """
    cleaned = _clean_display_text(value)
    return cleaned or None


def clean_profile_picture(value: Any) -> str | None:
    """
    Clean a profile picture URL.

    Args:
        value: Raw URL value.

    Returns:
        A normalized URL string if it looks valid, otherwise None.
    """
    text = normalize_text(value)
    if not text:
        return None

    if text.startswith("http://") or text.startswith("https://"):
        return text

    return None


def clean_description(value: Any) -> str | None:
    """
    Clean a free-text description field.

    Args:
        value: Raw description value.

    Returns:
        A normalized description string, or None if the input is empty.
    """
    text = normalize_text(value)
    return text or None


def clean_preferences(
    value: Any,
    subjects_lookup: Mapping[str, str],
    threshold: float = 85.0,
) -> list[str]:
    """
    Clean and normalize a list of preferences/interests.

    Args:
        value: Raw preferences value. Can be a string, list, or other text-like value.
        subjects_lookup: Mapping of alias subject names to canonical subject names.
        threshold: Minimum fuzzy score required for accepting a fuzzy subject match.

    Returns:
        A deduplicated list of canonical preferences.
    """
    normalized_lookup = _build_normalized_lookup(subjects_lookup)
    if not normalized_lookup:
        return []

    raw_items: list[str] = []

    if isinstance(value, list):
        for item in value:
            raw_items.extend(re.split(r"[,\;/|،]+", normalize_text(item)))
    else:
        raw_text = normalize_text(value)
        raw_items.extend(re.split(r"[,\;/|،]+", raw_text))

    cleaned_items: list[str] = []
    seen: set[str] = set()

    for item in raw_items:
        candidate = normalize_text(item)
        if not candidate:
            continue

        lowered = candidate.lower()
        canonical = None

        if lowered in normalized_lookup:
            canonical = normalized_lookup[lowered]
        else:
            match = process.extractOne(
                lowered,
                list(normalized_lookup.keys()),
                scorer=fuzz.ratio,
            )
            if match is not None:
                matched_alias, score, _ = match
                if float(score) >= threshold:
                    canonical = normalized_lookup[matched_alias]

        if canonical is None:
            canonical = smart_title(candidate)

        dedupe_key = canonical.lower()
        if dedupe_key not in seen:
            seen.add(dedupe_key)
            cleaned_items.append(canonical)

    return cleaned_items