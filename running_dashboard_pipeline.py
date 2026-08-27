from __future__ import annotations

import getpass
import io
import json
import math
import os
import re
import shutil
import time
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from garmin_fit_sdk import Decoder, Stream
from garminconnect import (
    Garmin,
    GarminConnectAuthenticationError,
    GarminConnectConnectionError,
    GarminConnectTooManyRequestsError,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent
USER_DIR = PROJECT_DIR / "data" / "users" / "sietse"

GARMIN_EMAIL = os.getenv("GARMIN_EMAIL", "YOURMAIL@GMAIL.COM").strip()
TOKEN_STORE = Path.home() / ".garminconnect" / "running-insights"

RAW_DIR = USER_DIR / "raw"
SILVER_DIR = USER_DIR / "silver"
SUMMARY_DIR = RAW_DIR / "activity_summaries"
WEATHER_DIR = RAW_DIR / "activity_weather"
MAX_METRICS_DIR = RAW_DIR / "garmin_max_metrics"
RHR_DIR = RAW_DIR / "garmin_resting_hr"
ORIGINAL_DIR = RAW_DIR / "original_activities"
FIT_DIR = RAW_DIR / "fit"
RECORDS_DIR = SILVER_DIR / "records"
LAPS_DIR = SILVER_DIR / "laps"
SESSIONS_DIR = SILVER_DIR / "sessions"
MANIFEST_DIR = SILVER_DIR / "manifests"
ACTIVITY_INDEX_FILE = MANIFEST_DIR / "running_activity_index.parquet"
EXPORT_MANIFEST_FILE = MANIFEST_DIR / "activity_export_manifest.parquet"
GARMIN_VO2MAX_FILE = MANIFEST_DIR / "garmin_vo2max_history.parquet"
GARMIN_RHR_FILE = MANIFEST_DIR / "garmin_resting_hr_history.parquet"

BATCH_SIZE = 100
REQUEST_DELAY_SECONDS = 0.25
FORCE_REEXPORT = False
RHR_EXPORT_LOOKBACK_DAYS = 400

# Azure publication settings.
# The analytics pipeline still writes all website JSON locally first.
# After a successful build, those JSON files are uploaded to the public
# dashboard container in Azure Blob Storage.
AZURE_PUBLISH_ENABLED = os.getenv("AZURE_PUBLISH_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
AZURE_PUBLIC_STORAGE_ACCOUNT = os.getenv(
    "AZURE_PUBLIC_STORAGE_ACCOUNT",
    "runningdashboardpublic",
).strip()
AZURE_PUBLIC_CONTAINER = os.getenv(
    "AZURE_PUBLIC_CONTAINER",
    "dashboard-data",
).strip()
AZURE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_STORAGE_CONNECTION_STRING",
    "",
).strip()

# Private persistent pipeline state.
#
# A GitHub-hosted runner starts with an empty disk on every run. The
# private account therefore stores only the state needed to continue the
# incremental pipeline:
#   - data/users/sietse/raw/
#   - data/users/sietse/silver/
#   - ~/.garminconnect/running-insights/ (Garmin auth tokens)
#
# Gold/dashboard outputs are deliberately excluded because they are
# reproducible from the persisted raw/silver state.
AZURE_PRIVATE_SYNC_ENABLED = os.getenv(
    "AZURE_PRIVATE_SYNC_ENABLED",
    "1",
).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
AZURE_PRIVATE_STORAGE_ACCOUNT = os.getenv(
    "AZURE_PRIVATE_STORAGE_ACCOUNT",
    "runningdashboardprivate",
).strip()
AZURE_PRIVATE_CONTAINER = os.getenv(
    "AZURE_PRIVATE_CONTAINER",
    "pipeline-state",
).strip()
AZURE_PRIVATE_STORAGE_CONNECTION_STRING = os.getenv(
    "AZURE_PRIVATE_STORAGE_CONNECTION_STRING",
    "",
).strip()
AZURE_PRIVATE_FORCE_RESTORE = os.getenv(
    "AZURE_PRIVATE_FORCE_RESTORE",
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

PRIVATE_STATE_ROOTS = (
    (RAW_DIR, "user/raw"),
    (SILVER_DIR, "user/silver"),
    (TOKEN_STORE, "auth/garminconnect"),
)

DASHBOARD_JSON_FILES = (
    "summary.json",
    "runs.json",
    "weekly.json",
    "monthly.json",
    "zones_weekly.json",
    "zones_monthly.json",
    "heart_rate_zones_weekly.json",
    "pace_zones_weekly.json",
    "race_predictions.json",
    "training_context.json",
)

# Leave these as None to export the complete Garmin running history.
EARLIEST_ACTIVITY_DATE: date | None = None
LATEST_ACTIVITY_DATE: date | None = None

RUNNING_ACTIVITY_TYPES = {
    "running",
    "run",
    "street_running",
    "trail_running",
    "trail_run",
    "treadmill_running",
    "indoor_running",
    "track_running",
    "virtual_running",
    "virtual_run",
    "ultra_run",
    "obstacle_run",
}


# =============================================================================
# General helpers
# =============================================================================


def create_directories() -> None:
    for directory in [
        USER_DIR,
        TOKEN_STORE,
        SUMMARY_DIR,
        WEATHER_DIR,
        MAX_METRICS_DIR,
        RHR_DIR,
        ORIGINAL_DIR,
        FIT_DIR,
        RECORDS_DIR,
        LAPS_DIR,
        SESSIONS_DIR,
        MANIFEST_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)



def json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.hex()
    return str(value)



def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )



# =============================================================================
# Single-user paths
# =============================================================================


def token_store_has_files() -> bool:
    return TOKEN_STORE.exists() and any(TOKEN_STORE.iterdir())




def parquet_safe_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=json_default,
        )
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value



def make_parquet_safe(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    for column in dataframe.columns:
        if dataframe[column].dtype != "object":
            continue

        dataframe[column] = dataframe[column].map(parquet_safe_value)
        inferred_type = pd.api.types.infer_dtype(
            dataframe[column].dropna(),
            skipna=True,
        )

        if inferred_type in {
            "mixed",
            "mixed-integer",
            "mixed-integer-float",
            "bytes",
            "unknown-array",
        }:
            dataframe[column] = dataframe[column].map(
                lambda value: None if pd.isna(value) else str(value)
            )

    return dataframe



def write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    make_parquet_safe(dataframe).to_parquet(path, index=False)



def first_present(
    payload: dict[str, Any],
    keys: list[str],
    default: Any = None,
) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return default


# =============================================================================
# Garmin connection
# =============================================================================


def prompt_for_mfa() -> str:
    """Ask for Garmin's one-time MFA code when a fresh login requires it."""
    while True:
        code = input("Garmin MFA code: ").strip()
        if code:
            return code
        print("The MFA code cannot be empty.")



def connect_to_garmin() -> Garmin:
    garmin_email = GARMIN_EMAIL or input("Garmin email address: ").strip()
    if not garmin_email:
        raise ValueError("Set GARMIN_EMAIL or enter a Garmin email address.")

    tokenstore_path = str(TOKEN_STORE.expanduser().resolve())

    if token_store_has_files():
        try:
            api = Garmin(
                retry_attempts=3,
                retry_min_wait=1.0,
                retry_max_wait=10.0,
            )
            api.login(tokenstore_path)
            print("Authenticated using saved Garmin tokens.")
            return api
        except (
            GarminConnectAuthenticationError,
            GarminConnectConnectionError,
        ) as exc:
            print(f"Saved Garmin tokens could not be used: {exc}")
            print("Starting a fresh credential login.")

    password = os.getenv("GARMIN_PASSWORD")
    if not password:
        password = getpass.getpass(f"Garmin password for {garmin_email}: ")

    api = Garmin(
        email=garmin_email,
        password=password,
        prompt_mfa=prompt_for_mfa,
        retry_attempts=3,
        retry_min_wait=1.0,
        retry_max_wait=10.0,
    )
    api.login(tokenstore_path)
    print("Fresh Garmin login completed; tokens have been saved.")
    return api


# =============================================================================
# Find every running activity
# =============================================================================


def normalize_activity_batch(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ["activities", "activityList", "items"]:
            rows = payload.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    raise TypeError(
        f"Unexpected get_activities response: {type(payload).__name__}"
    )



def get_activity_id(activity: dict[str, Any]) -> str:
    for key in ["activityId", "activity_id", "id"]:
        if activity.get(key) is not None:
            return str(activity[key])
    raise KeyError("Activity contains no recognised activity ID.")



def get_activity_type(activity: dict[str, Any]) -> str:
    activity_type = activity.get("activityType")

    if isinstance(activity_type, dict):
        value = (
            activity_type.get("typeKey")
            or activity_type.get("type_key")
            or activity_type.get("key")
        )
    else:
        value = activity_type

    if value is None:
        value = activity.get("activityTypeKey") or activity.get("typeKey")

    return str(value or "").strip().lower()



def is_running_activity(activity: dict[str, Any]) -> bool:
    activity_type = get_activity_type(activity)
    return (
        activity_type in RUNNING_ACTIVITY_TYPES
        or "running" in activity_type
        or activity_type.endswith("_run")
    )



def get_activity_date(activity: dict[str, Any]) -> date | None:
    for key in [
        "startTimeLocal",
        "startTimeGMT",
        "activityStartTimeLocal",
        "activity_start_time_local",
    ]:
        value = activity.get(key)
        if value is None:
            continue
        timestamp = pd.to_datetime(value, errors="coerce")
        if not pd.isna(timestamp):
            return timestamp.date()
    return None



def in_requested_date_range(activity: dict[str, Any]) -> bool:
    activity_date = get_activity_date(activity)

    if activity_date is None:
        return True
    if EARLIEST_ACTIVITY_DATE and activity_date < EARLIEST_ACTIVITY_DATE:
        return False
    if LATEST_ACTIVITY_DATE and activity_date > LATEST_ACTIVITY_DATE:
        return False
    return True



def get_all_activities(api: Garmin) -> list[dict[str, Any]]:
    """Paginate until Garmin returns no further unique activities."""
    activities: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    start = 0

    while True:
        batch = normalize_activity_batch(
            api.get_activities(start, BATCH_SIZE)
        )

        if not batch:
            break

        new_rows = 0

        for activity in batch:
            activity_id = get_activity_id(activity)
            if activity_id in seen_ids:
                continue
            seen_ids.add(activity_id)
            activities.append(activity)
            new_rows += 1

        print(
            f"Retrieved {len(activities):,} unique activities "
            f"after offset {start:,}."
        )

        if new_rows == 0:
            print("No new activity IDs returned; stopping pagination.")
            break

        start += len(batch)

        if len(batch) < BATCH_SIZE:
            break

    return activities



def get_all_running_activities(api: Garmin) -> list[dict[str, Any]]:
    all_activities = get_all_activities(api)

    running_activities = [
        activity
        for activity in all_activities
        if is_running_activity(activity)
        and in_requested_date_range(activity)
    ]

    running_activities.sort(
        key=lambda activity: (
            get_activity_date(activity) or date.min,
            get_activity_id(activity),
        ),
        reverse=True,
    )

    print(f"All Garmin activities: {len(all_activities):,}")
    print(f"Running activities: {len(running_activities):,}")

    return running_activities


# =============================================================================
# Download raw summary and FIT file
# =============================================================================


def get_activity_summary(
    api: Garmin,
    activity: dict[str, Any],
) -> dict[str, Any]:
    activity_id = get_activity_id(activity)
    output_path = SUMMARY_DIR / f"{activity_id}.json"

    if output_path.exists() and not FORCE_REEXPORT:
        with output_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        if isinstance(summary, dict):
            return summary

    detailed_summary = api.get_activity(activity_id)
    summary = dict(activity)

    if isinstance(detailed_summary, dict):
        summary.update(detailed_summary)
    else:
        summary["detailed_summary"] = detailed_summary

    write_json(output_path, summary)
    return summary



def get_activity_weather(
    api: Garmin,
    activity_id: str,
) -> dict[str, Any]:
    """Fetch and cache Garmin Connect weather data for one activity.

    Weather is optional enrichment. A missing weather response must not make
    the activity export fail. Rate-limit errors are still re-raised.
    """
    output_path = WEATHER_DIR / f"{activity_id}.json"

    if output_path.exists() and not FORCE_REEXPORT:
        try:
            with output_path.open("r", encoding="utf-8") as file:
                weather = json.load(file)
            if isinstance(weather, dict):
                return weather
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    try:
        weather = api.get_activity_weather(activity_id)
    except GarminConnectTooManyRequestsError:
        raise
    except GarminConnectConnectionError as exc:
        print(f"Weather unavailable for {activity_id}: {exc}")
        return {}

    if not isinstance(weather, dict):
        return {}

    write_json(output_path, weather)
    return weather


def weather_metadata(weather: dict[str, Any]) -> dict[str, Any]:
    """Return analysis-friendly activity-level weather fields."""
    weather_type = weather.get("weatherTypeDTO")
    weather_station = weather.get("weatherStationDTO")

    if not isinstance(weather_type, dict):
        weather_type = {}
    if not isinstance(weather_station, dict):
        weather_station = {}

    return {
        "weather_temperature": first_present(weather, ["temp", "temperature"]),
        "weather_apparent_temperature": first_present(
            weather,
            ["apparentTemp", "apparentTemperature"],
        ),
        "weather_dew_point": first_present(weather, ["dewPoint"]),
        "weather_relative_humidity_pct": first_present(
            weather,
            ["relativeHumidity"],
        ),
        "weather_wind_speed": first_present(weather, ["windSpeed"]),
        "weather_wind_direction_deg": first_present(
            weather,
            ["windDirection"],
        ),
        "weather_wind_direction_compass": first_present(
            weather,
            ["windDirectionCompassPoint"],
        ),
        "weather_condition": first_present(
            weather_type,
            ["desc", "description"],
        ),
        "weather_station_name": first_present(
            weather_station,
            ["name"],
        ),
        "weather_issue_date": first_present(weather, ["issueDate"]),
    }


def add_standard_record_columns(records: pd.DataFrame) -> pd.DataFrame:
    """Add stable names for key point-by-point running variables.

    Garmin FIT files may expose either altitude or enhanced_altitude. We keep
    the original FIT columns and additionally create altitude_m, preferring
    enhanced_altitude when available.
    """
    records = records.copy()

    altitude = pd.Series(np.nan, index=records.index, dtype="float64")

    if "enhanced_altitude" in records.columns:
        altitude = pd.to_numeric(
            records["enhanced_altitude"],
            errors="coerce",
        )

    if "altitude" in records.columns:
        fallback_altitude = pd.to_numeric(
            records["altitude"],
            errors="coerce",
        )
        altitude = altitude.fillna(fallback_altitude)

    records["altitude_m"] = altitude
    return records


def add_session_enrichment(
    sessions: pd.DataFrame,
    summary: dict[str, Any],
    weather: dict[str, Any],
) -> pd.DataFrame:
    """Add standardized elevation totals and Garmin weather to sessions."""
    sessions = sessions.copy()

    # Prefer FIT session totals if present; fall back to Garmin activity summary.
    if "total_ascent" in sessions.columns:
        sessions["elevation_gain_m"] = pd.to_numeric(
            sessions["total_ascent"],
            errors="coerce",
        )
    else:
        sessions["elevation_gain_m"] = np.nan

    if "total_descent" in sessions.columns:
        sessions["elevation_loss_m"] = pd.to_numeric(
            sessions["total_descent"],
            errors="coerce",
        )
    else:
        sessions["elevation_loss_m"] = np.nan

    summary_gain = first_present(
        summary,
        ["elevationGain", "totalAscent", "elevation_gain"],
    )
    summary_loss = first_present(
        summary,
        ["elevationLoss", "totalDescent", "elevation_loss"],
    )

    if summary_gain is not None:
        sessions["elevation_gain_m"] = sessions["elevation_gain_m"].fillna(
            pd.to_numeric(pd.Series([summary_gain]), errors="coerce").iloc[0]
        )
    if summary_loss is not None:
        sessions["elevation_loss_m"] = sessions["elevation_loss_m"].fillna(
            pd.to_numeric(pd.Series([summary_loss]), errors="coerce").iloc[0]
        )

    for column, value in weather_metadata(weather).items():
        sessions[column] = value

    return sessions


def is_fit_file(content: bytes) -> bool:
    return len(content) >= 12 and content[8:12] == b".FIT"



def extract_fit_from_zip(content: bytes, activity_id: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        fit_members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and member.filename.lower().endswith(".fit")
        ]

        if not fit_members:
            raise ValueError(
                f"No FIT file found in original download for {activity_id}."
            )

        # The activity FIT file is normally the largest FIT member.
        member = max(fit_members, key=lambda item: item.file_size)
        return archive.read(member)



def get_fit_file(api: Garmin, activity_id: str) -> Path:
    fit_path = FIT_DIR / f"{activity_id}.fit"
    zip_path = ORIGINAL_DIR / f"{activity_id}.zip"
    original_fit_path = ORIGINAL_DIR / f"{activity_id}.fit"

    if fit_path.exists() and not FORCE_REEXPORT:
        return fit_path

    if zip_path.exists() and not FORCE_REEXPORT:
        fit_path.write_bytes(
            extract_fit_from_zip(zip_path.read_bytes(), activity_id)
        )
        return fit_path

    if original_fit_path.exists() and not FORCE_REEXPORT:
        content = original_fit_path.read_bytes()
        if not is_fit_file(content):
            raise ValueError(f"Invalid stored FIT file: {original_fit_path}")
        fit_path.write_bytes(content)
        return fit_path

    content = api.download_activity(
        activity_id,
        dl_fmt=Garmin.ActivityDownloadFormat.ORIGINAL,
    )

    if not isinstance(content, (bytes, bytearray)):
        raise TypeError(
            f"Expected bytes for {activity_id}, got {type(content).__name__}."
        )

    content = bytes(content)

    if zipfile.is_zipfile(io.BytesIO(content)):
        zip_path.write_bytes(content)
        fit_content = extract_fit_from_zip(content, activity_id)
    elif is_fit_file(content):
        original_fit_path.write_bytes(content)
        fit_content = content
    else:
        raise ValueError(
            f"Original download for {activity_id} is neither ZIP nor FIT."
        )

    fit_path.write_bytes(fit_content)
    return fit_path


# =============================================================================
# Decode FIT and write silver tables
# =============================================================================


def decode_fit(fit_path: Path) -> tuple[dict[str, Any], list[str]]:
    stream = Stream.from_file(str(fit_path))
    decoder = Decoder(stream)

    messages, errors = decoder.read(
        apply_scale_and_offset=True,
        convert_datetimes_to_dates=True,
        convert_types_to_strings=True,
        enable_crc_check=True,
        expand_sub_fields=True,
        expand_components=True,
        merge_heart_rates=True,
    )

    if not isinstance(messages, dict):
        raise TypeError("FIT decoder did not return a message dictionary.")

    return messages, [str(error) for error in errors]



def build_metadata(
    activity_id: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "garmin_activity_id": activity_id,
        "activity_name": first_present(
            summary,
            ["activityName", "activity_name", "name"],
        ),
        "activity_type": get_activity_type(summary),
        "activity_start_time_local": first_present(
            summary,
            ["startTimeLocal", "activityStartTimeLocal"],
        ),
        "activity_start_time_gmt": first_present(
            summary,
            ["startTimeGMT", "activityStartTimeGMT"],
        ),
    }



def message_dataframe(
    messages: dict[str, Any],
    key: str,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    rows = messages.get(key, [])
    dataframe = pd.DataFrame(rows if isinstance(rows, list) else [])

    for column, value in metadata.items():
        dataframe[column] = value

    dataframe.insert(0, "fit_message_index", range(len(dataframe)))
    return dataframe



def fallback_session(
    activity_id: str,
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fit_message_index": 0,
                **metadata,
                "session_source": "garmin_summary_fallback",
                "start_time": first_present(
                    summary,
                    ["startTimeGMT", "startTimeLocal"],
                ),
                "total_timer_time": first_present(
                    summary,
                    ["duration", "movingDuration", "elapsedDuration"],
                ),
                "total_elapsed_time": first_present(
                    summary,
                    ["elapsedDuration", "duration"],
                ),
                "total_distance": first_present(
                    summary,
                    ["distance", "totalDistance"],
                ),
                "avg_heart_rate": first_present(
                    summary,
                    ["averageHR", "averageHeartRate", "avgHeartRate"],
                ),
                "max_heart_rate": first_present(
                    summary,
                    ["maxHR", "maxHeartRate", "maximumHeartRate"],
                ),
                "garmin_activity_id": activity_id,
            }
        ]
    )



def write_activity_tables(
    activity_id: str,
    summary: dict[str, Any],
    weather: dict[str, Any],
    messages: dict[str, Any],
) -> dict[str, int]:
    metadata = build_metadata(activity_id, summary)

    records = message_dataframe(messages, "record_mesgs", metadata)
    laps = message_dataframe(messages, "lap_mesgs", metadata)
    sessions = message_dataframe(messages, "session_mesgs", metadata)

    # Point-by-point altitude used later to calculate smoothed grade.
    records = add_standard_record_columns(records)

    if sessions.empty:
        sessions = fallback_session(activity_id, summary, metadata)
    else:
        sessions["session_source"] = "fit_session_message"

    sessions = add_session_enrichment(sessions, summary, weather)

    write_parquet(records, RECORDS_DIR / f"{activity_id}.parquet")
    write_parquet(laps, LAPS_DIR / f"{activity_id}.parquet")
    write_parquet(sessions, SESSIONS_DIR / f"{activity_id}.parquet")

    return {
        "record_rows": len(records),
        "lap_rows": len(laps),
        "session_rows": len(sessions),
    }


# =============================================================================
# Index and manifest
# =============================================================================


def activity_is_complete(activity_id: str) -> bool:
    return all(
        path.exists()
        for path in [
            RECORDS_DIR / f"{activity_id}.parquet",
            LAPS_DIR / f"{activity_id}.parquet",
            SESSIONS_DIR / f"{activity_id}.parquet",
        ]
    )



def save_activity_index(activities: list[dict[str, Any]]) -> None:
    rows = []

    for activity in activities:
        activity_id = get_activity_id(activity)
        rows.append(
            {
                "garmin_activity_id": activity_id,
                "activity_name": first_present(
                    activity,
                    ["activityName", "activity_name", "name"],
                ),
                "activity_type": get_activity_type(activity),
                "activity_date": get_activity_date(activity),
                "start_time_local": first_present(
                    activity,
                    ["startTimeLocal", "activityStartTimeLocal"],
                ),
                "distance_m": first_present(
                    activity,
                    ["distance", "totalDistance"],
                ),
                "duration_s": first_present(
                    activity,
                    ["duration", "elapsedDuration"],
                ),
                "average_hr_bpm": first_present(
                    activity,
                    ["averageHR", "averageHeartRate"],
                ),
                "max_hr_bpm": first_present(
                    activity,
                    ["maxHR", "maxHeartRate"],
                ),
                "already_exported": activity_is_complete(activity_id),
            }
        )

    index = pd.DataFrame(rows)
    write_parquet(index, ACTIVITY_INDEX_FILE)
    index.to_csv(
        ACTIVITY_INDEX_FILE.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )



def save_manifest(rows: list[dict[str, Any]]) -> None:
    new_manifest = pd.DataFrame(rows)

    if EXPORT_MANIFEST_FILE.exists():
        old_manifest = pd.read_parquet(EXPORT_MANIFEST_FILE)
        manifest = pd.concat(
            [old_manifest, new_manifest],
            ignore_index=True,
            sort=False,
        )
    else:
        manifest = new_manifest

    manifest["exported_at"] = pd.to_datetime(
        manifest["exported_at"],
        errors="coerce",
    )

    manifest = (
        manifest
        .sort_values("exported_at")
        .drop_duplicates("garmin_activity_id", keep="last")
    )

    write_parquet(manifest, EXPORT_MANIFEST_FILE)
    manifest.to_csv(
        EXPORT_MANIFEST_FILE.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# Garmin benchmark health metrics
# =============================================================================


def decode_possible_json(value: Any) -> Any:
    """Garmin GraphQL scalar responses can occasionally contain JSON strings."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def extract_graphql_vo2_entries(payload: Any) -> list[dict[str, Any]]:
    """
    Extract Garmin running VO2max history from the GraphQL vo2MaxScalar response.

    Expected shape:
        {
          "data": {
            "vo2MaxScalar": [
              {
                "generic": {
                  "calendarDate": "...",
                  "vo2MaxPreciseValue": ...,
                  "vo2MaxValue": ...
                }
              }
            ]
          }
        }
    """
    payload = decode_possible_json(payload)

    if not isinstance(payload, dict):
        return []

    data = decode_possible_json(payload.get("data"))
    if not isinstance(data, dict):
        return []

    scalar = decode_possible_json(data.get("vo2MaxScalar"))

    if isinstance(scalar, dict):
        entries = [scalar]
    elif isinstance(scalar, list):
        entries = scalar
    else:
        return []

    rows: list[dict[str, Any]] = []

    for entry in entries:
        entry = decode_possible_json(entry)
        if not isinstance(entry, dict):
            continue

        generic = decode_possible_json(entry.get("generic"))
        if not isinstance(generic, dict):
            continue

        calendar_date = first_present(
            generic,
            ["calendarDate", "calendar_date", "date"],
        )
        precise = first_present(
            generic,
            ["vo2MaxPreciseValue", "vo2maxPreciseValue"],
        )
        displayed = first_present(
            generic,
            ["vo2MaxValue", "vo2maxValue"],
        )

        precise_num = pd.to_numeric(
            pd.Series([precise]),
            errors="coerce",
        ).iloc[0]
        displayed_num = pd.to_numeric(
            pd.Series([displayed]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(precise_num) and not pd.isna(displayed_num):
            precise_num = displayed_num

        if calendar_date is None or (
            pd.isna(precise_num) and pd.isna(displayed_num)
        ):
            continue

        acclimation = decode_possible_json(
            entry.get("heatAltitudeAcclimation")
        )
        if not isinstance(acclimation, dict):
            acclimation = {}

        rows.append(
            {
                "calendar_date": calendar_date,
                "garmin_vo2max_precise": precise_num,
                "garmin_vo2max_displayed": displayed_num,
                "garmin_heat_acclimation_pct": pd.to_numeric(
                    pd.Series(
                        [acclimation.get("heatAcclimationPercentage")]
                    ),
                    errors="coerce",
                ).iloc[0],
                "garmin_current_altitude_m": pd.to_numeric(
                    pd.Series([acclimation.get("currentAltitude")]),
                    errors="coerce",
                ).iloc[0],
                "source": "graphql_vo2MaxScalar",
            }
        )

    return rows


def fetch_vo2_graphql_range(
    api: Garmin,
    start_day: date,
    end_day: date,
) -> list[dict[str, Any]]:
    """
    Fetch Garmin VO2max history for a date range in one GraphQL request.

    The response is cached so repeated pipeline runs normally do not hit
    Garmin again for the same historical range.
    """
    cache_path = (
        MAX_METRICS_DIR
        / f"vo2max_graphql_{start_day.isoformat()}_{end_day.isoformat()}.json"
    )

    if cache_path.exists() and not FORCE_REEXPORT:
        try:
            payload = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
            return extract_graphql_vo2_entries(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    if not hasattr(api, "query_garmin_graphql"):
        return []

    query = {
        "query": (
            "query { "
            f'vo2MaxScalar(startDate:"{start_day.isoformat()}", '
            f'endDate:"{end_day.isoformat()}")'
            " }"
        )
    }

    payload = api.query_garmin_graphql(query)
    write_json(cache_path, payload)

    return extract_graphql_vo2_entries(payload)


def get_max_metrics_for_date(api: Garmin, activity_date: date) -> Any:
    """Fallback source when the GraphQL VO2 endpoint is unavailable."""
    date_string = activity_date.isoformat()
    path = MAX_METRICS_DIR / f"max_metrics_{date_string}.json"

    if path.exists() and not FORCE_REEXPORT:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    payload = api.get_max_metrics(date_string)
    write_json(path, payload)
    return payload


def find_vo2_in_max_metrics(payload: Any) -> dict[str, Any] | None:
    """
    Parse several max-metrics response shapes.

    Newer Garmin responses may expose generic.metricType/value rather than
    vo2MaxPreciseValue directly, so support both formats.
    """
    candidates: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        value = decode_possible_json(value)

        if isinstance(value, dict):
            generic = decode_possible_json(value.get("generic"))

            if isinstance(generic, dict):
                if (
                    generic.get("vo2MaxPreciseValue") is not None
                    or generic.get("vo2MaxValue") is not None
                ):
                    candidates.append(generic)

                metric_type = str(
                    generic.get("metricType") or ""
                ).upper()
                metric_value = generic.get("value")

                if "VO2" in metric_type and metric_value is not None:
                    candidates.append(
                        {
                            "calendarDate": first_present(
                                generic,
                                ["calendarDate", "date"],
                            ),
                            "vo2MaxPreciseValue": metric_value,
                            "vo2MaxValue": metric_value,
                        }
                    )

            if (
                value.get("vo2MaxPreciseValue") is not None
                or value.get("vo2MaxValue") is not None
            ):
                candidates.append(value)

            metric_type = str(
                value.get("metricType") or ""
            ).upper()
            if "VO2" in metric_type and value.get("value") is not None:
                candidates.append(
                    {
                        "calendarDate": first_present(
                            value,
                            ["calendarDate", "date"],
                        ),
                        "vo2MaxPreciseValue": value.get("value"),
                        "vo2MaxValue": value.get("value"),
                    }
                )

            for nested in value.values():
                walk(nested)

        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return candidates[-1] if candidates else None


def export_garmin_vo2max_history(
    api: Garmin,
    activities: list[dict[str, Any]],
) -> None:
    """
    Export Garmin's own running VO2max history.

    Primary source:
        GraphQL vo2MaxScalar over date ranges.

    Fallback:
        get_max_metrics() on running dates.

    This writes:
        silver/manifests/garmin_vo2max_history.parquet
        silver/manifests/garmin_vo2max_history.csv
    """
    activity_dates = sorted(
        {
            d
            for activity in activities
            if (d := get_activity_date(activity)) is not None
        }
    )

    if not activity_dates:
        print("No running dates available for Garmin VO2max history.")
        return

    start_day = activity_dates[0]
    end_day = max(activity_dates[-1], date.today())

    print("\nRetrieving Garmin VO2max benchmark history")
    print("-" * 70)
    print(
        f"Requested range: {start_day.isoformat()} "
        f"to {end_day.isoformat()}"
    )

    rows: list[dict[str, Any]] = []

    # Use 366-day chunks. This keeps requests small while avoiding one API
    # call per run/day.
    current_start = start_day

    try:
        while current_start <= end_day:
            current_end = min(
                current_start + timedelta(days=365),
                end_day,
            )

            chunk_rows = fetch_vo2_graphql_range(
                api,
                current_start,
                current_end,
            )
            rows.extend(chunk_rows)

            print(
                f"GraphQL {current_start.isoformat()} "
                f"to {current_end.isoformat()}: "
                f"{len(chunk_rows):,} VO2max records."
            )

            current_start = current_end + timedelta(days=1)

            if current_start <= end_day:
                time.sleep(REQUEST_DELAY_SECONDS)

    except GarminConnectTooManyRequestsError:
        raise
    except Exception as exc:
        print(
            "Garmin GraphQL VO2max history unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    # Fallback only if GraphQL did not return any usable VO2max values.
    if not rows:
        print(
            "No VO2max records returned by GraphQL. "
            "Trying get_max_metrics() on running dates."
        )

        for number, day in enumerate(activity_dates, start=1):
            try:
                metric = find_vo2_in_max_metrics(
                    get_max_metrics_for_date(api, day)
                )

                if metric is not None:
                    precise = pd.to_numeric(
                        pd.Series(
                            [metric.get("vo2MaxPreciseValue")]
                        ),
                        errors="coerce",
                    ).iloc[0]
                    displayed = pd.to_numeric(
                        pd.Series([metric.get("vo2MaxValue")]),
                        errors="coerce",
                    ).iloc[0]

                    if pd.isna(precise) and not pd.isna(displayed):
                        precise = displayed

                    if not pd.isna(precise) or not pd.isna(displayed):
                        rows.append(
                            {
                                "calendar_date": first_present(
                                    metric,
                                    ["calendarDate", "date"],
                                    day,
                                ),
                                "garmin_vo2max_precise": precise,
                                "garmin_vo2max_displayed": displayed,
                                "garmin_heat_acclimation_pct": np.nan,
                                "garmin_current_altitude_m": np.nan,
                                "source": "get_max_metrics",
                            }
                        )

            except GarminConnectTooManyRequestsError:
                raise
            except Exception as exc:
                print(
                    f"VO2max fallback failed for {day}: "
                    f"{type(exc).__name__}: {exc}"
                )

            if number % 25 == 0 or number == len(activity_dates):
                print(
                    f"Fallback checked {number:,}/"
                    f"{len(activity_dates):,} running dates."
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(rows)

    if df.empty:
        print(
            "WARNING: Garmin returned no VO2max history. "
            "Check raw/garmin_max_metrics for the cached responses."
        )
        return

    df["calendar_date"] = pd.to_datetime(
        df["calendar_date"],
        errors="coerce",
    ).dt.date

    df["garmin_vo2max_precise"] = pd.to_numeric(
        df["garmin_vo2max_precise"],
        errors="coerce",
    )
    df["garmin_vo2max_displayed"] = pd.to_numeric(
        df["garmin_vo2max_displayed"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=["calendar_date", "garmin_vo2max_precise"]
        )
        .sort_values("calendar_date")
        .drop_duplicates("calendar_date", keep="last")
        .reset_index(drop=True)
    )

    if df.empty:
        print(
            "WARNING: VO2max responses existed, but no numeric "
            "running VO2max values could be parsed."
        )
        return

    write_parquet(df, GARMIN_VO2MAX_FILE)
    df.round(2).to_csv(
        GARMIN_VO2MAX_FILE.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {len(df):,} Garmin VO2max records "
        f"({df['calendar_date'].min()} to "
        f"{df['calendar_date'].max()})."
    )



def get_daily_hr(api: Garmin, day: date) -> Any:
    """Single-day fallback for resting HR."""
    path = RHR_DIR / f"{day.isoformat()}.json"

    if path.exists() and not FORCE_REEXPORT:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    # get_rhr_day is a more direct source for resting HR when available.
    if hasattr(api, "get_rhr_day"):
        payload = api.get_rhr_day(day.isoformat())
    else:
        payload = api.get_heart_rates(day.isoformat())

    write_json(path, payload)
    return payload


def extract_resting_hr(payload: Any) -> float | None:
    """Extract one plausible resting-HR value from a Garmin response."""
    payload = decode_possible_json(payload)

    if isinstance(payload, dict):
        for key in [
            "restingHeartRate",
            "resting_heart_rate",
            "restingHR",
            "value",
        ]:
            value = pd.to_numeric(
                pd.Series([payload.get(key)]),
                errors="coerce",
            ).iloc[0]

            if not pd.isna(value) and 20 <= float(value) <= 120:
                return float(value)

        for nested in payload.values():
            result = extract_resting_hr(nested)
            if result is not None:
                return result

    elif isinstance(payload, list):
        for nested in payload:
            result = extract_resting_hr(nested)
            if result is not None:
                return result

    return None


def extract_rhr_history_rows(payload: Any) -> list[dict[str, Any]]:
    """
    Extract dated resting-HR values from Garmin's wellness/daily metric range.

    The request is made with metricId=60, so nested calendarDate/value pairs
    represent resting-HR observations.
    """
    payload = decode_possible_json(payload)
    rows: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        value = decode_possible_json(value)

        if isinstance(value, dict):
            calendar_date = first_present(
                value,
                ["calendarDate", "calendar_date", "date"],
            )

            raw_hr = first_present(
                value,
                [
                    "restingHeartRate",
                    "resting_heart_rate",
                    "restingHR",
                    "value",
                ],
            )

            hr = pd.to_numeric(
                pd.Series([raw_hr]),
                errors="coerce",
            ).iloc[0]

            if (
                calendar_date is not None
                and not pd.isna(hr)
                and 20 <= float(hr) <= 120
            ):
                rows.append(
                    {
                        "calendar_date": calendar_date,
                        "resting_hr_bpm": float(hr),
                    }
                )

            for nested in value.values():
                walk(nested)

        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(payload)
    return rows


def fetch_rhr_range(
    api: Garmin,
    start_day: date,
    end_day: date,
) -> list[dict[str, Any]]:
    """
    Fetch a range of Garmin resting-HR values in one request where supported.
    """
    cache_path = (
        RHR_DIR
        / f"rhr_range_{start_day.isoformat()}_{end_day.isoformat()}.json"
    )

    if cache_path.exists() and not FORCE_REEXPORT:
        try:
            payload = json.loads(
                cache_path.read_text(encoding="utf-8")
            )
            return extract_rhr_history_rows(payload)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    rhr_url = getattr(api, "garmin_connect_rhr_url", None)
    display_name = getattr(api, "display_name", None)

    if not rhr_url or not display_name:
        return []

    url = f"{rhr_url}/{display_name}"
    params = {
        "fromDate": start_day.isoformat(),
        "untilDate": end_day.isoformat(),
        "metricId": 60,
    }

    payload = api.connectapi(url, params=params)
    write_json(cache_path, payload)

    return extract_rhr_history_rows(payload)


def export_garmin_resting_hr_history(api: Garmin) -> None:
    """
    Export recent Garmin resting-HR history.

    Primary method uses date ranges, avoiding hundreds of one-day requests.
    Single-day calls are only a fallback.
    """
    today = date.today()
    start_day = today - timedelta(
        days=RHR_EXPORT_LOOKBACK_DAYS - 1
    )

    print("\nRetrieving Garmin resting-heart-rate history")
    print("-" * 70)
    print(
        f"Requested range: {start_day.isoformat()} "
        f"to {today.isoformat()}"
    )

    rows: list[dict[str, Any]] = []

    # Moderate chunks are gentle on Garmin and cache well.
    current_start = start_day

    try:
        while current_start <= today:
            current_end = min(
                current_start + timedelta(days=179),
                today,
            )

            chunk_rows = fetch_rhr_range(
                api,
                current_start,
                current_end,
            )
            rows.extend(chunk_rows)

            print(
                f"RHR {current_start.isoformat()} "
                f"to {current_end.isoformat()}: "
                f"{len(chunk_rows):,} records."
            )

            current_start = current_end + timedelta(days=1)

            if current_start <= today:
                time.sleep(REQUEST_DELAY_SECONDS)

    except GarminConnectTooManyRequestsError:
        raise
    except Exception as exc:
        print(
            "Garmin RHR range history unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

    # Fallback to day-by-day only if the range endpoint returned nothing.
    if not rows:
        print(
            "No RHR range data returned. "
            "Falling back to daily requests."
        )

        dates = [
            start_day + timedelta(days=i)
            for i in range(
                (today - start_day).days + 1
            )
        ]

        for number, day in enumerate(dates, start=1):
            try:
                rhr = extract_resting_hr(
                    get_daily_hr(api, day)
                )
                if rhr is not None:
                    rows.append(
                        {
                            "calendar_date": day,
                            "resting_hr_bpm": rhr,
                        }
                    )

            except GarminConnectTooManyRequestsError:
                raise
            except Exception:
                pass

            if number % 25 == 0 or number == len(dates):
                print(
                    f"RHR fallback checked {number:,}/"
                    f"{len(dates):,} days."
                )

            time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(rows)

    if df.empty:
        print("WARNING: Garmin returned no resting-HR history.")
        return

    df["calendar_date"] = pd.to_datetime(
        df["calendar_date"],
        errors="coerce",
    ).dt.date
    df["resting_hr_bpm"] = pd.to_numeric(
        df["resting_hr_bpm"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=["calendar_date", "resting_hr_bpm"]
        )
        .sort_values("calendar_date")
        .drop_duplicates("calendar_date", keep="last")
        .reset_index(drop=True)
    )

    write_parquet(df, GARMIN_RHR_FILE)
    df.round(2).to_csv(
        GARMIN_RHR_FILE.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved {len(df):,} resting-HR records "
        f"({df['calendar_date'].min()} to "
        f"{df['calendar_date'].max()})."
    )


# =============================================================================
# Export
# =============================================================================


def export_activity(
    api: Garmin,
    activity: dict[str, Any],
) -> dict[str, Any]:
    activity_id = get_activity_id(activity)

    manifest_row: dict[str, Any] = {
        "garmin_activity_id": activity_id,
        "activity_name": first_present(
            activity,
            ["activityName", "activity_name", "name"],
        ),
        "activity_type": get_activity_type(activity),
        "activity_date": get_activity_date(activity),
        "exported_at": datetime.now(),
        "status": None,
        "record_rows": None,
        "lap_rows": None,
        "session_rows": None,
        "decode_errors": None,
        "error": None,
    }

    if activity_is_complete(activity_id) and not FORCE_REEXPORT:
        manifest_row["status"] = "skipped_existing"
        return manifest_row

    try:
        summary = get_activity_summary(api, activity)
        weather = get_activity_weather(api, activity_id)
        fit_path = get_fit_file(api, activity_id)
        messages, decode_errors = decode_fit(fit_path)
        row_counts = write_activity_tables(
            activity_id,
            summary,
            weather,
            messages,
        )

        manifest_row.update(row_counts)
        manifest_row["status"] = "success"
        manifest_row["decode_errors"] = json.dumps(
            decode_errors,
            ensure_ascii=False,
        )

    except GarminConnectTooManyRequestsError:
        raise

    except Exception as exc:
        manifest_row["status"] = "failed"
        manifest_row["error"] = f"{type(exc).__name__}: {exc}"

    return manifest_row



def export_garmin_data() -> None:
    create_directories()

    print("\nGarmin export")
    print("-" * 70)
    print(f"Project: {PROJECT_DIR}")
    print(f"User data: {USER_DIR}")
    print(f"Tokens: {TOKEN_STORE}")

    api = connect_to_garmin()
    print("Authenticated with Garmin Connect.")

    activities = get_all_running_activities(api)

    if not activities:
        raise ValueError("No running activities found.")

    save_activity_index(activities)

    export_garmin_vo2max_history(api, activities)
    export_garmin_resting_hr_history(api)

    manifest_rows: list[dict[str, Any]] = []

    total_activities = len(activities)

    for number, activity in enumerate(activities, start=1):
        activity_id = get_activity_id(activity)
        activity_name = first_present(
            activity,
            ["activityName", "activity_name", "name"],
            "Running activity",
        )

        result = export_activity(api, activity)
        manifest_rows.append(result)
        save_manifest(manifest_rows)

        if result["status"] == "success":
            print(
                f"Downloaded [{number}/{total_activities}] "
                f"{activity_id} - {activity_name} | "
                f"records={result['record_rows']:,}, "
                f"laps={result['lap_rows']:,}, "
                f"sessions={result['session_rows']:,}"
            )
        elif result["status"] == "failed":
            print(
                f"FAILED [{number}/{total_activities}] "
                f"{activity_id} - {activity_name}: "
                f"{result['error']}"
            )

        if (
            number % 25 == 0
            or number == total_activities
        ):
            processed = pd.DataFrame(manifest_rows)
            successful = int(
                (processed["status"] == "success").sum()
            )
            skipped = int(
                (processed["status"] == "skipped_existing").sum()
            )
            failed = int(
                (processed["status"] == "failed").sum()
            )
            print(
                f"Export progress: {number}/{total_activities} | "
                f"new={successful}, existing={skipped}, failed={failed}"
            )

        if result["status"] != "skipped_existing":
            time.sleep(REQUEST_DELAY_SECONDS)

    current_run = pd.DataFrame(manifest_rows)

    print("\nExport complete")
    print("-" * 70)
    print(f"Activities considered: {len(activities):,}")
    print(f"Successful: {(current_run['status'] == 'success').sum():,}")
    print(f"Skipped existing: {(current_run['status'] == 'skipped_existing').sum():,}")
    print(f"Failed: {(current_run['status'] == 'failed').sum():,}")
    print(f"Record files: {len(list(RECORDS_DIR.glob('*.parquet'))):,}")
    print(f"Lap files: {len(list(LAPS_DIR.glob('*.parquet'))):,}")
    print(f"Session files: {len(list(SESSIONS_DIR.glob('*.parquet'))):,}")


# =============================================================================
# ADVANCED ANALYTICS
# =============================================================================


# =============================================================================
# Configuration
# =============================================================================

# PROJECT_DIR and USER_DIR are shared with the export stage.

# Automatic HR parameter estimation.
HR_MAX_FALLBACK = 200
RESTING_HR_FALLBACK: float | None = None
# HRmax changes relatively slowly and genuinely maximal efforts are
# infrequent, so the model uses a fixed two-year trailing window.
HRMAX_LOOKBACK_DAYS = 730

RHR_LOOKBACK_DAYS = 28

# Sustained HRmax evidence.
#
# Each running activity is reduced to one credible HRmax candidate from its
# record-level heart-rate series. Short isolated spikes are not allowed to
# determine the candidate unless the surrounding sustained HR supports them.
HRMAX_MIN_VALID_EXERCISE_HR = 40
HRMAX_MAX_VALID_EXERCISE_HR = 230
HRMAX_MIN_VALID_ACTIVITY_SECONDS = 300
HRMAX_MAX_INTERPOLATION_GAP_SECONDS = 2

# A raw/session maximum is accepted only when sustained HR is close enough.
# Otherwise the run candidate is simply the highest 10-second average HR.
HRMAX_SESSION_MAX_10S_TOLERANCE_BPM = 3
HRMAX_SESSION_MAX_30S_TOLERANCE_BPM = 7

# Cache record-level evidence because historical weekly/monthly snapshots call
# estimate_hrmax_as_of many times.
_HRMAX_EVIDENCE_CACHE: dict[str, pd.DataFrame] = {}


# Pace equivalents are observed directly from recent running data.
# Each valid moving record contributes its travelled distance as weight.
# For every historical HR zone we calculate the distance-weighted P20, P50
# and P80 of actual pace over the latest 90 days.
PACE_ZONE_LOOKBACK_DAYS = 90
PACE_ZONE_MIN_SPEED_MPS = 1.5
PACE_ZONE_MAX_SPEED_MPS = 12.0
PACE_ZONE_MAX_RECORD_GAP_SECONDS = 10

# Garmin VO2max is used as provided by Garmin. No custom VO2max is calculated.
GARMIN_REFERENCE_MAX_AGE_DAYS = 7

# Set dynamically in main().
HR_MAX = HR_MAX_FALLBACK
RESTING_HR: float | None = RESTING_HR_FALLBACK


# =============================================================================
# Helpers
# =============================================================================


def choose_user_dir() -> Path:
    if not USER_DIR.exists():
        raise FileNotFoundError(f"Garmin user directory not found: {USER_DIR}")
    return USER_DIR


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = weights.sum() / 2.0
    return float(values[np.searchsorted(np.cumsum(weights), cutoff)])


def weighted_percentile(
    values: np.ndarray,
    weights: np.ndarray,
    percentile: float,
) -> float:
    """Distance-weighted percentile for percentile in [0, 1]."""
    mask = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    values = values[mask]
    weights = weights[mask]

    if len(values) == 0:
        return float("nan")

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = float(percentile) * float(weights.sum())
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    index = min(index, len(values) - 1)

    return float(values[index])


def pace_seconds_to_text(value: float | None) -> str | None:
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    minutes = int(value // 60)
    seconds = int(round(value - minutes * 60))
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def temperature_to_celsius(value: Any, column_name: str = "") -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    numeric = float(numeric)
    if column_name.lower().endswith("_c"):
        return numeric
    if 170 <= numeric <= 350:
        return numeric - 273.15
    if numeric > 45:
        return (numeric - 32.0) * 5.0 / 9.0
    return numeric


# =============================================================================
# Automatic HR parameters
# =============================================================================


def load_session_max_hr_lookup(user_dir: Path) -> dict[str, float]:
    """Return the session-level maximum HR for each running activity."""
    sessions_dir = user_dir / "silver" / "sessions"
    lookup: dict[str, float] = {}

    for path in sorted(sessions_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue

        if df.empty:
            continue

        hr_col = first_existing_column(
            df,
            [
                "max_heart_rate",
                "max_hr_bpm",
                "max_hr",
                "maximum_heart_rate",
                "maxHeartRate",
            ],
        )
        if hr_col is None:
            continue

        values = pd.to_numeric(df[hr_col], errors="coerce").dropna()
        values = values[
            values.between(
                HRMAX_MIN_VALID_EXERCISE_HR,
                HRMAX_MAX_VALID_EXERCISE_HR,
                inclusive="both",
            )
        ]

        if not values.empty:
            lookup[str(path.stem)] = float(values.max())

    return lookup


def calculate_sustained_hrmax_evidence(
    record_path: Path,
    session_max_lookup: dict[str, float],
) -> dict[str, Any] | None:
    """
    Build one credible HRmax candidate for one run.

    Heart rate is resampled to one-second resolution. Only gaps of at most two
    seconds are interpolated. Full 10- and 30-second rolling windows are
    required.

    The highest available raw/session maximum is accepted only when:
      peak_10s >= observed_max - 3 bpm
      peak_30s >= observed_max - 7 bpm

    If either check fails, the one-second/raw maximum is treated as unsupported
    and the activity candidate is simply the highest 10-second average HR.
    """
    try:
        raw = pd.read_parquet(record_path)
    except Exception:
        return None

    timestamp_col = first_existing_column(
        raw,
        ["timestamp", "record_timestamp", "time"],
    )
    hr_col = first_existing_column(
        raw,
        ["heart_rate", "heartRate", "hr"],
    )

    if timestamp_col is None or hr_col is None:
        return None

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                raw[timestamp_col],
                errors="coerce",
                utc=True,
            ),
            "heart_rate": pd.to_numeric(
                raw[hr_col],
                errors="coerce",
            ),
        }
    )

    df = df[
        df["heart_rate"].between(
            HRMAX_MIN_VALID_EXERCISE_HR,
            HRMAX_MAX_VALID_EXERCISE_HR,
            inclusive="both",
        )
    ]

    df = (
        df.dropna(subset=["timestamp", "heart_rate"])
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
    )

    if df.empty:
        return None

    hr_1s = (
        df.set_index("timestamp")["heart_rate"]
        .resample("1s")
        .mean()
        .interpolate(
            method="time",
            limit=HRMAX_MAX_INTERPOLATION_GAP_SECONDS,
            limit_area="inside",
        )
    )

    if int(hr_1s.notna().sum()) < HRMAX_MIN_VALID_ACTIVITY_SECONDS:
        return None

    peak_10s = hr_1s.rolling(
        window=10,
        min_periods=10,
    ).mean().max()

    peak_30s = hr_1s.rolling(
        window=30,
        min_periods=30,
    ).mean().max()

    if any(pd.isna(v) for v in [peak_10s, peak_30s]):
        return None

    raw_record_max = float(df["heart_rate"].max())
    activity_id = str(record_path.stem)

    session_max = session_max_lookup.get(activity_id, np.nan)
    session_max = pd.to_numeric(
        pd.Series([session_max]),
        errors="coerce",
    ).iloc[0]

    valid_maxima = [raw_record_max]

    if (
        np.isfinite(session_max)
        and HRMAX_MIN_VALID_EXERCISE_HR
        <= float(session_max)
        <= HRMAX_MAX_VALID_EXERCISE_HR
    ):
        valid_maxima.append(float(session_max))

    # Use the highest available raw/session maximum, then let the sustained
    # 10s/30s evidence decide whether that maximum is credible.
    observed_max = float(max(valid_maxima))

    observed_max_supported = bool(
        float(peak_10s)
        >= observed_max - HRMAX_SESSION_MAX_10S_TOLERANCE_BPM
        and float(peak_30s)
        >= observed_max - HRMAX_SESSION_MAX_30S_TOLERANCE_BPM
    )

    if observed_max_supported:
        activity_candidate = observed_max
        candidate_source = "supported_raw_or_session_max"
    else:
        activity_candidate = float(peak_10s)
        candidate_source = "peak_10s_average"

    # Keep the run-level candidate at full precision. The final HRmax estimate
    # is rounded only after combining the top run candidates.
    activity_candidate = float(
        np.clip(
            activity_candidate,
            120,
            HRMAX_MAX_VALID_EXERCISE_HR,
        )
    )

    possible_spike = bool(
        not observed_max_supported
        and observed_max
        > float(peak_10s)
        + HRMAX_SESSION_MAX_10S_TOLERANCE_BPM
    )

    activity_date = (
        df["timestamp"]
        .min()
        .tz_localize(None)
        .normalize()
    )

    return {
        "garmin_activity_id": activity_id,
        "activity_date": activity_date,
        "raw_record_max_hr_bpm": raw_record_max,
        "session_max_hr_bpm": (
            float(session_max)
            if np.isfinite(session_max)
            else np.nan
        ),
        "peak_10s_hr_bpm": float(peak_10s),
        "peak_30s_hr_bpm": float(peak_30s),
        "observed_max_hr_bpm": observed_max,
        "observed_max_supported": observed_max_supported,
        "possible_spike": possible_spike,
        "activity_hrmax_candidate_bpm": activity_candidate,
        "candidate_source": candidate_source,
    }



def build_sustained_hrmax_evidence(user_dir: Path) -> pd.DataFrame:
    """
    Build/cache one sustained-HR evidence row per running activity.

    This table contains all historical runs. estimate_hrmax_as_of applies the
    730-day trailing window and no-look-ahead filter afterwards.
    """
    cache_key = str(user_dir.expanduser().resolve())

    if cache_key in _HRMAX_EVIDENCE_CACHE:
        return _HRMAX_EVIDENCE_CACHE[cache_key].copy()

    records_dir = user_dir / "silver" / "records"
    record_files = sorted(records_dir.glob("*.parquet"))

    if not record_files:
        evidence = pd.DataFrame()
        _HRMAX_EVIDENCE_CACHE[cache_key] = evidence
        return evidence.copy()

    session_lookup = load_session_max_hr_lookup(user_dir)
    rows: list[dict[str, Any]] = []

    for record_path in record_files:
        row = calculate_sustained_hrmax_evidence(
            record_path,
            session_lookup,
        )
        if row is not None:
            rows.append(row)

    evidence = pd.DataFrame(rows)

    if not evidence.empty:
        evidence["activity_date"] = pd.to_datetime(
            evidence["activity_date"],
            errors="coerce",
        )
        evidence["activity_hrmax_candidate_bpm"] = pd.to_numeric(
            evidence["activity_hrmax_candidate_bpm"],
            errors="coerce",
        )
        evidence = (
            evidence.dropna(
                subset=[
                    "activity_date",
                    "activity_hrmax_candidate_bpm",
                ]
            )
            .sort_values(
                [
                    "activity_date",
                    "activity_hrmax_candidate_bpm",
                ]
            )
            .reset_index(drop=True)
        )

    _HRMAX_EVIDENCE_CACHE[cache_key] = evidence.copy()
    return evidence


def estimate_hrmax_as_of(
    user_dir: Path,
    as_of_date: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Estimate HRmax from record-level running heart-rate data.

    Per-activity validation:
      - heart rate is resampled to one-second resolution;
      - the raw/session maximum is considered supported only when:
            peak_10s >= observed_max - 3 bpm
            peak_30s >= observed_max - 7 bpm
      - if the maximum is not supported, the 10-second peak remains useful
        fallback evidence for that activity.

    Final HRmax selection:
      1. Use only evidence from the previous 730 days and never look forward
         beyond as_of_date.
      2. If one or more supported raw/session maxima exist, select the highest
         supported maximum.
      3. If no supported maximum exists, select the highest credible 10-second
         peak in the 730-day window.
      4. If no usable evidence exists, use the fallback HRmax.

    The fixed two-year window reflects that HRmax changes relatively slowly,
    while genuinely maximal efforts may occur infrequently in normal training.
    """
    fallback = {
        "selected_hrmax_bpm": float(HR_MAX_FALLBACK),
        "hrmax_source": "fallback_no_sustained_evidence",
        "hrmax_activities_used": 0,
        "hrmax_top_values": None,
    }

    evidence = build_sustained_hrmax_evidence(user_dir)
    if evidence.empty:
        return fallback

    if as_of_date is None:
        as_of = evidence["activity_date"].max()
    else:
        as_of = pd.Timestamp(as_of_date)
        if as_of.tzinfo is not None:
            as_of = as_of.tz_localize(None)
        as_of = as_of.normalize()

    lookback_start = (
        as_of
        - timedelta(days=HRMAX_LOOKBACK_DAYS - 1)
    )

    window = evidence[
        (evidence["activity_date"] >= lookback_start)
        & (evidence["activity_date"] <= as_of)
    ].copy()

    if window.empty:
        return fallback

    # Prefer a maximum that passed the within-activity 10s/30s support checks.
    supported = window[
        window["observed_max_supported"] == True  # noqa: E712
    ].copy()

    if not supported.empty:
        supported["observed_max_hr_bpm"] = pd.to_numeric(
            supported["observed_max_hr_bpm"],
            errors="coerce",
        )
        supported = supported.dropna(
            subset=["observed_max_hr_bpm"]
        )

    if not supported.empty:
        selected_row = supported.sort_values(
            [
                "observed_max_hr_bpm",
                "peak_30s_hr_bpm",
                "peak_10s_hr_bpm",
            ],
            ascending=False,
        ).iloc[0]

        selected = float(
            selected_row["observed_max_hr_bpm"]
        )

        return {
            "selected_hrmax_bpm": float(round(selected)),
            "hrmax_source": "highest_supported_max_730d",
            "hrmax_activities_used": 1,
            "hrmax_top_values": str(int(round(selected))),
        }

    # No supported raw/session maximum was available. Use the strongest
    # sustained 10-second evidence rather than an isolated one-second peak.
    window["peak_10s_hr_bpm"] = pd.to_numeric(
        window["peak_10s_hr_bpm"],
        errors="coerce",
    )
    peak10 = window.dropna(
        subset=["peak_10s_hr_bpm"]
    ).sort_values(
        [
            "peak_10s_hr_bpm",
            "peak_30s_hr_bpm",
        ],
        ascending=False,
    )

    if not peak10.empty:
        selected = float(
            peak10.iloc[0]["peak_10s_hr_bpm"]
        )

        return {
            "selected_hrmax_bpm": float(round(selected)),
            "hrmax_source": "highest_peak10_730d_no_supported_max",
            "hrmax_activities_used": 1,
            "hrmax_top_values": str(int(round(selected))),
        }

    return fallback

def estimate_resting_hr_as_of(user_dir: Path, as_of_date: pd.Timestamp | None = None) -> dict[str, Any]:
    """28-day median Garmin resting HR using only dates up to as_of_date."""
    path = user_dir / "silver" / "manifests" / "garmin_resting_hr_history.parquet"

    fallback = {
        "selected_resting_hr_bpm": np.nan if RESTING_HR_FALLBACK is None else RESTING_HR_FALLBACK,
        "resting_hr_source": "fallback" if RESTING_HR_FALLBACK is not None else "unavailable",
        "resting_hr_days_used": 0,
    }

    if not path.exists():
        return fallback

    df = pd.read_parquet(path)
    if "calendar_date" not in df.columns or "resting_hr_bpm" not in df.columns:
        return fallback

    df["calendar_date"] = pd.to_datetime(df["calendar_date"], errors="coerce")
    if getattr(df["calendar_date"].dt, "tz", None) is not None:
        df["calendar_date"] = df["calendar_date"].dt.tz_localize(None)
    df["resting_hr_bpm"] = pd.to_numeric(df["resting_hr_bpm"], errors="coerce")
    df = df[df["resting_hr_bpm"].between(25, 100)].dropna(
        subset=["calendar_date", "resting_hr_bpm"]
    )
    if df.empty:
        return fallback

    if as_of_date is None:
        as_of = df["calendar_date"].max()
    else:
        as_of = pd.Timestamp(as_of_date)
        if as_of.tzinfo is not None:
            as_of = as_of.tz_localize(None)

    recent = df[
        (df["calendar_date"] <= as_of)
        & (df["calendar_date"] >= as_of - timedelta(days=RHR_LOOKBACK_DAYS - 1))
    ]
    if recent.empty:
        return fallback

    selected = float(recent["resting_hr_bpm"].median())
    days = int(recent["calendar_date"].dt.date.nunique())

    return {
        "selected_resting_hr_bpm": round(selected, 1),
        "resting_hr_source": "garmin_28d_median",
        "resting_hr_days_used": days,
    }


def build_heart_rate_parameters(user_dir: Path, as_of_date: pd.Timestamp | None = None) -> pd.DataFrame:
    return pd.DataFrame([{
        **estimate_hrmax_as_of(user_dir, as_of_date),
        **estimate_resting_hr_as_of(user_dir, as_of_date),
    }])


# =============================================================================
# Record/session preparation
# =============================================================================


def normalise_records(raw: pd.DataFrame) -> pd.DataFrame:
    timestamp_col = first_existing_column(raw, ["timestamp", "record_timestamp", "time"])
    hr_col = first_existing_column(raw, ["heart_rate", "heartRate", "hr"])
    distance_col = first_existing_column(raw, ["distance", "total_distance", "distance_m"])
    speed_col = first_existing_column(raw, ["enhanced_speed", "speed", "velocity"])
    altitude_col = first_existing_column(raw, ["altitude_m", "enhanced_altitude", "altitude"])

    if timestamp_col is None or hr_col is None or distance_col is None:
        raise ValueError("FIT records need timestamp, heart rate and distance.")

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[timestamp_col], errors="coerce"),
        "heart_rate": pd.to_numeric(raw[hr_col], errors="coerce"),
        "distance_m": pd.to_numeric(raw[distance_col], errors="coerce"),
        "speed_mps": pd.to_numeric(raw[speed_col], errors="coerce") if speed_col else np.nan,
        "altitude_m": pd.to_numeric(raw[altitude_col], errors="coerce") if altitude_col else np.nan,
    })

    df = df.dropna(subset=["timestamp", "heart_rate", "distance_m"]).sort_values("timestamp")
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    if df.empty:
        return df

    dt = df["timestamp"].diff().dt.total_seconds()
    dd = df["distance_m"].diff()
    derived = (dd / dt).where((dt > 0) & (dt <= 10) & (dd / dt).between(0.5, 12.0))
    df["speed_mps"] = df["speed_mps"].where(df["speed_mps"].between(0.5, 12.0), derived)
    df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def extract_temperature_c(session: pd.DataFrame) -> float | None:
    if session.empty:
        return None
    column = first_existing_column(session, [
        "weather_temperature_c",
        "weather_temperature",
        "weather_temperature_raw",
        "temperature",
        "avg_temperature",
    ])
    if column is None:
        return None
    values = pd.to_numeric(session[column], errors="coerce").dropna()
    if values.empty:
        return None
    return temperature_to_celsius(float(values.iloc[0]), column)


def add_grade(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["grade_pct"] = 0.0
    valid = result["distance_m"].notna() & result["altitude_m"].notna()
    if valid.sum() < 20:
        return result

    source = result.loc[valid, ["distance_m", "altitude_m"]].groupby("distance_m", as_index=False).median()
    source = source.sort_values("distance_m")
    if source["distance_m"].max() - source["distance_m"].min() < 150:
        return result

    grid = np.arange(source["distance_m"].min(), source["distance_m"].max() + 20, 20)
    altitude = np.interp(grid, source["distance_m"], source["altitude_m"])
    altitude = pd.Series(altitude).rolling(3, center=True, min_periods=1).mean().to_numpy()
    grade = np.full(len(grid), np.nan)
    for i in range(3, len(grid) - 3):
        grade[i] = 100.0 * (altitude[i + 3] - altitude[i - 3]) / (grid[i + 3] - grid[i - 3])

    good = np.isfinite(grade)
    if good.sum() >= 2:
        result.loc[valid, "grade_pct"] = np.interp(result.loc[valid, "distance_m"], grid[good], grade[good])
    result["grade_pct"] = result["grade_pct"].clip(-10, 10)
    return result



def equivalent_flat_pace(pace_s: float, grade_pct: float) -> float:
    """
    Conservative grade correction.

    Small gradients (within +/-2%) are left untouched.
    Only the excess gradient beyond 2% is corrected.
    """
    grade = float(np.clip(grade_pct, -MAX_USABLE_GRADE_PCT, MAX_USABLE_GRADE_PCT))

    if grade > HILL_CORRECTION_START_PCT:
        excess = grade - HILL_CORRECTION_START_PCT
        return pace_s - excess * UPHILL_SECONDS_PER_KM_PER_EXCESS_PERCENT

    if grade < -HILL_CORRECTION_START_PCT:
        excess = abs(grade) - HILL_CORRECTION_START_PCT
        return pace_s + excess * DOWNHILL_SECONDS_PER_KM_PER_EXCESS_PERCENT

    return pace_s


def heat_hr_correction_bpm(temp_c: float | None) -> float:
    """
    Small heuristic correction for clearly hot runs.

    We intentionally do nothing below 25 C.
    """
    if temp_c is None or not np.isfinite(temp_c):
        return 0.0

    if temp_c <= HEAT_HR_CORRECTION_START_C:
        return 0.0

    correction = (
        (temp_c - HEAT_HR_CORRECTION_START_C)
        * HEAT_HR_CORRECTION_BPM_PER_C
    )
    return float(np.clip(correction, 0.0, MAX_HEAT_HR_CORRECTION_BPM))


def environment_weight(
    temp_c: float | None,
    median_altitude_m: float | None,
    grade_pct: float,
) -> float:
    """Confidence weight only; no aggressive physiology correction."""
    weight = 1.0

    if temp_c is not None and np.isfinite(temp_c):
        if temp_c > 34:
            weight *= 0.65
        elif temp_c > 30:
            weight *= 0.75
        elif temp_c > 26:
            weight *= 0.88
        elif temp_c > HEAT_CONFIDENCE_START_C:
            weight *= 0.95

    if (
        median_altitude_m is not None
        and np.isfinite(median_altitude_m)
        and median_altitude_m > HIGH_ALTITUDE_START_M
    ):
        if median_altitude_m > 2000:
            weight *= 0.65
        elif median_altitude_m > 1500:
            weight *= 0.75
        else:
            weight *= 0.88

    grade_abs = abs(float(grade_pct))
    if grade_abs > MAX_USABLE_GRADE_PCT:
        return 0.0
    if grade_abs > HILL_CORRECTION_START_PCT:
        weight *= 0.80

    return float(weight)


# =============================================================================
# Distance-weighted pace observations within HR zones
# =============================================================================


def build_pace_observations(
    records: pd.DataFrame,
    activity_id: str,
    activity_date: pd.Timestamp,
    hr_max: float,
) -> pd.DataFrame:
    """
    Convert one run to distance-weighted pace observations.

    Every valid moving record interval contributes actual pace, HR zone and
    travelled distance as its statistical weight.
    """
    if records.empty or not np.isfinite(hr_max) or hr_max <= 0:
        return pd.DataFrame()

    df = records.copy().sort_values("timestamp").reset_index(drop=True)

    df["dt_s"] = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    df["distance_weight_m"] = (
        pd.to_numeric(
            df["distance_m"],
            errors="coerce",
        )
        .diff()
    )

    df["speed_mps"] = pd.to_numeric(
        df["speed_mps"],
        errors="coerce",
    )

    df["heart_rate"] = pd.to_numeric(
        df["heart_rate"],
        errors="coerce",
    )

    valid = (
        df["dt_s"].gt(0)
        & df["dt_s"].le(
            PACE_ZONE_MAX_RECORD_GAP_SECONDS
        )
        & df["distance_weight_m"].gt(0)
        & df["speed_mps"].between(
            PACE_ZONE_MIN_SPEED_MPS,
            PACE_ZONE_MAX_SPEED_MPS,
            inclusive="both",
        )
        & df["heart_rate"].between(
            0.50 * hr_max,
            hr_max,
            inclusive="both",
        )
    )

    df = df.loc[valid].copy()

    if df.empty:
        return pd.DataFrame()

    df["pace_s_per_km"] = (
        1000.0
        / df["speed_mps"]
    )

    hr_fraction = (
        df["heart_rate"]
        / float(hr_max)
    )

    conditions = [
        (hr_fraction >= 0.50) & (hr_fraction < 0.60),
        (hr_fraction >= 0.60) & (hr_fraction < 0.70),
        (hr_fraction >= 0.70) & (hr_fraction < 0.80),
        (hr_fraction >= 0.80) & (hr_fraction < 0.90),
        (hr_fraction >= 0.90) & (hr_fraction <= 1.00),
    ]

    choices = [
        "Z1",
        "Z2",
        "Z3",
        "Z4",
        "Z5",
    ]

    df["zone"] = np.select(
        conditions,
        choices,
        default=None,
    )

    df = df[df["zone"].notna()].copy()

    if df.empty:
        return pd.DataFrame()

    df["garmin_activity_id"] = str(activity_id)
    df["activity_date"] = pd.Timestamp(activity_date)
    df["hrmax_used_bpm"] = float(hr_max)

    return df[
        [
            "garmin_activity_id",
            "activity_date",
            "zone",
            "heart_rate",
            "pace_s_per_km",
            "distance_weight_m",
            "hrmax_used_bpm",
        ]
    ].reset_index(drop=True)


# =============================================================================
# Garmin VO2max history
# =============================================================================


def normalize_calendar_timestamp(values: pd.Series) -> pd.Series:
    """
    Return timezone-free midnight timestamps with an explicit nanosecond dtype.

    Pandas may preserve different datetime resolutions from different input
    sources (for example datetime64[us] vs datetime64[s]). merge_asof requires
    the join keys to have exactly the same dtype, so we normalize both the
    timezone and the datetime unit here.
    """
    parsed = pd.to_datetime(values, errors="coerce", utc=True)
    normalized = parsed.dt.tz_localize(None).dt.normalize()
    return normalized.astype("datetime64[ns]")


def read_garmin_vo2max_history(user_dir: Path) -> pd.DataFrame:
    path = (
        user_dir
        / "silver"
        / "manifests"
        / "garmin_vo2max_history.parquet"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=["calendar_date", "garmin_vo2max_precise"]
        )

    df = pd.read_parquet(path)

    if "calendar_date" not in df.columns:
        return pd.DataFrame(
            columns=["calendar_date", "garmin_vo2max_precise"]
        )

    df["calendar_date"] = normalize_calendar_timestamp(
        df["calendar_date"]
    )

    df["garmin_vo2max_precise"] = pd.to_numeric(
        df.get("garmin_vo2max_precise"),
        errors="coerce",
    )

    return (
        df.dropna(subset=["calendar_date", "garmin_vo2max_precise"])
        .sort_values("calendar_date")
        .drop_duplicates("calendar_date", keep="last")
        .reset_index(drop=True)
    )


def attach_garmin_vo2max_to_runs(
    runs: pd.DataFrame,
    garmin: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the latest Garmin VO2max on or before each run (maximum age 7 days)."""
    if runs.empty:
        return runs.copy()

    left = runs.copy()
    left["calendar_date"] = normalize_calendar_timestamp(
        left["activity_date"]
    )

    if garmin.empty:
        left["garmin_reference_date"] = pd.NaT
        left["garmin_vo2max_precise"] = np.nan
        return left.drop(columns=["calendar_date"])

    right = garmin[["calendar_date", "garmin_vo2max_precise"]].copy()
    right = right.rename(columns={"calendar_date": "garmin_reference_date"})

    left["calendar_date"] = pd.to_datetime(
        left["calendar_date"], errors="coerce"
    ).astype("datetime64[ns]")
    right["garmin_reference_date"] = pd.to_datetime(
        right["garmin_reference_date"], errors="coerce"
    ).astype("datetime64[ns]")

    valid_left = left.dropna(subset=["calendar_date"]).sort_values("calendar_date")
    invalid_left = left[left["calendar_date"].isna()].copy()

    right = right.dropna(subset=["garmin_reference_date"]).sort_values(
        "garmin_reference_date"
    )

    merged = pd.merge_asof(
        valid_left,
        right,
        left_on="calendar_date",
        right_on="garmin_reference_date",
        direction="backward",
        tolerance=pd.Timedelta(days=GARMIN_REFERENCE_MAX_AGE_DAYS),
    )

    if not invalid_left.empty:
        invalid_left["garmin_reference_date"] = pd.NaT
        invalid_left["garmin_vo2max_precise"] = np.nan
        merged = pd.concat([merged, invalid_left], ignore_index=True, sort=False)

    return (
        merged.drop(columns=["calendar_date"], errors="ignore")
        .sort_values(["activity_date", "garmin_activity_id"])
        .reset_index(drop=True)
    )


def latest_garmin_vo2_on_or_before(
    garmin: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[float, pd.Timestamp | pd.NaT]:
    if garmin.empty:
        return np.nan, pd.NaT

    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)

    dates = pd.to_datetime(garmin["calendar_date"], errors="coerce")
    values = pd.to_numeric(garmin["garmin_vo2max_precise"], errors="coerce")
    mask = dates.notna() & values.notna() & (dates <= as_of)

    if not mask.any():
        return np.nan, pd.NaT

    subset = pd.DataFrame({
        "date": dates[mask],
        "value": values[mask],
    }).sort_values("date")

    row = subset.iloc[-1]
    return float(row["value"]), pd.Timestamp(row["date"])


def build_garmin_vo2_snapshot_history(
    garmin: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """Weekly/monthly snapshots containing Garmin VO2max only."""
    rows: list[dict[str, Any]] = []

    for _, period in calendar.iterrows():
        value, reference_date = latest_garmin_vo2_on_or_before(
            garmin,
            pd.Timestamp(period["as_of_date"]),
        )

        row = period.to_dict()
        row.update({
            "garmin_vo2max_precise": value,
            "garmin_reference_date": reference_date,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Zones
# =============================================================================


def build_heart_rate_zones(hr_max: float) -> pd.DataFrame:
    boundaries = [0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    rows = []
    hr_max_int = int(round(hr_max))
    for i in range(5):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        lower_bpm = int(np.ceil(hr_max_int * lower))
        upper_bpm = int(np.ceil(hr_max_int * upper)) - 1 if i < 4 else hr_max_int
        rows.append({
            "zone": f"Z{i + 1}",
            "lower_percent_hrmax": int(lower * 100),
            "upper_percent_hrmax": int(upper * 100),
            "lower_bpm": lower_bpm,
            "upper_bpm": upper_bpm,
        })
    return pd.DataFrame(rows)


def select_pace_zone_observations(
    observations: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Return only observations from the previous 90 days, with no look-ahead."""
    if observations.empty:
        return observations.copy()

    df = observations.copy()

    df["activity_date"] = pd.to_datetime(
        df["activity_date"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)

    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)
    as_of = as_of.normalize()

    lookback_start = (
        as_of
        - timedelta(
            days=PACE_ZONE_LOOKBACK_DAYS - 1
        )
    )

    return df[
        (df["activity_date"] <= as_of)
        & (df["activity_date"] >= lookback_start)
    ].copy()


def build_pace_zones(
    observations: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Observed actual-pace distribution within each historical %HRmax zone.

    P20/P50/P80 are weighted by travelled distance, not sample count.
    """
    usable = select_pace_zone_observations(
        observations,
        as_of_date,
    )

    rows: list[dict[str, Any]] = []

    for zone in ["Z1", "Z2", "Z3", "Z4", "Z5"]:
        subset = usable[
            usable["zone"] == zone
        ].copy()

        pace = pd.to_numeric(
            subset.get("pace_s_per_km"),
            errors="coerce",
        ).to_numpy(float)

        distance = pd.to_numeric(
            subset.get("distance_weight_m"),
            errors="coerce",
        ).to_numpy(float)

        p20 = weighted_percentile(
            pace,
            distance,
            0.20,
        )
        p50 = weighted_percentile(
            pace,
            distance,
            0.50,
        )
        p80 = weighted_percentile(
            pace,
            distance,
            0.80,
        )

        valid_distance = (
            np.isfinite(distance)
            & (distance > 0)
        )

        total_distance_km = (
            float(
                np.nansum(
                    distance[valid_distance]
                )
            )
            / 1000.0
            if len(distance)
            else 0.0
        )

        has_data = (
            np.isfinite(p20)
            and np.isfinite(p50)
            and np.isfinite(p80)
        )

        rows.append({
            "zone": zone,
            "pace_p20_s_per_km": (
                p20 if has_data else np.nan
            ),
            "pace_median_s_per_km": (
                p50 if has_data else np.nan
            ),
            "pace_p80_s_per_km": (
                p80 if has_data else np.nan
            ),
            "pace_median": (
                f"{pace_seconds_to_text(p50)}/km"
                if has_data
                else None
            ),
            "pace_range": (
                f"{pace_seconds_to_text(p20)}–{pace_seconds_to_text(p80)}/km"
                if has_data
                else None
            ),
            "pace_description": (
                f"{pace_seconds_to_text(p50)}/km "
                f"(P20–P80: {pace_seconds_to_text(p20)}–{pace_seconds_to_text(p80)}/km)"
                if has_data
                else "No pace data in the latest 90 days."
            ),
            "distance_km_used": total_distance_km,
            "lookback_days": PACE_ZONE_LOOKBACK_DAYS,
        })

    return pd.DataFrame(rows)



def make_snapshot_calendar(
    first_date: pd.Timestamp,
    last_date: pd.Timestamp,
    frequency: str,
) -> pd.DataFrame:
    first = pd.Timestamp(first_date).normalize()
    last = pd.Timestamp(last_date).normalize()
    if first.tzinfo is not None:
        first = first.tz_localize(None)
    if last.tzinfo is not None:
        last = last.tz_localize(None)

    if frequency == "weekly":
        periods = pd.period_range(first, last, freq="W-SUN")
    elif frequency == "monthly":
        periods = pd.period_range(first, last, freq="M")
    else:
        raise ValueError("frequency must be 'weekly' or 'monthly'")

    rows = []
    for period in periods:
        period_start = period.start_time.normalize()
        period_end = period.end_time.normalize()
        as_of = min(period_end, last)
        rows.append({
            "frequency": frequency,
            "period_label": str(period),
            "period_start": period_start,
            "period_end": period_end,
            "as_of_date": as_of,
        })
    return pd.DataFrame(rows)


def build_parameter_history(
    user_dir: Path,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, period in calendar.iterrows():
        params = build_heart_rate_parameters(user_dir, pd.Timestamp(period["as_of_date"]))
        row = params.iloc[0].to_dict()
        row.update(period.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def build_hr_zone_history(parameter_history: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, row in parameter_history.iterrows():
        zones = build_heart_rate_zones(float(row["selected_hrmax_bpm"]))
        for column in ["frequency", "period_label", "period_start", "period_end", "as_of_date"]:
            zones[column] = row[column]
        zones["selected_hrmax_bpm"] = row["selected_hrmax_bpm"]
        frames.append(zones)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_pace_zone_history(
    pace_observations: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for _, period in calendar.iterrows():
        zones = build_pace_zones(
            pace_observations,
            pd.Timestamp(period["as_of_date"]),
        )
        for column in ["frequency", "period_label", "period_start", "period_end", "as_of_date"]:
            zones[column] = period[column]
        frames.append(zones)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_value_on_or_before(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    as_of_date: pd.Timestamp,
) -> float:
    if df.empty or value_column not in df.columns:
        return np.nan
    dates = pd.to_datetime(df[date_column], errors="coerce", utc=True).dt.tz_localize(None)
    as_of = pd.Timestamp(as_of_date)
    if as_of.tzinfo is not None:
        as_of = as_of.tz_localize(None)
    values = pd.to_numeric(df[value_column], errors="coerce")
    mask = (dates <= as_of) & values.notna()
    if not mask.any():
        return np.nan
    subset = pd.DataFrame({"date": dates[mask], "value": values[mask]}).sort_values("date")
    return float(subset.iloc[-1]["value"])


# =============================================================================
# Output
# =============================================================================


def rounded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round every numeric CSV column to 2 decimals.
    Internal calculations and Parquet output keep full precision.
    """
    result = df.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(2)
    return result


def write_table(df: pd.DataFrame, base: Path) -> None:
    df.to_parquet(base.with_suffix(".parquet"), index=False)
    rounded_csv(df).to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")


# =============================================================================
# Main
# =============================================================================


def remove_legacy_custom_vo2_outputs(output_dir: Path) -> None:
    """Delete outputs from the retired custom VO2max model, if they exist."""
    legacy_basenames = [
        "vo2max_history",
        "vo2max_model_validation",
        "vo2max_weekly",
        "vo2max_monthly",
    ]

    for basename in legacy_basenames:
        for suffix in [".parquet", ".csv"]:
            path = output_dir / f"{basename}{suffix}"
            if path.exists():
                path.unlink()


def run_advanced_analytics() -> None:
    global HR_MAX
    global RESTING_HR

    user_dir = choose_user_dir()
    records_dir = user_dir / "silver" / "records"
    sessions_dir = user_dir / "silver" / "sessions"
    output_dir = user_dir / "gold" / "advanced_analytics"
    output_dir.mkdir(parents=True, exist_ok=True)

    remove_legacy_custom_vo2_outputs(output_dir)

    record_files = sorted(records_dir.glob("*.parquet"))
    if not record_files:
        raise FileNotFoundError(f"No record files found in {records_dir}.")

    current_parameters = build_heart_rate_parameters(user_dir)
    HR_MAX = int(round(float(current_parameters.loc[0, "selected_hrmax_bpm"])))
    rhr = pd.to_numeric(
        pd.Series([current_parameters.loc[0, "selected_resting_hr_bpm"]]),
        errors="coerce",
    ).iloc[0]
    RESTING_HR = None if pd.isna(rhr) else float(rhr)

    print("\nAdvanced Running Analytics")
    print("-" * 70)
    print(f"User: {user_dir.name}")
    print(f"Current HRmax: {HR_MAX} bpm")
    print(f"Current resting HR: {RESTING_HR if RESTING_HR is not None else 'unavailable'}")
    print(f"Running activities: {len(record_files):,}\n")

    run_rows: list[dict[str, Any]] = []
    pace_observation_frames: list[pd.DataFrame] = []
    analytics_failed = 0
    total_record_files = len(record_files)

    for number, record_path in enumerate(record_files, start=1):
        activity_id = record_path.stem

        try:
            records = normalise_records(pd.read_parquet(record_path))
            if records.empty:
                raise ValueError("No usable FIT records.")

            activity_date = records["timestamp"].min()
            activity_as_of = pd.Timestamp(activity_date)
            if activity_as_of.tzinfo is not None:
                activity_as_of = activity_as_of.tz_localize(None)

            run_parameters = build_heart_rate_parameters(user_dir, activity_as_of)
            run_hrmax = int(round(float(run_parameters.loc[0, "selected_hrmax_bpm"])))
            run_rhr = pd.to_numeric(
                pd.Series([run_parameters.loc[0, "selected_resting_hr_bpm"]]),
                errors="coerce",
            ).iloc[0]

            pace_observations = build_pace_observations(
                records,
                activity_id,
                activity_date,
                run_hrmax,
            )

            if not pace_observations.empty:
                pace_observation_frames.append(
                    pace_observations
                )

            observed_distance_km = (
                float(
                    pace_observations[
                        "distance_weight_m"
                    ].sum()
                )
                / 1000.0
                if not pace_observations.empty
                else 0.0
            )

            run_rows.append({
                "garmin_activity_id": activity_id,
                "activity_date": activity_date,
                "selected_hrmax_bpm": run_hrmax,
                "selected_resting_hr_bpm": (
                    np.nan if pd.isna(run_rhr) else float(run_rhr)
                ),
                "pace_observation_distance_km": observed_distance_km,
            })

        except Exception as exc:
            analytics_failed += 1
            print(
                f"Analytics FAILED [{number}/{total_record_files}] "
                f"{activity_id}: {type(exc).__name__}: {exc}"
            )
            run_rows.append({
                "garmin_activity_id": activity_id,
                "activity_date": pd.NaT,
                "selected_hrmax_bpm": np.nan,
                "selected_resting_hr_bpm": np.nan,
                "pace_observation_distance_km": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
            })

        if (
            number % 25 == 0
            or number == total_record_files
        ):
            print(
                f"Analytics progress: "
                f"{number}/{total_record_files} activities"
            )

    run_history = pd.DataFrame(run_rows)
    run_history["activity_date"] = pd.to_datetime(
        run_history["activity_date"], errors="coerce", utc=True
    )

    pace_observations = (
        pd.concat(
            pace_observation_frames,
            ignore_index=True,
            sort=False,
        )
        if pace_observation_frames
        else pd.DataFrame()
    )

    garmin = read_garmin_vo2max_history(user_dir)
    garmin_by_run = attach_garmin_vo2max_to_runs(
        run_history[["garmin_activity_id", "activity_date"]],
        garmin,
    )

    valid_dates = run_history["activity_date"].dropna()
    if valid_dates.empty:
        raise ValueError("No valid activity dates were available for historical snapshots.")

    first_date = valid_dates.min().tz_localize(None)
    last_date = valid_dates.max().tz_localize(None)

    weekly_calendar = make_snapshot_calendar(first_date, last_date, "weekly")
    monthly_calendar = make_snapshot_calendar(first_date, last_date, "monthly")

    parameters_weekly = build_parameter_history(user_dir, weekly_calendar)
    parameters_monthly = build_parameter_history(user_dir, monthly_calendar)

    hr_zones_weekly = build_hr_zone_history(parameters_weekly)
    hr_zones_monthly = build_hr_zone_history(parameters_monthly)

    pace_zones_weekly = build_pace_zone_history(
        pace_observations,
        weekly_calendar,
    )
    pace_zones_monthly = build_pace_zone_history(
        pace_observations,
        monthly_calendar,
    )

    garmin_vo2_weekly = build_garmin_vo2_snapshot_history(
        garmin, weekly_calendar
    )
    garmin_vo2_monthly = build_garmin_vo2_snapshot_history(
        garmin, monthly_calendar
    )

    current_hr_zones = build_heart_rate_zones(HR_MAX)
    current_pace_zones = build_pace_zones(
        pace_observations,
        last_date,
    )

    write_table(current_parameters, output_dir / "heart_rate_parameters")
    write_table(
        run_history[
            [
                "garmin_activity_id",
                "activity_date",
                "selected_hrmax_bpm",
                "selected_resting_hr_bpm",
            ]
        ],
        output_dir / "heart_rate_parameters_by_run",
    )
    write_table(garmin_by_run, output_dir / "garmin_vo2max_by_run")
    write_table(current_hr_zones, output_dir / "heart_rate_zones")
    write_table(current_pace_zones, output_dir / "pace_zones")

    write_table(parameters_weekly, output_dir / "heart_rate_parameters_weekly")
    write_table(parameters_monthly, output_dir / "heart_rate_parameters_monthly")
    write_table(hr_zones_weekly, output_dir / "heart_rate_zones_weekly")
    write_table(hr_zones_monthly, output_dir / "heart_rate_zones_monthly")
    write_table(pace_zones_weekly, output_dir / "pace_zones_weekly")
    write_table(pace_zones_monthly, output_dir / "pace_zones_monthly")
    write_table(garmin_vo2_weekly, output_dir / "garmin_vo2max_weekly")
    write_table(garmin_vo2_monthly, output_dir / "garmin_vo2max_monthly")

    print("\nComplete")
    print("-" * 70)
    print(f"Activities processed: {len(record_files):,}")
    print(f"Failed activities:    {analytics_failed:,}")
    print(f"Weekly snapshots:     {len(weekly_calendar):,}")
    print(f"Monthly snapshots:    {len(monthly_calendar):,}")
    print(f"Output folder: {output_dir}")
    print(
        "Created weekly/monthly histories for HR parameters, HR zones, "
        "pace zones and Garmin VO2max."
    )

    if not garmin.empty:
        latest_garmin = garmin.sort_values("calendar_date").iloc[-1]
        print(
            f"Current Garmin VO2max: "
            f"{float(latest_garmin['garmin_vo2max_precise']):.1f}"
        )


# =============================================================================
# DASHBOARD BUILD
# =============================================================================


# =============================================================================
# SETTINGS
# =============================================================================

# PROJECT_DIR and USER_DIR are shared with the earlier stages.

# Optional:
# Point this to the data folder inside your LOCAL clone of your personal
# website repository. If set, the website-ready JSON files are copied there
# automatically every time this script runs.
#
# Example:
# WEBSITE_DATA_DIR = Path(
#     r"C:\Users\chess\Documents\sietse-van-meer.github.io\data\running"
# )
WEBSITE_DATA_DIR: Path | None = PROJECT_DIR / "data" / "running"

# Whether to classify record-level time into historical HR and pace zones.
# This is the slowest part of the script, but useful for dashboard charts.
BUILD_ZONE_DISTRIBUTION = True

# Gaps larger than this are not counted as active running time.
MAX_RECORD_GAP_SECONDS = 10.0

# Website bundle stays compact.
WEBSITE_MAX_RUNS = 250

# CSV/dashboard presentation precision.
CSV_DECIMALS = 2

# Edwards TRIMP / Summated Heart Rate Zone load.
# Load is calculated per run and summed for weekly/monthly totals.
EDWARDS_ZONE_WEIGHTS = {
    "Z1": 1,
    "Z2": 2,
    "Z3": 3,
    "Z4": 4,
    "Z5": 5,
}

# Current race predictions.
RACE_PREDICTION_LOOKBACK_DAYS = 90
RIEGEL_EXPONENT = 1.07

# Distances scanned as recent performance anchors.
RACE_ANCHORS = (
    ("2_5k", "2.5K", 2500.0),
    ("5k", "5K", 5000.0),
    ("10k", "10K", 10000.0),
    ("10_miles", "10 miles", 16093.44),
    ("half_marathon", "Half marathon", 21097.5),
)

# Distances displayed as predictions.
RACE_TARGETS = (
    ("5k", "5K", 5000.0),
    ("10k", "10K", 10000.0),
    ("10_miles", "10 miles", 16093.44),
    ("half_marathon", "Half marathon", 21097.5),
)

# Short anchors are deliberately prevented from driving long-distance
# predictions. This keeps 2K useful as a current-speed signal without letting
# it produce a half-marathon estimate.
RACE_ALLOWED_ANCHORS = {
    "5k": {
        "2_5k",
        "5k",
        "10k",
        "10_miles",
        "half_marathon",
    },
    "10k": {
        "2_5k",
        "5k",
        "10k",
        "10_miles",
        "half_marathon",
    },
    "10_miles": {
        "5k",
        "10k",
        "10_miles",
        "half_marathon",
    },
    "half_marathon": {
        "5k",
        "10k",
        "10_miles",
        "half_marathon",
    },
}


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def choose_user_dir() -> Path:
    if not USER_DIR.exists():
        raise FileNotFoundError(f"Garmin user directory not found: {USER_DIR}")
    return USER_DIR


def first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def first_valid_value(
    dataframe: pd.DataFrame,
    candidates: list[str],
    default: Any = np.nan,
) -> Any:
    column = first_existing_column(dataframe, candidates)

    if column is None:
        return default

    values = dataframe[column].dropna()

    if values.empty:
        return default

    return values.iloc[0]


def numeric_value(
    dataframe: pd.DataFrame,
    candidates: list[str],
    default: float = np.nan,
) -> float:
    column = first_existing_column(dataframe, candidates)

    if column is None:
        return default

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return default

    return float(values.iloc[0])


def sum_numeric(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = first_existing_column(dataframe, candidates)

    if column is None:
        return np.nan

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.sum())


def max_numeric(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float:
    column = first_existing_column(dataframe, candidates)

    if column is None:
        return np.nan

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).dropna()

    if values.empty:
        return np.nan

    return float(values.max())


def weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")

    mask = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not mask.any():
        return np.nan

    return float(
        np.average(
            values[mask].to_numpy(float),
            weights=weights[mask].to_numpy(float),
        )
    )


def to_datetime_naive(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(
        values,
        errors="coerce",
        utc=True,
    )
    return parsed.dt.tz_localize(None)


def round_numeric_output(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    numeric_columns = result.select_dtypes(
        include=[np.number]
    ).columns

    result[numeric_columns] = result[
        numeric_columns
    ].round(CSV_DECIMALS)

    return result


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, CSV_DECIMALS)

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if pd.isna(value):
        return None

    return value


def dataframe_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    df = round_numeric_output(dataframe)

    records: list[dict[str, Any]] = []

    for row in df.to_dict(orient="records"):
        records.append(
            {
                str(key): json_safe(value)
                for key, value in row.items()
            }
        )

    return records


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def write_table(
    dataframe: pd.DataFrame,
    base_path: Path,
) -> None:
    base_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        base_path.with_suffix(".parquet"),
        index=False,
    )

    round_numeric_output(
        dataframe
    ).to_csv(
        base_path.with_suffix(".csv"),
        index=False,
        encoding="utf-8-sig",
    )


def read_optional_parquet(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def temperature_to_celsius(
    value: float,
) -> float:
    if not np.isfinite(value):
        return np.nan

    # Garmin weather values in some responses are Fahrenheit.
    if 170 <= value <= 350:
        return float(value - 273.15)

    if value > 45:
        return float(
            (value - 32.0) * 5.0 / 9.0
        )

    return float(value)


def pace_seconds_to_text(
    pace_s_per_km: float,
) -> str | None:
    if (
        not np.isfinite(pace_s_per_km)
        or pace_s_per_km <= 0
    ):
        return None

    minutes = int(pace_s_per_km // 60)
    seconds = int(
        round(
            pace_s_per_km - minutes * 60
        )
    )

    if seconds == 60:
        minutes += 1
        seconds = 0

    return f"{minutes}:{seconds:02d}"


# =============================================================================
# RUN TABLE
# =============================================================================

def build_runs_table(
    user_dir: Path,
) -> pd.DataFrame:
    sessions_dir = (
        user_dir
        / "silver"
        / "sessions"
    )

    if not sessions_dir.exists():
        raise FileNotFoundError(
            f"Sessions directory not found: {sessions_dir}"
        )

    rows: list[dict[str, Any]] = []

    for path in sorted(
        sessions_dir.glob("*.parquet")
    ):
        activity_id = path.stem

        try:
            sessions = pd.read_parquet(path)
        except Exception as exc:
            print(
                f"Skipping session {activity_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if sessions.empty:
            continue

        start_value = first_valid_value(
            sessions,
            [
                "activity_start_time_local",
                "start_time",
                "timestamp",
                "activity_start_time_gmt",
            ],
            default=pd.NaT,
        )

        start_time = pd.to_datetime(
            start_value,
            errors="coerce",
        )

        if (
            isinstance(start_time, pd.Timestamp)
            and start_time.tzinfo is not None
        ):
            start_time = start_time.tz_localize(
                None
            )

        distance_m = sum_numeric(
            sessions,
            [
                "total_distance",
                "distance",
                "distance_m",
            ],
        )

        duration_s = sum_numeric(
            sessions,
            [
                "total_timer_time",
                "total_elapsed_time",
                "duration",
                "duration_s",
            ],
        )

        elevation_gain_m = sum_numeric(
            sessions,
            [
                "elevation_gain_m",
                "total_ascent",
            ],
        )

        elevation_loss_m = sum_numeric(
            sessions,
            [
                "elevation_loss_m",
                "total_descent",
            ],
        )

        avg_hr_col = first_existing_column(
            sessions,
            [
                "avg_heart_rate",
                "average_heart_rate",
                "average_hr",
                "averageHR",
            ],
        )

        timer_col = first_existing_column(
            sessions,
            [
                "total_timer_time",
                "total_elapsed_time",
                "duration",
            ],
        )

        if (
            avg_hr_col is not None
            and timer_col is not None
        ):
            avg_hr_bpm = weighted_average(
                sessions[avg_hr_col],
                sessions[timer_col],
            )
        else:
            avg_hr_bpm = numeric_value(
                sessions,
                [
                    "avg_heart_rate",
                    "average_heart_rate",
                    "average_hr",
                    "averageHR",
                ],
            )

        max_hr_bpm = max_numeric(
            sessions,
            [
                "max_heart_rate",
                "maximum_heart_rate",
                "max_hr",
                "maxHR",
            ],
        )

        temperature_raw = numeric_value(
            sessions,
            [
                "weather_temperature_c",
                "weather_temperature",
                "temperature",
                "avg_temperature",
            ],
        )

        if np.isfinite(temperature_raw):
            temp_column = first_existing_column(
                sessions,
                [
                    "weather_temperature_c",
                    "weather_temperature",
                    "temperature",
                    "avg_temperature",
                ],
            )

            if (
                temp_column is not None
                and temp_column.endswith("_c")
            ):
                temperature_c = temperature_raw
            else:
                temperature_c = temperature_to_celsius(
                    temperature_raw
                )
        else:
            temperature_c = np.nan

        distance_km = (
            distance_m / 1000.0
            if np.isfinite(distance_m)
            else np.nan
        )

        duration_min = (
            duration_s / 60.0
            if np.isfinite(duration_s)
            else np.nan
        )

        pace_s_per_km = (
            duration_s / distance_km
            if (
                np.isfinite(duration_s)
                and np.isfinite(distance_km)
                and distance_km > 0
            )
            else np.nan
        )

        rows.append(
            {
                "garmin_activity_id": str(
                    activity_id
                ),
                "activity_date": (
                    start_time.normalize()
                    if isinstance(
                        start_time,
                        pd.Timestamp,
                    )
                    else pd.NaT
                ),
                "start_time_local": start_time,
                "activity_name": first_valid_value(
                    sessions,
                    [
                        "activity_name",
                        "name",
                    ],
                    default=None,
                ),
                "activity_type": first_valid_value(
                    sessions,
                    ["activity_type"],
                    default="running",
                ),
                "distance_km": distance_km,
                "duration_min": duration_min,
                "duration_hours": (
                    duration_s / 3600.0
                    if np.isfinite(duration_s)
                    else np.nan
                ),
                "pace_s_per_km": pace_s_per_km,
                "pace": pace_seconds_to_text(
                    pace_s_per_km
                ),
                "avg_hr_bpm": avg_hr_bpm,
                "max_hr_bpm": max_hr_bpm,
                "elevation_gain_m": elevation_gain_m,
                "elevation_loss_m": elevation_loss_m,
                "temperature_c": temperature_c,
            }
        )

    runs = pd.DataFrame(rows)

    if runs.empty:
        return runs

    runs["activity_date"] = pd.to_datetime(
        runs["activity_date"],
        errors="coerce",
    )

    runs["start_time_local"] = pd.to_datetime(
        runs["start_time_local"],
        errors="coerce",
    )

    runs = (
        runs.sort_values(
            [
                "activity_date",
                "start_time_local",
                "garmin_activity_id",
            ]
        )
        .drop_duplicates(
            "garmin_activity_id",
            keep="last",
        )
        .reset_index(drop=True)
    )

    return runs


def add_advanced_metrics_to_runs(
    runs: pd.DataFrame,
    advanced_dir: Path,
) -> pd.DataFrame:
    if runs.empty:
        return runs

    result = runs.copy()

    garmin_vo2 = read_optional_parquet(
        advanced_dir
        / "garmin_vo2max_by_run.parquet"
    )

    if not garmin_vo2.empty:
        keep_columns = [
            column
            for column in [
                "garmin_activity_id",
                "garmin_reference_date",
                "garmin_vo2max_precise",
            ]
            if column in garmin_vo2.columns
        ]

        garmin_vo2 = garmin_vo2[keep_columns].copy()
        garmin_vo2["garmin_activity_id"] = (
            garmin_vo2["garmin_activity_id"]
            .astype(str)
        )

        result = result.merge(
            garmin_vo2,
            on="garmin_activity_id",
            how="left",
        )

    parameters_by_run = read_optional_parquet(
        advanced_dir
        / "heart_rate_parameters_by_run.parquet"
    )

    if not parameters_by_run.empty:
        keep_columns = [
            column
            for column in [
                "garmin_activity_id",
                "selected_hrmax_bpm",
                "selected_resting_hr_bpm",
            ]
            if column in parameters_by_run.columns
        ]

        parameters_by_run = (
            parameters_by_run[keep_columns]
            .copy()
        )
        parameters_by_run[
            "garmin_activity_id"
        ] = parameters_by_run[
            "garmin_activity_id"
        ].astype(str)

        result["garmin_activity_id"] = (
            result["garmin_activity_id"]
            .astype(str)
        )

        result = result.merge(
            parameters_by_run,
            on="garmin_activity_id",
            how="left",
        )

    else:
        # Backward-compatible fallback for an older advanced-analytics output.
        parameters = read_optional_parquet(
            advanced_dir
            / "heart_rate_parameters_weekly.parquet"
        )

        if not parameters.empty:
            result = attach_period_parameters(
                result,
                parameters,
            )

    return result


def attach_period_parameters(
    runs: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    result = runs.copy()

    params = parameters.copy()

    for column in [
        "period_start",
        "period_end",
        "as_of_date",
    ]:
        if column in params.columns:
            params[column] = pd.to_datetime(
                params[column],
                errors="coerce",
            )

    columns = [
        column
        for column in [
            "period_start",
            "period_end",
            "selected_hrmax_bpm",
            "selected_resting_hr_bpm",
        ]
        if column in params.columns
    ]

    params = params[columns]

    parameter_rows = []

    for _, run in result.iterrows():
        date = pd.Timestamp(
            run["activity_date"]
        )

        if pd.isna(date):
            parameter_rows.append({})
            continue

        matching = params[
            (params["period_start"] <= date)
            & (params["period_end"] >= date)
        ]

        if matching.empty:
            earlier = params[
                params["period_end"] <= date
            ].sort_values("period_end")

            matching = earlier.tail(1)

        if matching.empty:
            parameter_rows.append({})
        else:
            row = (
                matching
                .iloc[-1]
                .drop(
                    labels=[
                        "period_start",
                        "period_end",
                    ],
                    errors="ignore",
                )
                .to_dict()
            )
            parameter_rows.append(row)

    extra = pd.DataFrame(
        parameter_rows,
        index=result.index,
    )

    for column in extra.columns:
        result[column] = extra[column]

    return result


# =============================================================================
# PERIOD AGGREGATION
# =============================================================================

def add_period_columns(
    runs: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    df = runs.copy()

    if frequency == "weekly":
        periods = df[
            "activity_date"
        ].dt.to_period("W-SUN")
    elif frequency == "monthly":
        periods = df[
            "activity_date"
        ].dt.to_period("M")
    else:
        raise ValueError(
            "frequency must be weekly or monthly"
        )

    df["period_label"] = periods.astype(
        str
    )
    df["period_start"] = periods.apply(
        lambda value: (
            value.start_time.normalize()
            if pd.notna(value)
            else pd.NaT
        )
    )
    df["period_end"] = periods.apply(
        lambda value: (
            value.end_time.normalize()
            if pd.notna(value)
            else pd.NaT
        )
    )

    return df


def aggregate_runs(
    runs: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    df = add_period_columns(
        runs,
        frequency,
    )

    rows: list[dict[str, Any]] = []

    for (
        period_label,
        period_start,
        period_end,
    ), group in df.groupby(
        [
            "period_label",
            "period_start",
            "period_end",
        ],
        dropna=False,
    ):
        total_distance_km = pd.to_numeric(
            group["distance_km"],
            errors="coerce",
        ).sum(min_count=1)

        total_duration_min = pd.to_numeric(
            group["duration_min"],
            errors="coerce",
        ).sum(min_count=1)

        total_elevation_gain_m = (
            pd.to_numeric(
                group["elevation_gain_m"],
                errors="coerce",
            )
            .sum(min_count=1)
        )

        overall_pace_s = (
            total_duration_min
            * 60.0
            / total_distance_km
            if (
                np.isfinite(total_duration_min)
                and np.isfinite(total_distance_km)
                and total_distance_km > 0
            )
            else np.nan
        )

        avg_hr = weighted_average(
            group["avg_hr_bpm"],
            group["duration_min"],
        )

        latest = (
            group.sort_values(
                "activity_date"
            )
            .iloc[-1]
        )

        training_load_edwards = (
            pd.to_numeric(
                group["training_load_edwards"],
                errors="coerce",
            ).sum(min_count=1)
            if "training_load_edwards" in group.columns
            else np.nan
        )

        row = {
            "frequency": frequency,
            "period_label": period_label,
            "period_start": period_start,
            "period_end": period_end,
            "runs": int(len(group)),
            "total_distance_km": total_distance_km,
            "total_duration_hours": (
                total_duration_min / 60.0
                if np.isfinite(
                    total_duration_min
                )
                else np.nan
            ),
            "average_run_distance_km": pd.to_numeric(
                group["distance_km"],
                errors="coerce",
            ).mean(),
            "longest_run_km": pd.to_numeric(
                group["distance_km"],
                errors="coerce",
            ).max(),
            "average_pace_s_per_km": overall_pace_s,
            "average_pace": pace_seconds_to_text(
                overall_pace_s
            ),
            "duration_weighted_avg_hr_bpm": avg_hr,
            "total_elevation_gain_m": (
                total_elevation_gain_m
            ),
            "average_temperature_c": pd.to_numeric(
                group["temperature_c"],
                errors="coerce",
            ).mean(),
            "training_load_edwards": training_load_edwards,
        }

        for column in [
            "garmin_vo2max_precise",
            "selected_hrmax_bpm",
            "selected_resting_hr_bpm",
        ]:
            if column in group.columns:
                values = (
                    group[
                        [
                            "activity_date",
                            column,
                        ]
                    ]
                    .dropna(
                        subset=[column]
                    )
                    .sort_values(
                        "activity_date"
                    )
                )

                row[column] = (
                    values.iloc[-1][column]
                    if not values.empty
                    else np.nan
                )

        rows.append(row)

    result = pd.DataFrame(rows)

    return result.sort_values(
        "period_start"
    ).reset_index(drop=True)


def merge_snapshot_table(
    period_table: pd.DataFrame,
    snapshot: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if (
        period_table.empty
        or snapshot.empty
    ):
        return period_table

    result = period_table.copy()
    snap = snapshot.copy()

    if "period_label" not in snap.columns:
        return result

    keep = [
        column
        for column in (
            ["period_label"] + columns
        )
        if column in snap.columns
    ]

    snap = (
        snap[keep]
        .drop_duplicates(
            "period_label",
            keep="last",
        )
    )

    for column in keep:
        if (
            column != "period_label"
            and column in result.columns
        ):
            result = result.drop(
                columns=[column]
            )

    return result.merge(
        snap,
        on="period_label",
        how="left",
    )


# =============================================================================
# HISTORICAL ZONE LOOKUP
# =============================================================================

def zone_rows_for_date(
    zones: pd.DataFrame,
    date: pd.Timestamp,
) -> pd.DataFrame:
    if zones.empty:
        return pd.DataFrame()

    df = zones.copy()

    df["period_start"] = pd.to_datetime(
        df["period_start"],
        errors="coerce",
    )
    df["period_end"] = pd.to_datetime(
        df["period_end"],
        errors="coerce",
    )

    matching = df[
        (df["period_start"] <= date)
        & (df["period_end"] >= date)
    ]

    if not matching.empty:
        return matching.copy()

    earlier = df[
        df["period_end"] <= date
    ]

    if earlier.empty:
        return pd.DataFrame()

    latest_end = earlier[
        "period_end"
    ].max()

    return earlier[
        earlier["period_end"]
        == latest_end
    ].copy()


def hr_zones_from_hrmax(
    hr_max: float,
) -> pd.DataFrame:
    """Build the five fixed %HRmax zones for one activity."""
    if not np.isfinite(hr_max):
        return pd.DataFrame()

    hr_max_int = int(round(float(hr_max)))
    boundaries = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ]

    rows = []

    for index in range(5):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        rows.append({
            "zone": f"Z{index + 1}",
            "lower_bpm": int(
                np.ceil(
                    hr_max_int * lower
                )
            ),
            "upper_bpm": (
                int(
                    np.ceil(
                        hr_max_int * upper
                    )
                )
                - 1
                if index < 4
                else hr_max_int
            ),
        })

    return pd.DataFrame(rows)


def classify_hr_zone(
    hr_values: pd.Series,
    zones: pd.DataFrame,
) -> pd.Series:
    result = pd.Series(
        None,
        index=hr_values.index,
        dtype="object",
    )

    hr = pd.to_numeric(
        hr_values,
        errors="coerce",
    )

    for _, zone in zones.iterrows():
        if (
            zone.get("zone")
            not in {
                "Z1",
                "Z2",
                "Z3",
                "Z4",
                "Z5",
            }
        ):
            continue

        lower = pd.to_numeric(
            pd.Series(
                [zone.get("lower_bpm")]
            ),
            errors="coerce",
        ).iloc[0]

        upper = pd.to_numeric(
            pd.Series(
                [zone.get("upper_bpm")]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            pd.isna(lower)
            or pd.isna(upper)
        ):
            continue

        mask = (
            hr.between(
                float(lower),
                float(upper),
                inclusive="both",
            )
        )

        result.loc[mask] = zone["zone"]

    return result


def classify_pace_zone(
    pace_s_per_km: pd.Series,
    zones: pd.DataFrame,
) -> pd.Series:
    result = pd.Series(
        None,
        index=pace_s_per_km.index,
        dtype="object",
    )

    pace = pd.to_numeric(
        pace_s_per_km,
        errors="coerce",
    )

    for _, zone in zones.iterrows():
        zone_name = zone.get("zone")

        if zone_name not in {
            "Z1",
            "Z2",
            "Z3",
            "Z4",
            "Z5",
        }:
            continue

        slower = pd.to_numeric(
            pd.Series(
                [
                    zone.get(
                        "slower_than_s_per_km"
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        faster = pd.to_numeric(
            pd.Series(
                [
                    zone.get(
                        "faster_than_s_per_km"
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if zone_name == "Z1":
            mask = (
                pace >= slower
                if not pd.isna(slower)
                else False
            )

        elif zone_name == "Z5":
            mask = (
                pace < faster
                if not pd.isna(faster)
                else False
            )

        else:
            if (
                pd.isna(slower)
                or pd.isna(faster)
            ):
                continue

            # Example Z2:
            # slower boundary 6:00/km,
            # faster boundary 5:20/km.
            mask = (
                (pace < slower)
                & (pace >= faster)
            )

        result.loc[mask] = zone_name

    return result


def prepare_record_data(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    timestamp_col = first_existing_column(
        raw,
        [
            "timestamp",
            "record_timestamp",
            "time",
        ],
    )

    hr_col = first_existing_column(
        raw,
        [
            "heart_rate",
            "heartRate",
            "hr",
        ],
    )

    distance_col = first_existing_column(
        raw,
        [
            "distance",
            "distance_m",
            "total_distance",
        ],
    )

    speed_col = first_existing_column(
        raw,
        [
            "enhanced_speed",
            "speed",
            "velocity",
        ],
    )

    if timestamp_col is None:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                raw[timestamp_col],
                errors="coerce",
            ),
        }
    )

    df["heart_rate"] = (
        pd.to_numeric(
            raw[hr_col],
            errors="coerce",
        )
        if hr_col
        else np.nan
    )

    df["distance_m"] = (
        pd.to_numeric(
            raw[distance_col],
            errors="coerce",
        )
        if distance_col
        else np.nan
    )

    df["speed_mps"] = (
        pd.to_numeric(
            raw[speed_col],
            errors="coerce",
        )
        if speed_col
        else np.nan
    )

    df = (
        df.dropna(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    if df.empty:
        return df

    delta_t = (
        df["timestamp"]
        .shift(-1)
        .sub(df["timestamp"])
        .dt.total_seconds()
    )

    median_gap = delta_t[
        (delta_t > 0)
        & (
            delta_t
            <= MAX_RECORD_GAP_SECONDS
        )
    ].median()

    if (
        pd.isna(median_gap)
        or median_gap <= 0
    ):
        median_gap = 1.0

    df["seconds"] = delta_t.where(
        (delta_t > 0)
        & (
            delta_t
            <= MAX_RECORD_GAP_SECONDS
        ),
        np.nan,
    )

    df["seconds"] = df[
        "seconds"
    ].fillna(median_gap)

    # Derive speed if necessary.
    if (
        df["speed_mps"].isna().all()
        and df["distance_m"].notna().any()
    ):
        dt = (
            df["timestamp"]
            .diff()
            .dt.total_seconds()
        )
        dd = df["distance_m"].diff()

        derived = dd / dt

        df["speed_mps"] = derived.where(
            (dt > 0)
            & (
                dt
                <= MAX_RECORD_GAP_SECONDS
            )
            & derived.between(
                0.5,
                10.0,
            )
        )

    df["pace_s_per_km"] = (
        1000.0
        / df["speed_mps"]
    )

    df.loc[
        ~df["speed_mps"].between(
            0.5,
            10.0,
        ),
        "pace_s_per_km",
    ] = np.nan

    return df


def build_zone_distribution(
    user_dir: Path,
    runs: pd.DataFrame,
    advanced_dir: Path,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()

    if "selected_hrmax_bpm" not in runs.columns:
        print(
            "HR parameters and pace-zone history are unavailable; "
            "skipping time-in-zone."
        )
        return pd.DataFrame()

    records_dir = (
        user_dir
        / "silver"
        / "records"
    )

    rows: list[dict[str, Any]] = []

    run_dates = (
        runs[
            [
                "garmin_activity_id",
                "activity_date",
            ]
        ]
        .dropna(
            subset=["activity_date"]
        )
        .copy()
    )

    run_dates[
        "garmin_activity_id"
    ] = run_dates[
        "garmin_activity_id"
    ].astype(str)

    run_date_lookup = dict(
        zip(
            run_dates[
                "garmin_activity_id"
            ],
            pd.to_datetime(
                run_dates["activity_date"]
            ),
        )
    )

    run_hrmax_lookup: dict[str, float] = {}

    if "selected_hrmax_bpm" in runs.columns:
        hrmax_rows = runs[
            [
                "garmin_activity_id",
                "selected_hrmax_bpm",
            ]
        ].copy()

        hrmax_rows[
            "garmin_activity_id"
        ] = hrmax_rows[
            "garmin_activity_id"
        ].astype(str)

        hrmax_rows[
            "selected_hrmax_bpm"
        ] = pd.to_numeric(
            hrmax_rows[
                "selected_hrmax_bpm"
            ],
            errors="coerce",
        )

        run_hrmax_lookup = {
            str(row["garmin_activity_id"]): float(
                row["selected_hrmax_bpm"]
            )
            for _, row in hrmax_rows.dropna(
                subset=["selected_hrmax_bpm"]
            ).iterrows()
        }

    files = sorted(
        records_dir.glob("*.parquet")
    )

    print("\nBuilding time-in-zone")
    print("-" * 70)

    for number, path in enumerate(
        files,
        start=1,
    ):
        activity_id = path.stem
        activity_date = run_date_lookup.get(
            activity_id
        )

        if activity_date is None:
            continue

        try:
            raw = pd.read_parquet(path)
            records = prepare_record_data(
                raw
            )

            if records.empty:
                continue

            run_hrmax = run_hrmax_lookup.get(
                activity_id
            )

            hr_for_run = (
                hr_zones_from_hrmax(
                    run_hrmax
                )
                if (
                    run_hrmax is not None
                    and np.isfinite(run_hrmax)
                )
                else pd.DataFrame()
            )

            records["hr_zone"] = (
                classify_hr_zone(
                    records["heart_rate"],
                    hr_for_run,
                )
                if not hr_for_run.empty
                else None
            )

            hr_total = records.loc[
                records["hr_zone"].notna(),
                "seconds",
            ].sum()

            for zone in [
                "Z1",
                "Z2",
                "Z3",
                "Z4",
                "Z5",
            ]:
                hr_seconds = records.loc[
                    records[
                        "hr_zone"
                    ]
                    == zone,
                    "seconds",
                ].sum()

                rows.append(
                    {
                        "garmin_activity_id": activity_id,
                        "activity_date": activity_date,
                        "zone": zone,
                        "hr_seconds": hr_seconds,
                        "hr_minutes": hr_seconds
                        / 60.0,
                        "hr_percent": (
                            100.0
                            * hr_seconds
                            / hr_total
                            if hr_total > 0
                            else np.nan
                        ),
                    }
                )

        except Exception as exc:
            print(
                f"  Zone classification failed "
                f"for {activity_id}: "
                f"{type(exc).__name__}: {exc}"
            )

        if (
            number % 25 == 0
            or number == len(files)
        ):
            print(
                f"Processed "
                f"{number:,}/{len(files):,} "
                f"activities."
            )

    return pd.DataFrame(rows)


def aggregate_zone_distribution(
    zone_distribution: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    if zone_distribution.empty:
        return pd.DataFrame()

    df = zone_distribution.copy()

    df["activity_date"] = pd.to_datetime(
        df["activity_date"],
        errors="coerce",
    )

    if frequency == "weekly":
        period = df[
            "activity_date"
        ].dt.to_period("W-SUN")
    elif frequency == "monthly":
        period = df[
            "activity_date"
        ].dt.to_period("M")
    else:
        raise ValueError(
            "frequency must be weekly or monthly"
        )

    df["period_label"] = period.astype(str)
    df["period_start"] = period.apply(
        lambda p: (
            p.start_time.normalize()
            if pd.notna(p)
            else pd.NaT
        )
    )
    df["period_end"] = period.apply(
        lambda p: (
            p.end_time.normalize()
            if pd.notna(p)
            else pd.NaT
        )
    )

    grouped = (
        df.groupby(
            [
                "period_label",
                "period_start",
                "period_end",
                "zone",
            ],
            as_index=False,
        )["hr_seconds"]
        .sum()
    )

    grouped["hr_minutes"] = (
        grouped["hr_seconds"] / 60.0
    )

    grouped[
        "hr_period_total_seconds"
    ] = grouped.groupby(
        "period_label"
    )["hr_seconds"].transform("sum")

    grouped["hr_percent"] = np.where(
        grouped[
            "hr_period_total_seconds"
        ] > 0,
        100.0
        * grouped["hr_seconds"]
        / grouped[
            "hr_period_total_seconds"
        ],
        np.nan,
    )

    return grouped.drop(
        columns=[
            "hr_period_total_seconds",
        ]
    )




# =============================================================================
# EDWARDS TRAINING LOAD
# =============================================================================

def build_edwards_load_by_run(
    zone_distribution: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Edwards TRIMP / Summated Heart Rate Zone load per activity.

    Load = 1*minutes_Z1 + 2*minutes_Z2 + ... + 5*minutes_Z5.

    Only time classified into the five historical %HRmax zones contributes to
    the load. Weekly and monthly values are sums of these per-run loads.
    """
    if zone_distribution.empty:
        return pd.DataFrame(
            columns=[
                "garmin_activity_id",
                "activity_date",
                "training_load_edwards",
                "classified_hr_minutes",
            ]
        )

    df = zone_distribution.copy()
    df = df[df["zone"].isin(EDWARDS_ZONE_WEIGHTS)].copy()

    if df.empty:
        return pd.DataFrame()

    df["hr_minutes"] = pd.to_numeric(
        df["hr_minutes"],
        errors="coerce",
    )
    df["edwards_weight"] = df["zone"].map(
        EDWARDS_ZONE_WEIGHTS
    ).astype(float)

    df["edwards_load_component"] = (
        df["hr_minutes"]
        * df["edwards_weight"]
    )

    grouped = (
        df.groupby(
            [
                "garmin_activity_id",
                "activity_date",
            ],
            as_index=False,
        )
        .agg(
            training_load_edwards=(
                "edwards_load_component",
                "sum",
            ),
            classified_hr_minutes=(
                "hr_minutes",
                "sum",
            ),
        )
    )

    return grouped


def attach_edwards_load_to_runs(
    runs: pd.DataFrame,
    load_by_run: pd.DataFrame,
) -> pd.DataFrame:
    result = runs.copy()

    if load_by_run.empty:
        result["training_load_edwards"] = np.nan
        result["classified_hr_minutes"] = np.nan
        return result

    load = load_by_run.copy()
    load["garmin_activity_id"] = (
        load["garmin_activity_id"]
        .astype(str)
    )

    result["garmin_activity_id"] = (
        result["garmin_activity_id"]
        .astype(str)
    )

    keep = [
        "garmin_activity_id",
        "training_load_edwards",
        "classified_hr_minutes",
    ]

    return result.merge(
        load[keep],
        on="garmin_activity_id",
        how="left",
    )


# =============================================================================
# RACE PREDICTIONS
# =============================================================================

def fastest_continuous_segment(
    records: pd.DataFrame,
    target_distance_m: float,
) -> dict[str, float] | None:
    """
    Find the fastest continuous elapsed-time window covering target_distance_m.

    The watch is assumed to keep recording. Recovery jogging or standing time
    between intervals therefore remains inside a continuous target-distance
    window and cannot be removed or stitched around.

    Start points are record timestamps (normally about one second apart).
    The end timestamp is linearly interpolated at the exact target distance.
    """
    if records.empty:
        return None

    df = records[
        [
            "timestamp",
            "distance_m",
        ]
    ].copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )
    df["distance_m"] = pd.to_numeric(
        df["distance_m"],
        errors="coerce",
    )

    df = (
        df.dropna(
            subset=[
                "timestamp",
                "distance_m",
            ]
        )
        .sort_values("timestamp")
        .drop_duplicates(
            "timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    if len(df) < 2:
        return None

    distance = df["distance_m"].to_numpy(float)
    # Garmin distance is cumulative. cummax makes tiny GPS/FIT reversals harmless.
    distance = np.maximum.accumulate(distance)
    distance = distance - distance[0]

    elapsed = (
        df["timestamp"]
        - df["timestamp"].iloc[0]
    ).dt.total_seconds().to_numpy(float)

    if (
        not np.isfinite(distance[-1])
        or distance[-1] < target_distance_m
    ):
        return None

    best: dict[str, float] | None = None
    n = len(df)

    for i in range(n - 1):
        target = distance[i] + target_distance_m

        if target > distance[-1]:
            break

        j = int(
            np.searchsorted(
                distance,
                target,
                side="left",
            )
        )

        if j <= i or j >= n:
            continue

        if distance[j] == target:
            end_elapsed = elapsed[j]
        else:
            previous = j - 1
            d0 = distance[previous]
            d1 = distance[j]

            if d1 <= d0:
                continue

            fraction = (
                (target - d0)
                / (d1 - d0)
            )

            end_elapsed = (
                elapsed[previous]
                + fraction
                * (
                    elapsed[j]
                    - elapsed[previous]
                )
            )

        duration = end_elapsed - elapsed[i]

        if (
            not np.isfinite(duration)
            or duration <= 0
        ):
            continue

        if (
            best is None
            or duration < best["duration_s"]
        ):
            best = {
                "duration_s": float(duration),
                "start_elapsed_s": float(elapsed[i]),
                "end_elapsed_s": float(end_elapsed),
            }

    return best


def build_fastest_recent_race_segments(
    user_dir: Path,
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp | pd.NaT]:
    """
    For each anchor distance, keep the fastest continuous segment found in the
    most recent 90 days of available running data.
    """
    if runs.empty:
        return pd.DataFrame(), pd.NaT

    run_info = runs[
        [
            "garmin_activity_id",
            "activity_date",
        ]
    ].copy()

    run_info["garmin_activity_id"] = (
        run_info["garmin_activity_id"]
        .astype(str)
    )
    run_info["activity_date"] = pd.to_datetime(
        run_info["activity_date"],
        errors="coerce",
    )

    run_info = run_info.dropna(
        subset=["activity_date"]
    )

    if run_info.empty:
        return pd.DataFrame(), pd.NaT

    as_of = pd.Timestamp(
        run_info["activity_date"].max()
    ).normalize()

    lookback_start = (
        as_of
        - pd.Timedelta(
            days=RACE_PREDICTION_LOOKBACK_DAYS - 1
        )
    )

    recent = run_info[
        run_info["activity_date"].between(
            lookback_start,
            as_of,
            inclusive="both",
        )
    ].copy()

    records_dir = (
        user_dir
        / "silver"
        / "records"
    )

    best_by_target: dict[
        str,
        dict[str, Any],
    ] = {}

    print("\nBuilding 90-day race prediction anchors")
    print("-" * 70)

    for _, run in recent.iterrows():
        activity_id = str(
            run["garmin_activity_id"]
        )
        path = (
            records_dir
            / f"{activity_id}.parquet"
        )

        if not path.exists():
            continue

        try:
            raw = pd.read_parquet(path)
            records = prepare_record_data(raw)

            if records.empty:
                continue

            for (
                target_key,
                target_label,
                target_distance_m,
            ) in RACE_ANCHORS:
                segment = fastest_continuous_segment(
                    records,
                    target_distance_m,
                )

                if segment is None:
                    continue

                candidate = {
                    "target_key": target_key,
                    "distance_label": target_label,
                    "distance_km": (
                        target_distance_m
                        / 1000.0
                    ),
                    "observed_time_s": segment[
                        "duration_s"
                    ],
                    "observed_pace_s_per_km": (
                        segment["duration_s"]
                        / (
                            target_distance_m
                            / 1000.0
                        )
                    ),
                    "source_activity_id": activity_id,
                    "source_activity_date": pd.Timestamp(
                        run["activity_date"]
                    ).normalize(),
                    "segment_start_elapsed_s": segment[
                        "start_elapsed_s"
                    ],
                    "segment_end_elapsed_s": segment[
                        "end_elapsed_s"
                    ],
                }

                current = best_by_target.get(
                    target_key
                )

                if (
                    current is None
                    or candidate["observed_time_s"]
                    < current["observed_time_s"]
                ):
                    best_by_target[
                        target_key
                    ] = candidate

        except Exception as exc:
            print(
                f"  Race-segment scan failed "
                f"for {activity_id}: "
                f"{type(exc).__name__}: {exc}"
            )

    observations = pd.DataFrame(
        best_by_target.values()
    )

    if not observations.empty:
        order = {
            key: index
            for index, (
                key,
                _,
                _,
            ) in enumerate(
                RACE_ANCHORS
            )
        }
        observations["_order"] = (
            observations["target_key"]
            .map(order)
        )
        observations = (
            observations.sort_values(
                "_order"
            )
            .drop(
                columns="_order"
            )
            .reset_index(drop=True)
        )

    return observations, as_of


def build_riegel_predictions(
    observations: pd.DataFrame,
    as_of_date: pd.Timestamp | pd.NaT,
) -> pd.DataFrame:
    """
    Convert allowed recent performance anchors to each target distance using
    Riegel k=1.07, then keep the fastest (most optimistic) estimate per target.

    Anchor restrictions:
      - 5K may use 2.5K and 5K+ evidence.
      - 10K may use 2.5K and 5K+ evidence.
      - 10 miles and half marathon may use only 5K+ evidence.
    """
    if observations.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for (
        target_key,
        target_label,
        target_distance_m,
    ) in RACE_TARGETS:
        candidates: list[
            dict[str, Any]
        ] = []

        allowed_anchor_keys = RACE_ALLOWED_ANCHORS[target_key]

        for _, anchor in observations.iterrows():
            anchor_key = str(anchor["target_key"])
            if anchor_key not in allowed_anchor_keys:
                continue

            anchor_distance_m = (
                float(anchor["distance_km"])
                * 1000.0
            )
            anchor_time_s = float(
                anchor["observed_time_s"]
            )

            predicted_time_s = (
                anchor_time_s
                * (
                    target_distance_m
                    / anchor_distance_m
                )
                ** RIEGEL_EXPONENT
            )

            direct = bool(
                abs(
                    target_distance_m
                    - anchor_distance_m
                )
                < 1.0
            )

            candidates.append({
                "target_key": target_key,
                "target_label": target_label,
                "target_distance_km": (
                    target_distance_m
                    / 1000.0
                ),
                "predicted_time_s": float(
                    predicted_time_s
                ),
                "predicted_pace_s_per_km": float(
                    predicted_time_s
                    / (
                        target_distance_m
                        / 1000.0
                    )
                ),
                "source_distance_label": anchor[
                    "distance_label"
                ],
                "source_distance_km": float(
                    anchor["distance_km"]
                ),
                "source_observed_time_s": anchor_time_s,
                "source_observed_pace_s_per_km": float(
                    anchor[
                        "observed_pace_s_per_km"
                    ]
                ),
                "source_activity_id": anchor[
                    "source_activity_id"
                ],
                "source_activity_date": anchor[
                    "source_activity_date"
                ],
                "method": (
                    "direct"
                    if direct
                    else "riegel"
                ),
                "riegel_exponent": (
                    np.nan
                    if direct
                    else RIEGEL_EXPONENT
                ),
                "lookback_days": (
                    RACE_PREDICTION_LOOKBACK_DAYS
                ),
                "as_of_date": as_of_date,
            })

        if not candidates:
            continue

        # The user's chosen rule: use the most optimistic estimate generated by
        # the available direct observations.
        best = min(
            candidates,
            key=lambda row: row[
                "predicted_time_s"
            ],
        )

        rows.append(best)

    return pd.DataFrame(rows)


def build_race_prediction_payload(
    observations: pd.DataFrame,
    predictions: pd.DataFrame,
    as_of_date: pd.Timestamp | pd.NaT,
) -> dict[str, Any]:
    return {
        "as_of_date": json_safe(
            as_of_date
        ),
        "lookback_days": (
            RACE_PREDICTION_LOOKBACK_DAYS
        ),
        "riegel_exponent": (
            RIEGEL_EXPONENT
        ),
        "observations": (
            dataframe_to_records(
                observations
            )
            if not observations.empty
            else []
        ),
        "predictions": (
            dataframe_to_records(
                predictions
            )
            if not predictions.empty
            else []
        ),
    }


# =============================================================================
# WEBSITE SUMMARY
# =============================================================================

def latest_non_null(
    dataframe: pd.DataFrame,
    column: str,
) -> Any:
    if (
        dataframe.empty
        or column not in dataframe.columns
    ):
        return None

    values = dataframe[
        [
            "activity_date",
            column,
        ]
    ].dropna(
        subset=[column]
    )

    if values.empty:
        return None

    values = values.sort_values(
        "activity_date"
    )

    return values.iloc[-1][column]


def build_summary(
    runs: pd.DataFrame,
) -> dict[str, Any]:
    if runs.empty:
        return {
            "generated_from_data": None,
            "latest_run": None,
        }

    df = runs.copy()

    df["activity_date"] = pd.to_datetime(
        df["activity_date"],
        errors="coerce",
    )

    latest_date = df[
        "activity_date"
    ].max()

    if pd.isna(latest_date):
        return {
            "generated_from_data": None,
            "latest_run": None,
        }

    last_7 = df[
        df["activity_date"]
        >= latest_date
        - pd.Timedelta(days=6)
    ]

    last_28 = df[
        df["activity_date"]
        >= latest_date
        - pd.Timedelta(days=27)
    ]

    latest_run = (
        df.sort_values(
            [
                "activity_date",
                "start_time_local",
            ]
        )
        .iloc[-1]
    )

    payload = {
        "generated_from_data": latest_date,
        "current": {
            "garmin_vo2max": latest_non_null(
                df,
                "garmin_vo2max_precise",
            ),
            "hrmax_bpm": latest_non_null(
                df,
                "selected_hrmax_bpm",
            ),
            "resting_hr_bpm": latest_non_null(
                df,
                "selected_resting_hr_bpm",
            ),
        },
        "last_7_days": {
            "distance_km": pd.to_numeric(
                last_7["distance_km"],
                errors="coerce",
            ).sum(),
            "runs": int(len(last_7)),
        },
        "last_28_days": {
            "distance_km": pd.to_numeric(
                last_28["distance_km"],
                errors="coerce",
            ).sum(),
            "runs": int(len(last_28)),
            "duration_hours": pd.to_numeric(
                last_28[
                    "duration_hours"
                ],
                errors="coerce",
            ).sum(),
        },
        "latest_run": {
            "date": latest_run[
                "activity_date"
            ],
            "distance_km": latest_run.get(
                "distance_km"
            ),
            "pace": latest_run.get(
                "pace"
            ),
            "avg_hr_bpm": latest_run.get(
                "avg_hr_bpm"
            ),
            "temperature_c": latest_run.get(
                "temperature_c"
            ),
            "elevation_gain_m": latest_run.get(
                "elevation_gain_m"
            ),
        },
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
            }

        return json_safe(value)

    return clean(payload)


# =============================================================================
# AI TRAINING CONTEXT
# =============================================================================

def build_recent_laps_context(
    runs: pd.DataFrame,
    laps_dir: Path,
    max_runs: int = 30,
) -> list[dict[str, Any]]:
    """Return compact Garmin FIT lap data for the most recent runs.

    This is only an AI read-model helper. It does not alter lap data, running
    metrics or any analytical method. Garmin activity IDs are used internally
    to locate the corresponding lap parquet file, but are deliberately omitted
    from the public training context.
    """
    if runs.empty or not laps_dir.exists() or max_runs <= 0:
        return []

    df = runs.copy()

    if "garmin_activity_id" not in df.columns or "activity_date" not in df.columns:
        return []

    df["activity_date"] = pd.to_datetime(
        df["activity_date"],
        errors="coerce",
    )

    if "start_time_local" in df.columns:
        df["start_time_local"] = pd.to_datetime(
            df["start_time_local"],
            errors="coerce",
        )

    sort_columns = ["activity_date"]
    if "start_time_local" in df.columns:
        sort_columns.append("start_time_local")

    recent = (
        df.dropna(subset=["activity_date"])
        .sort_values(sort_columns, ascending=False)
        .head(max_runs)
        .sort_values(sort_columns)
    )

    def row_numeric(
        row: pd.Series,
        candidates: list[str],
    ) -> float:
        for column in candidates:
            if column not in row.index:
                continue
            value = pd.to_numeric(
                pd.Series([row.get(column)]),
                errors="coerce",
            ).iloc[0]
            if not pd.isna(value):
                return float(value)
        return np.nan

    def row_value(
        row: pd.Series,
        candidates: list[str],
    ) -> Any:
        for column in candidates:
            if column not in row.index:
                continue
            value = row.get(column)
            if value is not None and not pd.isna(value):
                return value
        return None

    output: list[dict[str, Any]] = []

    for _, run in recent.iterrows():
        activity_id = str(run["garmin_activity_id"])
        lap_path = laps_dir / f"{activity_id}.parquet"

        if not lap_path.is_file():
            continue

        try:
            laps = pd.read_parquet(lap_path)
        except Exception as exc:
            print(
                f"AI lap context skipped {activity_id}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue

        if laps.empty:
            continue

        if "fit_message_index" in laps.columns:
            laps = laps.copy()
            laps["_lap_order"] = pd.to_numeric(
                laps["fit_message_index"],
                errors="coerce",
            )
            laps = laps.sort_values(
                "_lap_order",
                kind="stable",
            ).drop(columns="_lap_order")
        else:
            laps = laps.reset_index(drop=True)

        lap_records: list[dict[str, Any]] = []

        for lap_number, (_, lap) in enumerate(
            laps.iterrows(),
            start=1,
        ):
            distance_m = row_numeric(
                lap,
                [
                    "total_distance",
                    "distance",
                    "distance_m",
                ],
            )
            duration_s = row_numeric(
                lap,
                [
                    "total_timer_time",
                    "total_elapsed_time",
                    "duration",
                    "duration_s",
                ],
            )
            avg_hr = row_numeric(
                lap,
                [
                    "avg_heart_rate",
                    "average_heart_rate",
                    "average_hr",
                    "averageHR",
                ],
            )
            max_hr = row_numeric(
                lap,
                [
                    "max_heart_rate",
                    "maximum_heart_rate",
                    "max_hr",
                    "maxHR",
                ],
            )
            elevation_gain_m = row_numeric(
                lap,
                ["total_ascent", "elevation_gain_m"],
            )
            elevation_loss_m = row_numeric(
                lap,
                ["total_descent", "elevation_loss_m"],
            )

            distance_km = (
                distance_m / 1000.0
                if np.isfinite(distance_m)
                else np.nan
            )
            pace_s_per_km = (
                duration_s / distance_km
                if (
                    np.isfinite(duration_s)
                    and np.isfinite(distance_km)
                    and distance_km > 0
                )
                else np.nan
            )

            lap_start = row_value(
                lap,
                [
                    "start_time",
                    "timestamp",
                    "lap_start_time",
                ],
            )
            lap_start = pd.to_datetime(
                lap_start,
                errors="coerce",
            )

            lap_records.append(
                {
                    "lap_number": lap_number,
                    "start_time": json_safe(lap_start),
                    "distance_km": json_safe(distance_km),
                    "duration_s": json_safe(duration_s),
                    "duration_min": json_safe(
                        duration_s / 60.0
                        if np.isfinite(duration_s)
                        else np.nan
                    ),
                    "pace_s_per_km": json_safe(pace_s_per_km),
                    "pace": (
                        pace_seconds_to_text(pace_s_per_km)
                        if np.isfinite(pace_s_per_km)
                        else None
                    ),
                    "avg_hr_bpm": json_safe(avg_hr),
                    "max_hr_bpm": json_safe(max_hr),
                    "elevation_gain_m": json_safe(elevation_gain_m),
                    "elevation_loss_m": json_safe(elevation_loss_m),
                }
            )

        if not lap_records:
            continue

        output.append(
            {
                "activity_date": json_safe(run.get("activity_date")),
                "start_time_local": (
                    json_safe(run.get("start_time_local"))
                    if "start_time_local" in run.index
                    else None
                ),
                "run_distance_km": json_safe(run.get("distance_km")),
                "run_pace": json_safe(run.get("pace")),
                "laps": lap_records,
            }
        )

    return output


def build_training_context(
    runs: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    zone_weekly: pd.DataFrame,
    race_prediction_payload: dict[str, Any],
    advanced_dir: Path,
) -> dict[str, Any]:
    """Build one compact, machine-readable snapshot for AI training advice.

    This is a read-only output layer. It reuses the metrics already calculated
    by the pipeline and does not change HRmax, zones, training load, VO2max or
    race-prediction methods.
    """
    if runs.empty:
        return {
            "schema_version": 2,
            "generated_at": datetime.now().astimezone().isoformat(),
            "data_as_of": None,
            "current": {},
            "training_windows": {},
            "weekly_history": [],
            "monthly_history": [],
            "recent_runs": [],
            "recent_laps": [],
            "recent_hr_zone_distribution": [],
            "race_fitness": race_prediction_payload,
        }

    df = runs.copy()
    df["activity_date"] = pd.to_datetime(
        df["activity_date"],
        errors="coerce",
    )
    df = df.dropna(subset=["activity_date"]).copy()

    if df.empty:
        latest_activity_date = pd.NaT
    else:
        latest_activity_date = pd.Timestamp(
            df["activity_date"].max()
        ).normalize()

    # Rolling training windows are anchored to the day on which the endpoint
    # is generated. This means rest days genuinely age older training out of
    # the 7/14/28/etc. day summaries.
    window_as_of = pd.Timestamp(date.today()).normalize()

    def numeric_series(
        dataframe: pd.DataFrame,
        column: str,
    ) -> pd.Series:
        if column not in dataframe.columns:
            return pd.Series(
                np.nan,
                index=dataframe.index,
                dtype="float64",
            )
        return pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    def window_summary(days: int) -> dict[str, Any]:
        start = window_as_of - pd.Timedelta(days=days - 1)
        subset = df[
            df["activity_date"].between(
                start,
                window_as_of,
                inclusive="both",
            )
        ].copy()

        distance = numeric_series(subset, "distance_km")
        duration_hours = numeric_series(subset, "duration_hours")
        duration_min = numeric_series(subset, "duration_min")
        avg_hr = numeric_series(subset, "avg_hr_bpm")
        edwards = numeric_series(subset, "training_load_edwards")

        weighted_hr = (
            weighted_average(avg_hr, duration_min)
            if len(subset)
            else np.nan
        )

        return {
            "days": days,
            "start_date": start.date().isoformat(),
            "end_date": window_as_of.date().isoformat(),
            "runs": int(len(subset)),
            "distance_km": json_safe(
                distance.sum(min_count=1)
            ),
            "duration_hours": json_safe(
                duration_hours.sum(min_count=1)
            ),
            "edwards_load": json_safe(
                edwards.sum(min_count=1)
            ),
            "longest_run_km": json_safe(
                distance.max()
            ),
            "duration_weighted_avg_hr_bpm": json_safe(
                weighted_hr
            ),
        }

    heart_rate_zones = read_optional_parquet(
        advanced_dir / "heart_rate_zones.parquet"
    )
    pace_zones = read_optional_parquet(
        advanced_dir / "pace_zones.parquet"
    )

    recent_run_columns = [
        column
        for column in [
            "activity_date",
            "distance_km",
            "duration_min",
            "pace_s_per_km",
            "pace",
            "avg_hr_bpm",
            "max_hr_bpm",
            "training_load_edwards",
            "classified_hr_minutes",
            "selected_hrmax_bpm",
            "selected_resting_hr_bpm",
            "garmin_vo2max_precise",
            "elevation_gain_m",
            "temperature_c",
        ]
        if column in df.columns
    ]

    sort_columns = ["activity_date"]
    if "start_time_local" in df.columns:
        sort_columns.append("start_time_local")

    recent_runs = (
        df.sort_values(
            sort_columns,
            ascending=False,
        )
        .head(30)[recent_run_columns]
        .sort_values("activity_date")
        if recent_run_columns
        else pd.DataFrame()
    )

    recent_laps = build_recent_laps_context(
        df,
        LAPS_DIR,
        max_runs=30,
    )

    weekly_history = (
        weekly.sort_values("period_start").tail(16)
        if not weekly.empty and "period_start" in weekly.columns
        else weekly.tail(16)
        if not weekly.empty
        else pd.DataFrame()
    )
    monthly_history = (
        monthly.sort_values("period_start").tail(12)
        if not monthly.empty and "period_start" in monthly.columns
        else monthly.tail(12)
        if not monthly.empty
        else pd.DataFrame()
    )

    if (
        not zone_weekly.empty
        and {"period_start", "period_label", "zone"}.issubset(
            zone_weekly.columns
        )
    ):
        recent_periods = (
            zone_weekly[["period_label", "period_start"]]
            .drop_duplicates("period_label", keep="last")
            .sort_values("period_start")
            .tail(16)["period_label"]
            .tolist()
        )
        recent_zone_history = (
            zone_weekly[
                zone_weekly["period_label"].isin(recent_periods)
            ]
            .sort_values(["period_start", "zone"])
            .copy()
        )
    else:
        recent_zone_history = pd.DataFrame()

    # Remove activity identifiers from the AI read model. They are not needed
    # for training advice even though race_predictions.json keeps its original
    # source metadata for the dashboard.
    race_context = {
        "as_of_date": race_prediction_payload.get("as_of_date"),
        "lookback_days": race_prediction_payload.get("lookback_days"),
        "riegel_exponent": race_prediction_payload.get("riegel_exponent"),
        "observations": [],
        "predictions": [],
    }

    for key in ["observations", "predictions"]:
        for row in race_prediction_payload.get(key, []) or []:
            race_context[key].append(
                {
                    name: value
                    for name, value in row.items()
                    if name != "source_activity_id"
                }
            )

    current = {
        "hrmax_bpm": json_safe(
            latest_non_null(df, "selected_hrmax_bpm")
        ),
        "resting_hr_bpm_28d": json_safe(
            latest_non_null(df, "selected_resting_hr_bpm")
        ),
        "garmin_vo2max": json_safe(
            latest_non_null(df, "garmin_vo2max_precise")
        ),
        "heart_rate_zones": (
            dataframe_to_records(heart_rate_zones)
            if not heart_rate_zones.empty
            else []
        ),
        "pace_zones_90d": (
            dataframe_to_records(pace_zones)
            if not pace_zones.empty
            else []
        ),
    }

    anchor_order = [
        key
        for key, _, _ in RACE_ANCHORS
    ]

    methodology = {
        "hrmax": {
            "lookback_days": HRMAX_LOOKBACK_DAYS,
            "method": (
                "highest supported raw/session maximum; if none exists, "
                "highest credible 10-second peak"
            ),
            "support_rule": {
                "peak_10s_within_bpm": HRMAX_SESSION_MAX_10S_TOLERANCE_BPM,
                "peak_30s_within_bpm": HRMAX_SESSION_MAX_30S_TOLERANCE_BPM,
            },
        },
        "resting_hr": {
            "lookback_days": RHR_LOOKBACK_DAYS,
            "method": "median of Garmin resting-heart-rate observations",
        },
        "heart_rate_zones": {
            "method": "fixed percentage of HRmax",
            "boundaries_percent": [50, 60, 70, 80, 90, 100],
        },
        "pace_zones": {
            "lookback_days": PACE_ZONE_LOOKBACK_DAYS,
            "method": (
                "distance-weighted P20/P50/P80 of observed pace within "
                "historical HR zones"
            ),
        },
        "training_load": {
            "method": "Edwards summated heart-rate-zone load",
            "zone_weights": EDWARDS_ZONE_WEIGHTS,
        },
        "laps": {
            "source": "Garmin FIT lap messages",
            "coverage": "most recent 30 runs with available lap data",
            "pace_method": "lap timer time divided by lap distance",
            "privacy": "Garmin activity IDs are omitted from training_context.json",
        },
        "race_predictions": {
            "lookback_days": RACE_PREDICTION_LOOKBACK_DAYS,
            "riegel_exponent": RIEGEL_EXPONENT,
            "anchors": [
                {
                    "key": key,
                    "label": label,
                    "distance_km": distance_m / 1000.0,
                }
                for key, label, distance_m in RACE_ANCHORS
            ],
            "targets": [
                {
                    "key": key,
                    "label": label,
                    "distance_km": distance_m / 1000.0,
                    "allowed_anchors": [
                        anchor_key
                        for anchor_key in anchor_order
                        if anchor_key in RACE_ALLOWED_ANCHORS[key]
                    ],
                }
                for key, label, distance_m in RACE_TARGETS
            ],
        },
    }

    return {
        "schema_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "data_as_of": (
            latest_activity_date.date().isoformat()
            if pd.notna(latest_activity_date)
            else None
        ),
        "source": "Garmin Connect via Running Insights pipeline",
        "current": current,
        "training_windows": {
            f"{days}d": window_summary(days)
            for days in [7, 14, 28, 56, 84]
        },
        "weekly_history": (
            dataframe_to_records(weekly_history)
            if not weekly_history.empty
            else []
        ),
        "monthly_history": (
            dataframe_to_records(monthly_history)
            if not monthly_history.empty
            else []
        ),
        "recent_runs": (
            dataframe_to_records(recent_runs)
            if not recent_runs.empty
            else []
        ),
        "recent_laps": recent_laps,
        "recent_hr_zone_distribution": (
            dataframe_to_records(recent_zone_history)
            if not recent_zone_history.empty
            else []
        ),
        "race_fitness": race_context,
        "methodology": methodology,
    }


# =============================================================================
# WEBSITE EXPORT
# =============================================================================

def export_website_bundle(
    dashboard_dir: Path,
    summary: dict[str, Any],
    runs: pd.DataFrame,
    weekly: pd.DataFrame,
    monthly: pd.DataFrame,
    zone_weekly: pd.DataFrame,
    zone_monthly: pd.DataFrame,
    race_prediction_payload: dict[str, Any],
    advanced_dir: Path,
) -> Path:
    website_dir = (
        dashboard_dir
        / "website"
    )

    website_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    recent_runs = (
        runs.sort_values(
            "activity_date",
            ascending=False,
        )
        .head(WEBSITE_MAX_RUNS)
        .sort_values(
            "activity_date"
        )
    )

    write_json(
        website_dir / "summary.json",
        summary,
    )

    write_json(
        website_dir / "runs.json",
        dataframe_to_records(
            recent_runs
        ),
    )

    write_json(
        website_dir / "weekly.json",
        dataframe_to_records(
            weekly
        ),
    )

    write_json(
        website_dir / "monthly.json",
        dataframe_to_records(
            monthly
        ),
    )

    write_json(
        website_dir
        / "zones_weekly.json",
        dataframe_to_records(
            zone_weekly
        ),
    )

    write_json(
        website_dir
        / "zones_monthly.json",
        dataframe_to_records(
            zone_monthly
        ),
    )

    write_json(
        website_dir
        / "race_predictions.json",
        race_prediction_payload,
    )

    training_context = build_training_context(
        runs=runs,
        weekly=weekly,
        monthly=monthly,
        zone_weekly=zone_weekly,
        race_prediction_payload=race_prediction_payload,
        advanced_dir=advanced_dir,
    )

    write_json(
        website_dir
        / "training_context.json",
        training_context,
    )

    for source_name, target_name in [
        (
            "heart_rate_zones_weekly.parquet",
            "heart_rate_zones_weekly.json",
        ),
        (
            "pace_zones_weekly.parquet",
            "pace_zones_weekly.json",
        ),
    ]:
        source = (
            advanced_dir
            / source_name
        )

        if source.exists():
            dataframe = pd.read_parquet(
                source
            )

            write_json(
                website_dir
                / target_name,
                dataframe_to_records(
                    dataframe
                ),
            )

    # Remove JSON from the retired custom VO2max model if an older build left it behind.
    legacy_validation_json = website_dir / "vo2max_model_validation.json"
    if legacy_validation_json.exists():
        legacy_validation_json.unlink()

    if WEBSITE_DATA_DIR is not None:
        WEBSITE_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        stale_validation = WEBSITE_DATA_DIR / "vo2max_model_validation.json"
        if stale_validation.exists():
            stale_validation.unlink()

        for file in website_dir.glob(
            "*.json"
        ):
            shutil.copy2(
                file,
                WEBSITE_DATA_DIR
                / file.name,
            )

        print(
            f"Website JSON copied to:\n"
            f"{WEBSITE_DATA_DIR}"
        )

    return website_dir


# =============================================================================
# MAIN
# =============================================================================

def build_dashboard_data() -> None:
    user_dir = choose_user_dir()

    advanced_dir = (
        user_dir
        / "gold"
        / "advanced_analytics"
    )

    dashboard_dir = (
        user_dir
        / "gold"
        / "dashboard"
    )

    dashboard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nBuild Dashboard Data")
    print("-" * 70)
    print(f"User: {user_dir.name}")

    runs = build_runs_table(
        user_dir
    )

    if runs.empty:
        raise ValueError(
            "No run sessions available."
        )

    runs = add_advanced_metrics_to_runs(
        runs,
        advanced_dir,
    )

    # -----------------------------------------------------------------
    # Historical zone classification + Edwards TRIMP
    # -----------------------------------------------------------------
    zone_distribution = pd.DataFrame()
    zone_weekly = pd.DataFrame()
    zone_monthly = pd.DataFrame()
    edwards_load_by_run = pd.DataFrame()

    if BUILD_ZONE_DISTRIBUTION:
        zone_distribution = (
            build_zone_distribution(
                user_dir,
                runs,
                advanced_dir,
            )
        )

        edwards_load_by_run = (
            build_edwards_load_by_run(
                zone_distribution
            )
        )

        runs = attach_edwards_load_to_runs(
            runs,
            edwards_load_by_run,
        )

        zone_weekly = (
            aggregate_zone_distribution(
                zone_distribution,
                "weekly",
            )
        )

        zone_monthly = (
            aggregate_zone_distribution(
                zone_distribution,
                "monthly",
            )
        )
    else:
        runs["training_load_edwards"] = np.nan
        runs["classified_hr_minutes"] = np.nan

    # Weekly/monthly load is a SUM of per-run Edwards load.
    weekly = aggregate_runs(
        runs,
        "weekly",
    )

    monthly = aggregate_runs(
        runs,
        "monthly",
    )

    # Add richer snapshot values calculated by advanced_analytics.
    weekly = merge_snapshot_table(
        weekly,
        read_optional_parquet(
            advanced_dir
            / "garmin_vo2max_weekly.parquet"
        ),
        [
            "garmin_vo2max_precise",
        ],
    )

    monthly = merge_snapshot_table(
        monthly,
        read_optional_parquet(
            advanced_dir
            / "garmin_vo2max_monthly.parquet"
        ),
        [
            "garmin_vo2max_precise",
        ],
    )

    weekly = merge_snapshot_table(
        weekly,
        read_optional_parquet(
            advanced_dir
            / "heart_rate_parameters_weekly.parquet"
        ),
        [
            "selected_hrmax_bpm",
            "selected_resting_hr_bpm",
        ],
    )

    monthly = merge_snapshot_table(
        monthly,
        read_optional_parquet(
            advanced_dir
            / "heart_rate_parameters_monthly.parquet"
        ),
        [
            "selected_hrmax_bpm",
            "selected_resting_hr_bpm",
        ],
    )

    # -----------------------------------------------------------------
    # Current 90-day race predictions from 2.5K/5K+ anchors
    # -----------------------------------------------------------------
    race_observations, race_as_of = (
        build_fastest_recent_race_segments(
            user_dir,
            runs,
        )
    )

    race_predictions = (
        build_riegel_predictions(
            race_observations,
            race_as_of,
        )
    )

    race_prediction_payload = (
        build_race_prediction_payload(
            race_observations,
            race_predictions,
            race_as_of,
        )
    )

    summary = build_summary(
        runs
    )

    # Core dashboard tables.
    write_table(
        runs,
        dashboard_dir
        / "runs",
    )

    write_table(
        weekly,
        dashboard_dir
        / "weekly_running",
    )

    write_table(
        monthly,
        dashboard_dir
        / "monthly_running",
    )

    if not zone_distribution.empty:
        write_table(
            zone_distribution,
            dashboard_dir
            / "zone_distribution",
        )

        write_table(
            zone_weekly,
            dashboard_dir
            / "zone_distribution_weekly",
        )

        write_table(
            zone_monthly,
            dashboard_dir
            / "zone_distribution_monthly",
        )

    if not edwards_load_by_run.empty:
        write_table(
            edwards_load_by_run,
            dashboard_dir
            / "edwards_training_load_by_run",
        )

    if not race_observations.empty:
        write_table(
            race_observations,
            dashboard_dir
            / "race_fastest_segments_90d",
        )

    if not race_predictions.empty:
        write_table(
            race_predictions,
            dashboard_dir
            / "race_predictions",
        )

    website_dir = export_website_bundle(
        dashboard_dir=dashboard_dir,
        summary=summary,
        runs=runs,
        weekly=weekly,
        monthly=monthly,
        zone_weekly=zone_weekly,
        zone_monthly=zone_monthly,
        race_prediction_payload=(
            race_prediction_payload
        ),
        advanced_dir=advanced_dir,
    )

    print("\nComplete")
    print("-" * 70)
    print(
        f"Runs: {len(runs):,}"
    )
    print(
        f"Weekly rows: {len(weekly):,}"
    )
    print(
        f"Monthly rows: {len(monthly):,}"
    )
    print(
        f"Zone rows: "
        f"{len(zone_distribution):,}"
    )
    print(
        f"Race observations (90d): "
        f"{len(race_observations):,}"
    )

    print(
        f"\nDashboard data:\n"
        f"{dashboard_dir}"
    )

    print(
        f"\nWebsite-ready JSON:\n"
        f"{website_dir}"
    )

    print(
        "\nMain files:"
    )
    print(
        "  runs.csv"
    )
    print(
        "  weekly_running.csv"
    )
    print(
        "  monthly_running.csv"
    )
    print(
        "  edwards_training_load_by_run.csv"
    )
    print(
        "  race_predictions.csv"
    )
    print(
        "  website/summary.json"
    )
    print(
        "  website/weekly.json"
    )
    print(
        "  website/monthly.json"
    )
    print(
        "  website/race_predictions.json"
    )
    print(
        "  website/training_context.json"
    )

# =============================================================================
# Azure persistence and dashboard publication
# =============================================================================


def create_azure_blob_service_client(
    storage_account: str,
    connection_string: str = "",
):
    """Create an Azure BlobServiceClient and return it with an auth label.

    Authentication order:
      1. An explicitly supplied storage-account connection string.
      2. Microsoft Entra identity through DefaultAzureCredential.

    The second route is the intended production path for GitHub Actions:
    GitHub will authenticate to Azure with OIDC, so no permanent storage
    account key needs to be stored in the repository or workflow.
    """
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as exc:
        raise RuntimeError(
            "Azure support requires 'azure-storage-blob' and "
            "'azure-identity'. Install them with:\n"
            "  pip install azure-storage-blob azure-identity"
        ) from exc

    account_url = (
        f"https://{storage_account}.blob.core.windows.net"
    )

    if connection_string:
        client = BlobServiceClient.from_connection_string(
            connection_string
        )
        auth_mode = "connection string"
    else:
        credential = DefaultAzureCredential(
            exclude_interactive_browser_credential=False,
        )
        client = BlobServiceClient(
            account_url=account_url,
            credential=credential,
        )
        auth_mode = "Microsoft Entra / DefaultAzureCredential"

    return client, auth_mode, account_url



def local_private_user_state_exists() -> bool:
    """Return True when enough local state exists to continue incrementally."""
    if ACTIVITY_INDEX_FILE.is_file() or EXPORT_MANIFEST_FILE.is_file():
        return True

    return (
        RECORDS_DIR.exists()
        and any(RECORDS_DIR.glob("*.parquet"))
    )



def private_state_blob_name(
    local_root: Path,
    blob_prefix: str,
    file_path: Path,
) -> str:
    relative = file_path.relative_to(local_root)
    return (
        f"{blob_prefix.rstrip('/')}/"
        f"{relative.as_posix()}"
    )



def list_private_state_files() -> list[tuple[Path, str]]:
    """Return local private-state files and their destination blob names."""
    files: list[tuple[Path, str]] = []

    for local_root, blob_prefix in PRIVATE_STATE_ROOTS:
        if not local_root.exists():
            continue

        for file_path in sorted(local_root.rglob("*")):
            if not file_path.is_file():
                continue

            files.append(
                (
                    file_path,
                    private_state_blob_name(
                        local_root,
                        blob_prefix,
                        file_path,
                    ),
                )
            )

    return files



def download_private_prefix(
    container_client,
    blob_prefix: str,
    destination_root: Path,
) -> tuple[int, int]:
    """Download one blob prefix and preserve Azure last-modified times."""
    prefix = blob_prefix.rstrip("/") + "/"
    blobs = list(
        container_client.list_blobs(
            name_starts_with=prefix
        )
    )

    downloaded = 0
    downloaded_bytes = 0

    for blob in blobs:
        relative_text = blob.name[len(prefix):]
        if not relative_text:
            continue

        relative_path = Path(relative_text)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
        ):
            raise ValueError(
                f"Unsafe blob path in private state: {blob.name}"
            )

        destination = destination_root / relative_path
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        blob_client = container_client.get_blob_client(
            blob.name
        )

        with destination.open("wb") as output:
            blob_client.download_blob().readinto(output)

        # This is useful for the incremental upload step: a restored file
        # keeps the time of the cloud copy, while files modified by the
        # pipeline receive a newer local mtime.
        if getattr(blob, "last_modified", None) is not None:
            timestamp = blob.last_modified.timestamp()
            os.utime(
                destination,
                (timestamp, timestamp),
            )

        downloaded += 1
        downloaded_bytes += destination.stat().st_size

    return downloaded, downloaded_bytes



def restore_private_state_from_azure() -> None:
    """Restore raw/silver data and Garmin tokens when local state is absent.

    On a normal developer machine the existing local state is left alone.
    On a fresh GitHub runner both the user state and Garmin token store are
    absent, so they are restored automatically before Garmin is contacted.

    Set AZURE_PRIVATE_FORCE_RESTORE=1 only when an explicit overwrite from
    Azure is desired for testing/recovery.
    """
    if not AZURE_PRIVATE_SYNC_ENABLED:
        print(
            "\nPrivate Azure state sync disabled "
            "(AZURE_PRIVATE_SYNC_ENABLED=0)."
        )
        return

    restore_user = (
        AZURE_PRIVATE_FORCE_RESTORE
        or not local_private_user_state_exists()
    )
    restore_tokens = (
        AZURE_PRIVATE_FORCE_RESTORE
        or not token_store_has_files()
    )

    if not restore_user and not restore_tokens:
        print("\n" + "=" * 70)
        print("RESTORE PRIVATE PIPELINE STATE FROM AZURE")
        print("=" * 70)
        print(
            "Local raw/silver state and Garmin tokens already exist; "
            "Azure restore skipped."
        )
        return

    blob_service, auth_mode, _ = (
        create_azure_blob_service_client(
            AZURE_PRIVATE_STORAGE_ACCOUNT,
            AZURE_PRIVATE_STORAGE_CONNECTION_STRING,
        )
    )
    container_client = blob_service.get_container_client(
        AZURE_PRIVATE_CONTAINER
    )

    print("\n" + "=" * 70)
    print("RESTORE PRIVATE PIPELINE STATE FROM AZURE")
    print("=" * 70)
    print(
        f"Storage account: {AZURE_PRIVATE_STORAGE_ACCOUNT}"
    )
    print(f"Container:       {AZURE_PRIVATE_CONTAINER}")
    print(f"Authentication:  {auth_mode}")

    total_files = 0
    total_bytes = 0

    if restore_user:
        for local_root, blob_prefix in PRIVATE_STATE_ROOTS[:2]:
            count, byte_count = download_private_prefix(
                container_client,
                blob_prefix,
                local_root,
            )
            total_files += count
            total_bytes += byte_count
            print(
                f"  restored {blob_prefix}: "
                f"{count:,} files"
            )

    if restore_tokens:
        token_root, token_prefix = PRIVATE_STATE_ROOTS[2]
        count, byte_count = download_private_prefix(
            container_client,
            token_prefix,
            token_root,
        )
        total_files += count
        total_bytes += byte_count
        print(
            f"  restored {token_prefix}: "
            f"{count:,} files"
        )

    print("-" * 70)

    if total_files == 0:
        print(
            "No private state was found in Azure. "
            "The pipeline will continue with the local/empty state."
        )
    else:
        print(
            f"Restored {total_files:,} files "
            f"({total_bytes / (1024 * 1024):.1f} MB)."
        )



def upload_private_state_to_azure() -> None:
    """Persist raw/silver data and Garmin tokens in the private container.

    Existing blobs are listed once. A local file is uploaded only when:
      - the blob does not exist;
      - its file size changed; or
      - its local modification time is newer than the blob.

    This avoids uploading the complete Garmin history on every daily run.
    """
    if not AZURE_PRIVATE_SYNC_ENABLED:
        print(
            "\nPrivate Azure state sync disabled "
            "(AZURE_PRIVATE_SYNC_ENABLED=0)."
        )
        return

    state_files = list_private_state_files()
    if not state_files:
        print(
            "\nNo local private pipeline state found to upload."
        )
        return

    blob_service, auth_mode, _ = (
        create_azure_blob_service_client(
            AZURE_PRIVATE_STORAGE_ACCOUNT,
            AZURE_PRIVATE_STORAGE_CONNECTION_STRING,
        )
    )
    container_client = blob_service.get_container_client(
        AZURE_PRIVATE_CONTAINER
    )

    print("\n" + "=" * 70)
    print("SAVE PRIVATE PIPELINE STATE TO AZURE")
    print("=" * 70)
    print(
        f"Storage account: {AZURE_PRIVATE_STORAGE_ACCOUNT}"
    )
    print(f"Container:       {AZURE_PRIVATE_CONTAINER}")
    print(f"Authentication:  {auth_mode}")

    relevant_prefixes = tuple(
        prefix.rstrip("/") + "/"
        for _, prefix in PRIVATE_STATE_ROOTS
    )

    remote_blobs = {
        blob.name: blob
        for blob in container_client.list_blobs()
        if blob.name.startswith(relevant_prefixes)
    }

    uploaded = 0
    skipped = 0
    uploaded_bytes = 0

    for local_path, blob_name in state_files:
        stat = local_path.stat()
        remote = remote_blobs.get(blob_name)

        unchanged = False
        if remote is not None:
            remote_size = getattr(remote, "size", None)
            remote_modified = getattr(
                remote,
                "last_modified",
                None,
            )

            same_size = (
                remote_size is not None
                and int(remote_size) == int(stat.st_size)
            )
            not_newer = (
                remote_modified is not None
                and stat.st_mtime
                <= remote_modified.timestamp() + 1.0
            )

            unchanged = same_size and not_newer

        if unchanged:
            skipped += 1
            continue

        blob_client = container_client.get_blob_client(
            blob_name
        )

        with local_path.open("rb") as source:
            blob_client.upload_blob(
                source,
                overwrite=True,
            )

        uploaded += 1
        uploaded_bytes += stat.st_size

        # Keep console output useful without printing hundreds of lines.
        if uploaded <= 10:
            print(f"  uploaded: {blob_name}")
        elif uploaded == 11:
            print("  ...")

    print("-" * 70)
    print(
        f"Private state: {uploaded:,} uploaded, "
        f"{skipped:,} unchanged."
    )
    print(
        f"Uploaded data: "
        f"{uploaded_bytes / (1024 * 1024):.1f} MB."
    )



def publish_dashboard_json_to_azure() -> None:
    """Upload the website JSON bundle to the public Azure Blob container."""
    if not AZURE_PUBLISH_ENABLED:
        print(
            "\nAzure publication disabled "
            "(AZURE_PUBLISH_ENABLED=0)."
        )
        return

    website_dir = PROJECT_DIR / "data" / "running"
    if not website_dir.exists():
        raise FileNotFoundError(
            f"Website JSON directory does not exist: {website_dir}"
        )

    missing_files = [
        filename
        for filename in DASHBOARD_JSON_FILES
        if not (website_dir / filename).is_file()
    ]
    if missing_files:
        missing_text = ", ".join(missing_files)
        raise FileNotFoundError(
            "Azure publication aborted because dashboard JSON files are "
            f"missing: {missing_text}"
        )

    try:
        from azure.storage.blob import ContentSettings
    except ImportError as exc:
        raise RuntimeError(
            "Azure publication requires 'azure-storage-blob'. "
            "Install it with:\n"
            "  pip install azure-storage-blob azure-identity"
        ) from exc

    blob_service, auth_mode, account_url = (
        create_azure_blob_service_client(
            AZURE_PUBLIC_STORAGE_ACCOUNT,
            AZURE_STORAGE_CONNECTION_STRING,
        )
    )

    container_client = blob_service.get_container_client(
        AZURE_PUBLIC_CONTAINER
    )

    print("\n" + "=" * 70)
    print("PUBLISH DASHBOARD JSON TO AZURE")
    print("=" * 70)
    print(f"Storage account: {AZURE_PUBLIC_STORAGE_ACCOUNT}")
    print(f"Container:       {AZURE_PUBLIC_CONTAINER}")
    print(f"Authentication:  {auth_mode}")

    uploaded = 0
    for filename in DASHBOARD_JSON_FILES:
        local_path = website_dir / filename
        blob_client = container_client.get_blob_client(
            filename
        )

        with local_path.open("rb") as file:
            blob_client.upload_blob(
                file,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=(
                        "application/json; charset=utf-8"
                    ),
                    cache_control="no-cache",
                ),
            )

        uploaded += 1
        print(f"  uploaded: {filename}")

    print("-" * 70)
    print(f"Uploaded {uploaded} dashboard JSON files.")
    print(
        "Dashboard base URL: "
        f"{account_url}/{AZURE_PUBLIC_CONTAINER}"
    )


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def main() -> None:
    print("=" * 70)
    print("RUNNING INSIGHTS PIPELINE")
    print("=" * 70)

    # On a fresh machine (such as a GitHub-hosted runner), restore the
    # incremental Garmin history and saved Garmin authentication first.
    restore_private_state_from_azure()

    export_garmin_data()
    run_advanced_analytics()
    build_dashboard_data()

    # Persist state only after the local pipeline has completed
    # successfully. Public dashboard JSON is then published separately.
    upload_private_state_to_azure()
    publish_dashboard_json_to_azure()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Website JSON: {PROJECT_DIR / 'data' / 'running'}")


if __name__ == "__main__":
    main()
