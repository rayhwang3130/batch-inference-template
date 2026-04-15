import json
import re
from pathlib import Path

import pandas as pd
import os


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
JSONL_OUTPUT_DIR = BASE_DIR / "data" / "jsonl_outputs"
EXCEL_OUTPUT_DIR = BASE_DIR / "data" / "excel_outputs"

TEXT_COLUMN_CANDIDATES = [
    "Message Content",
    "message",
    "text",
    "content",
    "post",
    "comment",
    "sentence",
]
KEY_COLUMN_CANDIDATES = ["key", "Key", "id", "ID"]

LABEL_COLUMNS = [
    "general_consumer",
    "social_content_creator",
    "field_recorder",
    "professional_video_production_worker",
]


def openable_path(path: Path) -> str:
    absolute_path = str(path.absolute())
    if os.name == "nt" and not absolute_path.startswith("\\\\?\\") and len(absolute_path) >= 240:
        return "\\\\?\\" + absolute_path
    return absolute_path


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def list_input_paths():
    return sorted(list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls")) + list(RAW_DIR.glob("*.csv")))


def list_prediction_paths():
    return sorted(JSONL_OUTPUT_DIR.rglob("*predictions.jsonl"))


def sanitize_stem(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")
    return cleaned or "input"


def find_prediction_paths(raw_input_path: Path, prediction_paths):
    stem = sanitize_stem(raw_input_path)
    matches = [path for path in prediction_paths if stem.lower() in path.as_posix().lower()]
    if matches:
        return matches
    raise FileNotFoundError(f"No prediction jsonl matched raw file '{raw_input_path.name}'")


def resolve_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    if len(df.columns) == 0:
        raise ValueError("Input file has no columns.")
    return str(df.columns[0])


def resolve_key_column(df: pd.DataFrame) -> str | None:
    for column in KEY_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    return None


def extract_key(payload):
    key = payload.get("key")
    if key is not None:
        return str(key)
    request = payload.get("request", {})
    if isinstance(request, dict) and request.get("key") is not None:
        return str(request["key"])
    return "unknown"


def extract_json_object(text_content: str):
    text_content = text_content.strip()
    if not text_content:
        return {}
    if text_content.startswith("```json"):
        text_content = text_content[7:]
    if text_content.startswith("```"):
        text_content = text_content[3:]
    if text_content.endswith("```"):
        text_content = text_content[:-3]
    text_content = text_content.strip()

    try:
        return json.loads(text_content)
    except json.JSONDecodeError:
        start = text_content.find("{")
        end = text_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text_content[start : end + 1])
            except json.JSONDecodeError:
                return {}
    return {}


def extract_prediction_parts(payload):
    response = payload.get("response", {})
    if not isinstance(response, dict):
        return {}

    candidates = response.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        first_candidate = candidates[0]
        content = first_candidate.get("content", {})
        parts = content.get("parts", [])
        if isinstance(parts, list) and parts:
            first_part = parts[0]
            if isinstance(first_part, dict):
                if isinstance(first_part.get("text"), str):
                    return extract_json_object(first_part["text"])
                if isinstance(first_part.get("inlineData"), dict):
                    inline_data = first_part["inlineData"]
                    if isinstance(inline_data.get("data"), str):
                        return extract_json_object(inline_data["data"])

    if isinstance(response.get("predictions"), list) and response["predictions"]:
        first_prediction = response["predictions"][0]
        if isinstance(first_prediction, dict):
            return first_prediction
        if isinstance(first_prediction, str):
            return extract_json_object(first_prediction)

    if isinstance(response.get("text"), str):
        return extract_json_object(response["text"])

    return {}


def normalize_binary(value) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "1.0", "true", "yes"}:
            return 1
    return 0


def normalize_prediction(predicted: dict) -> dict:
    normalized = {label: normalize_binary(predicted.get(label)) for label in LABEL_COLUMNS}
    positive_labels = [label for label, value in normalized.items() if value == 1]

    if len(positive_labels) <= 1:
        return normalized

    first_positive = positive_labels[0]
    return {label: 1 if label == first_positive else 0 for label in LABEL_COLUMNS}


def build_output_dataframe(raw_df: pd.DataFrame, prediction_rows):
    text_column = resolve_text_column(raw_df)
    key_column = resolve_key_column(raw_df)
    records = []

    for index, raw_row in raw_df.reset_index(drop=True).iterrows():
        key = str(raw_row[key_column]) if key_column and pd.notna(raw_row[key_column]) else str(index)
        predicted = normalize_prediction(prediction_rows.get(key, {}))

        output_row = dict(raw_row)
        output_row["Text Column Used"] = text_column
        for label in LABEL_COLUMNS:
            output_row[label] = predicted[label]
        records.append(output_row)

    preferred_columns = [text_column, "Text Column Used", *LABEL_COLUMNS]
    ordered_columns = preferred_columns + [column for column in records[0].keys() if column not in preferred_columns]
    return pd.DataFrame(records)[ordered_columns]


def read_prediction_rows(prediction_paths):
    prediction_rows = {}
    duplicate_keys = set()
    empty_prediction_count = 0
    total_lines = 0

    for prediction_path in prediction_paths:
        with open(openable_path(prediction_path), "r", encoding="utf-8-sig") as file:
            for line in file:
                if not line.strip():
                    continue

                total_lines += 1
                payload = json.loads(line)
                key = extract_key(payload)
                extracted = extract_prediction_parts(payload)

                if not extracted:
                    empty_prediction_count += 1
                if key in prediction_rows:
                    duplicate_keys.add(key)

                prediction_rows[key] = extracted

    return prediction_rows, total_lines, len(duplicate_keys), empty_prediction_count


def main():
    raw_input_paths = list_input_paths()
    prediction_paths = list_prediction_paths()

    if not raw_input_paths:
        raise FileNotFoundError(f"No raw input files found in {RAW_DIR.resolve()}")
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction jsonl files found in {JSONL_OUTPUT_DIR.resolve()}")

    EXCEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for raw_input_path in raw_input_paths:
        prediction_matches = find_prediction_paths(raw_input_path, prediction_paths)
        raw_df = load_input_dataframe(raw_input_path)
        prediction_rows, prediction_line_count, duplicate_key_count, empty_prediction_count = read_prediction_rows(
            prediction_matches
        )

        output_path = EXCEL_OUTPUT_DIR / f"{sanitize_stem(raw_input_path)}_user_labeling_output.xlsx"
        result_df = build_output_dataframe(raw_df, prediction_rows)
        key_column = resolve_key_column(raw_df)
        matched_prediction_count = sum(
            1
            for index, raw_row in raw_df.reset_index(drop=True).iterrows()
            if (
                str(raw_row[key_column]) if key_column and pd.notna(raw_row[key_column]) else str(index)
            ) in prediction_rows
        )

        with pd.ExcelWriter(
            output_path,
            engine="xlsxwriter",
            engine_kwargs={"options": {"strings_to_urls": False}},
        ) as writer:
            result_df.to_excel(writer, index=False)

        print(f"Converted {raw_input_path.name} using {len(prediction_matches)} prediction file(s)")
        for prediction_path in prediction_matches:
            print(f"Prediction file: {prediction_path.name}")
        print(f"Raw rows: {len(raw_df)}")
        print(f"Prediction lines read: {prediction_line_count}")
        print(f"Matched rows: {matched_prediction_count}")
        print(f"Duplicate prediction keys: {duplicate_key_count}")
        print(f"Empty prediction payloads: {empty_prediction_count}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
