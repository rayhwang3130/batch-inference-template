import json
import re
from pathlib import Path

import pandas as pd
import yaml
from jinja2 import Template


BASE_DIR = Path(__file__).resolve().parent
YAML_PATH = BASE_DIR / "config.yaml"
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "jsonl_inputs"

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


def load_config():
    with YAML_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_input_dataframe(input_path: Path) -> pd.DataFrame:
    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(input_path)
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def list_input_paths():
    return sorted(list(DATA_DIR.glob("*.xlsx")) + list(DATA_DIR.glob("*.xls")) + list(DATA_DIR.glob("*.csv")))


def sanitize_stem(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")
    return cleaned or "input"


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


def main():
    config = load_config()
    template = Template(config["system_prompt"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_paths = list_input_paths()
    if not input_paths:
        raise FileNotFoundError(f"No raw input files found in {DATA_DIR.resolve()}")

    for input_path in input_paths:
        output_path = OUTPUT_DIR / f"{sanitize_stem(input_path)}_user_labeling_requests.jsonl"
        df = load_input_dataframe(input_path).reset_index(drop=True)
        text_column = resolve_text_column(df)
        key_column = resolve_key_column(df)

        print(f"Input file: {input_path.name}")
        print(f"Rows: {len(df)}")
        print(f"Using text column: {text_column}")
        print(f"Using key column: {key_column if key_column else 'row_index'}")

        with output_path.open("w", encoding="utf-8") as file_out:
            for index, row in df.iterrows():
                rendered_prompt = template.render(
                    input_text="" if pd.isna(row[text_column]) else str(row[text_column]),
                )

                request_object = {
                    "key": str(row[key_column]) if key_column and pd.notna(row[key_column]) else str(index),
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": rendered_prompt}],
                            }
                        ],
                        "generationConfig": {
                            "temperature": 0,
                            "maxOutputTokens": 64,
                            "responseMimeType": "application/json",
                            "responseSchema": {
                                "type": "OBJECT",
                                "properties": {
                                    "general_consumer": {"type": "INTEGER"},
                                    "social_content_creator": {"type": "INTEGER"},
                                    "field_recorder": {"type": "INTEGER"},
                                    "professional_video_production_worker": {"type": "INTEGER"},
                                },
                                "required": [
                                    "general_consumer",
                                    "social_content_creator",
                                    "field_recorder",
                                    "professional_video_production_worker",
                                ],
                            },
                            "thinkingConfig": {"thinkingBudget": 0},
                        },
                    },
                }

                file_out.write(json.dumps(request_object, ensure_ascii=False) + "\n")

        print(f"Created: {output_path}")


if __name__ == "__main__":
    main()
