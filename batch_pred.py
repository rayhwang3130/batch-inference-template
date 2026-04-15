import datetime
import re
from pathlib import Path

import google.cloud.aiplatform as aiplatform


PROJECT_ID = "slcc-buzz-ai"
REGION = "global"
MODEL_NAME_STRING = "publishers/google/models/gemini-3.1-flash-lite-preview"

GCS_BUCKET = "gs://slcc-buzz-ai"
BASE_DIR = Path(__file__).resolve().parent
JSONL_INPUT_DIR = BASE_DIR / "data" / "jsonl_inputs"

MODEL_PARAMETERS = {
    "temperature": 0,
    "topP": 0.95,
    "topK": 40,
    "maxOutputTokens": 256,
    "thinkingConfig": {"thinkingBudget": 0},
}


def sanitize_stem(path: Path) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", path.stem).strip("_")
    return cleaned or "input"


def list_request_paths():
    return sorted(JSONL_INPUT_DIR.glob("*.jsonl"))


def start_batch_prediction_job():
    request_paths = list_request_paths()
    if not request_paths:
        raise FileNotFoundError(f"No request jsonl files found in {JSONL_INPUT_DIR.resolve()}")

    print(f"Starting batch prediction jobs for project: {PROJECT_ID}")
    aiplatform.init(project=PROJECT_ID, location=REGION)

    for request_path in request_paths:
        request_stem = sanitize_stem(request_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        job_display_name = f"{request_stem}-user-labeling-{timestamp}"
        gcs_source_uri = f"{GCS_BUCKET}/user_labeling_input/{request_path.name}"
        gcs_destination_uri_prefix = f"{GCS_BUCKET}/user_labeling_output/{request_stem}/"

        batch_job = aiplatform.BatchPredictionJob.create(
            job_display_name=job_display_name,
            model_name=MODEL_NAME_STRING,
            gcs_source=gcs_source_uri,
            gcs_destination_prefix=gcs_destination_uri_prefix,
            instances_format="jsonl",
            predictions_format="jsonl",
            model_parameters=MODEL_PARAMETERS
        )

        print(f"Job submitted for {request_path.name}")
        print(f"Job name: {job_display_name}")
        print(f"Destination: {gcs_destination_uri_prefix}")


if __name__ == "__main__":
    start_batch_prediction_job()
