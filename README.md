# Batch Inference Labeling Template

This repository is an example and master-format template for building a Vertex AI batch inference workflow that:

1. converts raw CSV/Excel files into JSONL batch requests,
2. submits those requests to a Gemini model on Vertex AI,
3. converts prediction JSONL outputs back into a reviewable Excel file.

It is intended to be copied and adapted for new labeling projects, not used as a product-specific repository as-is.

## What This Template Does

The current example classifies each input row into one of four user types:

- `general_consumer`
- `social_content_creator`
- `field_recorder`
- `professional_video_production_worker`

The generated model response is expected to follow this JSON shape:

```json
{
  "general_consumer": 0,
  "social_content_creator": 0,
  "field_recorder": 0,
  "professional_video_production_worker": 1
}
```

Default business rule:

- normally exactly one label should be `1`
- multiple `1` values are treated as invalid and normalized during conversion
- all `0` is allowed only when there is not enough evidence in the text

## Intended Use

Use this repository as a starter template when you need to:

- change the label schema
- replace the prompt in `config.yaml`
- point to a different Vertex AI project, bucket, or model
- reuse the same raw -> JSONL -> prediction -> Excel workflow for another classification problem

## Repository Structure

- `config.yaml`: prompt template used to build each request
- `create_jsonl.py`: converts raw input files into JSONL batch request files
- `batch_pred.py`: submits JSONL requests to Vertex AI batch prediction
- `convert_jsonl.py`: reads prediction JSONL files and writes Excel outputs
- `line-by-line-guide.txt`: process notes and operating reference
- `data/raw/`: put raw `.xlsx`, `.xls`, or `.csv` files here
- `data/jsonl_inputs/`: generated request files
- `data/jsonl_outputs/`: downloaded prediction files
- `data/excel_outputs/`: final review-friendly Excel outputs

## Expected Input Format

The scripts try to find a text column using these candidates:

- `Message Content`
- `message`
- `text`
- `content`
- `post`
- `comment`
- `sentence`

If none of them exist, the first column is used.

Optional key columns:

- `key`
- `Key`
- `id`
- `ID`

If no key column exists, row index is used.

## Setup

Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install pandas pyyaml jinja2 openpyxl xlsxwriter google-cloud-aiplatform
```

You also need:

- a Google Cloud project with Vertex AI enabled
- access to the target GCS bucket
- authenticated `gcloud` / Application Default Credentials

## Configuration You Must Replace

Before using this template in a new project, review and replace:

- prompt text in `config.yaml`
- label schema in `create_jsonl.py`
- label columns in `convert_jsonl.py`
- `PROJECT_ID` in `batch_pred.py`
- `GCS_BUCKET` in `batch_pred.py`
- `MODEL_NAME_STRING` in `batch_pred.py`

Do not upload project-specific credentials or private data to GitHub.

## Recommended Workflow

### 1. Put raw files into the input folder

Place source files in:

```text
data/raw/
```

### 2. Generate JSONL requests

```powershell
python .\create_jsonl.py
```

This writes batch request files to:

```text
data/jsonl_inputs/
```

### 3. Submit batch inference jobs

Update the values in `batch_pred.py`, then run:

```powershell
python .\batch_pred.py
```

This script submits one Vertex AI batch job per JSONL file.

### 4. Collect prediction outputs

Download the resulting prediction JSONL files into:

```text
data/jsonl_outputs/
```

### 5. Convert predictions into Excel

```powershell
python .\convert_jsonl.py
```

This creates review files in:

```text
data/excel_outputs/
```

## Files to Review First

If you are inheriting or repurposing this template, start here:

- `config.yaml`
- `create_jsonl.py`
- `batch_pred.py`
- `convert_jsonl.py`

## GitHub Publishing Notes

Before publishing:

- remove or anonymize real client/project names if needed
- remove sample raw data and model outputs if they are not shareable
- verify `batch_pred.py` does not point to a real production project unless intended
- confirm the prompt in `config.yaml` is safe to share publicly

## License / Usage

Add your team's preferred license before publishing if this repository will be shared outside the organization.
