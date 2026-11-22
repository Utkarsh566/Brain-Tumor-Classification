# Deployment Guide

## 1. Architecture

The project separates application hosting from model hosting.

- GitHub: Application source code and documentation.
- Hugging Face Hub: Trained Joblib model artifact.
- Streamlit Community Cloud: Web application hosting.

## 2. Hugging Face Model Hosting

The trained model is stored in a Hugging Face model repository.

Expected filename:

`brain_tumor_model.pkl`

The application uses `hf_hub_download` to retrieve the artifact.

Example:

```python
from huggingface_hub import hf_hub_download

MODEL_PATH = hf_hub_download(
    repo_id="YOUR_USERNAME/brain-tumor-classification-model",
    filename="brain_tumor_model.pkl"
)
```

Replace the repository ID with the actual repository.

The model repository must be publicly accessible if the deployed application downloads it without authentication.

## 3. GitHub Repository

The GitHub repository should contain:

- app.py
- requirements.txt
- README.md
- .gitignore
- docs/
- notebooks/
- assets/
- screenshots/

Do not upload the MRI dataset.

Do not commit the 154 MB model artifact to regular Git history. It is hosted separately on Hugging Face.

## 4. Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 5. Local Execution

Run:

```bash
streamlit run app.py
```

## 6. Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the project repository.
5. Choose `app.py` as the main file.
6. Deploy the application.

## 7. Troubleshooting

### Model download errors

Check:
- Repository ID.
- Model filename.
- Repository visibility.
- Hugging Face availability.

### Missing dependency errors

Check that the missing package is present in requirements.txt.

### Model loading errors

Ensure the application loads the correct Joblib artifact and uses compatible versions of the serialization-related libraries.

### Deployment limitations

The first application startup may take longer because the model needs to be downloaded. Hosting and runtime limits depend on the selected deployment service.
