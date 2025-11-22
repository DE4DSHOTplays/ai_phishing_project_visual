# Phishing URL Detector — Visual Edition

Welcome to the Visual Edition of the Phishing URL Detector — a compact, practical
tool that brings machine learning and human-friendly visuals together to help you
spot phishing URLs, experiment with models, and visualize results.

Think of this repo as a small lab:
- a trained model you can run locally,
- a Streamlit UI to explore and explain predictions,
- and scripts to retrain or extend the model with your own data.

✨ Highlights
- Fast single-URL checks with an explanation of extracted features
- Upload your own dataset (CSV with columns `url,label`) to evaluate or retrain
- Visual metrics: confusion matrix, ROC curve and other evaluation charts
- Lightweight Streamlit UI for interactive demos — no heavy infra required

Core files
- `app_visual.py` — Streamlit app (UI + visualizations)
- `train_and_save.py` — training script to produce `phish_model.joblib`
- `phish_model.joblib` — example trained model used by the app
- `requirements.txt` — Python dependencies
- `run_phishing_visual.bat` — Windows helper to launch the Streamlit app

Quick start (Windows PowerShell)
1. Create & activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Launch the visual app:

```powershell
streamlit run app_visual.py
```

Or simply double-click `run_phishing_visual.bat` to open the app.

Example: check a single URL
1. Open the Streamlit UI
2. Paste a URL into the single-check box and run
3. See: prediction (phishing / benign), probability score, and feature explanations

Training / Updating the Model
- Use `train_and_save.py` to train a new model on your CSV dataset. Typical CSV
	format: two columns `url,label` (where `label` is 1 for phishing, 0 for benign).
- After training, save or replace `phish_model.joblib`; the app will use the
	model file available in the repository root.

Best practices & notes
- `.gitignore` excludes virtual environments, caches, and common large binaries.
	Keep large datasets or final model artifacts out of the repository when sharing
	publicly — use releases, object storage, or dataset registries instead.
- Remove any sensitive or personal data before publishing the repository.

Want this README to be even more visual?
- I can add example screenshots, a small animated demo, or a short walkthrough
	in `docs/` (you can provide images or I can add placeholders).

Repository
- Remote: https://github.com/DE4DSHOTplays/ai_phishing_project_visual

Credits
- Built with Python, scikit-learn, Streamlit and a dash of curiosity.

If you'd like, I can also add a CONTRIBUTING guide, a detailed TRAINING.md, or
example unit tests for the training pipeline.