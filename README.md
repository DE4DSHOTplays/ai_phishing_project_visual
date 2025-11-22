# Phishing URL Detector — Visual Edition

A small project that detects phishing URLs using a trained ML model
and provides a Streamlit-based visual interface for exploring predictions,
uploading datasets, and evaluating results.

**Features:**
- Dataset upload (CSV with columns: `url,label`) to retrain or evaluate the model
- Visual evaluation: confusion matrix, ROC curve, accuracy and other metrics
- Single-URL check with feature explanation
- Streamlit UI for interactive demos and quick testing

**Files of interest:**
- `app_visual.py` : Streamlit app to run the UI
- `phish_model.joblib` : Trained model used by the app
- `train_and_save.py` : Script to train a model and save `phish_model.joblib`
- `requirements.txt` : Python dependencies
- `run_phishing_visual.bat` : Windows helper to start the Streamlit app

**Quick start (Windows / PowerShell)**
1. Create and activate a virtualenv (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the visual app:

```powershell
streamlit run app_visual.py
```

or double-click `run_phishing_visual.bat`.

**Training**
- To retrain or update the model, edit and run `train_and_save.py`. This will produce
	a new `phish_model.joblib` which the app will use.

**Notes & best practices**
- The repository includes a `.gitignore` to avoid committing virtual environments,
	caches, and large model binaries. Consider removing large files (e.g. Excel files)
	or storing models in a release or external storage if the repo will be public.
- If you share this repo publicly, remove or avoid committing sensitive data.

**Repository**
- Remote: https://github.com/DE4DSHOTplays/ai_phishing_project_visual

If you want, I can add a short usage gif/screenshot or expand the Training section.