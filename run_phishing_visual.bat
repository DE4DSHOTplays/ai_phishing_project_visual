@echo off
rem Ensure the script runs from its own directory
cd /d "%~dp0"

rem --- 1. Virtual Environment Setup ---
if not exist ".venv\Scripts\activate.bat" (
    echo Setting up virtual environment...
    python -m venv .venv
)

rem Activate the environment (using CALL is necessary to return to the script)
call .\.venv\Scripts\activate.bat

rem --- 2. Dependency Installation ---
echo Upgrading pip and installing/updating dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

rem --- 3. Run Streamlit App and Launch Browser ---
echo Starting Streamlit app...
rem Optional: Add a small delay to ensure the Streamlit server starts before the browser opens
timeout /t 3 /nobreak >nul

rem Open the browser to the specified address
start "" http://127.0.0.1:8501

rem Run the Streamlit application on the specified IP and Port
python -m streamlit run app_visual.py --server.address 127.0.0.1 --server.port 8501

rem Keep the window open after the server is shut down or if an error occurred
pause