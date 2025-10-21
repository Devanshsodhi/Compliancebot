@echo off
echo ========================================
echo   AI Compliance Bot - Streamlit UI
echo ========================================
echo.
echo Starting Streamlit application...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
call ..\venv\Scripts\activate
streamlit run ..\app.py
