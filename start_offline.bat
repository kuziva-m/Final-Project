@echo off
rem Start MSME Ledger on this computer. It works with no internet.
rem Phones on the same Wi-Fi open the address shown below in their browser.
cd /d "%~dp0"

rem No API key on the laptop. Once it is connected to the online app (Account
rem page), Claude reads pages there whenever there is internet; otherwise the
rem free reader (Tesseract + EasyOCR) reads every page.
set ANTHROPIC_API_KEY=
rem The online app the Account page offers to connect to.
set LEDGER_ONLINE_APP_URL=https://kmawoyo-msme-digitization.hf.space
rem No Hugging Face backup on a laptop; saved data stays in the data folder.
set HF_TOKEN=

rem Use the project's virtual environment if there is one (e.g. made by VS Code).
set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe
%PY% -m web.app
echo.
echo The app has stopped. Press any key to close this window.
pause >nul
