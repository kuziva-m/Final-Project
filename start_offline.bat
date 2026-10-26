@echo off
rem Start MSME Ledger on this computer, with no internet needed.
rem Phones on the same Wi-Fi open the address shown below in their browser.
cd /d "%~dp0"

rem Offline: no Claude, so the free reader (Tesseract + EasyOCR) reads every page.
set ANTHROPIC_API_KEY=
rem No Hugging Face backup on a laptop; saved data stays in the data folder.
set HF_TOKEN=

python -m web.app
echo.
echo The app has stopped. Press any key to close this window.
pause >nul
