# Running MSME Ledger offline on a laptop

The app can run on one Windows laptop with **no internet**. Phones in the shop
connect to the laptop over Wi-Fi and use the app in their browser (or the
phone app) exactly as they would online.

Offline, pages are read by the built-in free reader (Tesseract + EasyOCR)
instead of Claude, so:

- tables come out in fixed columns (Date, Item, Qty, Price, Total) rather than
  each document's own headings;
- reading is slower and less accurate on messy handwriting, so check every row.

Accounts, saving, editing, deleting, and Excel/CSV/JSON downloads all work the
same. Saved data stays in the `data` folder on the laptop.

You need the internet **once**, to install everything. After that it runs
without it.

## What you need

- A Windows 10 or 11 laptop with at least 8 GB of memory (16 GB is more
  comfortable), a 4-core processor or better, and about 15 GB of free disk.
- Wi-Fi that the laptop and the phones share. This doesn't need to be connected
  to the internet: a cheap Wi-Fi router works, or a phone's own hotspot with
  the laptop joined to it.

## 1. Install (once, with internet)

1. **Python 3.11**: download it from python.org. On the first installer screen,
   tick **"Add python.exe to PATH"**.
2. **Tesseract OCR**: install the Windows build from the UB Mannheim page
   (search "tesseract ub mannheim"). Keep the default folder,
   `C:\Program Files\Tesseract-OCR`; the app finds it there.
3. **The app**: copy the project folder onto the laptop, for example
   `C:\MSME-Ledger`. Either clone it from GitHub, or copy it from another
   computer on a USB stick.
4. Open **Command Prompt** in that folder and run:

   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   python scripts\prepare_offline.py
   ```

   The first command takes a while; it downloads about 1 GB. The last one
   downloads the reader's model files into `models\easyocr` and checks
   everything. It should finish with **"Ready."** If it lists something under
   "Not ready yet", fix that and run it again.

## 2. Start it (every time)

1. Connect the laptop to the shop's Wi-Fi, or to the phone's hotspot.
2. Double-click **`start_offline.bat`**.
3. The first time, Windows asks whether to allow Python on networks. Tick
   **Private networks** and click **Allow**. Without this, phones can't
   connect.
4. The window shows addresses like this:

   ```
   On this computer:            http://localhost:7860
   On phones on the same Wi-Fi: http://192.168.43.10:7860
   ```

   Leave the window open while the app is in use. Closing it stops the app.

## 3. Use it from a phone

- **In the browser**: type the "On phones" address exactly, including
  `http://` and `:7860`. Create an account, then scan as usual.
- **In the phone app**: on the sign-in screen, or from the menu (⋮) under
  **Change server**, enter the laptop's address, e.g. `192.168.43.10`. Accounts
  are made in the phone's browser first; then sign in in the app.

The laptop's address can change when it reconnects to the Wi-Fi, so check
the window each time you start the app.

## Good to know

- **Speed**: in a test on a 4-core processor with no internet, a phone photo
  took about 5 seconds and a full A4 PDF page about 16 seconds. A slower
  laptop takes longer. A PDF is read page by page with progress shown, up to
  40 pages. The reader uses about 4 GB of memory while it works.
- **Settings**: you can set these before `python -m web.app`, or edit
  `start_offline.bat`:
  - `LEDGER_MAX_PDF_PAGES` changes the page limit.
  - `LEDGER_TESSERACT_CMD` points to `tesseract.exe` if it's installed elsewhere.
  - `LEDGER_EASYOCR_GPU=1` uses an NVIDIA graphics card. That also needs the
    CUDA build of PyTorch.
- **Using Claude on the laptop**: if the laptop does have internet and you
  want Claude to read the pages, start the app with the key set instead of
  using `start_offline.bat`:

  ```
  set ANTHROPIC_API_KEY=your-key
  python -m web.app
  ```

- **Backups**: the online version backs up to Hugging Face; the laptop
  doesn't. Copy the `data` folder to a USB stick now and then.
- **Sign-in**: the laptop keeps its sign-in key in `data\secret_key`, so
  people stay signed in across restarts. Don't share that file.
