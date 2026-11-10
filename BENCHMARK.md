# Measuring the free reader

Claude drafts the correct contents of real ledger pages; you check them in
the app; the free reader is scored against them. Every change to the free
reader can then be measured instead of guessed.

## 1. Read pages with Claude

Claude needs either `ANTHROPIC_API_KEY` set on this computer, or the app
connected to the online app (Account page, "Read with Claude when there is
internet").

```
python scripts/claude_batch.py test_photos\ --split test
python scripts/claude_batch.py train_photos\ --split train
```

- **test** pages are kept aside for measuring. Never tune on them, or the
  scores will look better than the reader really is. 10-20 pages from several
  books and writers is a good start.
- **train** pages are for tuning, and later for training the reader.
- Each page becomes a document in the app: "Benchmark test: <file> p<n>".
  Add `--email you@example.com` if the app has more than one account.
- Each page is one Claude request. Running it again skips pages already read.

## 2. Check each page

Open each Benchmark document in **My documents**, fix anything Claude got
wrong against the photo (the amber cells first), and **Save**, even if
nothing needed fixing. Saving marks the page as checked. Only checked pages
are scored.

## 3. Score the free reader

```
python scripts/score_free_reader.py --note "baseline"
```

It prints, for the test pages: cells right (split into numbers and dates,
and words), cells to fix per page, character error rate, rows and columns
found, and seconds per page, with the change since the last run. Details go
to `data\benchmark\runs\<time>\`: `pages.csv`, `summary.json`, and the free
reader's table for every page.

- `--reader claude` scores Claude's first reading against your checked
  version: how accurate Claude was.
- `--split train` scores the training pages instead.
- `--note "..."` labels the run, so the comparison says what changed.

## Keep it private

`data\benchmark` holds real business records and is never committed (it is in
`.gitignore`). Ask the businesses before using their pages, and keep the
folder on your own machines.
