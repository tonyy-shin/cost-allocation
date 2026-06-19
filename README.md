# Common Cost Allocation Tool

A desktop tool that allocates shared (common) costs across cost centers using a
transfer-COA mapping and a multi-cycle distribution schedule, and produces two
complementary views of the result:

- **by_coa** — how much each common-cost COA distributed per cycle, keyed by sender and receiver.
- **by_cc** — each cost center's settled balance after every cycle.

All processing runs locally; no data is transmitted externally.

---

## Input Files

Four CSV files plus an output directory are selected at run time. File **encoding** is
auto-detected in the order UTF-8 (with or without BOM) → EUC-KR → CP949, so Excel's
default Korean "CSV (comma delimited)" files load correctly.

> **Excel cell formatting is tolerated.** Amount columns saved with a thousands
> separator (`5,000,000`), accounting parentheses (`(5,000,000)`), a trailing/unicode
> minus, or the `%` column saved as `30%` are all parsed back to numbers. Plain values
> (`5000000`, `0.3`) are equally fine.

Code columns (`COA`, `Cost Center`, `기존COA`, …) are normalized: values like `7832.0`
become `7832`. Values that cannot be parsed are reported as warnings.

### 1. COA Amount — `coa_amount.csv`

The master sheet. It is also the **single source of the cost-center list** — every CC
used by the pipeline comes from its `Cost Center` column. Rows with a blank
`Cost Center` are dropped.

| Column        | Type   | Description           |
|---------------|--------|-----------------------|
| `COA`         | string | Base COA code         |
| `Cost Center` | string | Cost center code      |
| `Amounts`     | number | Amount for (COA, CC)  |

```csv
COA,Cost Center,Amounts
6100,1001,5000000
7100,1002,8000000
7200,2002,2000000
```

### 2. Transfer COA Mapping — `mapping.csv`

Maps each base COA to its transfer COA. A COA listed here is a **common cost**; a COA
not listed is a **direct cost** and is excluded from allocation.

| Column    | Type   | Description       |
|-----------|--------|-------------------|
| `전기COA` | string | Transfer COA code |
| `기존COA` | string | Base COA code     |

```csv
전기COA,기존COA
E6100,6100
E6200,6200
```

### 3. Allocation Cycle — `cycle.csv` (wide format)

Authored as a matrix: `차수` and `Sender CC` are id columns, and **every remaining
column header is a Receiver CC code**; each cell is the allocation ratio from that
sender to that receiver in that cycle. The tool melts this grid to one row per
(sender, receiver) internally.

- Blank or `0` cells mean "no allocation" and are dropped.
- The `%` value accepts a decimal (`0.3`) or a percentage (`30%`).
- Within each `(차수, Sender CC)` group the ratios should sum to `1.0`. Tiny
  floating-point drift is auto-normalized (with a warning); a deviation of `0.005` or
  more raises an error so the input can be corrected.
- Cycles are applied in ascending `차수` order.

| Column      | Type   | Description                                  |
|-------------|--------|----------------------------------------------|
| `차수`      | int    | Cycle number (1, 2, …)                       |
| `Sender CC` | string | Cost center sending costs                    |
| *(per col)* | number | Header = Receiver CC; cell = allocation ratio |

```csv
차수,Sender CC,1001,1002,1003,3001
1,1001,,0.3,0.7,
2,2001,0.5,,,0.5
```

A receiver (or sender) CC that appears in `cycle.csv` but not in `coa_amount.csv` is
expected: it is inserted automatically with an amount of 0 so it still appears in the
by_cc output and receives its allocations.

### 4. Pre-allocation Amount — `pre_allocation.csv`

Same schema as `coa_amount.csv`. Only the per-CC total is used — amounts are summed by
`Cost Center` to populate the `배부전금액` (pre-allocation balance) column of the by_cc
output. The `COA` column is ignored.

```csv
COA,Cost Center,Amounts
6100,1001,600000
6100,1002,500000
```

---

## Output Files

Results are written under the selected output directory as a two-branch tree. All files
use UTF-8 with BOM for Excel compatibility.

```
<output_dir>/
  by_coa/
    result.csv
  by_cc/
    1차배부후.csv
    2차배부후.csv
    …            # one file per cycle
```

### `by_coa/result.csv`

One row per `(전기COA, 기존COA, Sender CC, Receiver CC)` for **common-cost** sender rows.
Each sender's per-cycle amount is split across its receivers by the cycle ratio, so a
`(전기COA, 기존COA, Sender CC)` group spans one row per receiver.

| Column                 | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `전기COA`              | Transfer COA                                                 |
| `기존COA`              | Base COA                                                     |
| `Sender CC`            | Sending cost center                                          |
| `Receiver CC`          | Receiving cost center                                        |
| `1차배부금액` … `n차배부금액` | Amount sent on this (sender → receiver) row in each cycle (the sender's cycle amount × the cycle's Sender→Receiver ratio) |
| *(empty column)*       | Blank separator                                              |
| `1차배부합계` … `n차배부합계` | Column-wide total for each cycle (placed in the first row only) |

### `by_cc/{n}차배부후.csv`

One file per cycle `n`, holding every cost center's settled balance after cycle `n`.

| Column                 | Description                                                      |
|------------------------|------------------------------------------------------------------|
| `CC`                   | Cost center code                                                 |
| `배부전금액`           | Pre-allocation balance (from `pre_allocation.csv`)               |
| `1차후금액` … `n차후금액` | Balance attributable to each cycle (the last column folds in the still-held original balance) |
| `배부합계`             | Row total of the 후금액 columns (the CC's final balance)         |

A **totals row** is appended at the bottom of each by_cc file: `CC` is labelled `합계`,
`배부전금액` and `배부합계` carry the integer-rounded column totals, and the remaining
cells are blank. Because allocation only moves money between cost centers, each file
satisfies `배부전금액` total == `배부합계` total.

---

## Pipeline Flow

The run is wired in [main.py](main.py) and flows through four modules:

1. **loader** ([src/loader.py](src/loader.py)) — read each CSV with encoding fallback,
   check required columns, normalize code columns, parse numeric/percent columns,
   validate and auto-normalize cycle ratios, and build the shared `CategoricalDtype`s
   used to harmonize code columns across sheets.
2. **prepare** ([src/prepare.py](src/prepare.py)) — `fill_missing_cycle_cc` adds
   zero-amount rows (`COA = NaN`) for cycle CCs absent from the master; `build_enriched`
   assigns each row its transfer COA (`전기COA`) — common costs get the mapped value,
   direct costs get an empty string.
3. **allocation** ([src/allocation.py](src/allocation.py)) — `build_by_coa` produces the
   by_coa table (each sender amount exploded into one row per receiver via the cycle
   ratios) and the per-(cycle, sender) totals; `build_by_cc` walks the cycles in
   order, crediting receivers and draining senders, and snapshots each CC's labelled
   balances into one frame per cycle.
4. **output** ([src/output.py](src/output.py)) — `save_results` writes the `by_coa/` and
   `by_cc/` tree; `append_total_row` adds the by_cc totals row.

Data-quality issues found during loading (non-numeric codes/amounts, percent values
above 1 without a `%` sign, auto-normalized ratios) are collected and shown in a
completion dialog; a missing required column or an unreadable encoding aborts the run
with an explanatory message.

---

## Usage

```bash
pip install -r requirements.txt
python main.py
```

`python main.py` opens a file-selection window. Choose the four input CSV files and the
output directory, then click **실행** (enabled once all paths are set). The dual-branch
result tree is written under the chosen directory, and a completion dialog reports
success, success-with-warnings, or failure.

The tool is also packaged as a standalone, windowed Windows executable
(`cost-allocation.exe`) built with PyInstaller from
[cost-allocation.spec](cost-allocation.spec).

---

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/        # or: pytest -v
```

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs `pytest -v` on Python
3.13 for every push and pull request targeting `main`.

---

## Project Structure

```
main.py                  # Entry point: UI → load → build outputs → save
src/
  loader.py              # CSV readers, code/number/percent parsing, dtype setup
  prepare.py             # Missing-CC fill, transfer-COA enrichment
  allocation.py          # by_coa table and by_cc per-cycle snapshots
  output.py              # Result tree writer + by_cc totals row
  ui.py                  # tkinter file-selection and completion dialogs
sample_data/             # Example inputs (coa_amount, mapping,
                         #   cycle, pre_allocation) for local testing
tests/                   # pytest suite
```

## Dependencies

- Python 3.13
- pandas 3.0.3, numpy 2.4.6 — see [requirements.txt](requirements.txt)
- pytest 9.1.0 (testing) — see [requirements-dev.txt](requirements-dev.txt)
