# Common Cost Allocation Tool

A desktop tool that allocates shared (common) costs across cost centers
based on transfer-COA rules and multi-cycle distribution schedules,
then decomposes the results back to original COA.

All processing runs locally. No data is transmitted externally for security reasons.

---

## Input Files

Four CSV files are required. Encoding must be **UTF-8 (with or without BOM)**.

### 1. CC Master — `cc.csv`

| Column | Type   | Description      |
|--------|--------|------------------|
| CC     | string | Cost center code |

```csv
CC
1001
1002
1003
```

### 2. COA Amount — `coa_amount.csv`

| Column      | Type   | Description                                                                 |
|-------------|--------|-----------------------------------------------------------------------------|
| COA         | string | Base COA code                                                               |
| Cost Center | string | Cost center code                                                            |
| Amounts     | number | Amount (common costs have a mapping entry; direct costs do not) |

```csv
COA,Cost Center,Amounts
6100,1001,5000000
7100,1002,8000000
```

### 3. Transfer COA Mapping — `mapping.csv`

Maps each base COA to its transfer COA. COAs not listed here are treated as direct costs.

| Column  | Type   | Description       |
|---------|--------|-------------------|
| 전기COA | string | Transfer COA code |
| 기존COA | string | Base COA code     |

```csv
전기COA,기존COA
E6100,6100
E6200,6200
```

### 4. Allocation Cycle — `cycle.csv`

Defines the allocation rules. Multiple cycles are applied in ascending `차수`(cycle) order.
The `%` column is a decimal (0.3 = 30%). All receiver percentages for a sender
within the same cycle should sum to 1.0.

| Column      | Type   | Description                 |
|-------------|--------|-----------------------------|
| 차수         | int    | Cycle number (1, 2, …)      |
| Sender CC   | string | Cost center sending costs   |
| Receiver CC | string | Cost center receiving costs |
| %           | number | Allocation ratio (decimal)  |

```csv
차수,Sender CC,Receiver CC,%
1,1001,1002,0.3
1,1001,1003,0.7
2,2001,1001,0.5
2,2001,3001,0.5
```

---

## Usage

1. Download the latest `.exe` from [GitHub Releases](../../releases).
2. Run the executable — no installation required.
3. In the file selection window:
   - Select each of the four input CSV files.
   - Select the output directory where the result will be saved.
   - Click **실행** (the button activates once all five paths are filled).
4. The result is saved as `result.csv` in the selected directory.

---

## Output

File: `result.csv`, encoding: UTF-8 with BOM (Excel-compatible).

| Column      | Description                                       |
|-------------|---------------------------------------------------|
| 전기COA     | Transfer COA (empty for direct costs)             |
| 기존COA     | Base COA                                          |
| 코스트센터  | Cost center                                       |
| 1차배분금액 | Amount allocated in cycle 1                       |
| …           | One column per cycle                              |
| 배부합계    | Total allocated amount (sum of all cycle columns) |

Every (base COA × cost center) combination from the input is present in the output;
combinations with no allocation are filled with 0.

---

## Notes

- **Unknown CC warning**: If a Sender or Receiver CC in the cycle sheet is not found
  in the CC master, a warning is shown and you can choose to continue or abort.
- **Conservation check**: After allocation, the tool verifies that sender balances
  reach 0 and that total amounts are preserved. A warning is emitted if either check
  fails — review the input data if this occurs.
- **No rounding**: All amounts are output as raw float values.
- **Sender 미등록 CC 경고**: 공통비를 보유한 CC가 cycle.csv에 Sender로
  등록되지 않은 경우 경고가 표시됩니다. 해당 금액은 배부합계에 포함되지
  않으므로 cycle.csv를 확인하세요.
- **CSV 컬럼 누락 오류**: 필수 컬럼이 없는 CSV를 선택하면 누락된 컬럼명과
  파일명을 포함한 오류 메시지가 표시되고 실행이 중단됩니다.
- **비숫자 코드 경고**: COA 또는 CC 컬럼에 숫자로 변환할 수 없는 값이 있으면
  경고가 표시됩니다. 해당 행은 매핑에서 제외됩니다.

---

## Developer Guide

### Project structure

```
main.py                  # Entry point
src/
  loader.py              # CSV readers and dtype setup (Steps 1–2)
  prepare.py             # Cost separation, aggregation, ratio calc (Steps 3–6)
  allocation.py          # Pivot build, allocation loop, decomposition (Steps 7–9)
  output.py              # Result assembly and CSV export (Steps 10–12)
  ui.py                  # tkinter file-selection window
sample_data/             # Example input CSVs for local testing
```

### Run locally

```bash
pip install -r requirements.txt
python main.py
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

CI runs automatically on every push and pull request to `main`
via GitHub Actions (`.github/workflows/ci.yml`).

### Dependencies

- Python 3.12+
- pandas
- numpy
- pytest (development/testing)

See `requirements.txt` for pinned versions, and `requirements-dev.txt`
for development/testing dependencies.

### Distribution

The `.exe` is built with PyInstaller and published to GitHub Releases.
Only source code is stored in the repository — no compiled binaries are committed.
