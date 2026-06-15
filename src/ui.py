from __future__ import annotations

import tkinter as tk
from tkinter import filedialog
from pathlib import Path


_FIELDS = [
    ("cc",         "CC 마스터",     "file"),
    ("coa_amount", "COA 금액",      "file"),
    ("mapping",    "전기 매핑",      "file"),
    ("cycle",      "Cycle",         "file"),
    ("output_dir", "결과 저장 폴더", "dir"),
]


def prompt_file_paths() -> dict[str, Path] | None:
    """Open a tkinter window to collect four input file paths and an output directory.

    Returns
    -------
    dict with keys:
        'cc'         : Path  -- CC master CSV
        'coa_amount' : Path  -- COA amount CSV
        'mapping'    : Path  -- transfer COA mapping CSV
        'cycle'      : Path  -- allocation cycle CSV
        'output_dir' : Path  -- directory where the result will be saved
    None
        If the user closes the window without completing the selection.
    """
    root = tk.Tk()
    root.title("공통비 배부")
    root.resizable(False, False)

    path_vars: dict[str, tk.StringVar] = {
        key: tk.StringVar() for key, *_ in _FIELDS
    }
    result: list[dict[str, Path] | None] = [None]

    def _browse(key: str, kind: str) -> None:
        if kind == "file":
            path = filedialog.askopenfilename(
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
        else:
            path = filedialog.askdirectory()
        if path:
            path_vars[key].set(path)

    def _update_run(*_) -> None:
        all_set = all(v.get() for v in path_vars.values())
        run_btn.config(state=tk.NORMAL if all_set else tk.DISABLED)

    for var in path_vars.values():
        var.trace_add("write", _update_run)

    for row, (key, label, kind) in enumerate(_FIELDS):
        tk.Label(root, text=label, anchor="w", width=14).grid(
            row=row, column=0, padx=(12, 4), pady=5, sticky="w"
        )
        tk.Entry(root, textvariable=path_vars[key], width=52, state="readonly").grid(
            row=row, column=1, padx=4
        )
        tk.Button(
            root, text="찾아보기",
            command=lambda k=key, t=kind: _browse(k, t)
        ).grid(row=row, column=2, padx=(4, 12))

    def _on_run() -> None:
        result[0] = {key: Path(path_vars[key].get()) for key, *_ in _FIELDS}
        root.destroy()

    run_btn = tk.Button(
        root, text="실행", state=tk.DISABLED, command=_on_run, width=12
    )
    run_btn.grid(row=len(_FIELDS), column=0, columnspan=3, pady=(8, 12))

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

    return result[0]
