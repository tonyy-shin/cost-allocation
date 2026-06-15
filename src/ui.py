from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
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


def show_completion(
    status: str,
    out_path: Path | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Show a modal dialog summarizing the pipeline outcome.

    A fresh hidden Tk root is created so this works after prompt_file_paths
    has already destroyed its own root, and under PyInstaller --windowed
    where no console is available. All user-facing text is in Korean.

    Parameters
    ----------
    status : {"success", "warning", "failure"}
        Drives which message box variant is shown.
    out_path : Path, optional
        Saved result path. Shown for "success" and "warning".
    warnings : list[str], optional
        Warning messages collected during the run. Shown for "warning".
    error : str, optional
        Error description. Shown for "failure".
    """
    root = tk.Tk()
    root.withdraw() 

    if status == "success":
        messagebox.showinfo(
            "완료",
            f"result.csv 생성이 완료되었습니다.\n\n저장 경로:\n{out_path}",
        )
    elif status == "warning":
        body = "\n".join(f"- {w}" for w in (warnings or []))
        messagebox.showwarning(
            "경고와 함께 완료",
            "result.csv는 생성되었으나 다음 경고가 발생했습니다:\n\n"
            f"{body}\n\n저장 경로:\n{out_path}",
        )
    else:  # "failure"
        messagebox.showerror("실패", f"처리에 실패했습니다:\n\n{error}")

    root.destroy()
