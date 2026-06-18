from __future__ import annotations

import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path


_FIELDS = [
    ("coa_amount",      "COA·CC 마스터",     "file"),
    ("override_amount", "실제 배부대상 금액", "file"),
    ("mapping",         "전기 매핑",         "file"),
    ("cycle",           "Cycle",            "file"),
    ("pre_allocation",  "배부전 금액",       "file"),
    ("output_dir",      "결과 저장 폴더",    "dir"),
]

_CONFIG_PATH = Path.home() / ".cost-allocation" / "last_paths.json"


def _load_last_paths() -> dict[str, str]:
    """Load previously saved paths from the config file.

    Returns an empty dict if the file does not exist, is not valid JSON,
    or cannot be read. Paths that no longer exist on disk are excluded.
    """
    try:
        data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        return {k: v for k, v in data.items() if Path(v).exists()}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_last_paths(paths: dict[str, str]) -> None:
    """Persist selected paths to the config file.

    Silently ignores write errors (e.g. permission denied).
    """
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_PATH.write_text(
            json.dumps(paths, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _display_name(path: str) -> str:
    """Return only the file or folder name for on-screen display.

    The Entry fields show this short name instead of the full path so the
    absolute path is not exposed on screen. The full path is still kept
    internally for reading inputs and saving the result. Returns "" for an
    empty path.
    """
    return Path(path).name if path else ""


def prompt_file_paths() -> dict[str, Path] | None:
    """Open a tkinter window to collect four input file paths and an output directory.

    Previously selected paths are pre-filled if they still exist on disk.

    Returns
    -------
    dict with keys:
        'coa_amount'     : Path  -- COA·CC master amount CSV
        'override_amount': Path  -- override amount CSV (corrects master Amounts)
        'mapping'        : Path  -- transfer COA mapping CSV
        'cycle'          : Path  -- allocation cycle CSV
        'pre_allocation' : Path  -- pre-allocation amount CSV (by_cc 배부전금액)
        'output_dir'     : Path  -- directory where the result will be saved
    None
        If the user closes the window without completing the selection.
    """
    root = tk.Tk()
    root.title("공통비 배부")
    root.resizable(False, False)

    last = _load_last_paths()
    # Full absolute paths are kept here; the Entry fields show only the file
    # or folder name so the full path is never exposed on screen.
    selected: dict[str, str] = {key: last.get(key, "") for key, *_ in _FIELDS}
    path_vars: dict[str, tk.StringVar] = {
        key: tk.StringVar(value=_display_name(selected[key])) for key, *_ in _FIELDS
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
            selected[key] = path
            path_vars[key].set(_display_name(path))

    def _update_run(*_) -> None:
        all_set = all(selected.values())
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
        _save_last_paths(selected)
        result[0] = {key: Path(v) for key, v in selected.items()}
        root.destroy()

    run_btn = tk.Button(
        root, text="실행", state=tk.DISABLED, command=_on_run, width=12
    )
    run_btn.grid(row=len(_FIELDS), column=0, columnspan=3, pady=(8, 12))

    root.protocol("WM_DELETE_WINDOW", root.destroy)

    _update_run()

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
        # A custom window keeps the warning list inside a fixed-height,
        # scrollable area so the dialog never grows past the screen no matter
        # how many warnings are collected. The hidden root is reused as the
        # window; closing it or the "확인" button calls root.destroy(), which
        # ends mainloop. This branch returns early so it does not hit the
        # shared root.destroy() at the end of the function.
        items = warnings or []
        body = "\n".join(f"- {w}" for w in items)

        root.deiconify()
        root.title("경고와 함께 완료")

        tk.Label(
            root,
            text="result.csv는 생성되었으나 다음 경고가 발생했습니다:",
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=12, pady=(12, 4))

        text = ScrolledText(
            root,
            width=70,
            height=min(max(len(items), 3), 15),
            wrap="none",
        )
        text.pack(fill="both", expand=True, padx=12)
        text.insert("1.0", body)
        text.config(state="disabled")

        tk.Label(
            root,
            text=f"저장 경로:\n{out_path}",
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=12, pady=(8, 4))

        tk.Button(root, text="확인", command=root.destroy, width=12).pack(
            pady=(0, 12)
        )

        root.protocol("WM_DELETE_WINDOW", root.destroy)

        # Size the window to its content, but cap the height at 80% of the
        # screen, then center it.
        root.update_idletasks()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        win_w = root.winfo_reqwidth()
        win_h = min(root.winfo_reqheight(), int(screen_h * 0.8))
        x = (screen_w - win_w) // 2
        y = (screen_h - win_h) // 2
        root.geometry(f"{win_w}x{win_h}+{x}+{y}")

        root.mainloop()
        return
    else:  # "failure"
        text = f"처리에 실패했습니다:\n\n{error}"
        body = "\n".join(f"- {w}" for w in (warnings or []))
        if body:
            text += f"\n\n{body}"
        messagebox.showerror("실패", text)

    root.destroy()
