from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional


def _resolve_yamnet_download_policy(*, experimental_lane_layout: bool) -> bool:
    """
    Return whether the GUI should allow TensorFlow Hub download for YAMNet.

    If the model is already local/cached, no network permission is needed. Otherwise,
    selected YAMNet lane layout should try TensorFlow Hub before failing.
    """
    if not bool(experimental_lane_layout):
        return False

    from aaf_speech_filter.speech_yamnet import (
        YamnetConfig,
        yamnet_model_available_locally,
    )

    if yamnet_model_available_locally(YamnetConfig()):
        return False
    return True


def _path_entry_state_for_busy(is_busy: bool) -> str:
    return "readonly" if bool(is_busy) else "normal"


def _default_experimental_lane_layout_enabled() -> bool:
    return True


def _reset_progress_display(prog_bar, pct_lbl, stage_lbl, timer_lbl, *, timer_text: str) -> None:
    prog_bar.stop()
    prog_bar.configure(mode="determinate")
    prog_bar["value"] = 0
    pct_lbl.config(text="0%")
    stage_lbl.config(text="")
    timer_lbl.config(text=timer_text)


def run_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    from aaf_pipeline import MSG_PIPELINE_STOPPED, _ru_removed_clips_phrase, run_aaf_pipeline

    root = tk.Tk()
    root.title("aaf-tool")
    root.minsize(560, 520)

    ui_progress_poll_ms = 1000
    indeterminate_animation_ms = 100

    path_var = tk.StringVar()
    experimental_lane_var = tk.BooleanVar(value=_default_experimental_lane_layout_enabled())
    remove_quiet_var = tk.BooleanVar(value=True)
    remove_dupes_var = tk.BooleanVar(value=True)
    quiet_db_var = tk.StringVar(value="-40")
    busy = {"v": False}
    cancel_event = threading.Event()
    active_worker: list[threading.Thread | None] = [None]
    closing = False

    frm = ttk.Frame(root, padding=10)
    frm.grid(row=0, column=0, sticky="nsew")
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    frm.columnconfigure(1, weight=1)

    ttk.Label(frm, text="Файл AAF:").grid(row=0, column=0, sticky="w", pady=(0, 4))
    ent = ttk.Entry(frm, textvariable=path_var, width=56)
    ent.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))

    def browse() -> None:
        p = filedialog.askopenfilename(
            title="Выберите AAF",
            filetypes=[("AAF", "*.aaf"), ("Все файлы", "*.*")],
        )
        if p:
            path_var.set(p)
            _reset_progress_display(prog_bar, pct_lbl, stage_lbl, timer_lbl, timer_text="")

    browse_btn = ttk.Button(frm, text="Обзор...", command=browse)
    browse_btn.grid(row=1, column=2, padx=(8, 0), pady=(0, 6))

    quiet_fr = ttk.LabelFrame(frm, text="Тихие клипы", padding=8)
    quiet_fr.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    remove_quiet_cb = ttk.Checkbutton(
        quiet_fr,
        text="Удалять тихие клипы",
        variable=remove_quiet_var,
    )
    remove_quiet_cb.grid(row=0, column=0, sticky="w", padx=(0, 12))
    ttk.Label(quiet_fr, text="Порог пика, dBFS:").grid(row=0, column=1, sticky="w", padx=(0, 4))
    quiet_db_ent = ttk.Entry(quiet_fr, textvariable=quiet_db_var, width=8)
    quiet_db_ent.grid(row=0, column=2, sticky="w")

    exp_fr = ttk.LabelFrame(frm, text="Упорядочение AAF", padding=8)
    exp_fr.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    remove_dupes_cb = ttk.Checkbutton(
        exp_fr,
        text="Удалять дубли",
        variable=remove_dupes_var,
    )
    remove_dupes_cb.grid(row=0, column=0, columnspan=3, sticky="w")

    experimental_lane_cb = ttk.Checkbutton(
        exp_fr,
        text="Раскладка по дорожкам (речь выше, шум в центре, музыка внизу)",
        variable=experimental_lane_var,
    )
    experimental_lane_cb.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

    prog_fr = ttk.Frame(frm)
    prog_fr.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(0, 6))
    prog_fr.columnconfigure(0, weight=1)
    prog_bar = ttk.Progressbar(prog_fr, maximum=100, mode="determinate")
    prog_bar.grid(row=0, column=0, sticky="ew", padx=(0, 10))
    pct_lbl = ttk.Label(prog_fr, text="0%", width=5)
    pct_lbl.grid(row=0, column=1, sticky="e")
    stage_lbl = ttk.Label(prog_fr, text="", anchor="w")
    stage_lbl.grid(row=1, column=0, sticky="w", pady=(6, 0))
    timer_lbl = ttk.Label(prog_fr, text="", width=12, anchor="e")
    timer_lbl.grid(row=1, column=1, sticky="e", pady=(6, 0))

    log_w = scrolledtext.ScrolledText(frm, height=10, wrap=tk.WORD, state=tk.NORMAL)
    log_w.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
    frm.rowconfigure(6, weight=1)

    menu = tk.Menu(root, tearoff=0)
    menu.add_command(label="Копировать", command=lambda: log_w.event_generate("<<Copy>>"))
    menu.add_command(label="Выделить всё", command=lambda: log_w.event_generate("<<SelectAll>>"))

    path_menu = tk.Menu(root, tearoff=0)
    path_menu.add_command(label="Копировать", command=lambda: ent.event_generate("<<Copy>>"))
    path_menu.add_command(label="Выделить всё", command=lambda: ent.event_generate("<<SelectAll>>"))

    def _popup_path_menu(ev) -> None:
        try:
            ent.focus_set()
            path_menu.tk_popup(ev.x_root, ev.y_root)
        finally:
            try:
                path_menu.grab_release()
            except Exception:
                pass

    ent.bind("<Button-3>", _popup_path_menu)
    ent.bind("<Control-a>", lambda _e: (ent.event_generate("<<SelectAll>>"), "break")[1])

    def _popup_menu(ev) -> None:
        try:
            log_w.focus_set()
            menu.tk_popup(ev.x_root, ev.y_root)
        finally:
            try:
                menu.grab_release()
            except Exception:
                pass

    log_w.bind("<Button-3>", _popup_menu)
    log_w.bind("<Control-a>", lambda _e: (log_w.event_generate("<<SelectAll>>"), "break")[1])

    def log_line(msg: str) -> None:
        log_w.insert(tk.END, msg + "\n")
        log_w.see(tk.END)

    def _set_widget_enabled(w, enabled: bool) -> None:
        try:
            if isinstance(w, tk.Text):
                return
        except Exception:
            pass
        try:
            if enabled:
                w.state(["!disabled"])
            else:
                w.state(["disabled"])
            return
        except Exception:
            pass
        try:
            w.configure(state=(tk.NORMAL if enabled else tk.DISABLED))
        except Exception:
            pass

    def set_busy(on: bool) -> None:
        busy["v"] = on
        enabled = not bool(on)
        ent.configure(state=_path_entry_state_for_busy(on))
        for w in (
            browse_btn,
            remove_quiet_cb,
            quiet_db_ent,
            experimental_lane_cb,
            remove_dupes_cb,
            btn,
        ):
            _set_widget_enabled(w, enabled)
        stop_btn.config(state=tk.NORMAL if on else tk.DISABLED)

    def on_stop() -> None:
        if busy["v"]:
            cancel_event.set()
            log_line("Запрошена остановка...")

    def on_close() -> None:
        nonlocal closing
        closing = True
        if busy["v"]:
            cancel_event.set()
            log_line("Остановка и освобождение ресурсов...")
            stop_btn.config(state=tk.DISABLED)
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def _format_mm_ss(sec: float) -> str:
        if sec < 0:
            return ""
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m}:{s:02d}"

    def _make_pipeline_progress():
        lock = threading.Lock()
        state = {
            "want_f": 0.0,
            "want_stage": "",
            "want_indet": False,
            "log_q": deque(),
        }

        class GuiPipelineProgress:
            def set_global_fraction(self, f: float) -> None:
                g = max(0.0, min(1.0, float(f)))
                with lock:
                    state["want_f"] = g

            def set_stage(self, label: str) -> None:
                t = (label or "").strip()
                with lock:
                    state["want_stage"] = t

            def set_indeterminate(self, active: bool) -> None:
                with lock:
                    state["want_indet"] = bool(active)

            def _enqueue_log_line(self, msg: str) -> None:
                with lock:
                    state["log_q"].append(("log", str(msg)))

            def complete(self, result) -> None:
                with lock:
                    state["log_q"].append(("complete", result))

        return GuiPipelineProgress(), state, lock

    def process() -> None:
        if busy["v"]:
            return
        raw = path_var.get().strip()
        if not raw:
            messagebox.showwarning("Внимание", "Укажите файл AAF.")
            return
        from aaf_speech_filter.thresholds import validate_quiet_peak_dbfs

        try:
            qdb = validate_quiet_peak_dbfs(quiet_db_var.get().replace(",", "."))
        except (TypeError, ValueError) as ex:
            messagebox.showerror("Ошибка", str(ex))
            return
        settings = {
            "experimental_yamnet_lane_layout": bool(experimental_lane_var.get()),
            "remove_quiet_clips": bool(remove_quiet_var.get()),
            "remove_duplicates": bool(remove_dupes_var.get()),
        }
        p = Path(raw)
        if not p.is_file():
            messagebox.showerror("Ошибка", "Файл не найден.")
            return
        if (
            not settings["remove_quiet_clips"]
            and not settings["experimental_yamnet_lane_layout"]
            and not settings["remove_duplicates"]
        ):
            messagebox.showwarning(
                "Внимание",
                "Включите хотя бы один режим: тихие клипы, раскладка дорожек или удаление дублей.",
            )
            return
        try:
            allow_yamnet_download = _resolve_yamnet_download_policy(
                experimental_lane_layout=settings["experimental_yamnet_lane_layout"],
            )
        except Exception as ex:
            messagebox.showerror("YAMNet", str(ex))
            return
        try:
            log_w.delete("1.0", tk.END)
        except Exception:
            pass
        log_line("Обработка...")
        _reset_progress_display(prog_bar, pct_lbl, stage_lbl, timer_lbl, timer_text="0:00")
        cancel_event.clear()
        set_busy(True)

        pipeline_ui, ui_state, ui_lock = _make_pipeline_progress()
        tick_after = [None]
        ui_tick_after = [None]
        t_start = [0.0]
        applied = {"pct": -1, "stage": None, "indet": None}

        def ui_tick() -> None:
            if not busy["v"]:
                ui_tick_after[0] = None
                return
            with ui_lock:
                want_f = float(ui_state.get("want_f") or 0.0)
                want_stage = str(ui_state.get("want_stage") or "")
                want_indet = bool(ui_state.get("want_indet"))
                q = ui_state["log_q"]
                lines = []
                terminal = None
                while q:
                    kind, payload = q[0]
                    if kind == "complete":
                        if active_worker[0] is not None and active_worker[0].is_alive():
                            break
                        q.popleft()
                        terminal = payload
                        break
                    q.popleft()
                    lines.append(payload)

            if want_stage != (applied["stage"] or ""):
                stage_lbl.configure(text=want_stage)
                applied["stage"] = want_stage

            if want_indet != bool(applied["indet"]):
                applied["indet"] = want_indet
                if want_indet:
                    try:
                        prog_bar.stop()
                    except Exception:
                        pass
                    prog_bar.configure(mode="indeterminate")
                    prog_bar.start(indeterminate_animation_ms)
                    pct_lbl.configure(text="...")
                else:
                    try:
                        prog_bar.stop()
                    except Exception:
                        pass
                    prog_bar.configure(mode="determinate")

            if not want_indet:
                pct = min(100, int(100.0 * max(0.0, min(1.0, want_f))))
                if pct != int(applied["pct"]):
                    applied["pct"] = pct
                    prog_bar.configure(value=pct)
                    pct_lbl.configure(text=f"{pct}%")

            for ln in lines:
                log_line(ln)

            if terminal is not None:
                _gui_done(*terminal)
                return
            ui_tick_after[0] = root.after(ui_progress_poll_ms, ui_tick)

        ui_tick_after[0] = root.after(ui_progress_poll_ms, ui_tick)

        def worker() -> None:
            try:
                def log_ui(msg: str) -> None:
                    pipeline_ui._enqueue_log_line(msg)

                _ue, sp, err, removed_n, hint = run_aaf_pipeline(
                    p,
                    yamnet_score_threshold=0.0035,
                    yamnet_frame_aggregate="max",
                    experimental_yamnet_lane_layout=settings["experimental_yamnet_lane_layout"],
                    allow_yamnet_download=bool(allow_yamnet_download),
                    remove_quiet_clips=settings["remove_quiet_clips"],
                    quiet_peak_dbfs=qdb,
                    remove_duplicates=settings["remove_duplicates"],
                    pipeline_progress=pipeline_ui,
                    cancel_event=cancel_event,
                    aaf_tools_dir=None,
                    log_callback=log_ui,
                )
            except Exception as ex:
                sp, err, removed_n, hint = None, str(ex), None, None
            elapsed = time.monotonic() - t_start[0]
            pipeline_ui.complete((sp, err, removed_n, hint, elapsed))

        def _gui_done(sp, err, removed_n=None, hint=None, elapsed: Optional[float] = None) -> None:
            if tick_after[0] is not None:
                try:
                    root.after_cancel(tick_after[0])
                except Exception:
                    pass
                tick_after[0] = None
            if ui_tick_after[0] is not None:
                try:
                    root.after_cancel(ui_tick_after[0])
                except Exception:
                    pass
                ui_tick_after[0] = None
            active_worker[0] = None
            set_busy(False)
            if closing:
                root.destroy()
                return
            try:
                prog_bar.stop()
            except Exception:
                pass
            prog_bar.configure(mode="determinate")
            if elapsed is not None:
                timer_lbl.configure(text=_format_mm_ss(elapsed))
            if err:
                prog_bar.configure(value=0)
                pct_lbl.configure(text="0%")
                stage_lbl.configure(text="")
                log_line(err)
                if err == MSG_PIPELINE_STOPPED:
                    messagebox.showinfo("Остановка", err)
                else:
                    messagebox.showerror("Ошибка", err)
                return
            prog_bar.configure(value=100)
            pct_lbl.configure(text="100%")
            log_line("Готово:")
            if sp:
                log_line(str(sp))

            moved_n = hint.get("moved") if isinstance(hint, dict) else None
            parts = []
            if removed_n is not None:
                parts.append(_ru_removed_clips_phrase(int(removed_n)))
            if moved_n is not None:
                try:
                    parts.append(f"Перемещено {int(moved_n)} клипов")
                except Exception:
                    pass
            summary_line = "; ".join(parts) if parts else None
            if summary_line:
                log_line(summary_line)

            time_line = None
            if elapsed is not None:
                time_line = f"Время выполнения: {_format_mm_ss(elapsed)} ({elapsed:.1f} с)"
                log_line(time_line)

            dialog_lines = []
            if summary_line:
                dialog_lines.append(summary_line)
            if time_line:
                dialog_lines.append(time_line)
            messagebox.showinfo("Готово", "\n".join(dialog_lines) if dialog_lines else "Готово!")

        t_start[0] = time.monotonic()

        def tick_timer() -> None:
            if not busy["v"]:
                tick_after[0] = None
                return
            dt = time.monotonic() - t_start[0]
            timer_lbl.configure(text=_format_mm_ss(dt))
            tick_after[0] = root.after(500, tick_timer)

        tick_after[0] = root.after(400, tick_timer)
        active_worker[0] = threading.Thread(target=worker, name="aaf-pipeline", daemon=False)
        active_worker[0].start()

    btn = ttk.Button(frm, text="Обработать", command=process)
    btn.grid(row=5, column=0, sticky="w", pady=(0, 8))
    stop_btn = ttk.Button(frm, text="Стоп", command=on_stop, state=tk.DISABLED)
    stop_btn.grid(row=5, column=1, sticky="w", padx=(8, 0), pady=(0, 8))

    root.mainloop()
