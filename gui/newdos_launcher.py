#!/usr/bin/env python3
"""
NewDOS — Graphical Desktop Environment
Boots the NewDOS kernel in QEMU and provides a full desktop shell:
taskbar, console, file manager, and settings.
"""

import subprocess
import threading
import shutil
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
KERNELS  = {
    "v0.1.3": BASE_DIR / "NewDOSv0_1_3.bin",
    "v0.1.2": BASE_DIR / "NewDOSv0_1_2.bin",
}

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG      = "#0a0e14"
PANEL   = "#0d1117"
CARD    = "#161b22"
SURFACE = "#21262d"
BORDER  = "#30363d"
BLUE    = "#58a6ff"
GREEN   = "#3fb950"
RED     = "#f85149"
MUTED   = "#8b949e"
TEXT    = "#e6edf3"
WHITE   = "#ffffff"

MONO  = ("Courier New", 10)
SMALL = ("Courier New", 9)
BOLD  = ("Courier New", 11, "bold")


# ---------------------------------------------------------------------------
# Boot animation (splash screen)
# ---------------------------------------------------------------------------

class BootSplash(tk.Toplevel):
    LOGO = [
        "  ███╗   ██╗███████╗██╗    ██╗██████╗  ██████╗ ███████╗",
        "  ████╗  ██║██╔════╝██║    ██║██╔══██╗██╔═══██╗██╔════╝",
        "  ██╔██╗ ██║█████╗  ██║ █╗ ██║██║  ██║██║   ██║███████╗",
        "  ██║╚██╗██║██╔══╝  ██║███╗██║██║  ██║██║   ██║╚════██║",
        "  ██║ ╚████║███████╗╚███╔███╔╝██████╔╝╚██████╔╝███████║",
        "  ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═════╝  ╚═════╝ ╚══════╝",
    ]
    STEPS = [
        (0.15, "Loading bootloader…"),
        (0.35, "Mapping memory…"),
        (0.55, "Starting interrupt handlers…"),
        (0.70, "Mounting in-memory filesystem…"),
        (0.85, "Starting CLI shell…"),
        (1.00, "Launching QEMU…"),
    ]

    def __init__(self, master, version):
        super().__init__(master)
        self.overrideredirect(True)
        w, h = 640, 340
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.configure(bg=BG)
        self.lift(); self.focus_force()

        tk.Label(self, text="\n".join(self.LOGO), bg=BG, fg=BLUE,
                 font=("Courier New", 9, "bold"), justify="left").pack(pady=(30, 0))
        tk.Label(self, text=f"NewDOS Base Kernel {version}",
                 bg=BG, fg=MUTED, font=MONO).pack(pady=(10, 0))

        self._canvas = tk.Canvas(self, width=400, height=6, bg=SURFACE,
                                 highlightthickness=0)
        self._canvas.pack(pady=24)
        self._fill = self._canvas.create_rectangle(0, 0, 0, 6,
                                                   fill=BLUE, outline="")
        self._msg = tk.Label(self, text="", bg=BG, fg=MUTED, font=SMALL)
        self._msg.pack()

        self._i = 0
        self._animate()

    def _animate(self):
        if self._i < len(self.STEPS):
            pct, msg = self.STEPS[self._i]
            self._i += 1
            self._canvas.coords(self._fill, 0, 0, 400 * pct, 6)
            self._msg.configure(text=msg)
            self.after(250, self._animate)
        else:
            self.after(180, self.destroy)


# ---------------------------------------------------------------------------
# Console tab
# ---------------------------------------------------------------------------

class ConsoleTab(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=CARD)

        hdr = tk.Frame(self, bg=CARD)
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(hdr, text="● CONSOLE", bg=CARD, fg=GREEN, font=SMALL).pack(side="left")
        tk.Button(hdr, text="Clear", bg=CARD, fg=MUTED, font=SMALL,
                  relief="flat", cursor="hand2",
                  activebackground=SURFACE,
                  command=self._clear).pack(side="right")

        self.out = tk.Text(self, bg="#050810", fg=GREEN, font=MONO,
                           relief="flat", padx=10, pady=8,
                           insertbackground=GREEN, state="disabled",
                           selectbackground=SURFACE, wrap="word")
        sb = ttk.Scrollbar(self, orient="vertical", command=self.out.yview)
        self.out.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.out.pack(fill="both", expand=True)

    def write(self, text):
        self.out.configure(state="normal")
        self.out.insert("end", text)
        self.out.see("end")
        self.out.configure(state="disabled")

    def _clear(self):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.configure(state="disabled")


# ---------------------------------------------------------------------------
# Files tab
# ---------------------------------------------------------------------------

class FilesTab(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg=CARD)

        hdr = tk.Frame(self, bg=CARD)
        hdr.pack(fill="x", padx=12, pady=(10, 6))
        tk.Label(hdr, text="◈ FILES", bg=CARD, fg=BLUE, font=SMALL).pack(side="left")
        tk.Label(hdr, text=str(BASE_DIR), bg=CARD, fg=MUTED,
                 font=SMALL).pack(side="left", padx=(10, 0))

        # Header row
        row = tk.Frame(self, bg=SURFACE)
        row.pack(fill="x", padx=12)
        for col, w in [("Name", 34), ("Size", 10), ("Type", 12)]:
            tk.Label(row, text=col, bg=SURFACE, fg=MUTED,
                     font=("Courier New", 9, "bold"),
                     width=w, anchor="w", padx=6, pady=4).pack(side="left")

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=12)

        scroll = tk.Frame(self, bg=CARD)
        scroll.pack(fill="both", expand=True, padx=12, pady=4)

        for path in sorted(BASE_DIR.iterdir()):
            if path.name.startswith("."):
                continue
            is_dir = path.is_dir()
            size   = "-" if is_dir else self._fmt(path.stat().st_size)
            ftype  = "Directory" if is_dir else (path.suffix.lstrip(".").upper() or "File")
            icon   = "📁" if is_dir else "📄"
            fg     = BLUE if path.suffix == ".bin" else TEXT

            r = tk.Frame(scroll, bg=CARD)
            r.pack(fill="x")
            r.bind("<Enter>", lambda e, f=r: f.configure(bg=SURFACE))
            r.bind("<Leave>", lambda e, f=r: f.configure(bg=CARD))

            tk.Label(r, text=f"{icon}  {path.name}", bg=CARD, fg=fg,
                     font=MONO, anchor="w", padx=6, pady=3,
                     width=34).pack(side="left")
            tk.Label(r, text=size,  bg=CARD, fg=MUTED,
                     font=SMALL, anchor="w", width=10).pack(side="left")
            tk.Label(r, text=ftype, bg=CARD, fg=MUTED,
                     font=SMALL, anchor="w", width=12).pack(side="left")

    @staticmethod
    def _fmt(n):
        for u in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.0f} {u}"
            n /= 1024
        return f"{n:.1f} TB"


# ---------------------------------------------------------------------------
# Settings tab
# ---------------------------------------------------------------------------

class SettingsTab(tk.Frame):
    def __init__(self, master, desktop):
        super().__init__(master, bg=CARD)
        self.desktop = desktop
        self._build()

    def _build(self):
        w = tk.Frame(self, bg=CARD)
        w.pack(fill="both", expand=True, padx=24, pady=16)

        def section(title):
            tk.Label(w, text=title, bg=CARD, fg=BLUE,
                     font=("Courier New", 10, "bold"), anchor="w").pack(
                         fill="x", pady=(14, 2))
            tk.Frame(w, bg=BORDER, height=1).pack(fill="x")

        section("KERNEL VERSION")
        vf = tk.Frame(w, bg=CARD)
        vf.pack(fill="x", pady=6)
        for label, val in [("v0.1.3 (latest)", "v0.1.3"), ("v0.1.2", "v0.1.2")]:
            ok = KERNELS[val].exists()
            tk.Radiobutton(vf, text=label,
                           variable=self.desktop.version_var, value=val,
                           bg=CARD, fg=TEXT if ok else MUTED, font=MONO,
                           selectcolor=BG, activebackground=CARD,
                           state="normal" if ok else "disabled").pack(
                               side="left", padx=(0, 24))

        section("MEMORY")
        mf = tk.Frame(w, bg=CARD)
        mf.pack(fill="x", pady=6)
        tk.Label(mf, text="RAM (MB):", bg=CARD, fg=TEXT, font=MONO).pack(side="left")
        tk.Spinbox(mf, from_=64, to=4096, increment=64,
                   textvariable=self.desktop.memory_var, width=7,
                   bg=SURFACE, fg=TEXT, font=MONO,
                   buttonbackground=SURFACE, relief="flat").pack(
                       side="left", padx=(10, 0))

        section("DISPLAY BACKEND")
        df = tk.Frame(w, bg=CARD)
        df.pack(fill="x", pady=6)
        for label, val in [("SDL", "sdl"), ("GTK", "gtk"),
                           ("VNC :0", "vnc"), ("None", "none")]:
            tk.Radiobutton(df, text=label,
                           variable=self.desktop.display_var, value=val,
                           bg=CARD, fg=TEXT, font=MONO,
                           selectcolor=BG, activebackground=CARD).pack(
                               side="left", padx=(0, 20))

        section("EXTRA QEMU FLAGS")
        ef = tk.Frame(w, bg=CARD)
        ef.pack(fill="x", pady=6)
        tk.Entry(ef, textvariable=self.desktop.flags_var, width=54,
                 bg=SURFACE, fg=TEXT, font=MONO,
                 insertbackground=TEXT, relief="flat").pack(side="left")

        section("ABOUT")
        for k, v in [
            ("Architecture", "x86_64 — bare metal, ring 0"),
            ("Language",     "Rust 2021 (nightly toolchain)"),
            ("Bootloader",   "bootloader v0.9 — BIOS only"),
            ("GUI",          "Python 3 + tkinter"),
        ]:
            r = tk.Frame(w, bg=CARD)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=k, bg=CARD, fg=MUTED, font=SMALL,
                     width=18, anchor="w").pack(side="left")
            tk.Label(r, text=v, bg=CARD, fg=TEXT, font=SMALL,
                     anchor="w").pack(side="left")


# ---------------------------------------------------------------------------
# Desktop
# ---------------------------------------------------------------------------

class NewDOSDesktop(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("NewDOS")
        self.geometry("1100x720")
        self.minsize(800, 560)
        self.configure(bg=BG)

        self.version_var = tk.StringVar(value="v0.1.3")
        self.memory_var  = tk.IntVar(value=128)
        self.display_var = tk.StringVar(value="sdl")
        self.flags_var   = tk.StringVar(value="")
        self.process     = None

        self._style()
        self._taskbar()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._workspace()
        self._statusbar()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("OS.TNotebook",     background=PANEL, tabmargins=[0, 0, 0, 0])
        s.configure("OS.TNotebook.Tab", background=PANEL, foreground=MUTED,
                    font=MONO, padding=[16, 8])
        s.map("OS.TNotebook.Tab",
              background=[("selected", CARD)],
              foreground=[("selected", BLUE)])

    # ---- taskbar ------------------------------------------------------------

    def _taskbar(self):
        bar = tk.Frame(self, bg=PANEL, height=52)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        tk.Label(bar, text="  NewDOS", bg=PANEL, fg=BLUE,
                 font=("Courier New", 15, "bold")).pack(side="left", padx=(6, 16))
        tk.Frame(bar, bg=BORDER, width=1).pack(side="left", fill="y", pady=10)

        self.boot_btn = tk.Button(
            bar, text="  ▶  Boot OS  ",
            bg=GREEN, fg="#000", font=BOLD,
            relief="flat", cursor="hand2",
            activebackground="#2ea043", activeforeground="#000",
            command=self._boot)
        self.boot_btn.pack(side="left", padx=(14, 6), pady=10)

        self.stop_btn = tk.Button(
            bar, text="  ■  Stop  ",
            bg=SURFACE, fg=RED, font=BOLD,
            relief="flat", cursor="hand2",
            activebackground=BORDER, activeforeground=RED,
            state="disabled", command=self._stop)
        self.stop_btn.pack(side="left", pady=10)

        self.clock = tk.Label(bar, text="", bg=PANEL, fg=MUTED, font=MONO)
        self.clock.pack(side="right", padx=14)
        self._tick()

        self.pill = tk.Label(bar, text=" ● Idle ", bg=SURFACE, fg=MUTED,
                             font=SMALL, padx=6, pady=2)
        self.pill.pack(side="right", padx=(0, 10))

    def _tick(self):
        self.clock.configure(text=time.strftime("  %H:%M:%S   %Y-%m-%d  "))
        self.after(1000, self._tick)

    # ---- workspace ----------------------------------------------------------

    def _workspace(self):
        self.nb = ttk.Notebook(self, style="OS.TNotebook")
        self.nb.pack(fill="both", expand=True)

        self.console  = ConsoleTab(self.nb)
        self.nb.add(self.console,  text="  Console  ")

        self.files    = FilesTab(self.nb)
        self.nb.add(self.files,    text="  Files  ")

        self.settings = SettingsTab(self.nb, self)
        self.nb.add(self.settings, text="  Settings  ")

        self.console.write(
            "NewDOS Desktop  —  x86_64 bare-metal OS\n"
            "────────────────────────────────────────\n"
            "Press  ▶ Boot OS  to start the kernel in QEMU.\n\n"
            "Inside the kernel:\n"
            "  pierre help          list all commands\n"
            "  pierre tui           text desktop UI  (W/S navigate, Enter open)\n"
            "  pierre edit <file>   editor  (F9 save, F10 exit)\n"
            "  suppiere gfx         graphics demo\n\n"
        )

    # ---- status bar ---------------------------------------------------------

    def _statusbar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        bar = tk.Frame(self, bg=PANEL)
        bar.pack(fill="x")
        self.status = tk.Label(bar, text="Ready", bg=PANEL, fg=MUTED,
                               font=SMALL, anchor="w", padx=12, pady=3)
        self.status.pack(side="left")
        ok  = bool(shutil.which("qemu-system-x86_64"))
        txt = "qemu ✓" if ok else "qemu not found — sudo apt install qemu-system-x86"
        tk.Label(bar, text=txt, bg=PANEL, fg=GREEN if ok else RED,
                 font=SMALL, padx=12, pady=3).pack(side="right")

    # ---- boot logic ---------------------------------------------------------

    def _boot(self):
        ver = self.version_var.get()
        bp  = KERNELS.get(ver)
        if not bp or not bp.exists():
            messagebox.showerror("Kernel Not Found",
                                 f"Binary not found:\n{bp}\n\nBuild with: cargo bootimage")
            return
        if not shutil.which("qemu-system-x86_64"):
            messagebox.showerror("QEMU Not Found",
                                 "Install with:  sudo apt install qemu-system-x86")
            return

        splash = BootSplash(self, ver)
        self.wait_window(splash)

        cmd = ["qemu-system-x86_64",
               "-drive", f"format=raw,file={bp}",
               "-m", str(self.memory_var.get()),
               "-display", self.display_var.get()]
        extra = self.flags_var.get().strip()
        if extra:
            cmd.extend(extra.split())

        self._set_running(True)
        self.status.configure(text=f"Running {ver}…")
        self.nb.select(self.console)
        self.console.write(f"$ {' '.join(str(c) for c in cmd)}\n\n")

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except Exception as exc:
            self.console.write(f"Error: {exc}\n")
            self._set_running(False)
            self.status.configure(text="Launch failed")
            return

        threading.Thread(target=self._stream, daemon=True).start()
        self.after(500, self._poll)

    def _stream(self):
        try:
            for line in self.process.stdout:
                self.after(0, self.console.write, line)
        except Exception:
            pass

    def _poll(self):
        if self.process and self.process.poll() is not None:
            rc = self.process.returncode
            self.console.write(f"\n── QEMU exited (code {rc}) ──\n")
            self._set_running(False)
            self.status.configure(text="Ready")
            self.process = None
        elif self.process:
            self.after(500, self._poll)

    def _stop(self):
        if self.process:
            self.process.terminate()
            self.console.write("\n── Stopped ──\n")
            self._set_running(False)
            self.status.configure(text="Ready")
            self.process = None

    def _set_running(self, running):
        if running:
            self.boot_btn.configure(state="disabled", bg=SURFACE, fg=MUTED)
            self.stop_btn.configure(state="normal",   bg=RED,     fg=WHITE)
            self.pill.configure(text=" ● Running ", bg=GREEN, fg="#000")
        else:
            self.boot_btn.configure(state="normal",   bg=GREEN,   fg="#000")
            self.stop_btn.configure(state="disabled", bg=SURFACE, fg=MUTED)
            self.pill.configure(text=" ● Idle ", bg=SURFACE, fg=MUTED)

    def _on_close(self):
        if self.process:
            self.process.terminate()
        self.destroy()


# ---------------------------------------------------------------------------

def main():
    NewDOSDesktop().mainloop()


if __name__ == "__main__":
    main()
