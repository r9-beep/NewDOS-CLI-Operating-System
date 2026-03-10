#!/usr/bin/env python3
"""NewDOS GUI Launcher — graphical launcher for the NewDOS Base Kernel."""

import subprocess
import threading
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ---------------------------------------------------------------------------
# Colour palette (dark terminal aesthetic)
# ---------------------------------------------------------------------------
BG_DARK    = "#0d1117"
BG_MID     = "#161b22"
BG_LIGHT   = "#21262d"
BORDER     = "#30363d"
TEXT_PRI   = "#e6edf3"
TEXT_SEC   = "#8b949e"
BLUE       = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#d29922"

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lbl(parent, text, fg=TEXT_PRI, bg=BG_LIGHT, font=("Courier New", 10), **kw):
    return tk.Label(parent, text=text, fg=fg, bg=bg, font=font, **kw)


def _card(parent, title):
    """Return a framed card with a blue header; also return its inner frame."""
    card = tk.Frame(parent, bg=BG_LIGHT)
    tk.Label(card, text=f"  {title}", bg=BG_LIGHT, fg=BLUE,
             font=("Courier New", 11, "bold"), anchor="w",
             pady=7, padx=6).pack(fill="x")
    tk.Frame(card, bg=BORDER, height=1).pack(fill="x")
    inner = tk.Frame(card, bg=BG_LIGHT)
    inner.pack(fill="both", expand=True, padx=12, pady=8)
    return card, inner


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class NewDOSLauncher(tk.Tk):

    # ---- init ---------------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.title("NewDOS Launcher")
        self.geometry("960x680")
        self.minsize(720, 520)
        self.configure(bg=BG_DARK)

        self.selected_version = tk.StringVar(value="v0.1.3")
        self.memory_mb        = tk.IntVar(value=128)
        self.display_mode     = tk.StringVar(value="sdl")
        self.extra_flags      = tk.StringVar(value="")
        self.process          = None

        self._style()
        self._header()
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self._notebook()
        self._statusbar()
        self._check_qemu()

    # ---- ttk style ----------------------------------------------------------

    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Dark.TNotebook",        background=BG_DARK,  tabmargins=[0,0,0,0])
        s.configure("Dark.TNotebook.Tab",    background=BG_MID,   foreground=TEXT_SEC,
                    font=("Courier New", 10), padding=[14, 6])
        s.map("Dark.TNotebook.Tab",
              background=[("selected", BG_LIGHT)],
              foreground=[("selected", BLUE)])
        s.configure("Mid.TFrame",   background=BG_MID)
        s.configure("Dark.TFrame",  background=BG_DARK)
        s.configure("Card.TFrame",  background=BG_LIGHT)
        s.configure("Dark.TRadiobutton", background=BG_LIGHT, foreground=TEXT_PRI,
                    font=("Courier New", 10))
        s.map("Dark.TRadiobutton", background=[("active", BG_LIGHT)])

    # ---- header bar ---------------------------------------------------------

    def _header(self):
        bar = tk.Frame(self, bg=BG_DARK)
        bar.pack(fill="x", padx=20, pady=(14, 10))

        tk.Label(bar, text="NewDOS", bg=BG_DARK, fg=BLUE,
                 font=("Courier New", 24, "bold")).pack(side="left")
        tk.Label(bar, text=" GUI Launcher ", bg=BLUE, fg="#000",
                 font=("Courier New", 10, "bold"), padx=4, pady=3).pack(
                     side="left", padx=(8, 0), pady=(10, 0))
        tk.Label(bar, text="  x86_64 bare-metal OS launcher",
                 bg=BG_DARK, fg=TEXT_SEC,
                 font=("Courier New", 10)).pack(side="left", pady=(10, 0))

    # ---- notebook -----------------------------------------------------------

    def _notebook(self):
        nb = ttk.Notebook(self, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=15, pady=(8, 0))

        for title, builder in [
            ("  Launch  ",  self._tab_launch),
            ("  Config  ",  self._tab_config),
            ("  Output  ",  self._tab_output),
            ("  About   ",  self._tab_about),
        ]:
            tab = ttk.Frame(nb, style="Mid.TFrame")
            nb.add(tab, text=title)
            builder(tab)

    # =========================================================================
    # Tab: Launch
    # =========================================================================

    def _tab_launch(self, parent):
        left  = tk.Frame(parent, bg=BG_MID)
        right = tk.Frame(parent, bg=BG_MID)
        left .pack(side="left",  fill="both", expand=True, padx=(15, 6), pady=15)
        right.pack(side="right", fill="y",    padx=(6, 15), pady=15)

        # -- version picker ---------------------------------------------------
        vc, vi = _card(left, "SELECT VERSION")
        vc.pack(fill="x", pady=(0, 10))

        versions = [
            ("NewDOS v0.1.3  (latest)", "v0.1.3", "NewDOSv0_1_3.bin"),
            ("NewDOS v0.1.2",           "v0.1.2", "NewDOSv0_1_2.bin"),
        ]
        for label, val, fname in versions:
            exists = (BASE_DIR / fname).exists()
            row = tk.Frame(vi, bg=BG_LIGHT)
            row.pack(fill="x", pady=3)
            rb = ttk.Radiobutton(row, text=label, variable=self.selected_version,
                                 value=val, style="Dark.TRadiobutton",
                                 state="normal" if exists else "disabled")
            rb.pack(side="left")
            tk.Label(row, text="✓ found" if exists else "✗ missing",
                     bg=BG_LIGHT, fg=GREEN if exists else RED,
                     font=("Courier New", 9)).pack(side="right")

        # -- feature list -----------------------------------------------------
        fc, fi = _card(left, "KERNEL FEATURES")
        fc.pack(fill="both", expand=True)

        features = [
            ("VGA Text Mode",  "80×25 character display"),
            ("CLI Interface",  "pierre / suppiere commands"),
            ("Memory Manager", "PMM + kernel heap"),
            ("In-Memory FS",   "mkdir / touch / write / cat"),
            ("Text UI",        "pierre tui — W/S/Enter/F1–F3"),
            ("Text Editor",    "pierre edit <file> — F9/F10"),
            ("PS/2 Input",     "keyboard + mouse interrupts"),
            ("PCI Scanning",   "AHCI controller detection"),
        ]
        for name, desc in features:
            row = tk.Frame(fi, bg=BG_LIGHT)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"▸ {name}", bg=BG_LIGHT, fg=GREEN,
                     font=("Courier New", 9, "bold"), width=18, anchor="w").pack(side="left")
            tk.Label(row, text=desc, bg=BG_LIGHT, fg=TEXT_SEC,
                     font=("Courier New", 9), anchor="w").pack(side="left")

        # -- controls panel ---------------------------------------------------
        ctrl = tk.Frame(right, bg=BG_LIGHT, width=230)
        ctrl.pack(fill="y", expand=True)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="  CONTROLS", bg=BG_LIGHT, fg=BLUE,
                 font=("Courier New", 11, "bold"), anchor="w",
                 pady=7, padx=6).pack(fill="x")
        tk.Frame(ctrl, bg=BORDER, height=1).pack(fill="x")

        inner = tk.Frame(ctrl, bg=BG_LIGHT)
        inner.pack(fill="both", expand=True, padx=14, pady=12)

        self.launch_btn = tk.Button(
            inner, text="▶  LAUNCH",
            bg=BLUE, fg="#fff", font=("Courier New", 14, "bold"),
            relief="flat", cursor="hand2", padx=20, pady=12,
            activebackground="#79b8ff", command=self._launch_qemu)
        self.launch_btn.pack(fill="x", pady=(0, 8))

        self.stop_btn = tk.Button(
            inner, text="■  STOP",
            bg=RED, fg="#fff", font=("Courier New", 11, "bold"),
            relief="flat", cursor="hand2", padx=20, pady=8,
            activebackground="#ff6b6b", state="disabled",
            command=self._stop_qemu)
        self.stop_btn.pack(fill="x", pady=(0, 18))

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(0, 10))

        # QEMU hotkeys
        tk.Label(inner, text="QEMU HOTKEYS", bg=BG_LIGHT, fg=TEXT_SEC,
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        for key, desc in [("Ctrl+Alt+G", "Release mouse"),
                           ("Ctrl+Alt+F", "Fullscreen"),
                           ("Ctrl+Alt+Q", "Quit QEMU")]:
            row = tk.Frame(inner, bg=BG_LIGHT)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=key, bg=BORDER, fg=YELLOW,
                     font=("Courier New", 8, "bold"), padx=4, pady=1).pack(side="left")
            tk.Label(row, text=f"  {desc}", bg=BG_LIGHT, fg=TEXT_SEC,
                     font=("Courier New", 8)).pack(side="left")

        tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", pady=(10, 8))

        # CLI quick-ref
        tk.Label(inner, text="CLI QUICK REF", bg=BG_LIGHT, fg=TEXT_SEC,
                 font=("Courier New", 8, "bold")).pack(anchor="w")
        for cmd in ["pierre help", "pierre dir", "pierre tui",
                    "pierre edit <file>", "suppiere gfx"]:
            tk.Label(inner, text=f"> {cmd}", bg=BG_LIGHT, fg=GREEN,
                     font=("Courier New", 8), anchor="w").pack(fill="x", pady=1)

    # =========================================================================
    # Tab: Config
    # =========================================================================

    def _tab_config(self, parent):
        wrap = tk.Frame(parent, bg=BG_MID)
        wrap.pack(fill="both", expand=True, padx=20, pady=15)

        # memory
        mc, mi = _card(wrap, "MEMORY")
        mc.pack(fill="x", pady=(0, 10))
        tk.Label(mi, text="RAM (MB):", bg=BG_LIGHT, fg=TEXT_PRI,
                 font=("Courier New", 10)).pack(side="left")
        tk.Spinbox(mi, from_=64, to=4096, increment=64,
                   textvariable=self.memory_mb, width=7,
                   bg=BG_DARK, fg=TEXT_PRI, font=("Courier New", 10),
                   insertbackground=TEXT_PRI, buttonbackground=BG_LIGHT,
                   relief="flat").pack(side="left", padx=(10, 0))
        tk.Label(mi, text="(default: 128 MB)", bg=BG_LIGHT, fg=TEXT_SEC,
                 font=("Courier New", 9)).pack(side="left", padx=(12, 0))

        # display
        dc, di = _card(wrap, "DISPLAY BACKEND")
        dc.pack(fill="x", pady=(0, 10))
        for label, val in [("SDL (default)", "sdl"),
                            ("GTK",          "gtk"),
                            ("VNC :0",       "vnc"),
                            ("None",         "none")]:
            ttk.Radiobutton(di, text=label, variable=self.display_mode,
                            value=val, style="Dark.TRadiobutton").pack(
                                side="left", padx=(0, 14))

        # extra flags
        ec, ei = _card(wrap, "EXTRA QEMU FLAGS")
        ec.pack(fill="x", pady=(0, 10))
        tk.Label(ei, text="Flags:", bg=BG_LIGHT, fg=TEXT_PRI,
                 font=("Courier New", 10)).pack(side="left")
        tk.Entry(ei, textvariable=self.extra_flags, width=52,
                 bg=BG_DARK, fg=TEXT_PRI, font=("Courier New", 10),
                 insertbackground=TEXT_PRI, relief="flat").pack(
                     side="left", padx=(10, 0))

        # command preview
        pvc, pvi = _card(wrap, "COMMAND PREVIEW")
        pvc.pack(fill="x")
        self.cmd_preview = tk.Text(pvi, height=3, bg=BG_DARK, fg=GREEN,
                                   font=("Courier New", 9), relief="flat",
                                   state="disabled")
        self.cmd_preview.pack(fill="x")
        tk.Button(pvi, text="Refresh", bg=BG_LIGHT, fg=TEXT_SEC,
                  font=("Courier New", 9), relief="flat", cursor="hand2",
                  activebackground=BORDER,
                  command=self._refresh_preview).pack(anchor="e", pady=(6, 0))
        self._refresh_preview()

    # =========================================================================
    # Tab: Output
    # =========================================================================

    def _tab_output(self, parent):
        wrap = tk.Frame(parent, bg=BG_MID)
        wrap.pack(fill="both", expand=True, padx=15, pady=15)

        hdr = tk.Frame(wrap, bg=BG_MID)
        hdr.pack(fill="x", pady=(0, 6))
        tk.Label(hdr, text="QEMU OUTPUT", bg=BG_MID, fg=BLUE,
                 font=("Courier New", 11, "bold")).pack(side="left")
        tk.Button(hdr, text="Clear", bg=BG_LIGHT, fg=TEXT_SEC,
                  font=("Courier New", 9), relief="flat", cursor="hand2",
                  activebackground=BORDER,
                  command=self._clear_output).pack(side="right")

        self.output_text = tk.Text(wrap, bg="#0d1117", fg=GREEN,
                                   font=("Courier New", 10), relief="flat",
                                   padx=8, pady=6, state="disabled",
                                   insertbackground=GREEN,
                                   selectbackground=BG_LIGHT)
        sb = ttk.Scrollbar(wrap, orient="vertical",
                           command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.output_text.pack(side="left", fill="both", expand=True)
        self._append_output("NewDOS Launcher ready — press LAUNCH to boot the kernel.\n")

    # =========================================================================
    # Tab: About
    # =========================================================================

    def _tab_about(self, parent):
        wrap = tk.Frame(parent, bg=BG_MID)
        wrap.pack(fill="both", expand=True, padx=20, pady=15)

        # ASCII logo
        lc, li = _card(wrap, "")
        lc.pack(fill="x", pady=(0, 10))
        ascii_logo = (
            "  ███╗   ██╗███████╗██╗    ██╗██████╗  ██████╗ ███████╗\n"
            "  ████╗  ██║██╔════╝██║    ██║██╔══██╗██╔═══██╗██╔════╝\n"
            "  ██╔██╗ ██║█████╗  ██║ █╗ ██║██║  ██║██║   ██║███████╗\n"
            "  ██║╚██╗██║██╔══╝  ██║███╗██║██║  ██║██║   ██║╚════██║\n"
            "  ██║ ╚████║███████╗╚███╔███╔╝██████╔╝╚██████╔╝███████║\n"
            "  ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═════╝  ╚═════╝ ╚══════╝"
        )
        tk.Label(lc, text=ascii_logo, bg=BG_LIGHT, fg=BLUE,
                 font=("Courier New", 7, "bold"), justify="left").pack(
                     padx=12, pady=(4, 2))
        tk.Label(lc, text="  Base Kernel  |  x86_64 bare-metal OS  |  Built with Rust",
                 bg=BG_LIGHT, fg=TEXT_SEC,
                 font=("Courier New", 9)).pack(pady=(0, 6))

        # Info grid
        ic, ii = _card(wrap, "PROJECT INFO")
        ic.pack(fill="x", pady=(0, 10))
        for label, value in [
            ("Architecture",   "x86_64 (bare metal, ring 0)"),
            ("Language",       "Rust 2021 — nightly toolchain"),
            ("Bootloader",     "bootloader v0.9 — BIOS only"),
            ("Display",        "VGA text mode (80×25) + framebuffer"),
            ("Input",          "PS/2 keyboard + mouse (IRQ-driven)"),
            ("Memory",         "Physical memory manager + kernel heap"),
            ("Filesystem",     "In-memory (no disk driver yet)"),
            ("Latest version", "v0.1.3"),
        ]:
            row = tk.Frame(ii, bg=BG_LIGHT)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=BG_LIGHT, fg=TEXT_SEC,
                     font=("Courier New", 9), width=20, anchor="w").pack(side="left")
            tk.Label(row, text=value, bg=BG_LIGHT, fg=TEXT_PRI,
                     font=("Courier New", 9), anchor="w").pack(side="left")

        # Build instructions
        bc, bi = _card(wrap, "BUILD FROM SOURCE")
        bc.pack(fill="x")
        build_cmds = (
            "rustup default nightly\n"
            "rustup target add x86_64-unknown-none\n"
            "cargo install bootimage\n"
            "cargo bootimage\n"
            "qemu-system-x86_64 -drive format=raw,"
            "file=target/x86_64-newdos/debug/bootimage-NewDOS-CLI-Operating-System.bin"
        )
        bt = tk.Text(bi, height=5, bg=BG_DARK, fg=GREEN,
                     font=("Courier New", 9), relief="flat", padx=8, pady=6)
        bt.insert("1.0", build_cmds)
        bt.configure(state="disabled")
        bt.pack(fill="x")

    # =========================================================================
    # Status bar
    # =========================================================================

    def _statusbar(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", side="bottom")
        bar = tk.Frame(self, bg=BG_MID)
        bar.pack(fill="x", side="bottom")
        self.status_lbl = tk.Label(bar, text="● Ready", bg=BG_MID, fg=GREEN,
                                   font=("Courier New", 9), anchor="w",
                                   padx=12, pady=4)
        self.status_lbl.pack(side="left")
        self.qemu_lbl = tk.Label(bar, text="", bg=BG_MID, fg=TEXT_SEC,
                                 font=("Courier New", 9), anchor="e",
                                 padx=12, pady=4)
        self.qemu_lbl.pack(side="right")

    # =========================================================================
    # Logic helpers
    # =========================================================================

    def _check_qemu(self):
        if shutil.which("qemu-system-x86_64"):
            self.qemu_lbl.configure(text="qemu-system-x86_64 ✓", fg=GREEN)
        else:
            self.qemu_lbl.configure(text="qemu-system-x86_64 not found", fg=RED)

    def _bin_path(self):
        ver  = self.selected_version.get()           # e.g. "v0.1.3"
        name = "NewDOSv" + ver[1:].replace(".", "_") + ".bin"
        return BASE_DIR / name

    def _build_cmd(self):
        cmd = [
            "qemu-system-x86_64",
            "-drive", f"format=raw,file={self._bin_path()}",
            "-m", str(self.memory_mb.get()),
            "-display", self.display_mode.get(),
        ]
        extra = self.extra_flags.get().strip()
        if extra:
            cmd.extend(extra.split())
        return cmd

    def _refresh_preview(self):
        preview = " ".join(str(c) for c in self._build_cmd())
        self.cmd_preview.configure(state="normal")
        self.cmd_preview.delete("1.0", "end")
        self.cmd_preview.insert("1.0", preview)
        self.cmd_preview.configure(state="disabled")

    def _launch_qemu(self):
        bp = self._bin_path()
        if not bp.exists():
            messagebox.showerror("File Not Found", f"Kernel binary not found:\n{bp}")
            return
        if not shutil.which("qemu-system-x86_64"):
            messagebox.showerror(
                "QEMU Not Found",
                "qemu-system-x86_64 is not installed.\n\n"
                "Install with:  sudo apt install qemu-system-x86")
            return

        cmd = self._build_cmd()
        self._append_output(f"\n$ {' '.join(str(c) for c in cmd)}\n")
        self._set_status("● Running", YELLOW)

        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
        except Exception as exc:
            self._append_output(f"Error: {exc}\n")
            self._set_status("● Error", RED)
            return

        self.launch_btn.configure(state="disabled")
        self.stop_btn  .configure(state="normal")

        threading.Thread(target=self._stream_output, daemon=True).start()
        self.after(500, self._poll)

    def _stream_output(self):
        try:
            for line in self.process.stdout:
                self.after(0, self._append_output, line)
        except Exception:
            pass

    def _poll(self):
        if self.process and self.process.poll() is not None:
            rc = self.process.returncode
            self._append_output(f"\nProcess exited (code {rc})\n")
            self._set_status("● Ready", GREEN)
            self.launch_btn.configure(state="normal")
            self.stop_btn  .configure(state="disabled")
            self.process = None
        elif self.process:
            self.after(500, self._poll)

    def _stop_qemu(self):
        if self.process:
            self.process.terminate()
            self._append_output("\n[Stopped by user]\n")
            self._set_status("● Ready", GREEN)
            self.launch_btn.configure(state="normal")
            self.stop_btn  .configure(state="disabled")
            self.process = None

    def _append_output(self, text):
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def _clear_output(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

    def _set_status(self, text, colour):
        self.status_lbl.configure(text=text, fg=colour)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = NewDOSLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
