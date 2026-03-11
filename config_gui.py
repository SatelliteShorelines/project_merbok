#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import yaml
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import tkinter.font as tkfont

# ---------------- Defaults ----------------
DEFAULTS = {
    "global_region": "1",
    "coastal_area": "4",
    "subregion": "00",
    "shoreline_section": "000",

    "gpu_id": "-1",
    "update": False,
    "reset": False,
    "planet_bool": False,

    "function": "validate",

    "dem": "",                     # TEXT BOX ONLY
    "slope": "",
    "smooth_slopes": False,
    "reference_elevation": "0",

    "resample_frequency": "365D",
    "model": "global",
    "estimate": "ensemble",

    "custom_sections": "",
    "waterline_filter": True,

    "year_min": "1984",
    "year_max": "2026",

    "ee_project": "ee-merbok",

    "home": "",
    "coastseg_roi_folder": "",
    "planet_folder": "",
}

FUNCTIONS = [
    "download_and_process", "process","rename_transects","transects","get_slope","prep_slope","rois","download",
    "find_rasters","check_planet","image_filter","reorg","pansharpen_coreg",
    "seg","seg_filter","extract","post_process","record_stats","validate",
    "merge_sections","merge_regions",
]
MODELS = ["global", "ak"]
ESTIMATES = ["ensemble", "rgb", "nir", "swir"]

# Smoother fonts—auto-pick first available
PREFERRED_FONTS = [
    "Inter",           # very smooth
    "Segoe UI",        # Windows
    "Helvetica Neue",  # macOS
    "SF Pro Text",     # macOS (if available)
    "Noto Sans",       # cross-platform
    "Ubuntu",
    "Arial",
]

def safe_load_yaml(path):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

class ConfigGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CoastSeg Batch Configuration Builder")
        self.geometry("900x900")
        self.minsize(880, 860)

        # Data vars
        self.vars = {k: tk.StringVar(value=str(v)) for k, v in DEFAULTS.items()}
        for key in ("update","reset","planet_bool","smooth_slopes","waterline_filter"):
            self.vars[key] = tk.BooleanVar(value=bool(DEFAULTS[key]))

        # Theme state
        self.dark_mode = tk.BooleanVar(value=True)

        # Fonts: set named Tk fonts so ttk updates live
        self._setup_named_fonts()

        # Styles & UI
        self._create_styles()
        self._build_ui()
        self._apply_theme(True)  # start dark

        # Track last saved path (default to config.yaml)
        self.last_saved_path = "config.yaml"

    def _choose_default_font(self):
        available = set(tkfont.families())
        for fam in PREFERRED_FONTS:
            if fam in available:
                return fam
        return tkfont.nametofont("TkDefaultFont").cget("family")

    def _setup_named_fonts(self):
        """Configure Tk named fonts (smooth family, larger default size)."""
        base_size = 14  # larger, readable default
        family = self._choose_default_font()

        self.tk_default = tkfont.nametofont("TkDefaultFont")
        self.tk_text    = tkfont.nametofont("TkTextFont")
        self.tk_menu    = tkfont.nametofont("TkMenuFont")
        self.tk_heading = tkfont.nametofont("TkHeadingFont")

        for f in (self.tk_default, self.tk_text, self.tk_menu):
            f.configure(family=family, size=base_size)
        self.tk_heading.configure(family=family, size=base_size + 3, weight="bold")

    def _create_styles(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure("TLabel", padding=4)
        self.style.configure("TEntry", padding=4)
        self.style.configure("TCombobox", padding=4)
        self.style.configure("TButton", padding=8)
        self.style.configure("Header.TLabel", font="TkHeadingFont")

        # Large, accent "Run" button style
        # We bind "Run.TButton" to the heading font to make it bigger
        self.style.configure("Run.TButton", font="TkHeadingFont", padding=10)

    def _apply_theme(self, dark: bool):
        if dark:
            BG, SUB_BG = "#0F172A", "#111827"
            FG, FG_MUTED = "#E5E7EB", "#9CA3AF"
            BTN_BG, BTN_BG_HOT = "#1F2937", "#374151"
            ACCENT, BORDER = "#60A5FA", "#374151"
            RUN_BG, RUN_BG_ACTIVE = "#2563EB", "#1D4ED8"  # brighter accent for Run
        else:
            BG, SUB_BG = "#F7FAFC", "#FFFFFF"
            FG, FG_MUTED = "#1F2937", "#6B7280"
            BTN_BG, BTN_BG_HOT = "#E5E7EB", "#D1D5DB"
            ACCENT, BORDER = "#2563EB", "#D1D5DB"
            RUN_BG, RUN_BG_ACTIVE = "#2563EB", "#1D4ED8"

        self.configure(bg=BG)
        self.style.configure("TFrame", background=BG)
        self.style.configure("TLabel", background=BG, foreground=FG)
        self.style.configure("Header.TLabel", background=BG, foreground=FG)
        self.style.configure("TEntry",
            fieldbackground=SUB_BG, foreground=FG,
            insertcolor=FG, bordercolor=BORDER, darkcolor=BORDER,
            lightcolor=ACCENT, background=SUB_BG
        )
        self.style.configure("TCombobox",
            fieldbackground=SUB_BG, foreground=FG,
            bordercolor=BORDER, darkcolor=BORDER, lightcolor=ACCENT,
            background=SUB_BG
        )
        self.style.map("TCombobox", fieldbackground=[("readonly", SUB_BG)])
        self.style.configure("TCheckbutton", background=BG, foreground=FG)
        self.style.configure("TButton", background=BTN_BG, foreground=FG, bordercolor=BORDER)
        self.style.map("TButton", background=[("active", BTN_BG_HOT)], foreground=[("disabled", FG_MUTED)])
        self.style.configure("Run.TButton", background=RUN_BG, foreground="#FFFFFF", bordercolor=BORDER)
        self.style.map("Run.TButton", background=[("active", RUN_BG_ACTIVE)])

        self.style.configure("Vertical.TScrollbar", background=BG, troughcolor=SUB_BG)

        if hasattr(self, "canvas"): self.canvas.configure(bg=BG)
        if hasattr(self, "form"): self.form.configure(style="TFrame")

    def _build_ui(self):
        # Toolbar (dark toggle + file actions)
        toolbar = ttk.Frame(self, padding=(12, 8))
        toolbar.pack(fill="x")

        ttk.Checkbutton(
            toolbar, text="Dark mode", variable=self.dark_mode,
            command=lambda: self._apply_theme(self.dark_mode.get())
        ).pack(side="left", padx=(2,10))

        ttk.Button(toolbar, text="Load config.yaml", command=self.load_yaml).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save", command=self.save_yaml).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save As...", command=self.save_as_yaml).pack(side="left", padx=4)

        # --- BIG RUN BUTTON on the right ---
        ttk.Button(toolbar, text="Run Function", style="Run.TButton",
                   command=self.run_function).pack(side="right", padx=8)

        ttk.Button(toolbar, text="Quit", command=self.destroy).pack(side="right", padx=4)

        # Status line (shows last run exit code)
        self.status_var = tk.StringVar(value="")
        status_bar = ttk.Frame(self, padding=(12, 2))
        status_bar.pack(fill="x")
        ttk.Label(status_bar, textvariable=self.status_var).pack(side="left")

        # Main area (scrollable)
        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=self.canvas.yview)
        self.form = ttk.Frame(self.canvas)

        self.form.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.form, anchor="nw")
        self.canvas.configure(yscrollcommand=vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        r = 0

        def header(text):
            nonlocal r
            ttk.Label(self.form, text=text, style="Header.TLabel").grid(
                row=r, column=0, columnspan=3, sticky="w", pady=(12, 6)
            )
            r += 1

        def entry(label, key, hint=None, width=28):
            nonlocal r
            ttk.Label(self.form, text=label).grid(row=r, column=0, sticky="w", padx=(2,2), pady=4)
            e = ttk.Entry(self.form, textvariable=self.vars[key], width=width)
            e.grid(row=r, column=1, sticky="ew", padx=(8,6), pady=4)
            if hint:
                ttk.Label(self.form, text=hint).grid(row=r, column=2, sticky="w", padx=(2,2), pady=4)
            r += 1

        def combo(label, key, values):
            nonlocal r
            ttk.Label(self.form, text=label).grid(row=r, column=0, sticky="w", padx=(2,2), pady=4)
            cb = ttk.Combobox(self.form, textvariable=self.vars[key], values=values, state="readonly")
            cb.grid(row=r, column=1, sticky="ew", padx=(8,6), pady=4)
            r += 1

        def check(label, key):
            nonlocal r
            ttk.Checkbutton(self.form, text=label, variable=self.vars[key]).grid(
                row=r, column=0, columnspan=2, sticky="w", padx=(2,2), pady=4
            )
            r += 1

        def path_entry(label, key):
            """Folder path selector (Browse...)."""
            nonlocal r
            ttk.Label(self.form, text=label).grid(row=r, column=0, sticky="w", padx=(2,2), pady=4)
            e = ttk.Entry(self.form, textvariable=self.vars[key], width=48)
            e.grid(row=r, column=1, sticky="ew", padx=(8,6), pady=4)
            ttk.Button(self.form, text="Browse...", command=lambda k=key: self._browse_dir(k)).grid(
                row=r, column=2, sticky="w", padx=(2,2), pady=4
            )
            r += 1

        # --- Sections ---
        header("Identifiers")
        entry("Global region (g)", "global_region", hint='e.g., "1"')
        entry("Coastal area (c)", "coastal_area", hint='e.g., "4"')
        entry("Subregion (rr)", "subregion", hint='e.g., "00"')
        entry("Shoreline section (sss)", "shoreline_section", hint='e.g., "014"; blank runs whole RR')

        header("Runtime Settings")
        entry("GPU ID", "gpu_id", hint='"-1" CPU; "0" GPU0')
        #check("Update imagery", "update")
        #check("Reset", "reset")
        check("Planet imagery", "planet_bool")
        combo("Function", "function", FUNCTIONS)
        entry("EE Project", "ee_project", hint='e.g., "ee-merbok, make sure EE is authenticated"')

        header("Models & DEM")
        entry("DEM (blank = ArcticDEM)", "dem")   # <-- TEXT BOX (no browse)
        entry("Constant slope", "slope", hint='e.g., "0.05"; blank for DEM-derived')
        check("Smooth slopes", "smooth_slopes")  # hint='smooth slopes if derived from transects on a DEM')
        entry("Reference elevation", "reference_elevation", hint='e.g., "0"')
        entry("Resample frequency", "resample_frequency", hint='"365D" yearly; "30D" monthly')
        combo("Model", "model", MODELS) # hint='global for globally trained model, ak for alaska model'
        combo("Estimate", "estimate", ESTIMATES) # hint='cycle different estimates when running validation'

        header("Filters & Date Range")
        entry("Custom sections", "custom_sections", hint='comma-separated (e.g., "001,002")')
        check("Waterline filter", "waterline_filter") # preferably leave on, inspect to see if it filters to heavily on edges of shoreline section")
        entry("Year min", "year_min", hint='"1984"')
        entry("Year max", "year_max", hint='"2026"')

        header("Paths")
        path_entry("Home directory (G/C/RR root)", "home")
        path_entry("CoastSeg ROI folder", "coastseg_roi_folder")
        path_entry("Planet folder", "planet_folder")

        for col in range(3):
            self.form.grid_columnconfigure(col, weight=1)

    # ----- Actions -----
    def _browse_dir(self, key):
        d = filedialog.askdirectory(title=f"Select folder for {key}")
        if d:
            self.vars[key].set(d)

    def to_dict(self):
        data = {}
        for k, var in self.vars.items():
            data[k] = bool(var.get()) if isinstance(var, tk.BooleanVar) else str(var.get()).strip()
        return data

    def load_yaml(self):
        path = filedialog.askopenfilename(
            title="Open config.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            cfg = safe_load_yaml(path)
            for k, var in self.vars.items():
                if k in cfg:
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(cfg[k]))
                    else:
                        var.set(str(cfg[k]))
            self.last_saved_path = path  # track path
            messagebox.showinfo("Loaded", f"Loaded configuration:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML:\n{e}")

    def save_yaml(self, path=None):
        # Save to specified path or last known path; default to "config.yaml"
        if path is None:
            path = self.last_saved_path or "config.yaml"
        try:
            with open(path, "w") as f:
                yaml.dump(self.to_dict(), f, sort_keys=False)
            self.last_saved_path = path
            messagebox.showinfo("Saved", f"Config written to:\n{os.path.abspath(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save YAML:\n{e}")

    def save_as_yaml(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
            initialfile="config.yaml",
            title="Save config.yaml as..."
        )
        if path:
            self.save_yaml(path)

    def run_function(self):
        """
        Save current config (to config.yaml) then run:
        os.system('python merbok_workflow.py --config config.yaml')
        """
        # Ensure we write to `config.yaml` as requested
        try:
            self.save_yaml("config.yaml")
        except Exception:
            # save_yaml already shows error dialogs
            return

        cmd = 'python merbok_workflow.py --config config.yaml'
        self.status_var.set(f"Running: {cmd}")
        # NOTE: os.system blocks the GUI until the process finishes.
        # If you want non-blocking behavior, we can move this to a thread.
        exit_code = os.system(cmd)
        self.status_var.set(f"Completed: exit code {exit_code}")

def main():
    app = ConfigGUI()
    app.mainloop()

if __name__ == "__main__":
    main()