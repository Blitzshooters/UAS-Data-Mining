"""Tab Project 2: Time Series Forecasting dengan LSTM."""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from data_generators import generate_air_pollution_like_dataset
from lstm_model import NumpyLSTM
from export_utils import export_dataframe


class Project2Tab(ttk.Frame):
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback or (lambda msg: None)
        self.df_raw = None
        self.df_series = None  # selected univariate/clean series (pandas Series indexed by date)
        self.scaler = None
        self.X_train = self.y_train = self.X_test = self.y_test = None
        self.model = None
        self.stop_flag = False
        self.eval_results = []
        self._build_layout()

    # ---------------------------------------------------------- layout
    def _build_layout(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(side="top", fill="x", padx=6, pady=4)
        ttk.Button(top_bar, text="Generate Dataset Sintetis (mirip Air Pollution China)",
                   command=self.load_synthetic).pack(side="left", padx=3)
        ttk.Button(top_bar, text="Load CSV...", command=self.load_csv).pack(side="left", padx=3)

        self.sub_nb = ttk.Notebook(self)
        self.sub_nb.pack(fill="both", expand=True, padx=6, pady=4)

        self.tab_viewer = ttk.Frame(self.sub_nb)
        self.tab_eda = ttk.Frame(self.sub_nb)
        self.tab_prep = ttk.Frame(self.sub_nb)
        self.tab_dl = ttk.Frame(self.sub_nb)
        self.tab_forecast = ttk.Frame(self.sub_nb)
        self.tab_eval = ttk.Frame(self.sub_nb)
        self.tab_viz = ttk.Frame(self.sub_nb)
        self.tab_summary = ttk.Frame(self.sub_nb)

        for tab, name in [(self.tab_viewer, "1. Dataset Viewer"),
                           (self.tab_eda, "2. EDA"),
                           (self.tab_prep, "3. Preprocessing"),
                           (self.tab_dl, "4. Deep Learning (LSTM)"),
                           (self.tab_forecast, "5. Forecasting"),
                           (self.tab_eval, "6. Evaluasi"),
                           (self.tab_viz, "7. Visualisasi Hasil"),
                           (self.tab_summary, "8. Ringkasan")]:
            self.sub_nb.add(tab, text=name)

        self._build_viewer_tab()
        self._build_eda_tab()
        self._build_prep_tab()
        self._build_dl_tab()
        self._build_forecast_tab()
        self._build_eval_tab()
        self._build_viz_tab()
        self._build_summary_tab()

    def _make_tree(self, parent, height=10):
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True)
        tree = ttk.Treeview(container, height=height)
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        return tree

    def _fill_tree(self, tree, df):
        tree.delete(*tree.get_children())
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        for col in df.columns:
            tree.heading(col, text=str(col))
            tree.column(col, width=100, stretch=False)
        for _, row in df.iterrows():
            tree.insert("", "end", values=[str(v) for v in row.tolist()])

    # ---------------------------------------------------------- data load
    def load_synthetic(self):
        self.status_callback("Membuat dataset polusi udara sintetis...")
        self.df_raw = generate_air_pollution_like_dataset()
        self.status_callback(f"Dataset dibuat: {self.df_raw.shape[0]} baris x {self.df_raw.shape[1]} kolom")
        self.refresh_viewer()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df_raw = pd.read_csv(path)
            self.status_callback(f"CSV dimuat: {self.df_raw.shape[0]} baris x {self.df_raw.shape[1]} kolom")
            self.refresh_viewer()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca CSV:\n{e}")

    # ---------------------------------------------------------- 1. viewer
    def _build_viewer_tab(self):
        info_frame = ttk.LabelFrame(self.tab_viewer, text="Informasi Dataset")
        info_frame.pack(fill="x", padx=6, pady=6)
        self.info_label = ttk.Label(info_frame, text="Belum ada dataset.", justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

        mode_frame = ttk.LabelFrame(self.tab_viewer, text="Sumber Sumbu Waktu")
        mode_frame.pack(fill="x", padx=6, pady=4)
        ttk.Label(mode_frame, text="Mode:").pack(side="left", padx=4)
        self.date_mode_var = tk.StringVar(value="Kolom Tanggal Asli")
        self.date_mode_combo = ttk.Combobox(mode_frame, textvariable=self.date_mode_var, state="readonly",
                                             values=["Kolom Tanggal Asli", "Bangun dari Year + Month (Agregasi Bulanan)"],
                                             width=42)
        self.date_mode_combo.pack(side="left", padx=4)
        self.date_mode_combo.bind("<<ComboboxSelected>>", self._on_date_mode_change)
        ttk.Label(mode_frame, text=(
            "  Pilih mode kedua jika dataset TIDAK punya kolom tanggal lengkap\n"
            "  (hanya ada Year, Month, Day of Week, Hour) — data akan dirata-ratakan\n"
            "  per Kota+Tahun+Bulan agar urutan waktunya valid (granularitas bulanan)."
        ), foreground="#555555", justify="left").pack(side="left", padx=8)

        sel_frame = ttk.Frame(self.tab_viewer)
        sel_frame.pack(fill="x", padx=6, pady=4)

        # --- Mode 1: kolom tanggal asli ---
        self.mode1_frame = ttk.Frame(sel_frame)
        ttk.Label(self.mode1_frame, text="Kolom Tanggal:").pack(side="left")
        self.date_col_combo = ttk.Combobox(self.mode1_frame, state="readonly", width=18)
        self.date_col_combo.pack(side="left", padx=4)

        # --- Mode 2: bangun dari Year + Month ---
        self.mode2_frame = ttk.Frame(sel_frame)
        ttk.Label(self.mode2_frame, text="Kolom Year:").pack(side="left")
        self.year_col_combo = ttk.Combobox(self.mode2_frame, state="readonly", width=12)
        self.year_col_combo.pack(side="left", padx=4)
        ttk.Label(self.mode2_frame, text="Kolom Month:").pack(side="left", padx=(8, 0))
        self.month_col_combo = ttk.Combobox(self.mode2_frame, state="readonly", width=12)
        self.month_col_combo.pack(side="left", padx=4)

        self.mode1_frame.pack(side="left")

        ttk.Label(sel_frame, text="Kolom Target (numerik):").pack(side="left", padx=(15, 3))
        self.target_col_combo = ttk.Combobox(sel_frame, state="readonly", width=18)
        self.target_col_combo.pack(side="left", padx=4)
        ttk.Label(sel_frame, text="Filter/Group Kolom (mis. City):").pack(side="left", padx=(15, 3))
        self.filter_col_combo = ttk.Combobox(sel_frame, state="readonly", width=14)
        self.filter_col_combo.pack(side="left", padx=4)
        self.filter_val_combo = ttk.Combobox(sel_frame, state="readonly", width=14)
        self.filter_val_combo.pack(side="left", padx=4)
        self.filter_col_combo.bind("<<ComboboxSelected>>", self._on_filter_col_change)
        ttk.Button(sel_frame, text="Siapkan Series", command=self.prepare_series).pack(side="left", padx=10)

        preview_frame = ttk.LabelFrame(self.tab_viewer, text="Preview Data")
        preview_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.viewer_tree = self._make_tree(preview_frame)

    def refresh_viewer(self):
        if self.df_raw is None:
            return
        df = self.df_raw
        self.info_label.config(text=(
            f"Jumlah data: {df.shape[0]:,} baris\n"
            f"Kolom: {', '.join(df.columns)}"
        ))
        self._fill_tree(self.viewer_tree, df.head(50))
        self.date_col_combo["values"] = list(df.columns)
        num_cols = list(df.select_dtypes(include=np.number).columns)
        self.target_col_combo["values"] = num_cols
        cat_cols = list(df.select_dtypes(exclude=np.number).columns)
        self.filter_col_combo["values"] = ["(tanpa filter)"] + cat_cols
        self.filter_col_combo.set("(tanpa filter)")
        self.filter_val_combo["values"] = []
        self.year_col_combo["values"] = list(df.columns)
        self.month_col_combo["values"] = list(df.columns)

        # smart defaults for known column names
        has_real_date = False
        for c in df.columns:
            if c.lower() in ("date", "tanggal", "datetime"):
                self.date_col_combo.set(c)
                has_real_date = True
        for c in num_cols:
            cl = c.lower().replace(" ", "").replace("(", "").replace(")", "")
            if cl.startswith("pm2.5") or cl.startswith("pm25") or cl == "value":
                self.target_col_combo.set(c)
        if not self.date_col_combo.get() and len(df.columns) > 0:
            self.date_col_combo.set(df.columns[0])
        if not self.target_col_combo.get() and num_cols:
            self.target_col_combo.set(num_cols[0])

        # Auto-detect Year/Month columns for mode 2, and auto-switch mode if no real date exists
        year_candidates = [c for c in df.columns if c.lower() == "year"]
        month_candidates = [c for c in df.columns if c.lower() == "month"]
        if year_candidates:
            self.year_col_combo.set(year_candidates[0])
        if month_candidates:
            self.month_col_combo.set(month_candidates[0])

        if not has_real_date and year_candidates and month_candidates:
            self.date_mode_var.set("Bangun dari Year + Month (Agregasi Bulanan)")
            self._on_date_mode_change()
            messagebox.showinfo(
                "Info",
                "Dataset ini tidak memiliki kolom tanggal lengkap (hanya Year/Month/Day of "
                "Week/Hour), sehingga mode otomatis dialihkan ke 'Bangun dari Year + Month "
                "(Agregasi Bulanan)'. Data akan dirata-ratakan per Kota+Tahun+Bulan agar "
                "urutan waktunya valid secara kronologis (granularitas bulanan, bukan harian)."
            )

    def _on_date_mode_change(self, event=None):
        if self.date_mode_var.get() == "Kolom Tanggal Asli":
            self.mode2_frame.pack_forget()
            self.mode1_frame.pack(side="left")
        else:
            self.mode1_frame.pack_forget()
            self.mode2_frame.pack(side="left")

    def _on_filter_col_change(self, event=None):
        col = self.filter_col_combo.get()
        if col and col != "(tanpa filter)" and self.df_raw is not None:
            self.filter_val_combo["values"] = sorted(self.df_raw[col].dropna().unique().tolist())
            if self.filter_val_combo["values"]:
                self.filter_val_combo.set(self.filter_val_combo["values"][0])

    def prepare_series(self):
        if self.df_raw is None:
            messagebox.showwarning("Peringatan", "Muat dataset dahulu.")
            return
        target_col = self.target_col_combo.get()
        if not target_col:
            messagebox.showwarning("Peringatan", "Pilih kolom target.")
            return
        df = self.df_raw.copy()
        filt_col, filt_val = self.filter_col_combo.get(), self.filter_val_combo.get()
        if filt_col and filt_col != "(tanpa filter)" and filt_val:
            df = df[df[filt_col] == filt_val]

        mode = self.date_mode_var.get()
        if mode == "Kolom Tanggal Asli":
            date_col = self.date_col_combo.get()
            if not date_col:
                messagebox.showwarning("Peringatan", "Pilih kolom tanggal.")
                return
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except Exception:
                messagebox.showerror("Error", f"Kolom '{date_col}' tidak bisa dikonversi ke tanggal.")
                return
            df = df.sort_values(date_col)
            series = df.set_index(date_col)[target_col]
            self.series_granularity = "harian/asli"
        else:
            year_col, month_col = self.year_col_combo.get(), self.month_col_combo.get()
            if not year_col or not month_col:
                messagebox.showwarning("Peringatan", "Pilih kolom Year dan Month.")
                return
            grouped = df.groupby([year_col, month_col])[target_col].mean().reset_index()
            grouped["__date__"] = pd.to_datetime(
                grouped[year_col].astype(int).astype(str) + "-" +
                grouped[month_col].astype(int).astype(str) + "-01"
            )
            grouped = grouped.sort_values("__date__")
            series = grouped.set_index("__date__")[target_col]
            self.series_granularity = "bulanan (agregasi rata-rata)"
            self.status_callback(
                "Catatan: dataset tidak punya kolom tanggal harian asli, sehingga series "
                "dibangun dari agregasi rata-rata per Year+Month (granularitas bulanan)."
            )

        self.df_series = series
        self.last_series_meta = {
            "target_col": target_col, "filter_col": filt_col, "filter_val": filt_val,
            "granularity": self.series_granularity, "n_points": len(series),
        }
        self.status_callback(f"Series siap: {len(series)} titik data dari kolom '{target_col}' ({self.series_granularity}).")
        self.refresh_eda()
        messagebox.showinfo(
            "Info",
            f"Series berhasil disiapkan: {len(series)} titik data.\n"
            f"Granularitas: {self.series_granularity}."
        )


    # ---------------------------------------------------------- 2. EDA
    def _build_eda_tab(self):
        left = ttk.Frame(self.tab_eda)
        left.pack(side="left", fill="both", expand=False, padx=6, pady=6)
        right = ttk.Frame(self.tab_eda)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        self.eda_stats_tree = self._make_tree(left, height=12)

        ctrl = ttk.Frame(right)
        ctrl.pack(fill="x")
        ttk.Label(ctrl, text="Grafik:").pack(side="left")
        self.eda_plot_choice = ttk.Combobox(ctrl, state="readonly", values=[
            "Line Chart", "Histogram", "Boxplot", "Trend Analysis", "Seasonal Analysis (mingguan)",
            "Rolling Mean", "Rolling Std", "Heatmap Korelasi (semua kolom numerik)"
        ])
        self.eda_plot_choice.set("Line Chart")
        self.eda_plot_choice.pack(side="left", padx=4)
        ttk.Label(ctrl, text="Window:").pack(side="left", padx=(10, 2))
        self.eda_window = tk.IntVar(value=30)
        ttk.Entry(ctrl, textvariable=self.eda_window, width=6).pack(side="left")
        ttk.Button(ctrl, text="Tampilkan", command=self.render_eda_plot).pack(side="left", padx=10)

        self.eda_fig = Figure(figsize=(7, 5))
        self.eda_canvas = FigureCanvasTkAgg(self.eda_fig, master=right)
        self.eda_canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_eda(self):
        if self.df_series is None:
            return
        s = self.df_series
        desc = s.describe()
        miss = s.isna().sum()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        n_out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
        rows = [{"Statistik": k, "Nilai": round(v, 3)} for k, v in desc.items()]
        rows.append({"Statistik": "Missing Value", "Nilai": int(miss)})
        rows.append({"Statistik": "Outlier (IQR)", "Nilai": int(n_out)})
        self._fill_tree(self.eda_stats_tree, pd.DataFrame(rows))
        self.render_eda_plot()

    def render_eda_plot(self):
        if self.df_series is None:
            return
        s = self.df_series.dropna()
        choice = self.eda_plot_choice.get()
        w = max(2, int(self.eda_window.get()))
        self.eda_fig.clear()
        ax = self.eda_fig.add_subplot(111)

        if choice == "Line Chart":
            ax.plot(s.index, s.values, color="#337ab7", linewidth=0.8)
            ax.set_title("Line Chart")
        elif choice == "Histogram":
            ax.hist(s.values, bins=40, color="#5cb85c")
            ax.set_title("Histogram")
        elif choice == "Boxplot":
            ax.boxplot(s.values)
            ax.set_title("Boxplot")
        elif choice == "Trend Analysis":
            ax.plot(s.index, s.values, color="#cccccc", linewidth=0.6, label="Data")
            trend = s.rolling(w, min_periods=1).mean()
            ax.plot(s.index, trend.values, color="#d9534f", linewidth=1.5, label=f"Trend (MA-{w})")
            ax.legend(); ax.set_title("Trend Analysis")
        elif choice == "Seasonal Analysis (mingguan)":
            try:
                weekly = s.groupby(s.index.dayofweek).mean()
                ax.bar(["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"], weekly.values, color="#f0ad4e")
                ax.set_title("Rata-rata per Hari dalam Minggu")
            except Exception:
                ax.text(0.5, 0.5, "Index bukan tanggal, tidak bisa analisis musiman", ha="center")
        elif choice == "Rolling Mean":
            roll = s.rolling(w, min_periods=1).mean()
            ax.plot(s.index, s.values, color="#cccccc", linewidth=0.5, label="Data")
            ax.plot(s.index, roll.values, color="#337ab7", label=f"Rolling Mean ({w})")
            ax.legend(); ax.set_title("Rolling Mean")
        elif choice == "Rolling Std":
            roll = s.rolling(w, min_periods=1).std()
            ax.plot(s.index, roll.values, color="#5bc0de")
            ax.set_title(f"Rolling Std ({w})")
        elif choice == "Heatmap Korelasi (semua kolom numerik)":
            num = self.df_raw.select_dtypes(include=np.number)
            corr = num.corr()
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=7)
            self.eda_fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Correlation Heatmap")

        self.eda_fig.tight_layout()
        self.eda_canvas.draw()

    # ---------------------------------------------------------- 3. preprocessing
    def _build_prep_tab(self):
        # ---- Baris atas: dua panel berdampingan ----
        top_panels = ttk.Frame(self.tab_prep)
        top_panels.pack(fill="x", padx=6, pady=4)

        # Panel Missing Value
        mv_frame = ttk.LabelFrame(top_panels, text="Penanganan Missing Value")
        mv_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))

        ttk.Label(mv_frame, text="Metode:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.miss_method = ttk.Combobox(mv_frame, state="readonly", width=22,
                                         values=["Interpolasi Linear", "Forward Fill",
                                                 "Backward Fill", "Isi dengan Median",
                                                 "Isi dengan Mean"])
        self.miss_method.set("Interpolasi Linear")
        self.miss_method.grid(row=0, column=1, padx=4)

        ttk.Button(mv_frame, text="Terapkan ke Series",
                   command=self.apply_series_missing).grid(row=1, column=0, columnspan=2,
                                                           pady=4, padx=4, sticky="ew")
        self.prep_mv_status = ttk.Label(mv_frame, text="", foreground="#337ab7")
        self.prep_mv_status.grid(row=2, column=0, columnspan=2, sticky="w", padx=4)

        # Panel Outlier
        out_frame = ttk.LabelFrame(top_panels, text="Penanganan Outlier Time Series")
        out_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))

        ttk.Label(out_frame, text="Deteksi:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.prep_outlier_detect_var = tk.StringVar(value="IQR (1.5×)")
        ttk.Combobox(out_frame, textvariable=self.prep_outlier_detect_var,
                     values=["IQR (1.5×)", "IQR (3×)", "Z-Score (>3)", "Z-Score (>2.5)",
                             "Rolling Z-Score (>3)"],
                     state="readonly", width=22).grid(row=0, column=1, padx=4)

        ttk.Label(out_frame, text="Penanganan:").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.prep_outlier_action_var = tk.StringVar(value="Interpolasi Linear")
        ttk.Combobox(out_frame, textvariable=self.prep_outlier_action_var,
                     values=["Interpolasi Linear", "Winsorizing (Clip ke Batas)",
                             "Ganti dengan Median Rolling", "Ganti dengan NaN",
                             "Hapus Titik Outlier"],
                     state="readonly", width=22).grid(row=1, column=1, padx=4)

        ttk.Button(out_frame, text="Terapkan ke Series",
                   command=self.apply_series_outlier).grid(row=2, column=0, columnspan=2,
                                                           pady=4, padx=4, sticky="ew")
        self.prep_out_status = ttk.Label(out_frame, text="", foreground="#337ab7")
        self.prep_out_status.grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        # ---- Panel konfigurasi LSTM prep ----
        lstm_frame = ttk.LabelFrame(self.tab_prep, text="Konfigurasi Windowing & Split")
        lstm_frame.pack(fill="x", padx=6, pady=4)

        ttk.Label(lstm_frame, text="Scaling:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.scale_method = ttk.Combobox(lstm_frame, state="readonly",
                                          values=["MinMaxScaler", "StandardScaler", "RobustScaler"])
        self.scale_method.set("MinMaxScaler")
        self.scale_method.grid(row=0, column=1, padx=4)

        ttk.Label(lstm_frame, text="Sequence Length (Windowing):").grid(row=0, column=2, padx=(16, 4), sticky="w")
        self.seq_len_var = tk.IntVar(value=14)
        ttk.Entry(lstm_frame, textvariable=self.seq_len_var, width=8).grid(row=0, column=3, padx=4, sticky="w")

        ttk.Label(lstm_frame, text="Train-Test Split (% train):").grid(row=0, column=4, padx=(16, 4), sticky="w")
        self.split_var = tk.IntVar(value=80)
        ttk.Entry(lstm_frame, textvariable=self.split_var, width=8).grid(row=0, column=5, padx=4, sticky="w")

        ttk.Button(lstm_frame, text="Jalankan Preprocessing (Scaling + Windowing)",
                   command=self.run_preprocessing).grid(row=1, column=0, columnspan=6,
                                                         pady=6, padx=4, sticky="w")
        self.prep_status_label = ttk.Label(lstm_frame, text="")
        self.prep_status_label.grid(row=2, column=0, columnspan=6, sticky="w", padx=4)

        # ---- Grafik preview ----
        plot_frame = ttk.LabelFrame(self.tab_prep, text="Preview Series (setelah handling)")
        plot_frame.pack(fill="both", expand=True, padx=6, pady=4)
        self.prep_fig = Figure(figsize=(7, 4))
        self.prep_canvas = FigureCanvasTkAgg(self.prep_fig, master=plot_frame)
        self.prep_canvas.get_tk_widget().pack(fill="both", expand=True)

    def apply_series_missing(self):
        """Tangani missing value pada df_series (series yang sudah disiapkan)."""
        if self.df_series is None:
            messagebox.showwarning("Peringatan", "Siapkan series dahulu di tab 1.")
            return
        s = self.df_series.copy()
        n_missing_before = int(s.isna().sum())
        if n_missing_before == 0:
            self.prep_mv_status.config(text="✓ Series tidak memiliki missing value.")
            return
        method = self.miss_method.get()
        if method == "Interpolasi Linear":
            s = s.interpolate(method="linear", limit_direction="both")
        elif method == "Forward Fill":
            s = s.ffill().bfill()
        elif method == "Backward Fill":
            s = s.bfill().ffill()
        elif method == "Isi dengan Median":
            s = s.fillna(s.median())
        elif method == "Isi dengan Mean":
            s = s.fillna(s.mean())
        n_missing_after = int(s.isna().sum())
        self.df_series = s
        self.prep_mv_status.config(
            text=f"✓ '{method}': {n_missing_before} missing → {n_missing_after} missing tersisa.")
        self._plot_prep_preview()

    def apply_series_outlier(self):
        """Tangani outlier pada df_series."""
        if self.df_series is None:
            messagebox.showwarning("Peringatan", "Siapkan series dahulu di tab 1.")
            return
        s = self.df_series.copy().dropna()
        detect = self.prep_outlier_detect_var.get()
        action = self.prep_outlier_action_var.get()

        # Deteksi
        if "IQR" in detect:
            mult = 1.5 if "1.5" in detect else 3.0
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            lower, upper = q1 - mult * (q3 - q1), q3 + mult * (q3 - q1)
            outlier_mask = (s < lower) | (s > upper)
        elif "Rolling" in detect:
            roll_mean = s.rolling(30, min_periods=5, center=True).mean()
            roll_std = s.rolling(30, min_periods=5, center=True).std()
            outlier_mask = (s - roll_mean).abs() > 3 * roll_std
            lower, upper = None, None
        else:
            threshold = 3.0 if ">3" in detect else 2.5
            z = (s - s.mean()) / s.std()
            outlier_mask = z.abs() > threshold
            lower, upper = s.mean() - threshold * s.std(), s.mean() + threshold * s.std()

        n_out = int(outlier_mask.sum())
        if n_out == 0:
            self.prep_out_status.config(text=f"✓ Tidak ditemukan outlier dengan metode {detect}.")
            return

        try:
            if action == "Interpolasi Linear":
                s[outlier_mask] = np.nan
                s = s.interpolate(method="linear", limit_direction="both")
            elif action == "Winsorizing (Clip ke Batas)" and lower is not None:
                s = s.clip(lower=lower, upper=upper)
            elif action == "Ganti dengan Median Rolling":
                roll_med = s.rolling(15, min_periods=3, center=True).median()
                s[outlier_mask] = roll_med[outlier_mask]
            elif action == "Ganti dengan NaN":
                s[outlier_mask] = np.nan
            elif action == "Hapus Titik Outlier":
                s = s[~outlier_mask]
            self.df_series = s
            self.prep_out_status.config(
                text=f"✓ {n_out} outlier ({detect}) → '{action}'. "
                     f"Series kini: {len(s)} titik.")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal menangani outlier:\n{e}")
            return

        self._plot_prep_preview()

    def _plot_prep_preview(self):
        """Tampilkan grafik preview series saat ini."""
        if self.df_series is None:
            return
        self.prep_fig.clear()
        ax = self.prep_fig.add_subplot(111)
        ax.plot(self.df_series.index, self.df_series.values, color="#337ab7", linewidth=0.8)
        ax.set_title("Preview Series (setelah handling missing value / outlier)")
        self.prep_fig.tight_layout()
        self.prep_canvas.draw()

    def run_preprocessing(self):
        if self.df_series is None:
            messagebox.showwarning("Peringatan", "Siapkan series dahulu di tab Dataset Viewer.")
            return
        s = self.df_series.copy()

        # Auto-handle missing yang tersisa sebelum scaling
        if s.isna().any():
            s = s.interpolate(limit_direction="both").ffill().bfill()

        scaler_cls = {"MinMaxScaler": MinMaxScaler, "StandardScaler": StandardScaler,
                      "RobustScaler": RobustScaler}[self.scale_method.get()]
        self.scaler = scaler_cls()
        values_scaled = self.scaler.fit_transform(s.values.reshape(-1, 1)).flatten()
        self.scaled_series = pd.Series(values_scaled, index=s.index)
        self.clean_series = s

        seq_len = max(2, int(self.seq_len_var.get()))
        X, y = [], []
        for i in range(len(values_scaled) - seq_len):
            X.append(values_scaled[i:i + seq_len])
            y.append(values_scaled[i + seq_len])
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)[..., np.newaxis]

        split_pct = max(10, min(95, int(self.split_var.get()))) / 100
        split_idx = int(len(X) * split_pct)
        self.X_train, self.y_train = X[:split_idx], y[:split_idx]
        self.X_test, self.y_test = X[split_idx:], y[split_idx:]
        self.seq_len = seq_len
        self.test_dates = s.index[seq_len + split_idx:]

        self.prep_status_label.config(text=(
            f"✓ Selesai. Total window: {len(X)} | Train: {len(self.X_train)} | "
            f"Test: {len(self.X_test)} | Seq len: {seq_len} | "
            f"Scaler: {self.scale_method.get()}"))

        self.prep_fig.clear()
        ax = self.prep_fig.add_subplot(111)
        ax.plot(self.scaled_series.index, self.scaled_series.values, color="#5cb85c", linewidth=0.8)
        ax.set_title("Data Setelah Scaling (siap untuk LSTM)")
        self.prep_fig.tight_layout()
        self.prep_canvas.draw()
        self.status_callback("Preprocessing selesai. Lanjut ke tab Deep Learning (LSTM).")

    # ---------------------------------------------------------- 4. deep learning
    def _build_dl_tab(self):
        frame = ttk.LabelFrame(self.tab_dl, text="Parameter LSTM")
        frame.pack(fill="x", padx=6, pady=6)

        params = [("Epoch", "epoch_var", 30), ("Batch Size", "batch_var", 32),
                  ("Learning Rate", "lr_var", 0.01), ("Neuron (Hidden Units)", "neuron_var", 32),
                  ("Dropout (0-1)", "dropout_var", 0.0)]
        for i, (label, attr, default) in enumerate(params):
            ttk.Label(frame, text=label + ":").grid(row=i, column=0, padx=4, pady=3, sticky="w")
            var = tk.DoubleVar(value=default) if isinstance(default, float) else tk.IntVar(value=default)
            setattr(self, attr, var)
            ttk.Entry(frame, textvariable=var, width=10).grid(row=i, column=1, padx=4, sticky="w")

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=0, column=2, rowspan=5, padx=20)
        ttk.Button(btn_frame, text="Mulai Training", command=self.start_training).pack(pady=4, fill="x")
        ttk.Button(btn_frame, text="Hentikan Training", command=self.stop_training).pack(pady=4, fill="x")
        ttk.Separator(btn_frame, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(btn_frame, text="↺ Reset Model",
                   command=self.reset_model).pack(pady=2, fill="x")
        ttk.Button(btn_frame, text="↺ Reset Riwayat Evaluasi",
                   command=self.reset_eval_history).pack(pady=2, fill="x")

        progress_frame = ttk.LabelFrame(self.tab_dl, text="Progress Training")
        progress_frame.pack(fill="x", padx=6, pady=6)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", padx=8, pady=6)
        self.train_status_label = ttk.Label(progress_frame, text="Status: belum mulai")
        self.train_status_label.pack(anchor="w", padx=8, pady=4)

        loss_frame = ttk.LabelFrame(self.tab_dl, text="Loss Training vs Validation")
        loss_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.loss_fig = Figure(figsize=(7, 4))
        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, master=loss_frame)
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def start_training(self):
        if self.X_train is None:
            messagebox.showwarning("Peringatan", "Jalankan preprocessing dahulu (tab 3).")
            return
        self.stop_flag = False
        epochs = max(1, int(self.epoch_var.get()))
        batch_size = max(1, int(self.batch_var.get()))
        lr = float(self.lr_var.get())
        neuron = max(2, int(self.neuron_var.get()))
        dropout = max(0.0, min(0.9, float(self.dropout_var.get())))

        self.model = NumpyLSTM(n_features=1, n_hidden=neuron, dropout=dropout)
        self.progress_bar["maximum"] = epochs
        self.progress_bar["value"] = 0
        self.train_status_label.config(text="Status: training dimulai...")
        self.status_callback("Training LSTM dimulai...")

        def progress_callback(ep, total, train_loss, val_loss):
            def update_ui():
                self.progress_bar["value"] = ep
                self.train_status_label.config(
                    text=f"Status: Epoch {ep}/{total} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
                self.loss_fig.clear()
                ax = self.loss_fig.add_subplot(111)
                ax.plot(self.model.history["loss"], label="Training Loss", color="#337ab7")
                ax.plot(self.model.history["val_loss"], label="Validation Loss", color="#d9534f")
                ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss"); ax.legend()
                ax.set_title("Loss Training vs Validation")
                self.loss_fig.tight_layout()
                self.loss_canvas.draw()
            self.after(0, update_ui)

        def train_thread():
            self.model.fit(self.X_train, self.y_train, self.X_test, self.y_test,
                            epochs=epochs, batch_size=batch_size, lr=lr,
                            progress_callback=progress_callback,
                            stop_flag=lambda: self.stop_flag)
            self.after(0, lambda: self.train_status_label.config(text="Status: training selesai."))
            self.after(0, lambda: self.status_callback("Training LSTM selesai."))

        threading.Thread(target=train_thread, daemon=True).start()

    def stop_training(self):
        self.stop_flag = True
        self.train_status_label.config(text="Status: training dihentikan oleh pengguna.")

    def reset_model(self):
        """Hapus model yang sudah dilatih dan hasil prediksi, agar bisa training ulang dari nol."""
        if self.model is None and not hasattr(self, "last_pred"):
            messagebox.showinfo("Info", "Tidak ada model yang perlu direset.")
            return
        if not messagebox.askyesno(
            "Konfirmasi Reset Model",
            "Model LSTM yang sudah dilatih, loss history, dan hasil prediksi akan dihapus.\n\n"
            "Data preprocessing (train/test split) TETAP tersimpan, sehingga Anda bisa langsung "
            "training ulang dengan parameter yang berbeda tanpa menjalankan preprocessing lagi.\n\n"
            "Lanjutkan?"
        ):
            return
        self.stop_flag = True  # hentikan jika sedang training
        self.model = None
        for attr in ("last_pred", "last_actual"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.progress_bar["value"] = 0
        self.train_status_label.config(text="Status: model direset. Siap training ulang.")
        self.loss_fig.clear()
        self.loss_canvas.draw()
        self.status_callback("Model LSTM direset. Ubah parameter lalu klik 'Mulai Training'.")

    def reset_eval_history(self):
        """Hapus riwayat evaluasi (tabel eval_results) tanpa menghapus model."""
        if not self.eval_results:
            messagebox.showinfo("Info", "Riwayat evaluasi sudah kosong.")
            return
        if not messagebox.askyesno(
            "Konfirmasi Reset Riwayat Evaluasi",
            "Tabel riwayat evaluasi LSTM akan dikosongkan.\n\n"
            "Model yang sudah dilatih TIDAK dihapus.\n\nLanjutkan?"
        ):
            return
        self.eval_results = []
        self._fill_tree(self.eval_tree, pd.DataFrame())
        self.update_summary()
        self.status_callback("Riwayat evaluasi dikosongkan.")

    # ---------------------------------------------------------- 5. forecasting
    def _build_forecast_tab(self):
        ctrl = ttk.Frame(self.tab_forecast)
        ctrl.pack(fill="x", padx=6, pady=6)
        ttk.Button(ctrl, text="Tampilkan Prediksi vs Aktual (Test Set)",
                   command=self.render_forecast_test).pack(side="left", padx=4)
        ttk.Label(ctrl, text="Forecast n-hari ke depan:").pack(side="left", padx=(20, 4))
        self.future_n_var = tk.IntVar(value=14)
        ttk.Entry(ctrl, textvariable=self.future_n_var, width=8).pack(side="left")
        ttk.Button(ctrl, text="Forecast Masa Depan", command=self.render_future_forecast).pack(side="left", padx=10)

        self.forecast_fig = Figure(figsize=(7, 5))
        self.forecast_canvas = FigureCanvasTkAgg(self.forecast_fig, master=self.tab_forecast)
        self.forecast_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def render_forecast_test(self):
        if self.model is None or self.X_test is None or len(self.X_test) == 0:
            messagebox.showwarning("Peringatan", "Latih model dahulu (tab 4) dan pastikan ada data test.")
            return
        pred_scaled = self.model.predict(self.X_test)
        pred = self.scaler.inverse_transform(pred_scaled)
        actual = self.scaler.inverse_transform(self.y_test)
        self.last_pred, self.last_actual = pred.flatten(), actual.flatten()

        self.forecast_fig.clear()
        ax = self.forecast_fig.add_subplot(111)
        ax.plot(self.test_dates[:len(actual)], actual.flatten(), label="Aktual", color="#337ab7")
        ax.plot(self.test_dates[:len(pred)], pred.flatten(), label="Prediksi", color="#d9534f", linestyle="--")
        ax.legend(); ax.set_title("Prediksi vs Aktual (Test Set)")
        self.forecast_fig.tight_layout()
        self.forecast_canvas.draw()
        self.compute_evaluation()

    def render_future_forecast(self):
        if self.model is None or self.scaled_series is None:
            messagebox.showwarning("Peringatan", "Latih model dahulu (tab 4).")
            return
        n_future = max(1, int(self.future_n_var.get()))
        seq = self.scaled_series.values[-self.seq_len:].tolist()
        future_preds = []
        for _ in range(n_future):
            x_in = np.array(seq[-self.seq_len:]).reshape(1, self.seq_len, 1)
            next_val = self.model.predict(x_in)[0, 0]
            future_preds.append(next_val)
            seq.append(next_val)
        future_preds = self.scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()

        last_date = self.clean_series.index[-1]
        try:
            future_dates = pd.date_range(start=last_date, periods=n_future + 1, freq="D")[1:]
        except Exception:
            future_dates = np.arange(len(self.clean_series), len(self.clean_series) + n_future)

        self.forecast_fig.clear()
        ax = self.forecast_fig.add_subplot(111)
        hist_tail = self.clean_series.iloc[-120:]
        ax.plot(hist_tail.index, hist_tail.values, label="Data Historis", color="#337ab7")
        ax.plot(future_dates, future_preds, label=f"Forecast {n_future} hari", color="#5cb85c", linestyle="--", marker="o", markersize=3)
        ax.legend(); ax.set_title("Forecast Masa Depan")
        self.forecast_fig.tight_layout()
        self.forecast_canvas.draw()

    # ---------------------------------------------------------- 6. evaluation
    def _build_eval_tab(self):
        frame = ttk.LabelFrame(self.tab_eval, text="Metrik Evaluasi (Test Set)")
        frame.pack(fill="x", padx=6, pady=6)
        self.eval_tree = self._make_tree(frame, height=4)

        ttk.Button(self.tab_eval, text="Hitung Ulang Evaluasi", command=self.compute_evaluation).pack(anchor="w", padx=6, pady=4)

        err_frame = ttk.LabelFrame(self.tab_eval, text="Distribusi Error")
        err_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.err_fig = Figure(figsize=(7, 4))
        self.err_canvas = FigureCanvasTkAgg(self.err_fig, master=err_frame)
        self.err_canvas.get_tk_widget().pack(fill="both", expand=True)

    def compute_evaluation(self):
        if not hasattr(self, "last_pred"):
            self.render_forecast_test()
            return
        pred, actual = self.last_pred, self.last_actual
        mae = np.mean(np.abs(pred - actual))
        mse = np.mean((pred - actual) ** 2)
        rmse = np.sqrt(mse)
        nonzero = actual != 0
        mape = np.mean(np.abs((actual[nonzero] - pred[nonzero]) / actual[nonzero])) * 100 if nonzero.any() else np.nan
        ss_res = np.sum((actual - pred) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        row = {"Model": "LSTM", "MAE": round(mae, 3), "MSE": round(mse, 3),
               "RMSE": round(rmse, 3), "MAPE (%)": round(mape, 3), "R2": round(r2, 3)}
        self.eval_results = [row]
        self._fill_tree(self.eval_tree, pd.DataFrame(self.eval_results))

        errors = pred - actual
        self.err_fig.clear()
        ax = self.err_fig.add_subplot(111)
        ax.hist(errors, bins=30, color="#5bc0de")
        ax.set_title("Distribusi Error (Prediksi - Aktual)")
        self.err_fig.tight_layout()
        self.err_canvas.draw()
        self.update_summary()

    # ---------------------------------------------------------- 7. visualization hub
    def _build_viz_tab(self):
        ctrl = ttk.Frame(self.tab_viz)
        ctrl.pack(fill="x", padx=6, pady=4)
        ttk.Label(ctrl, text="Pilih Grafik:").pack(side="left")
        self.viz2_choice = ttk.Combobox(ctrl, state="readonly", values=[
            "Loss Training", "Loss Validation", "Prediksi vs Aktual", "Error Distribution"])
        self.viz2_choice.set("Loss Training")
        self.viz2_choice.pack(side="left", padx=4)
        ttk.Button(ctrl, text="Tampilkan", command=self.render_viz2).pack(side="left", padx=10)

        self.viz2_fig = Figure(figsize=(7, 5))
        self.viz2_canvas = FigureCanvasTkAgg(self.viz2_fig, master=self.tab_viz)
        self.viz2_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def render_viz2(self):
        if self.model is None:
            messagebox.showwarning("Peringatan", "Latih model dahulu (tab 4).")
            return
        choice = self.viz2_choice.get()
        self.viz2_fig.clear()
        ax = self.viz2_fig.add_subplot(111)
        if choice == "Loss Training":
            ax.plot(self.model.history["loss"], color="#337ab7")
            ax.set_title("Loss Training")
        elif choice == "Loss Validation":
            ax.plot(self.model.history["val_loss"], color="#d9534f")
            ax.set_title("Loss Validation")
        elif choice == "Prediksi vs Aktual":
            if not hasattr(self, "last_pred"):
                self.render_forecast_test()
            ax.plot(self.last_actual, label="Aktual", color="#337ab7")
            ax.plot(self.last_pred, label="Prediksi", color="#d9534f", linestyle="--")
            ax.legend(); ax.set_title("Prediksi vs Aktual")
        elif choice == "Error Distribution":
            if not hasattr(self, "last_pred"):
                self.render_forecast_test()
            ax.hist(self.last_pred - self.last_actual, bins=30, color="#5bc0de")
            ax.set_title("Error Distribution")
        self.viz2_fig.tight_layout()
        self.viz2_canvas.draw()

    # ---------------------------------------------------------- 8. summary
    def _build_summary_tab(self):
        frame = ttk.LabelFrame(self.tab_summary, text="Ringkasan Hasil")
        frame.pack(fill="x", padx=6, pady=6)
        self.summary_tree = self._make_tree(frame, height=4)

        export_frame = ttk.Frame(self.tab_summary)
        export_frame.pack(fill="x", padx=6, pady=4)
        ttk.Button(export_frame, text="Export CSV", command=lambda: self._export("csv")).pack(side="left", padx=4)
        ttk.Button(export_frame, text="Export Excel", command=lambda: self._export("xlsx")).pack(side="left", padx=4)
        ttk.Button(export_frame, text="Export PDF", command=lambda: self._export("pdf")).pack(side="left", padx=4)

    def update_summary(self):
        self._fill_tree(self.summary_tree, pd.DataFrame(self.eval_results))

    def _export(self, fmt):
        if not self.eval_results:
            messagebox.showwarning("Peringatan", "Belum ada hasil evaluasi.")
            return
        export_dataframe(pd.DataFrame(self.eval_results), fmt, title="Ringkasan Hasil LSTM")
