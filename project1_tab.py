"""Tab Project 1: Data Preparation & Machine Learning."""

import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import (LabelEncoder, OneHotEncoder, StandardScaler,
                                    MinMaxScaler, RobustScaler)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              accuracy_score, confusion_matrix)

from data_generators import generate_accidents_like_dataset
from export_utils import export_dataframe


class Project1Tab(ttk.Frame):
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback or (lambda msg: None)
        self.df_raw = None
        self.df_work = None
        self.experiment_results = []  # list of dict rows for results table

        self._build_layout()

    # ---------------------------------------------------------- layout
    def _build_layout(self):
        top_bar = ttk.Frame(self)
        top_bar.pack(side="top", fill="x", padx=6, pady=4)

        ttk.Button(top_bar, text="Generate Dataset Sintetis (mirip US Accidents)",
                   command=self.load_synthetic).pack(side="left", padx=3)
        ttk.Button(top_bar, text="Load CSV...", command=self.load_csv).pack(side="left", padx=3)
        ttk.Button(top_bar, text="↺ Reset Data Kerja",
                   command=self.reset_work_data).pack(side="left", padx=3)
        ttk.Label(top_bar, text="Jumlah baris demo:").pack(side="left", padx=(15, 3))
        self.n_rows_var = tk.IntVar(value=60000)
        ttk.Entry(top_bar, textvariable=self.n_rows_var, width=10).pack(side="left")

        self.sub_nb = ttk.Notebook(self)
        self.sub_nb.pack(fill="both", expand=True, padx=6, pady=4)

        self.tab_viewer = ttk.Frame(self.sub_nb)
        self.tab_missing = ttk.Frame(self.sub_nb)
        self.tab_outlier = ttk.Frame(self.sub_nb)
        self.tab_encode = ttk.Frame(self.sub_nb)
        self.tab_transform = ttk.Frame(self.sub_nb)
        self.tab_feateng = ttk.Frame(self.sub_nb)
        self.tab_ml = ttk.Frame(self.sub_nb)
        self.tab_stats = ttk.Frame(self.sub_nb)
        self.tab_viz = ttk.Frame(self.sub_nb)

        for tab, name in [(self.tab_viewer, "1. Dataset Viewer"),
                           (self.tab_missing, "2. Missing Value"),
                           (self.tab_outlier, "3. Outlier"),
                           (self.tab_encode, "4. Encoding"),
                           (self.tab_transform, "5. Transformasi"),
                           (self.tab_feateng, "6. Feature Eng."),
                           (self.tab_ml, "7. Eksperimen ML"),
                           (self.tab_stats, "8. Statistik"),
                           (self.tab_viz, "9. Visualisasi")]:
            self.sub_nb.add(tab, text=name)

        self._build_viewer_tab()
        self._build_missing_tab()
        self._build_outlier_tab()
        self._build_encode_tab()
        self._build_transform_tab()
        self._build_feateng_tab()
        self._build_ml_tab()
        self._build_stats_tab()
        self._build_viz_tab()

    # ---------------------------------------------------------- data loading
    def load_synthetic(self):
        n = max(1000, int(self.n_rows_var.get()))
        self.status_callback(f"Membuat dataset sintetis ({n} baris)...")
        self.df_raw = generate_accidents_like_dataset(n_rows=n)
        self.df_work = self.df_raw.copy()
        self.status_callback(f"Dataset sintetis dibuat: {self.df_raw.shape[0]} baris x {self.df_raw.shape[1]} kolom")
        self.refresh_viewer()

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.status_callback("Membaca CSV...")
            self.df_raw = pd.read_csv(path)
            self.df_work = self.df_raw.copy()
            self.status_callback(f"CSV dimuat: {self.df_raw.shape[0]} baris x {self.df_raw.shape[1]} kolom")
            self.refresh_viewer()
        except Exception as e:
            messagebox.showerror("Error", f"Gagal membaca CSV:\n{e}")

    def reset_work_data(self):
        """Reset df_work ke kondisi awal (df_raw), tanpa mengulang load dataset."""
        if self.df_raw is None:
            messagebox.showwarning("Peringatan", "Belum ada dataset yang dimuat.")
            return
        if not messagebox.askyesno(
            "Konfirmasi Reset Data Kerja",
            "Semua langkah preprocessing (encoding, transformasi, feature engineering) "
            "akan dibatalkan dan data dikembalikan ke kondisi awal dataset.\n\n"
            "Hasil eksperimen ML yang sudah ada TIDAK dihapus.\n\n"
            "Lanjutkan?"
        ):
            return
        self.df_work = self.df_raw.copy()
        self._refresh_ml_column_choices()
        self.status_callback("Data kerja direset ke kondisi awal dataset.")
        messagebox.showinfo("Reset Berhasil",
                            "Data kerja telah dikembalikan ke kondisi awal.\n"
                            "Anda bisa memulai ulang langkah preprocessing dari tab 4 (Encoding).")

    # ---------------------------------------------------------- 1. viewer
    def _build_viewer_tab(self):
        info_frame = ttk.LabelFrame(self.tab_viewer, text="Informasi Dataset")
        info_frame.pack(fill="x", padx=6, pady=6)
        self.info_label = ttk.Label(info_frame, text="Belum ada dataset. Klik 'Generate Dataset Sintetis' atau 'Load CSV'.",
                                     justify="left")
        self.info_label.pack(anchor="w", padx=8, pady=6)

        preview_frame = ttk.LabelFrame(self.tab_viewer, text="Preview Data (50 baris pertama)")
        preview_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.viewer_tree = self._make_tree(preview_frame)

        dtype_frame = ttk.LabelFrame(self.tab_viewer, text="Tipe Data Kolom")
        dtype_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.dtype_tree = self._make_tree(dtype_frame, height=6)

    def _make_tree(self, parent, height=12):
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

    def refresh_viewer(self):
        if self.df_raw is None:
            return
        df = self.df_raw
        self.info_label.config(text=(
            f"Jumlah baris: {df.shape[0]:,}\n"
            f"Jumlah kolom: {df.shape[1]}\n"
            f"Kolom numerik: {len(df.select_dtypes(include=np.number).columns)}\n"
            f"Kolom kategorikal: {len(df.select_dtypes(exclude=np.number).columns)}"
        ))
        self._fill_tree(self.viewer_tree, df.head(50))

        dtype_df = pd.DataFrame({"Kolom": df.columns, "Tipe Data": [str(t) for t in df.dtypes]})
        self._fill_tree(self.dtype_tree, dtype_df)

        self.refresh_missing()
        self.refresh_outlier()
        self._refresh_ml_column_choices()

    # ---------------------------------------------------------- 2. missing value
    def _build_missing_tab(self):
        left = ttk.Frame(self.tab_missing)
        left.pack(side="left", fill="both", expand=False, padx=6, pady=6)
        right = ttk.Frame(self.tab_missing)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        self.missing_tree = self._make_tree(left)
        self.missing_fig = Figure(figsize=(5.5, 4.2))
        self.missing_canvas = FigureCanvasTkAgg(self.missing_fig, master=right)
        self.missing_canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_missing(self):
        if self.df_raw is None:
            return
        df = self.df_raw
        miss_count = df.isna().sum()
        miss_pct = (miss_count / len(df) * 100).round(2)
        result = pd.DataFrame({"Kolom": df.columns, "Jumlah Missing": miss_count.values,
                                "Persentase (%)": miss_pct.values})
        result = result.sort_values("Persentase (%)", ascending=False)
        self._fill_tree(self.missing_tree, result)

        self.missing_fig.clear()
        ax = self.missing_fig.add_subplot(111)
        plot_data = result[result["Jumlah Missing"] > 0]
        if len(plot_data) > 0:
            ax.barh(plot_data["Kolom"], plot_data["Persentase (%)"], color="#d9534f")
            ax.set_xlabel("Persentase Missing (%)")
            ax.set_title("Missing Value per Kolom")
        else:
            ax.text(0.5, 0.5, "Tidak ada missing value", ha="center", va="center")
        self.missing_fig.tight_layout()
        self.missing_canvas.draw()

    # ---------------------------------------------------------- 3. outlier
    def _build_outlier_tab(self):
        left = ttk.Frame(self.tab_outlier)
        left.pack(side="left", fill="both", expand=False, padx=6, pady=6)
        right = ttk.Frame(self.tab_outlier)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        self.outlier_tree = self._make_tree(left)

        sel_frame = ttk.Frame(right)
        sel_frame.pack(fill="x")
        ttk.Label(sel_frame, text="Kolom untuk Boxplot:").pack(side="left")
        self.outlier_col_var = tk.StringVar()
        self.outlier_col_combo = ttk.Combobox(sel_frame, textvariable=self.outlier_col_var, state="readonly")
        self.outlier_col_combo.pack(side="left", padx=4)
        ttk.Button(sel_frame, text="Tampilkan", command=self.plot_outlier_boxplot).pack(side="left", padx=4)

        self.outlier_fig = Figure(figsize=(5.5, 4))
        self.outlier_canvas = FigureCanvasTkAgg(self.outlier_fig, master=right)
        self.outlier_canvas.get_tk_widget().pack(fill="both", expand=True)

    def refresh_outlier(self):
        if self.df_raw is None:
            return
        df = self.df_raw
        num_cols = df.select_dtypes(include=np.number).columns
        rows = []
        for col in num_cols:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((s < lower) | (s > upper)).sum()
            rows.append({"Kolom": col, "Jumlah Outlier (IQR)": int(n_out),
                         "Lower Bound": round(lower, 2), "Upper Bound": round(upper, 2)})
        result = pd.DataFrame(rows)
        self._fill_tree(self.outlier_tree, result)
        self.outlier_col_combo["values"] = list(num_cols)
        if len(num_cols) > 0:
            self.outlier_col_var.set(num_cols[0])

    def plot_outlier_boxplot(self):
        if self.df_raw is None or not self.outlier_col_var.get():
            return
        col = self.outlier_col_var.get()
        self.outlier_fig.clear()
        ax = self.outlier_fig.add_subplot(111)
        ax.boxplot(self.df_raw[col].dropna(), vert=True)
        ax.set_title(f"Boxplot: {col}")
        self.outlier_fig.tight_layout()
        self.outlier_canvas.draw()

    # ---------------------------------------------------------- 4. encoding
    def _build_encode_tab(self):
        frame = ttk.LabelFrame(self.tab_encode, text="Data Encoding")
        frame.pack(fill="x", padx=6, pady=6)

        ttk.Label(frame, text="Metode:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.encode_method_var = tk.StringVar(value="Label Encoding")
        ttk.Combobox(frame, textvariable=self.encode_method_var,
                     values=["Label Encoding", "One Hot Encoding"], state="readonly").grid(row=0, column=1, padx=4)

        ttk.Label(frame, text="Kolom kategorikal:").grid(row=1, column=0, padx=4, pady=4, sticky="nw")
        self.encode_listbox = tk.Listbox(frame, selectmode="multiple", height=6, exportselection=False)
        self.encode_listbox.grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Button(frame, text="Terapkan Encoding", command=self.apply_encoding).grid(row=2, column=1, sticky="w", padx=4, pady=6)
        self.encode_status = ttk.Label(frame, text="")
        self.encode_status.grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        preview_frame = ttk.LabelFrame(self.tab_encode, text="Preview Hasil Encoding")
        preview_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.encode_preview_tree = self._make_tree(preview_frame)

    def _refresh_categorical_listbox(self):
        self.encode_listbox.delete(0, "end")
        if self.df_work is None:
            return
        cat_cols = self.df_work.select_dtypes(exclude=np.number).columns
        for c in cat_cols:
            self.encode_listbox.insert("end", c)

    def apply_encoding(self):
        if self.df_work is None:
            messagebox.showwarning("Peringatan", "Muat dataset terlebih dahulu.")
            return
        sel = [self.encode_listbox.get(i) for i in self.encode_listbox.curselection()]
        if not sel:
            messagebox.showwarning("Peringatan", "Pilih minimal satu kolom kategorikal.")
            return
        method = self.encode_method_var.get()
        df = self.df_work.copy()
        if method == "Label Encoding":
            for col in sel:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str).fillna("Missing"))
        else:
            df = pd.get_dummies(df, columns=sel, dummy_na=True)
        self.df_work = df
        self.encode_status.config(text=f"Encoding '{method}' diterapkan pada: {', '.join(sel)}. Kolom sekarang: {df.shape[1]}")
        self._fill_tree(self.encode_preview_tree, df.head(30))
        self._refresh_transform_listbox()
        self._refresh_ml_column_choices()

    # ---------------------------------------------------------- 5. transform
    def _build_transform_tab(self):
        frame = ttk.LabelFrame(self.tab_transform, text="Data Transformation")
        frame.pack(fill="x", padx=6, pady=6)

        ttk.Label(frame, text="Metode:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.transform_method_var = tk.StringVar(value="StandardScaler")
        ttk.Combobox(frame, textvariable=self.transform_method_var,
                     values=["StandardScaler", "MinMaxScaler", "RobustScaler"],
                     state="readonly").grid(row=0, column=1, padx=4)

        ttk.Label(frame, text="Kolom numerik:").grid(row=1, column=0, padx=4, pady=4, sticky="nw")
        self.transform_listbox = tk.Listbox(frame, selectmode="multiple", height=8, exportselection=False)
        self.transform_listbox.grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Button(frame, text="Terapkan Transformasi (isi missing dgn median dulu)",
                   command=self.apply_transform).grid(row=2, column=1, sticky="w", padx=4, pady=6)
        self.transform_status = ttk.Label(frame, text="")
        self.transform_status.grid(row=3, column=0, columnspan=2, sticky="w", padx=4)

        preview_frame = ttk.LabelFrame(self.tab_transform, text="Preview Hasil Transformasi")
        preview_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.transform_preview_tree = self._make_tree(preview_frame)

    def _refresh_transform_listbox(self):
        self.transform_listbox.delete(0, "end")
        if self.df_work is None:
            return
        num_cols = self.df_work.select_dtypes(include=np.number).columns
        for c in num_cols:
            self.transform_listbox.insert("end", c)

    def apply_transform(self):
        if self.df_work is None:
            messagebox.showwarning("Peringatan", "Muat dataset terlebih dahulu.")
            return
        sel = [self.transform_listbox.get(i) for i in self.transform_listbox.curselection()]
        if not sel:
            messagebox.showwarning("Peringatan", "Pilih minimal satu kolom numerik.")
            return
        method = self.transform_method_var.get()
        df = self.df_work.copy()
        for col in sel:
            df[col] = df[col].fillna(df[col].median())
        scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(),
                  "RobustScaler": RobustScaler()}[method]
        df[sel] = scaler.fit_transform(df[sel])
        self.df_work = df
        self.transform_status.config(text=f"Transformasi '{method}' diterapkan pada {len(sel)} kolom.")
        self._fill_tree(self.transform_preview_tree, df.head(30))
        self._refresh_ml_column_choices()

    # ---------------------------------------------------------- 6. feature engineering
    def _build_feateng_tab(self):
        frame = ttk.LabelFrame(self.tab_feateng, text="Feature Construction")
        frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(frame, text="Kolom A:").grid(row=0, column=0, padx=4, pady=4)
        self.fc_col_a = ttk.Combobox(frame, state="readonly")
        self.fc_col_a.grid(row=0, column=1, padx=4)
        ttk.Label(frame, text="Operasi:").grid(row=0, column=2, padx=4)
        self.fc_op = ttk.Combobox(frame, values=["+", "-", "*", "/"], state="readonly", width=4)
        self.fc_op.grid(row=0, column=3, padx=4)
        ttk.Label(frame, text="Kolom B:").grid(row=0, column=4, padx=4)
        self.fc_col_b = ttk.Combobox(frame, state="readonly")
        self.fc_col_b.grid(row=0, column=5, padx=4)
        ttk.Label(frame, text="Nama fitur baru:").grid(row=1, column=0, padx=4, pady=4)
        self.fc_new_name = ttk.Entry(frame)
        self.fc_new_name.insert(0, "fitur_baru")
        self.fc_new_name.grid(row=1, column=1, padx=4)
        ttk.Button(frame, text="Buat Fitur", command=self.apply_feature_construction).grid(row=1, column=2, padx=4)

        sel_frame = ttk.LabelFrame(self.tab_feateng, text="Feature Selection (SelectKBest, target = Severity/kolom target)")
        sel_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(sel_frame, text="Jumlah fitur top-K:").grid(row=0, column=0, padx=4, pady=4)
        self.fs_k = tk.IntVar(value=10)
        ttk.Entry(sel_frame, textvariable=self.fs_k, width=6).grid(row=0, column=1, padx=4)
        ttk.Button(sel_frame, text="Jalankan Feature Selection", command=self.apply_feature_selection).grid(row=0, column=2, padx=8)
        self.fs_result_label = ttk.Label(sel_frame, text="")
        self.fs_result_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=4)

        pca_frame = ttk.LabelFrame(self.tab_feateng, text="PCA")
        pca_frame.pack(fill="x", padx=6, pady=6)
        ttk.Label(pca_frame, text="Jumlah komponen:").grid(row=0, column=0, padx=4, pady=4)
        self.pca_n = tk.IntVar(value=5)
        ttk.Entry(pca_frame, textvariable=self.pca_n, width=6).grid(row=0, column=1, padx=4)
        ttk.Button(pca_frame, text="Jalankan PCA", command=self.apply_pca).grid(row=0, column=2, padx=8)
        self.pca_result_label = ttk.Label(pca_frame, text="")
        self.pca_result_label.grid(row=1, column=0, columnspan=3, sticky="w", padx=4)

        preview_frame = ttk.LabelFrame(self.tab_feateng, text="Preview Dataset Kerja")
        preview_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.feateng_preview_tree = self._make_tree(preview_frame)

    def _refresh_feateng_choices(self):
        if self.df_work is None:
            return
        num_cols = list(self.df_work.select_dtypes(include=np.number).columns)
        self.fc_col_a["values"] = num_cols
        self.fc_col_b["values"] = num_cols

    def apply_feature_construction(self):
        if self.df_work is None:
            return
        a, b, op, name = self.fc_col_a.get(), self.fc_col_b.get(), self.fc_op.get(), self.fc_new_name.get().strip()
        if not (a and b and op and name):
            messagebox.showwarning("Peringatan", "Lengkapi kolom A, operasi, kolom B, dan nama fitur baru.")
            return
        df = self.df_work.copy()
        try:
            if op == "+":
                df[name] = df[a] + df[b]
            elif op == "-":
                df[name] = df[a] - df[b]
            elif op == "*":
                df[name] = df[a] * df[b]
            elif op == "/":
                df[name] = df[a] / df[b].replace(0, np.nan)
            self.df_work = df
            self._fill_tree(self.feateng_preview_tree, df.head(30))
            self._refresh_feateng_choices()
            self._refresh_ml_column_choices()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_feature_selection(self):
        if self.df_work is None:
            return
        target_col = self.ml_target_var.get() if hasattr(self, "ml_target_var") else None
        if not target_col:
            messagebox.showwarning("Peringatan", "Pilih kolom target di tab 'Eksperimen ML' dahulu.")
            return
        df = self.df_work.select_dtypes(include=np.number).dropna()
        if target_col not in df.columns:
            messagebox.showwarning("Peringatan", "Kolom target harus numerik (encode dahulu bila kategorikal).")
            return
        X = df.drop(columns=[target_col])
        y = df[target_col]
        k = min(int(self.fs_k.get()), X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        selected = X.columns[selector.get_support()]
        self.fs_result_label.config(text=f"Fitur terpilih ({k}): {', '.join(selected)}")

    def apply_pca(self):
        if self.df_work is None:
            return
        df = self.df_work.select_dtypes(include=np.number).dropna()
        n = min(int(self.pca_n.get()), df.shape[1])
        pca = PCA(n_components=n)
        comps = pca.fit_transform(df)
        var_ratio = pca.explained_variance_ratio_
        comp_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(n)])
        self.df_work = pd.concat([self.df_work.reset_index(drop=True), comp_df], axis=1)
        self.pca_result_label.config(text=f"PCA selesai. Explained variance ratio: {np.round(var_ratio, 3).tolist()}")
        self._fill_tree(self.feateng_preview_tree, self.df_work.head(30))
        self._refresh_feateng_choices()
        self._refresh_ml_column_choices()

    # ---------------------------------------------------------- 7. ML experiments
    def _build_ml_tab(self):
        top = ttk.LabelFrame(self.tab_ml, text="Konfigurasi Eksperimen")
        top.pack(fill="x", padx=6, pady=6)

        ttk.Label(top, text="Kolom Target:").grid(row=0, column=0, padx=4, pady=4)
        self.ml_target_var = tk.StringVar()
        self.ml_target_combo = ttk.Combobox(top, textvariable=self.ml_target_var, state="readonly")
        self.ml_target_combo.grid(row=0, column=1, padx=4)

        ttk.Label(top, text="Catatan eksperimen:").grid(row=0, column=2, padx=4)
        self.ml_note_entry = ttk.Entry(top, width=25)
        self.ml_note_entry.insert(0, "baseline")
        self.ml_note_entry.grid(row=0, column=3, padx=4)

        ttk.Button(top, text="Jalankan Eksperimen (RF + LR)",
                   command=self.run_experiment).grid(row=0, column=4, padx=10)
        ttk.Button(top, text="Hapus Semua Hasil", command=self.clear_experiments).grid(row=0, column=5, padx=4)

        self.ml_status_label = ttk.Label(top, text="")
        self.ml_status_label.grid(row=1, column=0, columnspan=6, sticky="w", padx=4)

        result_frame = ttk.LabelFrame(self.tab_ml, text="Tabel Hasil Eksperimen")
        result_frame.pack(fill="both", expand=True, padx=6, pady=6)
        self.ml_result_tree = self._make_tree(result_frame, height=10)

        export_frame = ttk.Frame(self.tab_ml)
        export_frame.pack(fill="x", padx=6, pady=4)
        ttk.Button(export_frame, text="Export CSV", command=lambda: self._export_results("csv")).pack(side="left", padx=4)
        ttk.Button(export_frame, text="Export Excel", command=lambda: self._export_results("xlsx")).pack(side="left", padx=4)
        ttk.Button(export_frame, text="Export PDF", command=lambda: self._export_results("pdf")).pack(side="left", padx=4)

    def _refresh_ml_column_choices(self):
        if self.df_work is None:
            return
        cols = list(self.df_work.columns)
        self.ml_target_combo["values"] = cols
        if "Severity" in cols and not self.ml_target_var.get():
            self.ml_target_var.set("Severity")
        self._refresh_categorical_listbox()
        self._refresh_transform_listbox()
        self._refresh_feateng_choices()

    def run_experiment(self):
        if self.df_work is None:
            messagebox.showwarning("Peringatan", "Muat dataset dahulu.")
            return
        target = self.ml_target_var.get()
        if not target:
            messagebox.showwarning("Peringatan", "Pilih kolom target.")
            return

        self.ml_status_label.config(text="Menjalankan eksperimen, mohon tunggu...")
        self.update_idletasks()

        df = self.df_work.copy()
        num_df = df.select_dtypes(include=[np.number, bool]).copy()
        if target not in num_df.columns:
            le = LabelEncoder()
            num_df[target] = le.fit_transform(df[target].astype(str))
        num_df = num_df.fillna(num_df.median(numeric_only=True))
        num_df = num_df.replace([np.inf, -np.inf], np.nan).dropna()

        if num_df.shape[0] < 50 or target not in num_df.columns:
            messagebox.showerror("Error", "Data tidak cukup atau target tidak valid setelah pembersihan.")
            return

        # subsample for speed if huge
        if len(num_df) > 100000:
            num_df = num_df.sample(100000, random_state=42)

        X = num_df.drop(columns=[target])
        y = num_df[target]
        if y.nunique() > 15:
            y = pd.qcut(y, q=min(4, y.nunique()), labels=False, duplicates="drop")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if y.nunique() > 1 else None)

        results = {}
        for name, model in [("RF", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                             ("LR", LogisticRegression(max_iter=500))]:
            t0 = time.time()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            elapsed = time.time() - t0
            results[name] = {
                "Prec": precision_score(y_test, pred, average="macro", zero_division=0),
                "Rec": recall_score(y_test, pred, average="macro", zero_division=0),
                "F1": f1_score(y_test, pred, average="macro", zero_division=0),
                "Acc": accuracy_score(y_test, pred),
                "Time": elapsed,
                "y_test": y_test, "pred": pred, "model": model, "X_cols": list(X.columns),
            }

        row = {
            "No": len(self.experiment_results) + 1,
            "Catatan": self.ml_note_entry.get(),
            "Missing Value": int(self.df_raw.isna().sum().sum()) if self.df_raw is not None else 0,
            "Target": target,
            "Prec_RF": round(results["RF"]["Prec"], 3), "Rec_RF": round(results["RF"]["Rec"], 3),
            "F1_RF": round(results["RF"]["F1"], 3), "Acc_RF": round(results["RF"]["Acc"], 3),
            "Time_RF(s)": round(results["RF"]["Time"], 2),
            "Prec_LR": round(results["LR"]["Prec"], 3), "Rec_LR": round(results["LR"]["Rec"], 3),
            "F1_LR": round(results["LR"]["F1"], 3), "Acc_LR": round(results["LR"]["Acc"], 3),
            "Time_LR(s)": round(results["LR"]["Time"], 2),
        }
        self.experiment_results.append(row)
        self._last_experiment_detail = results
        self._fill_tree(self.ml_result_tree, pd.DataFrame(self.experiment_results))
        self.ml_status_label.config(text=f"Eksperimen #{row['No']} selesai.")
        self._refresh_stats()
        self._refresh_viz_choices()

    def clear_experiments(self):
        self.experiment_results = []
        self._fill_tree(self.ml_result_tree, pd.DataFrame())
        self._refresh_stats()

    def _export_results(self, fmt):
        if not self.experiment_results:
            messagebox.showwarning("Peringatan", "Belum ada hasil eksperimen.")
            return
        df = pd.DataFrame(self.experiment_results)
        export_dataframe(df, fmt, title="Hasil Eksperimen Machine Learning")

    # ---------------------------------------------------------- 8. statistics
    def _build_stats_tab(self):
        frame_avg = ttk.LabelFrame(self.tab_stats, text="Rata-Rata Performa")
        frame_avg.pack(fill="x", padx=6, pady=6)
        self.stats_avg_tree = self._make_tree(frame_avg, height=3)

        frame_min = ttk.LabelFrame(self.tab_stats, text="Performa Minimum")
        frame_min.pack(fill="x", padx=6, pady=6)
        self.stats_min_tree = self._make_tree(frame_min, height=3)

        frame_max = ttk.LabelFrame(self.tab_stats, text="Performa Maksimum")
        frame_max.pack(fill="x", padx=6, pady=6)
        self.stats_max_tree = self._make_tree(frame_max, height=3)

        placeholder = ttk.Label(
            self.tab_stats,
            text="Belum ada eksperimen yang dijalankan. Jalankan eksperimen di tab '7. Eksperimen ML' "
                 "terlebih dahulu untuk melihat statistik performa di sini.",
            foreground="#777777"
        )
        placeholder.pack(anchor="w", padx=10, pady=10)
        self._stats_placeholder = placeholder

    def _refresh_stats(self):
        if not self.experiment_results:
            for tree in [self.stats_avg_tree, self.stats_min_tree, self.stats_max_tree]:
                self._fill_tree(tree, pd.DataFrame())
            self._stats_placeholder.pack(anchor="w", padx=10, pady=10)
            return
        self._stats_placeholder.pack_forget()
        df = pd.DataFrame(self.experiment_results)
        rows_avg, rows_min, rows_max = [], [], []
        for model in ["RF", "LR"]:
            cols = {"Precision": f"Prec_{model}", "Recall": f"Rec_{model}",
                    "F1 Score": f"F1_{model}", "Accuracy": f"Acc_{model}"}
            rows_avg.append({"Model": model, **{k: round(df[v].mean(), 3) for k, v in cols.items()}})
            rows_min.append({"Model": model, **{k: round(df[v].min(), 3) for k, v in cols.items()}})
            rows_max.append({"Model": model, **{k: round(df[v].max(), 3) for k, v in cols.items()}})
        self._fill_tree(self.stats_avg_tree, pd.DataFrame(rows_avg))
        self._fill_tree(self.stats_min_tree, pd.DataFrame(rows_min))
        self._fill_tree(self.stats_max_tree, pd.DataFrame(rows_max))

    # ---------------------------------------------------------- 9. visualization
    def _build_viz_tab(self):
        top = ttk.Frame(self.tab_viz)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Pilih Visualisasi:").pack(side="left")
        self.viz_choice = ttk.Combobox(top, state="readonly", values=[
            "Histogram", "Boxplot", "Correlation Heatmap", "Feature Importance (RF)",
            "Confusion Matrix (RF)", "Confusion Matrix (LR)", "Perbandingan Accuracy Model"
        ])
        self.viz_choice.pack(side="left", padx=4)
        self.viz_choice.set("Histogram")
        ttk.Label(top, text="Kolom (untuk Histogram/Boxplot):").pack(side="left", padx=(15, 3))
        self.viz_col_combo = ttk.Combobox(top, state="readonly")
        self.viz_col_combo.pack(side="left", padx=4)
        ttk.Button(top, text="Tampilkan", command=self.render_viz).pack(side="left", padx=10)

        self.viz_fig = Figure(figsize=(7, 5))
        self.viz_canvas = FigureCanvasTkAgg(self.viz_fig, master=self.tab_viz)
        self.viz_canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    def _refresh_viz_choices(self):
        if self.df_work is not None:
            self.viz_col_combo["values"] = list(self.df_work.select_dtypes(include=np.number).columns)

    def render_viz(self):
        if self.df_work is None:
            messagebox.showwarning("Peringatan", "Muat dataset dahulu.")
            return
        choice = self.viz_choice.get()
        self.viz_fig.clear()
        ax = self.viz_fig.add_subplot(111)
        df = self.df_work

        if choice == "Histogram":
            col = self.viz_col_combo.get() or df.select_dtypes(include=np.number).columns[0]
            ax.hist(df[col].dropna(), bins=40, color="#5cb85c")
            ax.set_title(f"Histogram: {col}")
        elif choice == "Boxplot":
            col = self.viz_col_combo.get() or df.select_dtypes(include=np.number).columns[0]
            ax.boxplot(df[col].dropna())
            ax.set_title(f"Boxplot: {col}")
        elif choice == "Correlation Heatmap":
            num = df.select_dtypes(include=np.number).iloc[:, :20]
            corr = num.corr()
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
            ax.set_yticks(range(len(corr.columns))); ax.set_yticklabels(corr.columns, fontsize=6)
            self.viz_fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title("Correlation Heatmap")
        elif choice in ("Feature Importance (RF)", "Confusion Matrix (RF)", "Confusion Matrix (LR)"):
            if not hasattr(self, "_last_experiment_detail"):
                messagebox.showwarning("Peringatan", "Jalankan eksperimen ML dahulu (tab 7).")
                return
            detail = self._last_experiment_detail
            if choice == "Feature Importance (RF)":
                model = detail["RF"]["model"]
                cols = detail["RF"]["X_cols"]
                importances = model.feature_importances_
                order = np.argsort(importances)[-15:]
                ax.barh([cols[i] for i in order], importances[order], color="#337ab7")
                ax.set_title("Top 15 Feature Importance (Random Forest)")
            else:
                key = "RF" if "RF" in choice else "LR"
                cm = confusion_matrix(detail[key]["y_test"], detail[key]["pred"])
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title(f"Confusion Matrix ({key})")
                for (i, j), v in np.ndenumerate(cm):
                    ax.text(j, i, str(v), ha="center", va="center")
                self.viz_fig.colorbar(im, ax=ax, fraction=0.046)
        elif choice == "Perbandingan Accuracy Model":
            if not self.experiment_results:
                messagebox.showwarning("Peringatan", "Jalankan eksperimen ML dahulu (tab 7).")
                return
            edf = pd.DataFrame(self.experiment_results)
            x = np.arange(len(edf))
            ax.bar(x - 0.2, edf["Acc_RF"], width=0.4, label="Random Forest", color="#5bc0de")
            ax.bar(x + 0.2, edf["Acc_LR"], width=0.4, label="Logistic Regression", color="#f0ad4e")
            ax.set_xticks(x); ax.set_xticklabels(edf["No"])
            ax.set_xlabel("Eksperimen #"); ax.set_ylabel("Accuracy")
            ax.set_title("Perbandingan Accuracy Model"); ax.legend()

        self.viz_fig.tight_layout()
        self.viz_canvas.draw()
