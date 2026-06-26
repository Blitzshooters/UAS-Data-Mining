"""
Sistem Analisis Data dan Deep Learning Terintegrasi
=====================================================
Aplikasi desktop Tkinter yang menggabungkan:
  - Project 1: Data Preparation & Machine Learning
  - Project 2: Time Series Forecasting (LSTM)

Cara menjalankan:
    pip install pandas numpy matplotlib scikit-learn openpyxl
    python main.py
"""

import tkinter as tk
from tkinter import ttk

from project1_tab import Project1Tab
from project2_tab import Project2Tab
from report_export import run_export_dialog


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sistem Analisis Data dan Deep Learning Terintegrasi")
        self.geometry("1300x800")
        self.minsize(1100, 650)

        self._build_sidebar_and_main()
        self._build_status_bar()

    def _build_sidebar_and_main(self):
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        # ================= Sidebar =================
        sidebar = ttk.Frame(container, width=200, style="Sidebar.TFrame")
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Sidebar.TFrame",
            background="#23272f"
        )

        style.configure(
            "SidebarTitle.TLabel",
            background="#23272f",
            foreground="white",
            font=("Segoe UI", 12, "bold")
        )

        style.configure(
            "SidebarBtn.TButton",
            font=("Segoe UI", 10)
        )

        title_lbl = ttk.Label(
            sidebar,
            text="MENU NAVIGASI",
            style="SidebarTitle.TLabel"
        )
        title_lbl.pack(pady=(20, 10), padx=10, anchor="w")

        # ================= Main Area =================
        self.main_area = ttk.Frame(container)
        self.main_area.pack(side="right", fill="both", expand=True)

        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self.project1_tab = Project1Tab(
            self.notebook,
            status_callback=self.set_status
        )

        self.project2_tab = Project2Tab(
            self.notebook,
            status_callback=self.set_status
        )

        self.notebook.add(
            self.project1_tab,
            text="Project 1 — Data Prep & ML"
        )

        self.notebook.add(
            self.project2_tab,
            text="Project 2 — Time Series Forecasting (LSTM)"
        )

        # ================= Tombol Navigasi =================

        ttk.Button(
            sidebar,
            text="Project 1\nData Preparation & ML",
            style="SidebarBtn.TButton",
            command=lambda: self.notebook.select(self.project1_tab)
        ).pack(fill="x", padx=10, pady=6)

        ttk.Button(
            sidebar,
            text="Project 2\nTime Series Forecasting\n(LSTM)",
            style="SidebarBtn.TButton",
            command=lambda: self.notebook.select(self.project2_tab)
        ).pack(fill="x", padx=10, pady=6)

        # ================= Separator =================

        sep = ttk.Separator(sidebar, orient="horizontal")
        sep.pack(fill="x", padx=10, pady=12)

        # ================= Export =================

        report_lbl = ttk.Label(
            sidebar,
            text="LAPORAN GABUNGAN",
            style="SidebarTitle.TLabel",
            font=("Segoe UI", 10, "bold")
        )
        report_lbl.pack(padx=10, anchor="w")

        ttk.Label(
            sidebar,
            text=(
                "Gabungkan SEMUA hasil\n"
                "Project 1 + Project 2\n"
                "ke dalam satu file"
            ),
            style="SidebarTitle.TLabel",
            font=("Segoe UI", 8)
        ).pack(padx=10, pady=(2, 8), anchor="w")

        ttk.Button(
            sidebar,
            text="Export Laporan Lengkap\n(1 file Excel, multi-sheet)",
            style="SidebarBtn.TButton",
            command=lambda: self.export_full_report("xlsx")
        ).pack(fill="x", padx=10, pady=4)

        ttk.Button(
            sidebar,
            text="Export Laporan Lengkap\n(1 file PDF)",
            style="SidebarBtn.TButton",
            command=lambda: self.export_full_report("pdf")
        ).pack(fill="x", padx=10, pady=4)

        # ================= About =================

        about_lbl = ttk.Label(
            sidebar,
            text=(
                "Aplikasi UAS:\n"
                "Sistem Analisis Data\n"
                "dan Deep Learning\n"
                "Terintegrasi\n\n"
                "Project 1: Klasifikasi\n"
                "(Random Forest &\n"
                "Logistic Regression)\n\n"
                "Project 2: Forecasting\n"
                "(LSTM from scratch)"
            ),
            style="SidebarTitle.TLabel",
            font=("Segoe UI", 9)
        )

        about_lbl.pack(
            side="bottom",
            padx=10,
            pady=20,
            anchor="w"
        )

    def _build_status_bar(self):
        self.status_var = tk.StringVar(
            value="Status: Siap. Pilih Project 1 atau Project 2 untuk mulai."
        )

        status_bar = ttk.Label(
            self,
            textvariable=self.status_var,
            relief="sunken",
            anchor="w"
        )

        status_bar.pack(
            side="bottom",
            fill="x"
        )

    def set_status(self, message):
        self.status_var.set(f"Status: {message}")
        self.update_idletasks()

    def export_full_report(self, fmt):
        run_export_dialog(
            self.project1_tab,
            self.project2_tab,
            fmt=fmt
        )


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()