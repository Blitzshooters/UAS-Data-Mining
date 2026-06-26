"""Utility to export a pandas DataFrame to CSV, Excel, or PDF."""

from tkinter import filedialog, messagebox


def export_dataframe(df, fmt, title="Hasil"):
    if fmt == "csv":
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV", "*.csv")])
        if not path:
            return
        df.to_csv(path, index=False)
        messagebox.showinfo("Export", f"Berhasil disimpan ke:\n{path}")

    elif fmt == "xlsx":
        path = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                             filetypes=[("Excel", "*.xlsx")])
        if not path:
            return
        try:
            df.to_excel(path, index=False)
            messagebox.showinfo("Export", f"Berhasil disimpan ke:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal export Excel (perlu openpyxl):\n{e}")

    elif fmt == "pdf":
        path = filedialog.asksaveasfilename(defaultextension=".pdf",
                                             filetypes=[("PDF", "*.pdf")])
        if not path:
            return
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.pyplot as plt

            with PdfPages(path) as pdf:
                n_rows_per_page = 25
                n_pages = max(1, (len(df) + n_rows_per_page - 1) // n_rows_per_page)
                for p in range(n_pages):
                    chunk = df.iloc[p * n_rows_per_page:(p + 1) * n_rows_per_page]
                    fig, ax = plt.subplots(figsize=(11.7, 8.3))
                    ax.axis("off")
                    ax.set_title(f"{title} (halaman {p + 1}/{n_pages})")
                    tbl = ax.table(cellText=chunk.values, colLabels=chunk.columns,
                                    loc="center", cellLoc="center")
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(6)
                    tbl.scale(1, 1.2)
                    pdf.savefig(fig)
                    plt.close(fig)
            messagebox.showinfo("Export", f"Berhasil disimpan ke:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Gagal export PDF:\n{e}")
