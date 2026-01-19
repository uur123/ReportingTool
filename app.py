#!/usr/bin/env python3
import os
import io
import datetime
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re

# Required packages:
# pip install streamlit pandas fpdf2 Pillow
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF

# -----------------------------------------------------------------------------
ALL_MEASUREMENT_TECHNIQUES = [
    "ICP-OES", "XRF", "C, S, N Analysis", "F Analysis",
    "Metallic Al Analysis", "Metallic Si Analysis",
    "SEM-EDX", "Optical", "Porosity", "PSD", "BET",
    "XRD", "TGA", "Al Grain Size",
]

METHOD_HOURS = {
    "ICP-OES": 3.0,
    "XRF": 1.5,
    "XRD": 1.5,
    "SEM-EDX": 3.0,
    "Optical": 2.0,
    "Al Grain Size": 3.0,
    "Porosity": 1.0,
    "PSD": 2.0,
    "BET": 2.0,
    "TGA": 2.0,
    "F Analysis": 4.0,
    "C, S, N Analysis": 1.0,
    "Metallic Al Analysis": 1.0,
    "Metallic Si Analysis": 1.0,
}
REPORTING_HOURS = 1.0
CASTING_DEFECT_HOURS = 32.0

# ==============================================================================
# Helpers
# ==============================================================================
def pdf_safe_text(s: Any) -> str:
    """Normalize Unicode to latin-1-safe text for fpdf core fonts."""
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\u2014", "-")
         .replace("\u2013", "-")
         .replace("\u2212", "-")
         .replace("\u00a0", " ")
         .replace("\u2019", "'")
         .replace("\u2018", "'")
         .replace("\u201c", '"')
         .replace("\u201d", '"'))
    return s.encode("latin-1", "ignore").decode("latin-1")


def technique_output_hint(name: str) -> str:
    m = {
        "ICP-OES": "Elemental concentrations",
        "XRF": "Elemental analysis",
        "SEM-EDX": "Surface topography and chemistry",
        "Optical": "Microstructure images",
        "PSD": "Particle size distribution",
        "BET": "Surface area / porosity",
        "XRD": "Crystal Phase identification",
        "TGA": "Thermal behavior analysis",
        "C, S, N Analysis": "C/S/N content",
        "F Analysis": "Fluorine content",
        "Metallic Al Analysis": "Metallic Al %",
        "Metallic Si Analysis": "Metallic Si %",
        "Comparison Matrix": "Comparison tables",
    }
    return m.get(name, "-")


def compute_default_hours(technique_flags: Dict[str, bool], context: str) -> float:
    """
    Compute default effort hours:
    - Casting Defect Analysis -> fixed hours
    - otherwise: sum method hours + reporting if any selected
    """
    if context == "Casting Defect Analysis":
        return CASTING_DEFECT_HOURS
    base = 0.0
    any_selected = False
    for k, v in technique_flags.items():
        if v:
            any_selected = True
            base += float(METHOD_HOURS.get(k, 0.0))
    if any_selected:
        base += REPORTING_HOURS
    return round(base, 2)


def safe_key(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


def parse_excel_paste(text: str) -> Optional[pd.DataFrame]:
    if not text or not text.strip():
        return None
    s = text.strip()
    first_line = s.splitlines()[0]
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","
    try:
        df = pd.read_csv(io.StringIO(s), sep=sep, dtype=str, engine="python")
    except Exception:
        return None

    def normalize_number(val):
        if not isinstance(val, str):
            return val
        val = val.strip()
        val = re.sub(r"[^\d,.\-]", "", val)
        if not val:
            return val
        if "." in val and "," in val:
            if val.find(".") < val.find(","):
                val = val.replace(".", "").replace(",", ".")
            else:
                val = val.replace(",", "")
        elif "," in val:
            val = val.replace(",", ".")
        return val

    for col in df.columns:
        cleaned_col = df[col].apply(normalize_number)
        df[col] = pd.to_numeric(cleaned_col, errors="ignore")
    return df


# ==============================================================================
# PDF engine
# ==============================================================================
@dataclass
class Theme:
    primary: Tuple[int, int, int] = (26, 43, 60)
    header_fill: Tuple[int, int, int] = (230, 235, 240)
    zebra_fill: Tuple[int, int, int] = (250, 250, 250)
    text_dark: Tuple[int, int, int] = (30, 30, 30)
    text_gray: Tuple[int, int, int] = (100, 100, 100)
    border_gray: Tuple[int, int, int] = (200, 200, 200)
    font_main: str = "Helvetica"
    font_mono: str = "Courier"
    margin_mm: float = 25.0


class AnalyticalReportPDF(FPDF):
    def __init__(self, theme: Theme, meta: Dict[str, Any]):
        super().__init__()
        self.theme = theme
        self.meta = meta
        self.alias_nb_pages()
        self.set_auto_page_break(auto=True, margin=theme.margin_mm)
        self.set_margins(theme.margin_mm, theme.margin_mm, theme.margin_mm)

    def header(self):
        t = self.theme
        # single top stripe banner
        self.set_fill_color(220, 220, 220)
        self.rect(0, 0, self.w, 8, style="F")
        self.set_font(t.font_main, "B", 7)
        self.set_text_color(100, 100, 100)
        self.set_xy(0, 0)
        self.cell(self.w, 8, pdf_safe_text("INTERNAL USE ONLY - CONFIDENTIAL"), 0, 0, "C")

        # title + logo
        y_top = 10.0
        logo_h = 10.0
        self.set_font(t.font_main, "B", 13)
        self.set_text_color(*t.primary)
        self.set_xy(self.l_margin, y_top)
        self.cell(0, 10, pdf_safe_text("Technical Service Report"), 0, 0, "C")

        logo_path = self.meta.get("logo_path") or "static/logo2.png"
        if logo_path and os.path.exists(logo_path):
            try:
                x_logo = self.w - self.r_margin - 28
                self.image(logo_path, x=x_logo, y=y_top, h=logo_h)
            except Exception:
                pass

        # divider + reserve
        y_line = y_top + 11
        self.set_draw_color(*t.primary)
        self.set_line_width(0.5)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)
        self.set_y(y_line + 6.0)

    # subtitle, tables, images, kv boxes reused from earlier code (safe-ified)
    def subtitle(self, title: str):
        t = self.theme
        self.ln(2)
        self.set_font(t.font_main, "B", 10)
        self.set_text_color(*t.text_dark)
        self.cell(0, 6, pdf_safe_text(title), 0, 1, "L")
        y = self.get_y()
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.2)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2)
        self.set_text_color(*t.text_dark)

    def add_kv_box(self, items: List[Tuple[str, str]], title: str = "", cols: int = 2):
        t = self.theme
        if not items:
            return
        box_w = self.w - self.l_margin - self.r_margin
        x0 = self.l_margin
        y0 = self.get_y()
        line_h = 6.0
        pad = 2.5
        title_h = 6.5 if title else 0.0
        rows = (len(items) + cols - 1) // cols
        box_h = title_h + pad + rows * line_h + pad
        if self.get_y() + box_h > self.h - self.b_margin:
            self.add_page()
            x0 = self.l_margin
            y0 = self.get_y()
        self.set_fill_color(248, 249, 251)
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.2)
        self.rect(x0, y0, box_w, box_h, style="DF")
        cur_y = y0 + pad
        if title:
            self.set_xy(x0 + pad, cur_y)
            self.set_font(t.font_main, "B", 10)
            self.set_text_color(*t.primary)
            self.cell(box_w - 2 * pad, 6, pdf_safe_text(title), 0, 1, "L")
            cur_y = self.get_y()
        col_w = (box_w - 2 * pad) / cols
        label_w = col_w * 0.38
        value_w = col_w * 0.62
        self.set_font(t.font_main, "", 9)
        self.set_text_color(*t.text_dark)
        for r in range(rows):
            self.set_x(x0 + pad)
            y_row = cur_y + r * line_h
            self.set_y(y_row)
            for c in range(cols):
                i = r + c * rows
                if i >= len(items):
                    continue
                label, value = items[i]
                label = (label or "").strip()
                value = (value or "").strip()
                self.set_font(t.font_main, "B", 9)
                self.set_text_color(*t.text_gray)
                self.cell(label_w, line_h, pdf_safe_text(f"{label}:"), 0, 0, "L")
                self.set_font(t.font_main, "", 9)
                self.set_text_color(*t.text_dark)
                self.cell(value_w, line_h, pdf_safe_text(value), 0, 0, "L")
            self.ln(line_h)
        self.set_y(y0 + box_h + 4.0)

    def add_techniques_table(self, rows: List[Dict[str, str]], title: str = "Techniques included"):
        t = self.theme
        if not rows:
            return
        df = pd.DataFrame(rows, columns=["Technique", "Method/SOP", "Output"])
        est_h = 7 + min(len(df), 10) * 7 + 10
        if self.get_y() + est_h > self.h - self.b_margin:
            self.add_page()
        if title:
            self.set_font(t.font_main, "B", 10)
            self.set_text_color(*t.text_dark)
            self.cell(0, 6, pdf_safe_text(title), 0, 1, "L")
            self.ln(1)
        table_w = self.w - self.l_margin - self.r_margin
        w_tech = table_w * 0.25
        w_sop = table_w * 0.25
        w_out = table_w * 0.50
        widths = [w_tech, w_sop, w_out]
        self.set_font(t.font_main, "B", 9)
        self.set_fill_color(*t.header_fill)
        self.set_text_color(*t.primary)
        headers = ["Technique", "Method/SOP", "Output"]
        for h, w in zip(headers, widths):
            self.cell(w, 7, pdf_safe_text(h), 0, 0, "C", True)
        self.ln()
        self.set_font(t.font_main, "", 9)
        self.set_text_color(*t.text_dark)
        for i, row in df.iterrows():
            fill = (i % 2 == 1)
            if fill:
                self.set_fill_color(*t.zebra_fill)
            else:
                self.set_fill_color(255, 255, 255)
            vals = [str(row.get("Technique", "")), str(row.get("Method/SOP", "")), str(row.get("Output", ""))]
            for v, w in zip(vals, widths):
                v = pdf_safe_text(v.replace("\n", " ").strip())
                self.cell(w, 7, v, 0, 0, "C", True)
            self.ln()
        self.ln(2)

    def add_zebra_table(self, df: pd.DataFrame):
        if df is None or df.empty or len(df.columns) == 0:
            return
        t = self.theme
        col_width = (self.w - self.l_margin - self.r_margin) / len(df.columns)
        self.set_font(t.font_main, "B", 9)
        self.set_fill_color(*t.header_fill)
        self.set_text_color(*t.primary)
        for col in df.columns:
            self.cell(col_width, 7, pdf_safe_text(col), 0, 0, "C", True)
        self.ln()
        self.set_font(t.font_mono, "", 9)
        self.set_text_color(*t.text_dark)
        for i, row in df.iterrows():
            fill = (i % 2 == 1)
            self.set_fill_color(*t.zebra_fill) if fill else self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_width, 7, pdf_safe_text(val), 0, 0, "C", True)
            self.ln()
        self.ln(2)

    def add_framed_image(self, img_path: str, caption: str = ""):
        t = self.theme
        try:
            with Image.open(img_path) as pil_img:
                w_px, h_px = pil_img.size
                if w_px == 0:
                    return
                aspect = h_px / w_px
        except Exception:
            return
        max_w = self.w - self.l_margin - self.r_margin
        display_w = min(120, max_w)
        display_h = display_w * aspect
        if self.get_y() + display_h + 15 > self.h - t.margin_mm:
            self.add_page()
        x_pos = (self.w - display_w) / 2
        y_pos = self.get_y()
        self.image(img_path, x=x_pos, y=y_pos, w=display_w)
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.3)
        self.rect(x_pos, y_pos, display_w, display_h)
        self.set_y(y_pos + display_h + 2)
        if caption:
            self.set_font(t.font_main, "I", 8)
            self.set_text_color(*t.text_gray)
            self.cell(0, 5, pdf_safe_text(caption), 0, 1, "C")
        self.ln(5)

    def section_header_keep(self, title: str, first_block_mm: float = 20.0):
        ensure_space(self, needed_mm=12.0 + first_block_mm)
        self.section_header(title)

    def section_header(self, title: str):
        t = self.theme
        self.ln(5)
        self.set_font(t.font_main, "B", 12)
        self.set_fill_color(*t.primary)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, pdf_safe_text(f"  {title}"), 0, 1, "L", fill=True)
        self.ln(3)
        self.set_text_color(*t.text_dark)


# ==============================================================================
# Utilities used by main and PDF generation
# ==============================================================================
def ensure_space(pdf: FPDF, needed_mm: float):
    if pdf.get_y() + needed_mm > pdf.h - pdf.b_margin:
        pdf.add_page()


def pdf_operator_block(pdf: AnalyticalReportPDF, theme: Theme, operator: str, notes: str, title: str = "Analysis Notes"):
    operator = (operator or "").strip()
    notes = (notes or "").strip()
    if not operator and not notes:
        return
    pdf.set_x(pdf.l_margin)
    w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font(theme.font_main, "B", 9)
    pdf.set_text_color(*theme.text_dark)
    pdf.cell(w, 6, pdf_safe_text(f"{title}:"), 0, 1)
    pdf.set_font(theme.font_main, "", 9)
    if operator:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, pdf_safe_text(f"Operator: {operator}"))
    if notes:
        safe_notes = pdf_safe_text(notes)
        safe_notes = re.sub(r"\S{71,}", lambda m: " ".join(m.group(0)[i:i+70] for i in range(0, len(m.group(0)), 70)), safe_notes)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, pdf_safe_text(f"Notes: {safe_notes}"))
    pdf.ln(2)
    pdf.set_text_color(*theme.text_dark)


def save_upload(uploaded_file, force_ext: Optional[str] = None):
    if uploaded_file is None:
        return None
    try:
        suffix = force_ext or os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception:
        return None


def save_upload_as_jpg(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    try:
        img = Image.open(uploaded_file)
        if img.mode != "RGB":
            img = img.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name, format="JPEG", quality=95)
            return tmp.name
    except Exception:
        return None


def cleanup_file(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass


def estimate_table_height_mm(df: Optional[pd.DataFrame], row_h: float = 7.0, header_h: float = 7.0, min_rows: int = 2) -> float:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    n = max(min_rows, min(len(df), 12))
    return header_h + n * row_h + 2.0


# Comparison matrix small table editor (keeps state in session)
def comparison_matrix_editor(*, default_df: pd.DataFrame, editor_key: str, label: str) -> pd.DataFrame:
    store_key = f"{editor_key}__df"
    df_current = st.session_state.get(store_key)
    if not isinstance(df_current, pd.DataFrame) or df_current.empty:
        df_current = default_df.copy()

    c1, c2, c3 = st.columns([1.2, 1.2, 3.6])
    if c1.button("➕ Add column", key=f"{editor_key}__add_col"):
        base = "New column"
        idx = 1
        new_col = f"{base} {idx}"
        existing = set(map(str, df_current.columns))
        while new_col in existing:
            idx += 1
            new_col = f"{base} {idx}"
        df_current = df_current.copy()
        df_current[new_col] = ""

    with c2:
        try:
            with st.popover("➖ Remove column", use_container_width=True):
                st.caption("Select one or more columns to remove (at least 1 column must remain).")
                cols_now = list(map(str, df_current.columns))
                to_remove = st.multiselect("Columns to remove", options=cols_now, key=f"{editor_key}__rm_cols")
                if st.button("Remove selected", key=f"{editor_key}__rm_cols_btn"):
                    if not to_remove:
                        st.info("No columns selected.")
                    else:
                        remaining = [c for c in cols_now if c not in set(to_remove)]
                        if len(remaining) < 1:
                            st.warning("Cannot remove all columns. Keep at least one column.")
                        else:
                            df_current = df_current.copy()
                            df_current = df_current[remaining]
        except Exception:
            with st.expander("➖ Remove column (fallback)", expanded=False):
                st.caption("Select columns to remove")
                cols_now = list(map(str, df_current.columns))
                to_remove = st.multiselect("Columns to remove (expander)", options=cols_now, key=f"{editor_key}__rm_cols_fallback")
                if st.button("Remove selected (expander)", key=f"{editor_key}__rm_cols_btn_fallback"):
                    if not to_remove:
                        st.info("No columns selected.")
                    else:
                        remaining = [c for c in cols_now if c not in set(to_remove)]
                        if len(remaining) < 1:
                            st.warning("Cannot remove all columns. Keep at least one column.")
                        else:
                            df_current = df_current.copy()
                            df_current = df_current[remaining]

    # Import
    imported = None
    try:
        pop = st.popover("📋 Import from Excel", use_container_width=False)
        with pop:
            txt = st.text_area("Paste copied Excel range here", key=f"{editor_key}__import_txt", height=120)
            if st.button("Import", key=f"{editor_key}__import_btn"):
                imported = parse_excel_paste(txt)
    except Exception:
        with c3:
            with st.expander("📋 Import from Excel", expanded=False):
                txt = st.text_area("Paste copied Excel range here", key=f"{editor_key}__import_txt", height=120)
                if st.button("Import", key=f"{editor_key}__import_btn"):
                    imported = parse_excel_paste(txt)

    if imported is not None:
        if isinstance(imported, pd.DataFrame) and not imported.empty:
            df_current = imported
        else:
            st.warning("Import failed. Copy a rectangular range (including headers) and try again.")

    st.markdown("**Rename columns**")
    cols_now = list(df_current.columns)
    wrap = min(max(len(cols_now), 1), 6)
    rename_cols = st.columns(wrap)
    new_names = []
    for i, col in enumerate(cols_now):
        col_container = rename_cols[i % wrap]
        new_name = col_container.text_input(label=f"col_{i}", value=str(col), key=f"{editor_key}__colname_{i}", label_visibility="collapsed")
        new_names.append(new_name.strip() or str(col))
    if new_names != list(map(str, df_current.columns)):
        seen = {}
        fixed = []
        for name in new_names:
            if name not in seen:
                seen[name] = 1
                fixed.append(name)
            else:
                seen[name] += 1
                fixed.append(f"{name} ({seen[name]})")
        df_current = df_current.copy()
        df_current.columns = fixed

    df_out = st.data_editor(df_current, num_rows="dynamic", use_container_width=True, key=editor_key)
    st.session_state[store_key] = df_out
    return df_out


def editor_with_excel_paste(*, default_df: pd.DataFrame, editor_key: str, paste_key: str, label: str = "Paste from Excel (optional)", apply_button_text: str = "Apply paste to table", help_text: str = "Copy a cell range in Excel/Sheets and paste here. Tabs/newlines are supported.", height: int = 140, use_expander: bool = True) -> pd.DataFrame:
    store_key = f"{editor_key}__df"

    def paste_ui():
        st.markdown(f"**{label}**")
        st.caption(help_text)
        txt = st.text_area("Paste area", key=paste_key, height=height)
        if st.button(apply_button_text, key=f"{editor_key}__apply"):
            df_new = parse_excel_paste(txt)
            if df_new is None:
                st.warning("Could not parse pasted data. Copy a rectangular range (rows/columns) and paste again.")
            else:
                st.session_state[store_key] = df_new

    if use_expander:
        with st.expander(label, expanded=False):
            paste_ui()
    else:
        paste_ui()

    df_current = st.session_state.get(store_key)
    if not isinstance(df_current, pd.DataFrame) or df_current.empty:
        df_current = default_df

    df_out = st.data_editor(df_current, num_rows="dynamic", use_container_width=True, key=editor_key)
    st.session_state[store_key] = df_out
    return df_out


# ==============================================================================
# Streamlit UI (main)
# ==============================================================================
def main():
    st.set_page_config(page_title="EN-Report", layout="wide", page_icon="🔬", initial_sidebar_state="collapsed")
    if "report_data" not in st.session_state:
        st.session_state.report_data = {}

    st.markdown("<style>.stApp { background-color: #f8f9fa; }</style>", unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        lab_name = st.text_input("Lab Name", "Global R&D Analytical Service Lab")
        lab_addr = st.text_input("Address", "Pantheon 30, Enschede, NL")
        lab_no = st.text_input("Phone number", "+31 600504030")
        lab_mail = st.text_input("Mail address", "EN-Analytical@vesuvius.com")
        logo = st.file_uploader("Lab Logo", type=["png", "jpg"])
        st.divider()
        st.info("Status: Ready")

    st.title("🔬 Scientific Analysis Reporting Tool")
    st.markdown("---")

    # 1. Sample traceability
    st.subheader("1. Sample Traceability")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        project_title = c1.text_input("Project Title", "Benchmarking of HD1")
        requester = c2.text_input("Requester Name", "John Doe")
        context = c3.selectbox("Context", ["R&D project", "Operational Support", "Casting Defect Analysis"])
        sample_id = c4.text_input("TSR No (Project ID)", "", help="Mandatory: this is used as Project ID")

    with st.container():
        c1, c2, c3 = st.columns(3)
        rx_date = c1.date_input("Date Received")
        temp = c2.number_input("Temp (°C)", value=21.0)
        hum = c3.number_input("Humidity (%RH)", value=45.0)
        st.markdown("#### Sample condition / requester notes")
        sample_condition = st.text_area("Sample as received (packaging/condition) & requester comments", placeholder="e.g., Sample received in sealed bag...", height=90)

    st.markdown("#### Sample reception photo")
    sample_photo = st.file_uploader("Upload sample photo (as received)", type=["jpg", "jpeg", "png", "tif", "tiff"], key="sample_photo")
    st.divider()

    # 2. Techniques selection
    st.subheader("2. Select Analytical Techniques")
    def add_operator_fields(parent, key_prefix: str):
        parent.markdown("##### Operator traceability")
        c0, c1, c2 = parent.columns([1, 1, 2])
        method_ref = c0.text_input("Method/SOP ID", key=f"{key_prefix}_methodref", placeholder="e.g. WI-ICP-014")
        operator = c1.text_input("Operator", key=f"{key_prefix}_operator")
        op_notes = c2.text_area("Operator notes (short)", key=f"{key_prefix}_opnotes", height=70)
        return method_ref, operator, op_notes

    col1, col2, col3, col4 = st.columns(4)
    technique_flags = {}
    with col1:
        technique_flags["ICP-OES"] = st.checkbox("ICP-OES (Chemistry)")
        technique_flags["XRF"] = st.checkbox("XRF (Composition)")
        technique_flags["C, S, N Analysis"] = st.checkbox("C, S, N Analysis")
        technique_flags["Al Grain Size"] = st.checkbox("Al GSA")
    with col2:
        technique_flags["SEM-EDX"] = st.checkbox("SEM-EDX (Microscopy)")
        technique_flags["Optical"] = st.checkbox("Optical Microscopy")
        technique_flags["Metallic Al Analysis"] = st.checkbox("Metallic Al")
        technique_flags["Porosity"] = st.checkbox("Porosity")
    with col3:
        technique_flags["PSD"] = st.checkbox("Particle Size (PSD)")
        technique_flags["BET"] = st.checkbox("BET Surface Area")
        technique_flags["Metallic Si Analysis"] = st.checkbox("Metallic Si")
    with col4:
        technique_flags["XRD"] = st.checkbox("XRD (Phase ID)")
        technique_flags["TGA"] = st.checkbox("TGA-MS (Thermal)")
        technique_flags["F Analysis"] = st.checkbox("F Analysis")
        technique_flags["Comparison Matrix"] = st.checkbox("Comparison Matrix")

    # Effort & cost
    st.subheader("3. Effort & Cost (optional)")
    default_hours = compute_default_hours(technique_flags, context)
    if "est_hours" not in st.session_state:
        st.session_state.est_hours = default_hours
    if "hourly_rate" not in st.session_state:
        st.session_state.hourly_rate = 0.0
    if "effort_include" not in st.session_state:
        st.session_state.effort_include = True

    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    with c0:
        st.session_state.effort_include = st.checkbox("Include in report", value=st.session_state.effort_include, key="effort_include_chk")
    with c1:
        if st.button("Recalculate hours", key="recalc_hours_btn"):
            st.session_state.est_hours = default_hours
    with c2:
        est_hours = st.number_input("Estimated work hours", min_value=0.0, value=float(st.session_state.est_hours), step=0.5, key="est_hours_input")
        st.session_state.est_hours = float(est_hours)
    with c3:
        hourly_rate = st.number_input("Hourly cost (€ / h)", min_value=0.0, value=float(st.session_state.hourly_rate), step=5.0, key="hourly_rate_input")
        st.session_state.hourly_rate = float(hourly_rate)

    total_cost = round(st.session_state.est_hours * st.session_state.hourly_rate, 2)
    st.markdown(f"**Auto-calculated default:** {default_hours} h (includes {REPORTING_HOURS} h reporting)  \n**Estimated project cost:** € {total_cost:,.2f}")
    st.divider()

    report_data = st.session_state.report_data
    report_data.clear()

    # ---- Data entry blocks (unchanged structure; populate report_data) ----
    if technique_flags.get("ICP-OES"):
        with st.expander("🧪 ICP-OES Data", expanded=True):
            c_a, c_b = st.columns([1, 3])
            mass = c_a.number_input("Mass (g)", 0.100, format="%.3f")
            volume = c_a.number_input("Volume (mL)", 500, format="%.3f")
            df_in = pd.DataFrame([{"Element": "Fe", "Result (mg/kg)": 0.0}, {"Element": "Si", "Result (mg/kg)": 0.0}])
            df_icp = c_b.data_editor(df_in, num_rows="dynamic", use_container_width=True, key="icp_df")
            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="ICP_OES")
            report_data["ICP-OES"] = {"meta": f"Mass: {mass}g | Volume: {volume} mL", "table": df_icp, "method_ref": method_ref, "operator": operator, "op_notes": op_notes}

    # ... (other technique blocks are identical to previous code; omitted here in the comment for brevity)
    # For full working app please keep all technique blocks as in previous versions (XRF, SEM-EDX, Optical, PSD, XRD, TGA, BET, Porosity,
    # Al Grain Size, Comparison Matrix and Chemical methods). They are included above in prior responses and the comparison_matrix_editor is used.

    # (To keep this listing concise I assume those blocks are included unchanged here.)

    st.divider()
    # Conclusions
    st.subheader("4. Conclusions")
    conclusion_placeholder = "No specific conclusions provided. Standard analytical procedures applied; see results and tables above."
    summary = st.text_area("Final Remarks", height=100, placeholder=conclusion_placeholder)

    st.divider()
    # Sign-off
    st.subheader("5. Report Sign-off")
    c1, c2 = st.columns(2)
    with c1:
        reported_by = st.text_input("Reported by (name)", "", help="Mandatory")
        reported_title = st.text_input("Reported by (title)", "Analyst")
    with c2:
        reviewed_by = st.text_input("Reviewed by (name)", "", help="Mandatory")
        reviewed_title = st.text_input("Reviewed by (title)", "Reviewer")

    st.divider()
    rev_number = st.number_input("Revision number", min_value=0, value=0, step=1)

    # Generate PDF
    if st.button("🚀 Generate PDF Report", type="primary"):
        missing = []
        if not sample_id.strip():
            missing.append("TSR No (Project ID)")
        if not reported_by.strip():
            missing.append("Reported by (name)")
        if not reviewed_by.strip():
            missing.append("Reviewed by (name)")
        if missing:
            st.error(f"Please complete mandatory fields: {', '.join(missing)}")
            return

        # Build PDF
        theme = Theme()
        meta = {
            "lab_name": lab_name, "lab_addr": lab_addr, "lab_no": lab_no, "lab_mail": lab_mail,
            "project_title": project_title, "requester": requester,
            "report_date": str(datetime.date.today()), "logo_path": save_upload(logo),
            "project_id": sample_id, "revision": int(rev_number)
        }
        pdf = AnalyticalReportPDF(theme, meta)
        pdf.add_page()

        # Determine techniques_included (exclude "Comparison Matrix" as a listed technique)
        technique_order = ["ICP-OES", "XRF", "C, S, N Analysis", "F Analysis", "Metallic Al Analysis", "Metallic Si Analysis", "SEM-EDX", "Optical", "Porosity", "PSD", "BET", "XRD", "TGA", "Al Grain Size"]
        included_set = set(k for k, v in technique_flags.items() if v and k != "Comparison Matrix")

        cmp_block = report_data.get("Comparison Matrix") or {}
        cmp_tables = cmp_block.get("tables") or []
        cmp_tech_names = set()
        cmp_method_map: Dict[str, str] = {}
        for t in cmp_tables:
            tname = (t.get("technique") or "").strip()
            if tname:
                cmp_tech_names.add(tname)
                if t.get("method_ref"):
                    cmp_method_map.setdefault(tname, t.get("method_ref") or "")

        for tname in cmp_tech_names:
            included_set.add(tname)

        techniques_included = [t for t in technique_order if t in included_set]

        # Info block
        pdf.section_header_keep(f"1. Project: {context}", first_block_mm=70.0)
        metadata_items = [
            ("Project/Report ID", sample_id),
            ("Requester", requester),
            ("Sample received", str(rx_date)),
            ("Project title", project_title),
            ("Report date", str(datetime.date.today())),
            ("Revision", str(int(rev_number))),
            ("Environment", f"{temp}°C / {hum}% RH"),
        ]
        pdf.add_kv_box(metadata_items, title="Report and Sample Information", cols=2)
        pdf.set_font(theme.font_main, "", 10)
        pdf.set_text_color(*theme.text_dark)
        pdf.ln(2)

        sample_photo_path = save_upload_as_jpg(sample_photo) if sample_photo else None
        if sample_photo_path and os.path.exists(sample_photo_path):
            pdf.subtitle("Sample photo (as received)")
            pdf.add_framed_image(sample_photo_path, caption="Sample as received")
            cleanup_file(sample_photo_path)

        if sample_condition.strip():
            pdf.subtitle("Requester notes and Sample condition")
            pdf.set_font(theme.font_main, "", 10)
            pdf.set_text_color(*theme.text_dark)
            pdf.multi_cell(0, 6, pdf_safe_text(sample_condition.strip()))
            pdf.ln(1)

        # Techniques included
        pdf.subtitle("1.1 Techniques included")
        tech_rows = []
        for tech in techniques_included:
            d = report_data.get(tech, {}) or {}
            method_sop = (d.get("method_ref") or "").strip() or cmp_method_map.get(tech, "")
            tech_rows.append({"Technique": tech, "Method/SOP": method_sop, "Output": technique_output_hint(tech)})
        pdf.add_techniques_table(tech_rows, title="")

        pdf.ln(2)

        # Analysis results - skip printing a technique section if technique only appears in comparison tables and has no own data
        pdf.section_header_keep("2. Analysis results", first_block_mm=35.0)
        result_index = 1
        for tech in techniques_included:
            data = report_data.get(tech) or {}
            has_own_data = isinstance(data, dict) and (
                (isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty) or bool(data.get("images")) or bool((data.get("meta") or "").strip()) or bool((data.get("operator") or "").strip()) or bool((data.get("op_notes") or "").strip())
            )
            if (tech in cmp_tech_names) and not has_own_data:
                # skip here; the comparison table(s) will be printed later under the technique title
                continue

            first_block = 18.0
            if isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty:
                first_block = 28.0
            elif data.get("images"):
                first_block = 55.0
            elif data.get("meta"):
                first_block = 22.0
            ensure_space(pdf, 10.0 + first_block)

            pdf.subtitle(f"2.{result_index} {tech}")
            result_index += 1

            if data.get("meta"):
                pdf.set_font(theme.font_mono, "", 8)
                pdf.set_text_color(*theme.text_gray)
                pdf.multi_cell(0, 5, pdf_safe_text(str(data["meta"])))
                pdf.ln(1)
                pdf.set_text_color(*theme.text_dark)

            pdf_operator_block(pdf, theme, data.get("operator", ""), data.get("op_notes", ""))

            if isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty:
                pdf.add_zebra_table(data["table"])

            if data.get("images"):
                caption_list = (data.get("captions") or "").splitlines()
                tmp_paths = []
                try:
                    for i, img_file in enumerate(data["images"]):
                        tmp_path = save_upload_as_jpg(img_file)
                        if not tmp_path:
                            continue
                        tmp_paths.append(tmp_path)
                        cap = caption_list[i] if i < len(caption_list) and caption_list[i].strip() else f"Figure {i+1}"
                        pdf.add_framed_image(tmp_path, pdf_safe_text(cap))
                finally:
                    for p in tmp_paths:
                        cleanup_file(p)

        # Print comparison matrix tables separately (with method/SOP if provided)
        cmp_data = report_data.get("Comparison Matrix") or {}
        cmp_tables = cmp_data.get("tables") or []
        for block in cmp_tables:
            df_cmp = block.get("table") if isinstance(block.get("table"), pd.DataFrame) else None
            if df_cmp is None or df_cmp.empty:
                continue
            tech_used = (block.get("technique") or "").strip() or "Comparison"
            table_title = (block.get("title") or "").strip()
            method_ref_table = (block.get("method_ref") or "").strip()

            ensure_space(pdf, 10.0 + max(22.0, estimate_table_height_mm(df_cmp)))
            pdf.subtitle(f"2.{result_index} {tech_used}")
            result_index += 1

            if method_ref_table:
                pdf.set_font(theme.font_mono, "", 8)
                pdf.set_text_color(*theme.text_gray)
                pdf.cell(0, 5, pdf_safe_text(f"Method/SOP: {method_ref_table}"), 0, 1, "L")
                pdf.set_text_color(*theme.text_dark)

            if table_title:
                pdf.set_font(theme.font_main, "I", 9)
                pdf.set_text_color(*theme.text_gray)
                pdf.cell(0, 5, pdf_safe_text(table_title), 0, 1, "L")
                pdf.set_text_color(*theme.text_dark)

            pdf_operator_block(pdf, theme, cmp_data.get("operator", ""), cmp_data.get("op_notes", ""), title="Comparison Matrix traceability")
            pdf.add_zebra_table(df_cmp)
            pdf.ln(1)

        # Chemical group (if any)
        chem_items = [(k, v) for k, v in report_data.items() if v.get("group") == "Chemical analysis"]
        if chem_items:
            ensure_space(pdf, 18.0)
            pdf.subtitle(f"2.{result_index} Chemical analysis")
            result_index += 1
            sub_i = 1
            for method_name, data in chem_items:
                df = data.get("table") if isinstance(data.get("table"), pd.DataFrame) else None
                keep_method_with_first_content(pdf, df=df)
                pdf.set_font(theme.font_main, "B", 10)
                pdf.set_text_color(*theme.text_dark)
                pdf.cell(0, 6, pdf_safe_text(f"2.{result_index-1}.{sub_i} {method_name}"), 0, 1, "L")
                sub_i += 1
                if data.get("meta"):
                    pdf.set_font(theme.font_mono, "", 8)
                    pdf.set_text_color(*theme.text_gray)
                    pdf.multi_cell(0, 5, pdf_safe_text(str(data["meta"])))
                    pdf.set_text_color(*theme.text_dark)
                pdf_operator_block(pdf, theme, data.get("operator", ""), data.get("op_notes", ""), title="Method traceability")
                if df is not None:
                    pdf.add_zebra_table(df)

        # Conclusions
        pdf.section_header_keep("3. Conclusions / summary", first_block_mm=30.0)
        pdf.set_font(theme.font_main, "", 10)
        pdf.set_text_color(*theme.text_dark)
        conclusion_text = summary.strip() or conclusion_placeholder
        pdf.multi_cell(0, 6, pdf_safe_text(conclusion_text))

        # Sign-off
        pdf.section_header_keep("4. Sign-off", first_block_mm=25.0)
        pdf.add_signoff_two_columns(left_label="Reported by", left_name=reported_by, left_title=reported_title, right_label="Reviewed by", right_name=reviewed_by, right_title=reviewed_title)

        # Output
        try:
            out = pdf.output(dest="S")
            pdf_bytes = out.encode("latin-1", "ignore") if isinstance(out, str) else bytes(out)
            cleanup_file(meta["logo_path"])
            st.success("✅ Report Generated! Download below.")
            st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name=f"{sample_id}.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Error generating PDF: {e}")


if __name__ == "__main__":
    main()
