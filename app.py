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

# ----------------------------
# Constants / Defaults
# ----------------------------
ALL_MEASUREMENT_TECHNIQUES = [
    "ICP-OES", "XRF", "C, S, N Analysis", "F Analysis",
    "Metallic Al Analysis", "Metallic Si Analysis",
    "SEM-EDX", "Optical", "Porosity", "PSD", "BET",
    "XRD", "TGA", "Al Grain Size",
]

METHOD_HOURS = {
    "ICP-OES": 3.0, "XRF": 1.5, "XRD": 1.5, "SEM-EDX": 3.0,
    "Optical": 2.0, "Al Grain Size": 3.0, "Porosity": 1.0,
    "PSD": 2.0, "BET": 2.0, "TGA": 2.0, "F Analysis": 4.0,
    "C, S, N Analysis": 1.0, "Metallic Al Analysis": 1.0,
    "Metallic Si Analysis": 1.0,
}
REPORTING_HOURS = 1.0
CASTING_DEFECT_HOURS = 32.0

# ----------------------------
# Small helpers
# ----------------------------
def pdf_safe_text(s: Any) -> str:
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


# ----------------------------
# PDF engine
# ----------------------------
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
        # Top confidentiality stripe (single)
        self.set_fill_color(220, 220, 220)
        self.rect(0, 0, self.w, 8, style="F")
        self.set_font(t.font_main, "B", 7)
        self.set_text_color(100, 100, 100)
        self.set_xy(0, 0)
        self.cell(self.w, 8, pdf_safe_text("INTERNAL USE ONLY - CONFIDENTIAL"), 0, 0, "C")

        # Title + optional logo
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

        y_line = y_top + 11
        self.set_draw_color(*t.primary)
        self.set_line_width(0.5)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)

        self.set_y(y_line + 6.0)

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


# ----------------------------
# Utilities used by main
# ----------------------------
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


def estimate_table_height_mm(df: Optional[pd.DataFrame], row_h: float = 7.0, header_h: float = 7.0, min_rows: int = 2) -> float:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    n = max(min_rows, min(len(df), 12))
    return header_h + n * row_h + 2.0


# ----------------------------
# Small in-UI helpers for analysis blocks
# ----------------------------
def _ensure_blocks_state():
    if "analysis_blocks_n" not in st.session_state:
        st.session_state.analysis_blocks_n = 0
    # initialize per-block storage as lists/dicts when blocks exist
    for i in range(st.session_state.analysis_blocks_n):
        tables_key = f"block_{i}__tables"
        if tables_key not in st.session_state:
            st.session_state[tables_key] = []


def add_analysis_block():
    st.session_state.analysis_blocks_n = st.session_state.get("analysis_blocks_n", 0) + 1


def remove_analysis_block():
    n = st.session_state.get("analysis_blocks_n", 0)
    if n <= 0:
        return
    # cleanup block-associated session keys
    i = n - 1
    keys = [f"block_{i}__tables", f"block_{i}__images", f"block_{i}__captions",
            f"block_{i}__tech", f"block_{i}__tech_other", f"block_{i}__method_ref",
            f"block_{i}__operator", f"block_{i}__opnotes"]
    for k in keys:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.analysis_blocks_n = n - 1


def add_table_to_block(i: int):
    key = f"block_{i}__tables"
    lst = st.session_state.get(key, [])
    # default empty table with one row
    new = pd.DataFrame([{"Parameter": "", "Value": ""}])
    lst = list(lst)  # copy
    lst.append(new)
    st.session_state[key] = lst


def remove_table_from_block(i: int):
    key = f"block_{i}__tables"
    lst = st.session_state.get(key, [])
    if lst:
        lst = list(lst)
        lst.pop()
        st.session_state[key] = lst


# ----------------------------
# Main UI
# ----------------------------
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

    # 1. Header data
    st.subheader("1. Sample Traceability")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        project_title = c1.text_input("Project Title", "Benchmarking of HD1")
        requester = c2.text_input("Requester Name", "John Doe")
        context = c3.selectbox("Context", ["R&D project", "Operational Support", "Casting Defect Analysis"])
        sample_id = c4.text_input("TSR No (Project ID)", "", help="Mandatory: used as Project ID")

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

    # 2. Analyses blocks (new simplified workflow)
    st.subheader("2. Analyses (Add blocks)")
    _ensure_blocks_state()

    c_add, c_rm = st.columns([1, 1])
    with c_add:
        if st.button("➕ Add analysis block", key="add_block"):
            add_analysis_block()
    with c_rm:
        if st.button("➖ Remove last block", key="remove_block"):
            remove_analysis_block()

    n = st.session_state.analysis_blocks_n
    if n == 0:
        st.info("No analysis blocks yet. Click 'Add analysis block' to create one. Each block can contain multiple tables and images.")
    for i in range(n):
        st.markdown(f"### Analysis Block {i+1}")
        c1, c2, c3 = st.columns([2, 2, 2])
        tech = c1.selectbox("Technique", options=ALL_MEASUREMENT_TECHNIQUES + ["Other"], key=f"block_{i}__tech")
        tech_other = ""
        if st.session_state.get(f"block_{i}__tech") == "Other":
            tech_other = c1.text_input("Technique name", value="", key=f"block_{i}__tech_other")
        method_ref = c2.text_input("Method / SOP (optional)", value="", key=f"block_{i}__method_ref", placeholder="e.g. WI-ICP-014")
        # operator
        operator = c3.text_input("Operator", key=f"block_{i}__operator")
        op_notes = st.text_area("Operator notes (short)", key=f"block_{i}__opnotes", height=80)

        # Tables for this block (multiple)
        st.markdown("Tables")
        tb_col1, tb_col2 = st.columns([1, 1])
        with tb_col1:
            if st.button("➕ Add table to block", key=f"block_{i}__add_table"):
                add_table_to_block(i)
        with tb_col2:
            if st.button("➖ Remove last table", key=f"block_{i}__remove_table"):
                remove_table_from_block(i)

        tables_key = f"block_{i}__tables"
        if tables_key not in st.session_state:
            st.session_state[tables_key] = []
        tables = st.session_state[tables_key]

        for j, tbl in enumerate(tables):
            st.markdown(f"Table {j+1}")
            # allow paste import editor as an option
            with st.expander(f"Edit / Paste Table {j+1}", expanded=True):
                # provide a small paste box to import into this table
                paste_txt_key = f"block_{i}__table_{j}__paste"
                txt = st.text_area("Paste from Excel (optional)", key=paste_txt_key, height=80)
                if st.button("Apply paste to this table", key=f"{paste_txt_key}__apply"):
                    parsed = parse_excel_paste(st.session_state.get(paste_txt_key, ""))
                    if parsed is not None and not parsed.empty:
                        st.session_state[tables_key][j] = parsed
                    else:
                        st.warning("Couldn't parse pasted content.")
                # editable table UI
                df_out = st.data_editor(st.session_state[tables_key][j], use_container_width=True, num_rows="dynamic", key=f"block_{i}_table_{j}")
                st.session_state[tables_key][j] = df_out

        # Images for this block
        st.markdown("Images (optional)")
        img_files = st.file_uploader(f"Upload images for block {i+1}", type=["jpg", "jpeg", "png", "tif", "tiff"], accept_multiple_files=True, key=f"block_{i}__images")
        st.session_state[f"block_{i}__images"] = img_files
        captions = st.text_area("Image captions (one per line, optional)", key=f"block_{i}__captions", height=80)
        st.divider()

    # 3. Effort & cost (kept simple)
    st.subheader("3. Effort & Cost (optional)")
    # compute default hours from blocks selected (sum of METHOD_HOURS for blocks techniques)
    def compute_default_hours_from_blocks():
        if st.session_state.analysis_blocks_n == 0:
            return 0.0
        base = 0.0
        any_selected = False
        for i in range(st.session_state.analysis_blocks_n):
            tech = st.session_state.get(f"block_{i}__tech")
            if tech == "Other":
                # don't add unknown technique hours
                continue
            if tech:
                any_selected = True
                base += float(METHOD_HOURS.get(tech, 0.0))
        if any_selected:
            base += REPORTING_HOURS
        return round(base, 2)

    default_hours = compute_default_hours_from_blocks()
    if "est_hours" not in st.session_state:
        st.session_state.est_hours = default_hours
    if "hourly_rate" not in st.session_state:
        st.session_state.hourly_rate = 0.0
    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])
    with c0:
        st.session_state.effort_include = st.checkbox("Include in report", value=st.session_state.get("effort_include", True), key="effort_include_chk")
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
    st.markdown(f"**Auto-calculated default:** {default_hours} h  \n**Estimated project cost:** € {total_cost:,.2f}")
    st.divider()

    # Conclusions and sign-off
    st.subheader("4. Conclusions")
    conclusion_placeholder = "No specific conclusions provided. Standard analytical procedures applied; see results and tables above."
    summary = st.text_area("Final Remarks", height=100, placeholder=conclusion_placeholder)

    st.divider()
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
        else:
            # Build metadata
            meta = {
                "lab_name": st.session_state.get("lab_name", "") or lab_name,
                "lab_addr": lab_addr,
                "lab_no": lab_no,
                "lab_mail": lab_mail,
                "project_title": project_title,
                "requester": requester,
                "report_date": str(datetime.date.today()),
                "logo_path": save_upload(logo),
                "project_id": sample_id,
                "revision": int(rev_number),
            }

            theme = Theme()
            pdf = AnalyticalReportPDF(theme, meta)
            pdf.add_page()

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

            # Sample photo
            sample_photo_path = save_upload_as_jpg(sample_photo) if sample_photo else None
            if sample_photo_path and os.path.exists(sample_photo_path):
                pdf.subtitle("Sample photo (as received)")
                pdf.add_framed_image(sample_photo_path, caption="Sample as received")
                cleanup_file(sample_photo_path)

            # Sample condition notes
            if sample_condition.strip():
                pdf.subtitle("Requester notes and Sample condition")
                pdf.set_font(theme.font_main, "", 10)
                pdf.set_text_color(*theme.text_dark)
                pdf.multi_cell(0, 6, pdf_safe_text(sample_condition.strip()))
                pdf.ln(1)

            # Build a list of technique rows for "Techniques included" (collect techniques from blocks)
            included_set = []
            for i in range(st.session_state.analysis_blocks_n):
                tech = st.session_state.get(f"block_{i}__tech")
                tech_name = (st.session_state.get(f"block_{i}__tech_other") or "").strip() if tech == "Other" else tech
                if tech_name and tech_name not in included_set:
                    included_set.append(tech_name)

            tech_rows = []
            for t in included_set:
                # prefer a block-level method_ref if any block for this technique provided it
                method_sop = ""
                for i in range(st.session_state.analysis_blocks_n):
                    tech_i = st.session_state.get(f"block_{i}__tech")
                    tech_i_name = (st.session_state.get(f"block_{i}__tech_other") or "").strip() if tech_i == "Other" else tech_i
                    if tech_i_name == t:
                        method_sop = st.session_state.get(f"block_{i}__method_ref", "") or method_sop
                tech_rows.append({"Technique": t, "Method/SOP": method_sop or "", "Output": "See tables & images"})

            pdf.subtitle("1.1 Techniques included")
            pdf.add_techniques_table(tech_rows, title="")

            pdf.ln(2)

            # 2. Analysis results: print each block in order
            pdf.section_header_keep("2. Analysis results", first_block_mm=35.0)
            result_index = 1
            for i in range(st.session_state.analysis_blocks_n):
                tech = st.session_state.get(f"block_{i}__tech")
                tech_name = (st.session_state.get(f"block_{i}__tech_other") or "").strip() if tech == "Other" else tech
                method_ref = st.session_state.get(f"block_{i}__method_ref", "") or ""
                operator = st.session_state.get(f"block_{i}__operator", "") or ""
                op_notes = st.session_state.get(f"block_{i}__opnotes", "") or ""

                # Skip blocks that are completely empty (no tables, no images, no meta/op)
                tables = st.session_state.get(f"block_{i}__tables", []) or []
                images = st.session_state.get(f"block_{i}__images", []) or []
                captions = (st.session_state.get(f"block_{i}__captions", "") or "").splitlines()

                has_any = bool(tables) or bool(images) or bool(method_ref.strip()) or bool(operator.strip()) or bool(op_notes.strip())
                if not has_any:
                    continue

                ensure_space(pdf, 12.0 + 12.0)
                pdf.subtitle(f"2.{result_index} {tech_name or 'Analysis'}")
                result_index += 1

                if method_ref:
                    pdf.set_font(theme.font_mono, "", 8)
                    pdf.set_text_color(*theme.text_gray)
                    pdf.cell(0, 5, pdf_safe_text(f"Method/SOP: {method_ref}"), 0, 1, "L")
                    pdf.set_text_color(*theme.text_dark)

                pdf_operator_block(pdf, theme, operator, op_notes)

                # print all tables for this block
                for j, df_tbl in enumerate(tables):
                    df_tbl_clean = df_tbl if isinstance(df_tbl, pd.DataFrame) else None
                    if df_tbl_clean is None or df_tbl_clean.empty:
                        continue
                    ensure_space(pdf, 10.0 + estimate_table_height_mm(df_tbl_clean))
                    pdf.set_font(theme.font_main, "I", 9)
                    pdf.set_text_color(*theme.text_gray)
                    pdf.cell(0, 5, pdf_safe_text(f"Table {j+1}"), 0, 1, "L")
                    pdf.set_text_color(*theme.text_dark)
                    pdf.add_zebra_table(df_tbl_clean)

                # print images for this block
                tmp_paths = []
                try:
                    for k, img_file in enumerate(images):
                        tmp_path = save_upload_as_jpg(img_file)
                        if not tmp_path:
                            continue
                        tmp_paths.append(tmp_path)
                        cap = captions[k] if k < len(captions) and captions[k].strip() else f"Figure {k+1}"
                        pdf.add_framed_image(tmp_path, caption=cap)
                finally:
                    for p in tmp_paths:
                        cleanup_file(p)

            # 3. Conclusions
            pdf.section_header_keep("3. Conclusions / summary", first_block_mm=30.0)
            pdf.set_font(theme.font_main, "", 10)
            pdf.set_text_color(*theme.text_dark)
            concl = summary.strip() or conclusion_placeholder
            pdf.multi_cell(0, 6, pdf_safe_text(concl))

            # 4. Sign-off
            pdf.section_header_keep("4. Sign-off", first_block_mm=25.0)
            pdf.add_signoff_two_columns(
                left_label="Reported by",
                left_name=reported_by,
                left_title=reported_title,
                right_label="Reviewed by",
                right_name=reviewed_by,
                right_title=reviewed_title,
            )

            # Output PDF
            try:
                out = pdf.output(dest="S")
                pdf_bytes = out.encode("latin-1", "ignore") if isinstance(out, str) else bytes(out)
                cleanup_file(meta["logo_path"])
                st.success("✅ Report Generated! Download below.")
                st.download_button(label="📥 Download PDF", data=pdf_bytes, file_name=f"{sample_id or 'report'}.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")


if __name__ == "__main__":
    main()
