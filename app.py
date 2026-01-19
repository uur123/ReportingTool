#!/usr/bin/env python3
import os
import io
import datetime
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import re
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

# ----------------------------
# Small helpers
# ----------------------------
def pdf_safe_text(s: Any) -> str:
    if s is None: return ""
    s = str(s)
    s = (s.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
         .replace("\u00a0", " ").replace("\u2019", "'").replace("\u2018", "'")
         .replace("\u201c", '"').replace("\u201d", '"'))
    return s.encode("latin-1", "ignore").decode("latin-1")

def save_upload(uploaded_file, force_ext: Optional[str] = None):
    if uploaded_file is None: return None
    try:
        suffix = force_ext or os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception: return None

def save_upload_as_jpg(uploaded_file) -> Optional[str]:
    if uploaded_file is None: return None
    try:
        img = Image.open(uploaded_file)
        if img.mode != "RGB": img = img.convert("RGB")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name, format="JPEG", quality=95)
            return tmp.name
    except Exception: return None

def cleanup_file(path):
    if path and os.path.exists(path):
        try: os.unlink(path)
        except: pass

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
        self.set_auto_page_break(auto=True, margin=theme.margin_mm)
        self.set_margins(theme.margin_mm, theme.margin_mm, theme.margin_mm)

    def header(self):
        t = self.theme
        self.set_fill_color(220, 220, 220)
        self.rect(0, 0, self.w, 8, style="F")
        self.set_font(t.font_main, "B", 7)
        self.set_text_color(100, 100, 100)
        self.set_xy(0, 0)
        self.cell(self.w, 8, pdf_safe_text("INTERNAL USE ONLY - CONFIDENTIAL"), 0, 0, "C")

        y_top = 10.0
        self.set_font(t.font_main, "B", 13)
        self.set_text_color(*t.primary)
        self.set_xy(self.l_margin, y_top)
        self.cell(0, 10, pdf_safe_text("Technical Service Report"), 0, 0, "C")
        
        logo_path = self.meta.get("logo_path")
        if logo_path and os.path.exists(logo_path):
            self.image(logo_path, x=self.w - self.r_margin - 28, y=y_top, h=10)

        y_line = y_top + 11
        self.set_draw_color(*t.primary)
        self.set_line_width(0.5)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)
        self.set_y(y_line + 6.0)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.theme.font_main, "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def subtitle(self, title: str):
        t = self.theme
        self.ln(2)
        self.set_font(t.font_main, "B", 10)
        self.set_text_color(*t.text_dark)
        self.cell(0, 6, pdf_safe_text(title), 0, 1, "L")
        self.set_draw_color(*t.border_gray)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(2)

    def section_header_keep(self, title: str, first_block_mm: float = 20.0):
        if self.get_y() + 12.0 + first_block_mm > self.h - self.b_margin:
            self.add_page()
        t = self.theme
        self.ln(5)
        self.set_font(t.font_main, "B", 12)
        self.set_fill_color(*t.primary)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, pdf_safe_text(f"  {title}"), 0, 1, "L", fill=True)
        self.ln(3)
        self.set_text_color(*t.text_dark)

    def add_kv_box(self, items: List[Tuple[str, str]], title: str = "", cols: int = 2):
        t = self.theme
        box_w = self.w - self.l_margin - self.r_margin
        rows = (len(items) + cols - 1) // cols
        line_h = 6.0
        box_h = (rows * line_h) + 10
        if self.get_y() + box_h > self.h - self.b_margin: self.add_page()
        
        self.set_fill_color(248, 249, 251)
        self.set_draw_color(*t.border_gray)
        self.rect(self.l_margin, self.get_y(), box_w, box_h, style="DF")
        
        self.set_y(self.get_y() + 2)
        col_w = box_w / cols
        for i, (label, value) in enumerate(items):
            if i % cols == 0 and i > 0: self.ln(line_h)
            self.set_font(t.font_main, "B", 9)
            self.cell(col_w * 0.4, line_h, pdf_safe_text(f" {label}:"), 0, 0)
            self.set_font(t.font_main, "", 9)
            self.cell(col_w * 0.6, line_h, pdf_safe_text(value), 0, 0)
        self.ln(box_h - (rows * line_h))

    def add_zebra_table(self, df: pd.DataFrame):
        if df is None or df.empty: return
        t = self.theme
        col_width = (self.w - self.l_margin - self.r_margin) / len(df.columns)
        self.set_font(t.font_main, "B", 9)
        self.set_fill_color(*t.header_fill)
        for col in df.columns:
            self.cell(col_width, 7, pdf_safe_text(col), 1, 0, "C", True)
        self.ln()
        self.set_font(t.font_main, "", 9)
        for i, row in df.iterrows():
            fill = (i % 2 == 1)
            self.set_fill_color(*t.zebra_fill) if fill else self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_width, 7, pdf_safe_text(val), 1, 0, "C", True)
            self.ln()

    def add_signoff_two_columns(self, left_label, left_name, left_title, right_label, right_name, right_title):
        self.ln(10)
        t = self.theme
        w = (self.w - self.l_margin - self.r_margin) / 2
        
        curr_y = self.get_y()
        # Left side
        self.set_font(t.font_main, "B", 9)
        self.cell(w, 5, pdf_safe_text(left_label), 0, 1, "L")
        self.ln(10) # Signature space
        self.set_font(t.font_main, "", 9)
        self.cell(w, 5, pdf_safe_text(left_name), 0, 1, "L")
        self.cell(w, 5, pdf_safe_text(left_title), 0, 0, "L")
        
        # Right side
        self.set_xy(self.l_margin + w, curr_y)
        self.set_font(t.font_main, "B", 9)
        self.cell(w, 5, pdf_safe_text(right_label), 0, 1, "L")
        self.set_x(self.l_margin + w)
        self.ln(10)
        self.set_x(self.l_margin + w)
        self.set_font(t.font_main, "", 9)
        self.cell(w, 5, pdf_safe_text(right_name), 0, 1, "L")
        self.set_x(self.l_margin + w)
        self.cell(w, 5, pdf_safe_text(right_title), 0, 0, "L")

# ----------------------------
# Session State & Logic
# ----------------------------
def init_state():
    if "blocks" not in st.session_state:
        st.session_state.blocks = []

def add_block():
    st.session_state.blocks.append({
        "tech": ALL_MEASUREMENT_TECHNIQUES[0],
        "other": "",
        "method": "",
        "operator": "",
        "notes": "",
        "tables": [pd.DataFrame([["", ""]], columns=["Parameter", "Result"])],
        "images": []
    })

# ----------------------------
# Main UI
# ----------------------------
def main():
    st.set_page_config(page_title="EN-Report", layout="wide")
    init_state()

    st.title("🔬 Scientific Analysis Reporting Tool")
    
    # Sidebar
    with st.sidebar:
        st.header("Lab Info")
        lab_name = st.text_input("Lab Name", "Global Analytical Lab")
        logo = st.file_uploader("Logo", type=["png", "jpg"])

    # 1. Header
    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        sample_id = c1.text_input("TSR No (Project ID)*")
        project_title = c2.text_input("Project Title", "General Analysis")
        requester = c3.text_input("Requester", "John Doe")
        
        c4, c5 = st.columns(2)
        reported_by = c4.text_input("Reported By*")
        reviewed_by = c5.text_input("Reviewed By*")

    # 2. Blocks
    st.subheader("Analysis Blocks")
    for b_idx, block in enumerate(st.session_state.blocks):
        with st.expander(f"Block {b_idx+1}: {block['tech']}", expanded=True):
            c1, c2, c3 = st.columns(3)
            block['tech'] = c1.selectbox("Technique", ALL_MEASUREMENT_TECHNIQUES + ["Other"], key=f"t_{b_idx}")
            block['method'] = c2.text_input("Method/SOP", key=f"m_{b_idx}")
            block['operator'] = c3.text_input("Operator", key=f"o_{b_idx}")
            
            # --- Table Management ---
            st.markdown("#### Data Tables")
            for t_idx, df in enumerate(block['tables']):
                st.info(f"Table {t_idx+1} Structure Control")
                
                # Column Header Editing
                cols = list(df.columns)
                new_headers = []
                h_cols = st.columns(len(cols))
                for i, col_name in enumerate(cols):
                    new_h = h_cols[i].text_input(f"Header {i+1}", value=col_name, key=f"h_{b_idx}_{t_idx}_{i}")
                    new_headers.append(new_h)
                
                # Update headers if changed
                if new_headers != cols:
                    df.columns = new_headers
                    block['tables'][t_idx] = df

                # Add/Remove Columns
                tc1, tc2, tc3 = st.columns([1, 1, 4])
                if tc1.button("➕ Add Col", key=f"ac_{b_idx}_{t_idx}"):
                    df[f"New Col {len(df.columns)}"] = ""
                    st.rerun()
                if tc2.button("➖ Rem Col", key=f"rc_{b_idx}_{t_idx}") and len(df.columns) > 1:
                    df = df.drop(df.columns[-1], axis=1)
                    block['tables'][t_idx] = df
                    st.rerun()

                # The Data Editor
                block['tables'][t_idx] = st.data_editor(df, num_rows="dynamic", use_container_width=True, key=f"ed_{b_idx}_{t_idx}")

            if st.button("Add Another Table to this Block", key=f"at_{b_idx}"):
                block['tables'].append(pd.DataFrame([[""] * len(block['tables'][0].columns)], columns=block['tables'][0].columns))
                st.rerun()

    if st.button("➕ Add New Analysis Block"):
        add_block()
        st.rerun()

    # 3. Generate
    st.divider()
    if st.button("🚀 Generate PDF", type="primary"):
        if not sample_id or not reported_by or not reviewed_by:
            st.error("Please fill mandatory fields (ID, Reported By, Reviewed By)")
            return

        meta = {"logo_path": save_upload(logo), "project_id": sample_id}
        pdf = AnalyticalReportPDF(Theme(), meta)
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Info Box
        pdf.section_header_keep("1. Project Information")
        pdf.add_kv_box([
            ("Project ID", sample_id),
            ("Title", project_title),
            ("Requester", requester),
            ("Date", str(datetime.date.today()))
        ])
        
        # Analysis Results
        pdf.section_header_keep("2. Analysis Results")
        for b in st.session_state.blocks:
            pdf.subtitle(f"{b['tech']}")
            if b['method']:
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(0, 5, f"Method: {b['method']}", 0, 1)
            
            for df_table in b['tables']:
                pdf.add_zebra_table(df_table)
                pdf.ln(5)

        # Sign off
        pdf.section_header_keep("3. Authentication")
        pdf.add_signoff_two_columns(
            "Reported By", reported_by, "Analyst",
            "Reviewed By", reviewed_by, "Laboratory Manager"
        )

        # Output
        pdf_bytes = pdf.output()
        st.download_button("📥 Download Report", data=bytes(pdf_bytes), file_name=f"Report_{sample_id}.pdf", mime="application/pdf")
        cleanup_file(meta["logo_path"])

if __name__ == "__main__":
    main()
