import os
import io
import datetime
import tempfile
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import re

# ---------------------------------------------------------
# ENGINEERING NOTE: Ensure you have these installed:
# pip install streamlit pandas fpdf2 Pillow
# ---------------------------------------------------------
import pandas as pd
import streamlit as st
from PIL import Image
from fpdf import FPDF


# ==============================================================================
# 1. DESIGN SYSTEM (The "Face")
# ==============================================================================
@dataclass
class Theme:
    # Colors (R, G, B)
    primary: Tuple[int, int, int] = (26, 43, 60)       # Deep Navy (Brand)
    header_fill: Tuple[int, int, int] = (230, 235, 240) # Table Header
    zebra_fill: Tuple[int, int, int] = (250, 250, 250)  # Table Stripe
    text_dark: Tuple[int, int, int] = (30, 30, 30)
    text_gray: Tuple[int, int, int] = (100, 100, 100)
    border_gray: Tuple[int, int, int] = (200, 200, 200)

    # Typography & Sizing
    font_main: str = "Helvetica"
    font_mono: str = "Courier"
    margin_mm: float = 25.0

# ==============================================================================
# 2. PDF ENGINE (The "Core")
# ==============================================================================
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

        # --- 1) TOP CONFIDENTIAL STRIPE ---
        # Draw a gray rectangle from (0,0) to full width, 8mm high
        self.set_fill_color(220, 220, 220) # Light Grey
        self.rect(0, 0, self.w, 8, style="F")
    
        self.set_font(t.font_main, "B", 7)
        self.set_text_color(100, 100, 100) # Darker grey text
        self.set_xy(0, 0)
        self.cell(self.w, 8, "INTERNAL USE ONLY - CONFIDENTIAL", 0, 0, "C")

        # --- Fixed header geometry (professional + predictable) ---
        y_top = 10.0
        logo_h = 10.0
        #logo_pad_x = 2.0

    # 1) Lab name (left)
        self.set_font(t.font_main, "B", 13)
        self.set_text_color(*t.primary)
        self.set_xy(self.l_margin, y_top)
        #self.cell(110, 8, self.meta.get("lab_name", "Lab"), 0, 0, "L")
        self.cell(0, 10, "Technical Service Report", 0, 0, "C")

    # 2) Report ID (right)
        #self.set_font(t.font_mono, "", 9)
        #self.set_text_color(*t.text_gray)
        #report_id = self.meta.get("report_id", "DRAFT")
        #self.set_xy(self.l_margin, y_top)
        #self.cell(0, 8, f"ID: {report_id}", 0, 1, "R")

    # 3) Subheader line (address + requester)
        #self.set_font(t.font_main, "I", 9)
        #self.set_text_color(*t.text_gray)
        #self.cell(110, 5, self.meta.get("lab_addr", ""), 0, 0, "L")
        #req_name = self.meta.get("requester", "Unknown")
        #self.cell(0, 5, f"Req: {req_name} | {self.meta.get('report_date')}", 0, 1, "R")

    # 4) Logo (top-right) — place it BEFORE drawing divider line
        #logo_path = self.meta.get("logo_path")
        logo_path = "static/logo2.png"
        #if logo_path and os.path.exists(logo_path):
        if os.path.exists(logo_path):
            try:
                # Keep it in the header block, not drifting into body
                x_logo = self.w - self.r_margin - 28  # width-ish reservation
                self.image(logo_path, x=x_logo, y=y_top, h=logo_h)
            except Exception:
                pass

    # 5) Divider line below logos/subheader
        y_line = y_top + 11  # sits below subheader and logo
        self.set_draw_color(*t.primary)
        self.set_line_width(0.5)
        self.line(self.l_margin, y_line, self.w - self.r_margin, y_line)

    # 6) Reserve header height so content never overlaps
        self.set_y(y_line + 6.0)

    def subtitle(self, title: str):
        t = self.theme
        self.ln(2)
        self.set_font(t.font_main, "B", 10)
        self.set_text_color(*t.text_dark)
        self.cell(0, 6, title, 0, 1, "L")
        
        # subtle divider line under subtitle
        y = self.get_y()
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.2)
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(2)
        self.set_text_color(*t.text_dark)

    def add_signoff_two_columns(self, left_label: str, left_name: str, left_title: str,
                                right_label: str, right_name: str, right_title: str):
        t = self.theme
        box_w = self.w - self.l_margin - self.r_margin
        col_w = box_w / 2.0
        x0 = self.l_margin
        y0 = self.get_y()

        # Reserve enough space so it doesn't split awkwardly
        need = 22.0
        if self.get_y() + need > self.h - self.b_margin:
            self.add_page()
            x0 = self.l_margin
            y0 = self.get_y()

        # Top labels
        self.set_font(t.font_main, "B", 10)
        self.set_text_color(*t.primary)
        self.set_xy(x0, y0)
        self.cell(col_w, 6, left_label, 0, 0, "L")
        self.cell(col_w, 6, right_label, 0, 1, "L")

        # Names (same line)
        self.set_font(t.font_main, "", 10)
        self.set_text_color(*t.text_dark)
        self.set_x(x0)
        self.cell(col_w, 6, left_name or "—", 0, 0, "L")
        self.cell(col_w, 6, right_name or "—", 0, 1, "L")

        # Titles (same line)
        self.set_font(t.font_main, "I", 9)
        self.set_text_color(*t.text_gray)
        self.set_x(x0)
        self.cell(col_w, 6, left_title or "", 0, 0, "L")
        self.cell(col_w, 6, right_title or "", 0, 1, "L")

        self.ln(2)
        self.set_text_color(*t.text_dark)


    def footer(self):
        t = self.theme
        self.set_y(-18)
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        
        page_str = f"Page {self.page_no()}/{{nb}}"

        lab_name = (self.meta.get("lab_name") or "").strip() or "Analytical Lab"
        lab_mail = (self.meta.get("lab_mail") or "").strip()
        lab_addr = (self.meta.get("lab_addr") or "").strip()
        lab_no = (self.meta.get("lab_no") or "").strip()

        # Row 1: left lab name | center email | right page
        self.set_font(t.font_main, "", 7)
        self.set_text_color(*t.text_gray)

        row_w = self.w - self.l_margin - self.r_margin
        col_w = row_w / 3.0

        self.set_x(self.l_margin)
        self.cell(col_w, 5, lab_name, 0, 0, "L")
        self.cell(col_w, 5, lab_mail, 0, 0, "C")
        self.cell(col_w, 5, page_str, 0, 1, "R")

        contact = " | ".join([x for x in [lab_addr, lab_no] if x])
        if contact:
            self.set_x(self.l_margin)
            self.cell(0, 5, contact, 0, 1, "C")

        # Row 2: contact/address only on first page (optional, reduces noise)
        #if self.page_no() == 1:
        #    contact = " | ".join([x for x in [lab_addr, lab_no] if x])
         #   if contact:
          #      self.set_x(self.l_margin)
           #     self.cell(0, 5, contact, 0, 1, "C")

    def section_header(self, title: str):
        t = self.theme
        self.ln(5)
        self.set_font(t.font_main, "B", 12)
        self.set_fill_color(*t.primary)
        self.set_text_color(255, 255, 255)
        self.cell(0, 8, f"  {title}", 0, 1, "L", fill=True)
        self.ln(3)
        self.set_text_color(*t.text_dark)

    def add_kv_box(self, items: List[Tuple[str, str]], title: str = "", cols: int = 2):
        
        #Draws a clean metadata block with label/value pairs.
        #items: list of (label, value)
        
        t = self.theme
        if not items:
            return

        box_w = self.w - self.l_margin - self.r_margin
        x0 = self.l_margin
        y0 = self.get_y()

        # Layout
        line_h = 6.0
        pad = 2.5
        title_h = 6.5 if title else 0.0
        rows = (len(items) + cols - 1) // cols
        box_h = title_h + pad + rows * line_h + pad

        # Page-break safe
        if self.get_y() + box_h > self.h - self.b_margin:
            self.add_page()
            x0 = self.l_margin
            y0 = self.get_y()

        # Background + border
        self.set_fill_color(248, 249, 251)
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.2)
        self.rect(x0, y0, box_w, box_h, style="DF")

        # Title
        cur_y = y0 + pad
        if title:
            self.set_xy(x0 + pad, cur_y)
            self.set_font(t.font_main, "B", 10)
            self.set_text_color(*t.primary)
            self.cell(box_w - 2 * pad, 6, title, 0, 1, "L")
            cur_y = self.get_y()

        # Content
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

                # label
                self.set_font(t.font_main, "B", 9)
                self.set_text_color(*t.text_gray)
                self.cell(label_w, line_h, f"{label}:", 0, 0, "L")

                # value
                self.set_font(t.font_main, "", 9)
                self.set_text_color(*t.text_dark)
                self.cell(value_w, line_h, value, 0, 0, "L")

            self.ln(line_h)

        self.set_y(y0 + box_h + 4.0)

    def add_techniques_table(self, rows: List[Dict[str, str]], title: str = "Techniques included"):
        """
        rows: list of dicts with keys: Technique, Method/SOP, Output, Status
        """
        t = self.theme
        if not rows:
            return

        df = pd.DataFrame(rows, columns=["Technique", "Method/SOP", "Output"])

        # Reserve space so table doesn't start at bottom and continue immediately
        est_h = 7 + min(len(df), 10) * 7 + 10
        if self.get_y() + est_h > self.h - self.b_margin:
            self.add_page()

        if title:
            self.set_font(t.font_main, "B", 10)
            self.set_text_color(*t.text_dark)
            self.cell(0, 6, title, 0, 1, "L")
            self.ln(1)

        # custom widths: Technique 25%, SOP 25%, Output 45%
        table_w = self.w - self.l_margin - self.r_margin
        w_tech = table_w * 0.25
        w_sop  = table_w * 0.25
        w_out  = table_w * 0.50
        widths = [w_tech, w_sop, w_out]

         #header
        self.set_font(t.font_main, "B", 9)
        self.set_fill_color(*t.header_fill)
        self.set_text_color(*t.primary)
        headers = ["Technique", "Method/SOP", "Output"]
        for h, w in zip(headers, widths):
            self.cell(w, 7, pdf_safe_text(h), 0, 0, "C", True)
        self.ln()

        # rows
        self.set_font(t.font_main, "", 9)
        self.set_text_color(*t.text_dark)

        for i, row in df.iterrows():
            fill = (i % 2 == 1)
            if fill:
                self.set_fill_color(*t.zebra_fill)
            else:
                self.set_fill_color(255, 255, 255)

            vals = [str(row.get("Technique","")), str(row.get("Method/SOP","")), str(row.get("Output",""))]
            for v, w in zip(vals, widths):
                # no wrapping here; keep it clean and single-line
                v = pdf_safe_text(v.replace("\n", " ").strip())
                self.cell(w, 7, v, 0, 0, "C", True)
            self.ln()

        self.ln(2)


    def add_zebra_table(self, df: pd.DataFrame):
        if df is None or df.empty or len(df.columns) == 0:
            return
        t = self.theme
        col_width = (self.w - self.l_margin - self.r_margin) / len(df.columns)
        
        # Header
        self.set_font(t.font_main, "B", 9)
        self.set_fill_color(*t.header_fill)
        self.set_text_color(*t.primary)
        for col in df.columns:
            self.cell(col_width, 7, pdf_safe_text(col), 0, 0, "C", True)
        self.ln()

        # Rows
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


        # Max width inside margins
        max_w = self.w - self.l_margin - self.r_margin
        display_w = min(120, max_w) # Limit huge images to 120mm wide
        display_h = display_w * aspect
        
        # Check if page break needed
        if self.get_y() + display_h + 15 > self.h - t.margin_mm:
            self.add_page()

        x_pos = (self.w - display_w) / 2
        y_pos = self.get_y()

        self.image(img_path, x=x_pos, y=y_pos, w=display_w)
        
        # Draw Border
        self.set_draw_color(*t.border_gray)
        self.set_line_width(0.3)
        self.rect(x_pos, y_pos, display_w, display_h)

        self.set_y(y_pos + display_h + 2)
        if caption:
            self.set_font(t.font_main, "I", 8)
            self.set_text_color(*t.text_gray)
            self.cell(0, 5, caption, 0, 1, "C")
        self.ln(5)
    
    def section_header_keep(self, title: str, first_block_mm: float = 20.0):
        # Ensure the header + at least some content stays together
        ensure_space(self, needed_mm=12.0 + first_block_mm)
        self.section_header(title)


# ==============================================================================
# 3. UTILITIES
# ==============================================================================
def ensure_space(pdf: FPDF, needed_mm: float):
    # Works for portrait & landscape because it uses pdf.h
    if pdf.get_y() + needed_mm > pdf.h - pdf.b_margin:
        pdf.add_page()

def estimate_table_height_mm(df: Optional[pd.DataFrame], row_h: float = 7.0, header_h: float = 7.0, min_rows: int = 2) -> float:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return 0.0
    n = max(min_rows, min(len(df), 12))  # cap estimate so it doesn't over-reserve
    return header_h + n * row_h + 2.0

def keep_method_with_first_content(pdf: FPDF, df: Optional[pd.DataFrame] = None, extra_mm: float = 0.0):
    # subtitle line (~6) + small padding (~2) + operator block (~12) + table estimate
    need = 6.0 + 2.0 + 12.0 + estimate_table_height_mm(df) + extra_mm
    ensure_space(pdf, need)


def save_upload(uploaded_file, force_ext: Optional[str] = None):
    """Save upload to a temp file. Optionally override extension."""
    if uploaded_file is None:
        return None
    try:
        suffix = force_ext or os.path.splitext(uploaded_file.name)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            return tmp.name
    except Exception:
        return None

def cleanup_file(path):
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass

# Simple image size cache to avoid reopening the same temp path
_IMG_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}

def get_image_size(path: str) -> Optional[Tuple[int, int]]:
    if not path or not os.path.exists(path):
        return None
    if path in _IMG_SIZE_CACHE:
        return _IMG_SIZE_CACHE[path]
    try:
        with Image.open(path) as im:
            w, h = im.size
        _IMG_SIZE_CACHE[path] = (w, h)
        return w, h
    except Exception:
        return None
        
def clean_numeric_string(s: str) -> str:
    """
    Normalizes number strings from Excel.
    Example: '1.234,56' -> '1234.56'
    Example: '1,234.56' -> '1234.56'
    """
    if not isinstance(s, str):
        return s
    
    s = s.strip()
    # Check if it looks like a number with European formatting (1.000,00)
    if re.match(r'^-?\d+(\.\d{3})*,\d+$', s):
        s = s.replace('.', '').replace(',', '.')
    # Check if it looks like US formatting (1,000.00)
    elif re.match(r'^-?\d+(,\d{3})*\.\d+$', s):
        s = s.replace(',', '')
    # If it's just a comma decimal (1234,56)
    elif ',' in s and '.' not in s:
        s = s.replace(',', '.')
        
    return s
    
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

def safe_key(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))

def pdf_safe_text(s: Any) -> str:
    """
    FPDF core fonts can't encode many Unicode characters.
    Replace common offenders and drop anything still unsupported by latin-1.
    """
    if s is None:
        return ""
    s = str(s)

    # common unicode punctuation replacements
    s = (s.replace("\u2014", "-")   # em dash —
           .replace("\u2013", "-")   # en dash –
           .replace("\u2212", "-")   # minus −
           .replace("\u00a0", " ")   # non-breaking space
           .replace("\u2019", "'")   # ’
           .replace("\u2018", "'")   # ‘
           .replace("\u201c", '"')   # “
           .replace("\u201d", '"'))  # ”

    # keep only latin-1 encodable characters
    return s.encode("latin-1", "ignore").decode("latin-1")


def technique_output_hint(name: str) -> str:
    m = {
        "ICP-OES": "Elemental concentrations",
        "XRF": "Elemental analysis",
        "SEM-EDX": "Surface topograpy and Chemistry",
        "Optical Microscopy": "Microstructure images",
        "PSD": "Partical Size distribution",
        "BET": "Surface area / porosity",
        "XRD": "Crystal Phase identification",
        "TGA": "Thermal behavious analysis",
        "C, S, N Analysis": "C/S/N content",
        "F Analysis": "Fluorine content",
        "Metallic Al Analysis": "Metallic Al %",
        "Metallic Si Analysis": "Metallic Si %",
        "Comparison Matrix": "Comparison tables",

    }
    return m.get(name, "—")
METHOD_HOURS = {
    "ICP-OES": 3.0,
    "XRF": 1.5,
    "XRD": 1.5,
    "SEM-EDX": 3.0,
    "Optical": 2.0,                 # checkbox key
    "Al Grain Size": 3.0,
    "Porosity": 1.0,
    "PSD": 2.0,
    "BET": 2.0,
    "TGA": 2.0,
    "F Analysis": 4.0,
    "C, S, N Analysis": 1.0,        # treated as a method block
    "Metallic Al Analysis": 1.0,
    "Metallic Si Analysis": 1.0,
    # "Comparison Matrix": (optional) set if you want later
}

REPORTING_HOURS = 1.0  # always add this when any method is selected
CASTING_DEFECT_HOURS = 32.0

def compute_default_hours(
    technique_flags: Dict[str, bool],
    context: str
) -> float:
    """
    Compute default effort hours.
    - Casting Defect Analysis → fixed 32 h
    - Otherwise → sum of selected methods + reporting
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



def technique_status_from_data(data: Dict[str, Any]) -> str:
    # If a method is selected but user didn't fill anything: show "Selected – no data"
    has_table = isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty
    has_images = bool(data.get("images"))
    has_any = has_table or has_images or bool((data.get("meta") or "").strip())
    return "Performed" if has_any else "Selected – no data"
    
def default_comparison_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Parameter": "", "Sample A": "", "Sample B": "", "Unit": ""},
    ])

def default_porosity_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Porosity (%)", "Value": "", "Unit": "%", "Notes": ""},
        {"Metric": "Pore size (avg)", "Value": "", "Unit": "µm", "Notes": ""},
        {"Metric": "Measurement area", "Value": "", "Unit": "", "Notes": ""},
    ])

def default_gsa_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"Metric": "Grain size number (G)", "Value": "", "Unit": "", "Notes": ""},
        {"Metric": "Mean intercept length", "Value": "", "Unit": "µm", "Notes": ""},
        {"Metric": "n fields", "Value": "", "Unit": "", "Notes": ""},
    ])


def parse_excel_paste(text: str) -> Optional[pd.DataFrame]:
    """
    Parses pasted text from Excel/Sheets. 
    1. Detects delimiters (Tab, Semicolon, Comma).
    2. Normalizes numbers (handles 1.234,56 vs 1,234.56).
    3. Returns a clean DataFrame.
    """
    if not text or not text.strip():
        return None

    # --- STEP 1: DETECT SEPARATOR ---
    # Excel clipboard data is almost always Tab-separated (\t)
    s = text.strip()
    first_line = s.splitlines()[0]
    
    if "\t" in first_line:
        sep = "\t"
    elif ";" in first_line:
        sep = ";"
    else:
        sep = ","

    # --- STEP 2: LOAD INITIAL DATAFRAME ---
    try:
        # We read everything as strings initially to avoid pandas 
        # making wrong guesses before we clean the punctuation
        df = pd.read_csv(io.StringIO(s), sep=sep, dtype=str, engine="python")
    except Exception:
        return None

    # --- STEP 3: SMART NUMERIC CLEANING ---
    def normalize_number(val):
        if not isinstance(val, str):
            return val
        
        val = val.strip()
        
        # Remove currency symbols or whitespace often found in Excel
        val = re.sub(r'[^\d,.\-]', '', val)
        
        if not val:
            return val

        # Logic for Punctuation:
        # Case A: 1.234,56 (European) -> Remove dot, replace comma with dot
        if '.' in val and ',' in val:
            if val.find('.') < val.find(','):
                val = val.replace('.', '').replace(',', '.')
            # Case B: 1,234.56 (US/UK) -> Remove comma
            else:
                val = val.replace(',', '')
        
        # Case C: 1234,56 (Simple European) -> Replace comma with dot
        elif ',' in val:
            val = val.replace(',', '.')
            
        return val

    # Apply cleaning to every cell
    for col in df.columns:
        # Clean the string formatting
        cleaned_col = df[col].apply(normalize_number)
        
        # Try to convert to numeric, but leave as string if it's a text column
        # (errors='ignore' ensures 'Sample Name' remains 'Sample Name')
        df[col] = pd.to_numeric(cleaned_col, errors='ignore')

    return df


def comparison_matrix_editor(
    *,
    default_df: pd.DataFrame,
    editor_key: str,
    label: str,
    selected_techniques: List[str],
) -> pd.DataFrame:
    """
    Comparison Matrix table editor:
    - No always-visible paste box
    - Optional import from Excel clipboard in a popover/expander
    - Add column button
    - Rename columns via inputs
    """
    store_key = f"{editor_key}__df"
    df_current = st.session_state.get(store_key)

    if not isinstance(df_current, pd.DataFrame) or df_current.empty:
        df_current = default_df.copy()

    # ---- Controls row ----
    c1, c2, c3 = st.columns([1, 1, 4])

    # Add new column
    if c1.button("➕ Add column", key=f"{editor_key}__add_col"):
        new_col = f"New column {len(df_current.columns) + 1}"
        df_current[new_col] = ""

    # Optional import (hidden/compact UI)
    # Prefer popover if available; fallback to expander.
    imported = None
    try:
        pop = st.popover("📋 Import from Excel", use_container_width=False)
        with pop:
            txt = st.text_area("Paste copied Excel range here", key=f"{editor_key}__import_txt", height=120)
            if st.button("Import", key=f"{editor_key}__import_btn"):
                imported = parse_excel_paste(txt)
    except Exception:
        with c2:
            with st.expander("📋 Import from Excel", expanded=False):
                txt = st.text_area("Paste copied Excel range here", key=f"{editor_key}__import_txt", height=120)
                if st.button("Import", key=f"{editor_key}__import_btn"):
                    imported = parse_excel_paste(txt)

    if imported is not None:
        if isinstance(imported, pd.DataFrame) and not imported.empty:
            df_current = imported
        else:
            st.warning("Import failed. Copy a rectangular range (including headers) and try again.")

    # ---- Rename columns UI ----
    st.markdown("**Rename columns**")
    rename_cols = st.columns(min(len(df_current.columns), 6))  # wrap if many cols
    new_names = []
    for i, col in enumerate(df_current.columns):
        col_container = rename_cols[i % len(rename_cols)]
        new_name = col_container.text_input(
            label=f"col_{i}",
            value=str(col),
            key=f"{editor_key}__colname_{i}",
            label_visibility="collapsed",
        )
        new_names.append(new_name.strip() or str(col))

    if new_names != list(df_current.columns):
        df_current = df_current.copy()
        df_current.columns = new_names

    # ---- Table editor ----
    df_out = st.data_editor(
        df_current,
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key,
    )
    st.session_state[store_key] = df_out
    return df_out




def editor_with_excel_paste(
    *,
    default_df: pd.DataFrame,
    editor_key: str,
    paste_key: str,
    label: str = "Paste from Excel (optional)",
    apply_button_text: str = "Apply paste to table",
    help_text: str = "Copy a cell range in Excel/Sheets and paste here. Tabs/newlines are supported.",
    height: int = 140,
    use_expander: bool = True,
) -> pd.DataFrame:
    """
    Provides a paste area + Apply button + data_editor.
    If use_expander=False, it renders paste UI inline (safe inside other expanders).
    Keeps parsed DF in session_state so it survives reruns.
    """
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
# 4. STREAMLIT UI (The "Interaction")
# ==============================================================================
def main():
    st.set_page_config(page_title="EN-Report", layout="wide", page_icon="🔬", initial_sidebar_state="collapsed")
    if "report_data" not in st.session_state:
        st.session_state.report_data = {}

    
    # CSS for Checkboxes to look like nice clickable boxes
    st.markdown("""
        <style>
        .stApp { background-color: #f8f9fa; }
        </style>
    """, unsafe_allow_html=True)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        lab_name = st.text_input("Lab Name", "Global R&D Analytical Service Lab")
        lab_addr = st.text_input("Address", "Pantheon 30, Enschede, NL")
        lab_no = st.text_input("Phone number", "+31 600504030")
        lab_mail = st.text_input("Mail address", "EN-Analytical@vesuvius.com")
        logo = st.file_uploader("Lab Logo", type=['png', 'jpg'])
        st.divider()
        st.info("Status: Ready")

    st.title("🔬 Scientific Analysis Reporting Tool")
    st.markdown("---")

    # --- 1. HEADER DATA ---
    st.subheader("1. Sample Traceability")
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        project_title = c1.text_input("Project Title", "Benchmarking of HD1")
        #report_id = c1.text_input("Report ID", value=f"TSR-{datetime.date.today().year}-001")
        # NEW: Requester Name
        requester = c2.text_input("Requester Name", "John Doe") 
        context = c3.selectbox("Context", ["R&D project", "Operational Support", "Casting Defect Analysis"])
        sample_id = c4.text_input("TSR No", "XXXX")
    
    with st.container():
        c1, c2, c3 = st.columns(3)
        rx_date = c1.date_input("Date Received")
        temp = c2.number_input("Temp (°C)", value=21.0)
        hum = c3.number_input("Humidity (%RH)", value=45.0)
        st.markdown("#### Sample condition / requester notes")
        sample_condition = st.text_area( "Sample as received (packaging/condition) & requester comments", 
                                        placeholder="e.g., Sample received in sealed bag, slight oxidation visible. Requester asks to focus on inclusions.",
                                        height=90
                                        )


    
    st.markdown("#### Sample reception photo")
    sample_photo = st.file_uploader(
        "Upload sample photo (as received)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        key="sample_photo"
        )
    st.divider()

    # --- 2. SELECTABLE TEST BOXES ---
    st.subheader("2. Select Analytical Techniques")

    

    def operator_block(test_key: str):
        c1, c2 = st.columns([1, 2])
        operator = c1.text_input("Operator", key=f"{test_key}_operator")
        op_notes = c2.text_area("Operator notes (short)", key=f"{test_key}_opnotes", height=70)
        return operator, op_notes

    CHEM_METHODS = [
    "C, S, N Analysis",
    "F Analysis",
    "Metallic Al Analysis",
    "Metallic Si Analysis",
    ]
    
    def add_operator_fields(parent, key_prefix: str, method_name: str = ""):
        parent.markdown("##### Operator traceability")
        c0, c1, c2 = parent.columns([1, 1, 2])

        method_ref = c0.text_input("Method/SOP ID", key=f"{key_prefix}_methodref", placeholder="e.g. WI-ICP-014")
        operator = c1.text_input("Operator", key=f"{key_prefix}_operator")
        op_notes = c2.text_area("Operator notes (short)", key=f"{key_prefix}_opnotes", height=70)

        return method_ref, operator, op_notes
    

    def pdf_safe_text(text: str) -> str:
        if text is None:
            return ""
        # normalize common unicode that breaks latin-1 / fpdf fonts
        s = str(text)
        s = (s.replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2212", "-")   # minus sign
            .replace("\u00A0", " ")   # non-breaking space
        )
        # also avoid other sneaky whitespace
        s = s.replace("\t", " ")
        return s

    def _soft_wrap_unbreakable(s: str, every: int = 60) -> str:
        """
        Inserts break opportunities into very long runs (URLs, pasted Excel blobs, etc.)
        so fpdf2 can wrap them.
        """
        if not s:
            return ""
        # Break very long "words" (no spaces) by injecting a zero-width space marker.
        # fpdf doesn't understand ZWSP, so we insert a normal space after chunks.
        def break_word(m):
            w = m.group(0)
            return " ".join(w[i:i+every] for i in range(0, len(w), every))
        return re.sub(r"\S{%d,}" % (every + 1), break_word, s)

    def pdf_operator_block(pdf: AnalyticalReportPDF, theme: Theme, operator: str, notes: str,
                        title: str = "Analysis Notes"):
        operator = (operator or "").strip()
        notes = (notes or "").strip()
        if not operator and not notes:
            return

        # ✅ Critical: start at left margin so remaining width is never near-zero
        pdf.set_x(pdf.l_margin)

        # ✅ Use explicit width instead of w=0
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
            safe_notes = _soft_wrap_unbreakable(safe_notes, every=70)

            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(w, 5, pdf_safe_text(f"Notes: {safe_notes}"))

        pdf.ln(2)
        pdf.set_text_color(*theme.text_dark)


    # NEW: Clickable Boxes instead of Dropdown
    # We create a dictionary to store boolean states
    col1, col2, col3, col4 = st.columns(4)
    
    technique_flags = {}
    
    with col1:
        technique_flags["ICP-OES"] = st.checkbox("ICP-OES (Chemistry)") #done
        technique_flags["XRF"] = st.checkbox("XRF (Composition)") #done
        technique_flags["C, S, N Analysis"] = st.checkbox("C, S, N Analysis") #done
        technique_flags["Al Grain Size"] = st.checkbox("Al GSA")

    with col2:
        technique_flags["SEM-EDX"] = st.checkbox("SEM-EDX (Microscopy)") #done
        technique_flags["Optical"] = st.checkbox("Optical Microscopy")#done
        technique_flags["Metallic Al Analysis"] = st.checkbox("Metallic Al")
        technique_flags["Porosity"] = st.checkbox("Porosity") #done
    with col3:
        technique_flags["PSD"] = st.checkbox("Particle Size (PSD)")#done
        technique_flags["BET"] = st.checkbox("BET Surface Area")#done
        technique_flags["Metallic Si Analysis"] = st.checkbox("Metallic Si") #done

    with col4:
        technique_flags["XRD"] = st.checkbox("XRD (Phase ID)")#done
        technique_flags["TGA"] = st.checkbox("TGA-MS (Thermal)")#done
        technique_flags["F Analysis"] = st.checkbox("F Analysis") #done
        technique_flags["Comparison Matrix"] = st.checkbox("Comparison Matrix")
 

    st.subheader("3. Effort & Cost (optional)")

    default_hours = compute_default_hours(technique_flags, context)

    # Init session values once
    if "est_hours" not in st.session_state:
        st.session_state.est_hours = default_hours
    if "hourly_rate" not in st.session_state:
        st.session_state.hourly_rate = 0.0  # you can set a default like 85.0
    if "effort_include" not in st.session_state:
        st.session_state.effort_include = True

    c0, c1, c2, c3 = st.columns([1, 1, 1, 1])

    with c0:
        st.session_state.effort_include = st.checkbox(
            "Include in report",
            value=st.session_state.effort_include,
            key="effort_include_chk"
        )

    with c1:
        if st.button("Recalculate hours", key="recalc_hours_btn"):
            st.session_state.est_hours = default_hours

    with c2:
        est_hours = st.number_input(
            "Estimated work hours",
            min_value=0.0,
            value=float(st.session_state.est_hours),
            step=0.5,
            key="est_hours_input"
        )
        st.session_state.est_hours = float(est_hours)

    with c3:
        hourly_rate = st.number_input(
            "Hourly cost (€ / h)",
            min_value=0.0,
            value=float(st.session_state.hourly_rate),
            step=5.0,
            key="hourly_rate_input"
        )
        st.session_state.hourly_rate = float(hourly_rate)

    total_cost = round(st.session_state.est_hours * st.session_state.hourly_rate, 2)

    st.markdown(
        f"**Auto-calculated default:** {default_hours} h (includes {REPORTING_HOURS} h reporting)  \n"
        f"**Estimated project cost:** € {total_cost:,.2f}"
    )
    st.divider()

    report_data = st.session_state.report_data
    report_data.clear()  # rebuild fresh each run, but stable container



    # --- 3. DYNAMIC INPUT FORMS ---
    st.markdown("### 3. Data Entry")
    
    # A. CASTING DEFECT SPECIAL LOGIC
    # If context is Casting Defect, force show image uploaders even if not checked above?
    # Or just rely on user checking SEM/Optical. Let's rely on user checks but add a tip.
    if context == "Casting Defect Analysis":
        st.info("ℹ️ **Casting Defect Mode Active:** You can upload multiple images for SEM and Optical sections.")

    # --- MODULE: ICP-OES ---
    if technique_flags["ICP-OES"]:
        with st.expander("🧪 ICP-OES Data", expanded=True):
            c_a, c_b = st.columns([1, 3])
            mass = c_a.number_input("Mass (g)", 0.100, format="%.3f")
            volume = c_a.number_input("Volume (mL)", 500, format= "%.3f")
            
            

            #lod = c_a.text_input("LOD (mg/kg)", "0.01")
            
            df_in = pd.DataFrame([{"Element": "Fe", "Result (mg/kg)": 0.0}, {"Element": "Si", "Result (mg/kg)": 0.0}])
            df_icp = c_b.data_editor(df_in, num_rows="dynamic", use_container_width=True, key="icp_df")

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="ICP_OES", method_name="ICP-OES")

            report_data["ICP-OES"] = {"meta": f"Mass: {mass}g | Volume: {volume} mL", 
                                      "table": df_icp,
                                      "method_ref": method_ref,
                                      "operator": operator,
                                      "op_notes": op_notes
                                      }

    # --- MODULE: XRF ---
    if technique_flags["XRF"]:
        with st.expander("🧪 XRF Data", expanded=True):
            c_a, c_b = st.columns([1, 3])
            #mass = c_a.number_input("Mass (g)", 0.100, format="%.3f")
            #volume = c_a.number_input("Volume (mL)", 500, format= "%.3f")
            

            #lod = c_a.text_input("LOD (mg/kg)", "0.01")
            
            df_in_xrf = pd.DataFrame([{"Element": "Fe", "Result (mg/kg)": 0.0}, {"Element": "Si", "Result (mg/kg)": 0.0}])
            df_xrf = c_b.data_editor(df_in_xrf, num_rows="dynamic", use_container_width=True, key="xrf_df")

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="XRF", method_name="XRF")

            report_data["XRF"] = {"table": df_xrf,
                                  "method_ref": method_ref,
                                  "operator": operator,
                                  "op_notes": op_notes
                                      }
            
    # --- MODULE: SEM-EDX (With Multiple Images) ---
    if technique_flags["SEM-EDX"]:
        with st.expander("🔬 SEM-EDX Data", expanded=True):
            c_a, c_b = st.columns(2)
            kv = c_a.number_input("Voltage (kV)", 20)
            wd = c_b.number_input("Working Dist (mm)", 10.0)
            
            # NEW: Multiple Files Support
            sem_files = st.file_uploader("Upload SEM Images (Multi-select allowed)", 
                                       type=['jpg', 'png', 'tif'], accept_multiple_files=True, key="sem_files")
            sem_captions = st.text_area("Captions (One per line, optional)", 
                                      placeholder="e.g.\nFigure 1: Overview\nFigure 2: Inclusion detail",
                                      key = "sem_captions")
            
            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="SEM-EDX", method_name="SEM-EDX")
            report_data["SEM-EDX"] = {
                "meta": f"Acc. Voltage: {kv}kV | WD: {wd}mm",
                "images": sem_files, # List of files
                "captions": sem_captions,
                "method_ref": method_ref,
                "operator": operator, "op_notes": op_notes
            }

    # --- MODULE: OPTICAL MICROSCOPY (Multiple Images) ---
    if technique_flags["Optical"]:
        with st.expander("📷 Optical Microscopy", expanded=True):
            etchant = st.text_input("Etchant Used", "Nital 3%")
            opt_files = st.file_uploader("Upload Optical Images", 
                                       type=['jpg', 'png'], accept_multiple_files=True, key="opt_files")
            opt_captions = st.text_area("Observations", placeholder="Describe grain structure...", key="opt_captions")
            
            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="Optical", method_name="Optical")
            report_data["Optical Microscopy"] = {
                "meta": f"Etchant: {etchant} | Illumination: Brightfield",
                "images": opt_files,
                "captions": opt_captions,
                "method_ref": method_ref,
                "operator": operator, "op_notes": op_notes
            }

    # --- MODULE: PSD ---
    if technique_flags["PSD"]:
        with st.expander("📊 Particle Size Distribution", expanded=True):
            d10 = st.number_input("D10 (µm)", 0.0)
            d50 = st.number_input("D50 (µm)", 0.0)
            d90 = st.number_input("D90 (µm)", 0.0)
            dispersant = st.text_input("Dispersant agent")
            
            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="PSD", method_name="PSD")
            report_data["PSD"] = {
                "meta": "Method: Laser Diffraction", 
                "table": pd.DataFrame([
                    {"Param": "D10", "Value (µm)": d10},
                    {"Param": "D50", "Value (µm)": d50},
                    {"Param": "D90", "Value (µm)": d90}
                ]), "dispersant": dispersant,
                "method_ref": method_ref,
                "operator": operator, 
                "op_notes": op_notes
            }

    # --- MODULE: XRD (Multiple Images) ---
    if technique_flags["XRD"]:
        with st.expander("XRD", expanded=True):
        
            xrd_files = st.file_uploader("Upload XRD result", 
                                       type=['jpg', 'png'], accept_multiple_files=True, key="xrd_files")
            xrd_captions = st.text_area("Observations", placeholder="...", key="xrd_captions")
            
            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="XRD", method_name="XRD")
            report_data["XRD"] = {
                "images": xrd_files,
                "captions": xrd_captions,"method_ref": method_ref,"operator": operator, "op_notes": op_notes
            }

    # --- MODULE: TGA (Multiple Images)
    if technique_flags["TGA"]:
        with st.expander("TGA", expanded=True):            
            
            tga_files = st.file_uploader("Upload TGA result", 
                                       type=['jpg', 'png'], accept_multiple_files=True, key="tga_files")
            tga_captions = st.text_area("Observations", placeholder="...", key='tga_captions')

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="TGA", method_name="TGA")

            report_data["TGA"] = {"images": tga_files, "captions": tga_captions,"method_ref": method_ref,"operator": operator, "op_notes": op_notes}

    # --- MODULE: BET (Multiple Images)
    if technique_flags["BET"]:
        with st.expander("BET", expanded=True):
            surface_area = st.number_input("Surface area (m²/g)")
        
            bet_files = st.file_uploader("Upload BET result", 
                                       type=['jpg', 'png'], accept_multiple_files=True, key="bet_files")
            bet_captions = st.text_area("Observations", placeholder="...", key="bet_captions")

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="BET", method_name="BET")

            report_data["BET"] = {"surface area": surface_area ,"images": bet_files, "captions": bet_captions,"method_ref": method_ref,"operator": operator, "op_notes": op_notes}

    # --- MODULE: Porosity (Multiple Images + Table + Excel paste) ---
    if technique_flags["Porosity"]:
        with st.expander("🫧 Porosity", expanded=True):
            st.caption("Upload multiple images and record porosity results. You can paste a table range from Excel.")

            por_setup = st.text_input("Method / setup (optional)", "", key="por_setup")

            por_files = st.file_uploader(
                "Upload Porosity Images (Multi-select allowed)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                key="por_files"
            )

            por_captions = st.text_area(
                "Captions (One per line, optional)",
                placeholder="Figure 1: ...\nFigure 2: ...",
                key="por_captions"
            )

            df_por = editor_with_excel_paste(
                default_df=default_porosity_df(),
                editor_key="por_df",
                paste_key="por_paste",
                label="Paste Porosity table from Excel (optional)",
                apply_button_text="Apply paste to Porosity table",
                use_expander=False
            )

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="Porosity", method_name="Porosity")

            report_data["Porosity"] = {
                "meta": por_setup.strip(),
                "table": df_por,
                "images": por_files,
                "captions": por_captions,
                "method_ref": method_ref,
                "operator": operator,
                "op_notes": op_notes,
            }
# --- MODULE: Al Grain Size (Multiple Images + Table + Excel paste) ---
    if technique_flags["Al Grain Size"]:
        with st.expander("🌾 Al Grain Size (GSA)", expanded=True):
            st.caption("Upload multiple images and record grain size results. You can paste a table range from Excel.")

            gsa_standard = st.text_input("Standard / procedure", "ASTM E112", key="gsa_standard")
            gsa_etchant = st.text_input("Etchant / prep (optional)", "", key="gsa_etchant")
            gsa_mag = st.text_input("Magnification / scale (optional)", "", key="gsa_mag")

            gsa_files = st.file_uploader(
                "Upload Grain Size Images (Multi-select allowed)",
                type=["jpg", "jpeg", "png", "tif", "tiff"],
                accept_multiple_files=True,
                key="gsa_files"
            )

            gsa_captions = st.text_area(
                "Captions (One per line, optional)",
                placeholder="Figure 1: Field 1\nFigure 2: Field 2",
                key="gsa_captions"
            )

            df_gsa = editor_with_excel_paste(
                default_df=default_gsa_df(),
                editor_key="gsa_df",
                paste_key="gsa_paste",
                label="Paste Grain Size table from Excel (optional)",
                apply_button_text="Apply paste to Grain Size table",
                use_expander=False
            )

            st.divider()
            method_ref, operator, op_notes = add_operator_fields(st.container(), key_prefix="Al_Grain_Size", method_name="Al Grain Size")

            meta_parts = [f"Standard: {gsa_standard}"]
            if gsa_etchant.strip():
                meta_parts.append(f"Etchant/Prep: {gsa_etchant.strip()}")
            if gsa_mag.strip():
                meta_parts.append(f"Mag/Scale: {gsa_mag.strip()}")

            report_data["Al Grain Size"] = {
                "meta": " | ".join(meta_parts),
                "table": df_gsa,
                "images": gsa_files,
                "captions": gsa_captions,
                "method_ref": method_ref,
                "operator": operator,
                "op_notes": op_notes,
            }
    # --- MODULE: Comparison Matrix (MULTI TABLES + Excel paste) ---
    # --- MODULE: Comparison Matrix (MULTI TABLES, per-technique, no big paste area) ---
    if technique_flags["Comparison Matrix"]:
        with st.expander("📋 Comparison Matrix", expanded=True):
            st.caption("Add one or more comparison tables. Each table can be linked to a selected analysis technique.")

            if "cmp_tables_n" not in st.session_state:
                st.session_state.cmp_tables_n = 1

            selected_techniques_now = [k for k, v in technique_flags.items() if v and k != "Comparison Matrix"]
            if not selected_techniques_now:
                selected_techniques_now = ["Other"]

            cbtn1, cbtn2, _ = st.columns([1, 1, 4])
            if cbtn1.button("➕ Add table", key="cmp_add_table"):
                st.session_state.cmp_tables_n += 1
            if cbtn2.button("➖ Remove last", key="cmp_remove_table") and st.session_state.cmp_tables_n > 1:
                st.session_state.cmp_tables_n -= 1

            cmp_tables = []

            for i in range(st.session_state.cmp_tables_n):
                st.markdown(f"#### Comparison Table {i+1}")

                cA, cB = st.columns([1.2, 2.8])

                tech_choice = cA.selectbox(
                    "Analysis technique",
                    options=selected_techniques_now + (["Other"] if "Other" not in selected_techniques_now else []),
                    key=f"cmp_tech_{i}",
                )

                tech_other = ""
                if tech_choice == "Other":
                    tech_other = cA.text_input("Technique name", value="", key=f"cmp_tech_other_{i}")

                table_title = cB.text_input(
                    "Table title (optional)",
                    value="",
                    key=f"cmp_title_{i}"
                )

                df_cmp = comparison_matrix_editor(
                    default_df=default_comparison_df(),
                    editor_key=f"cmp_df_{i}",
                    label=f"Comparison table {i+1}",
                    selected_techniques=selected_techniques_now,
                )

                cmp_tables.append({
                    "technique": tech_other.strip() if tech_choice == "Other" else tech_choice,
                    "title": table_title,
                    "table": df_cmp
                })

                st.divider()

            st.markdown("##### Operator traceability (Comparison Matrix)")
            method_ref, operator, op_notes = add_operator_fields(
                st.container(), key_prefix="Comparison_Matrix", method_name=""
            )

            report_data["Comparison Matrix"] = {
                "meta": "",
                "tables": cmp_tables,
                "method_ref": method_ref,
                "operator": operator,
                "op_notes": op_notes,
            }

    
        
    # --- MODULE: Chemical analysis (grouped, multiple sub-methods) ---
    chem_selected = [m for m in CHEM_METHODS if technique_flags.get(m)]
    if chem_selected:
        with st.expander("🧪 Chemical analysis", expanded=True):
            st.caption("Add operator name + notes per method. Important far tracebility.")

            tabs = st.tabs(chem_selected)
            for tab, method in zip(tabs, chem_selected):
                with tab:
                    df_key = f"chem_{safe_key(method)}"
                    if method == "C, S, N Analysis":
                        df0 = pd.DataFrame([{"Element": "C", "Result (%)": 0.0},
                                            {"Element": "S", "Result (%)": 0.0},
                                            {"Element": "N", "Result (%)": 0.0}])
                    elif method == "F Analysis":
                        df0 = pd.DataFrame([{"Element": "F", "Result (mg/kg)": 0.0}])
                    elif method == "Metallic Al Analysis":
                        df0 = pd.DataFrame([{"Element": "Al(met)", "Result (%)": 0.0}])
                    else:  # Metallic Si Analysis
                        df0 = pd.DataFrame([{"Element": "Si(met)", "Result (%)": 0.0}]) 
                    df = st.data_editor(df0, num_rows="dynamic", use_container_width=True, key=df_key)

                    st.divider()
                    method_ref, operator, op_notes = add_operator_fields(tab, key_prefix=safe_key(method), method_name=method)

                    report_data[method] = {
                        "group": "Chemical analysis",
                        #"meta": f"Method: {method}",
                        "table": df,
                        "method_ref": method_ref,
                        "operator": operator,
                        "op_notes": op_notes,
                    }


    #st.subheader("Analyst & Analysis Notes")
    #c1, c2 = st.columns(2)
    #analyst_name = c1.text_input("Analyst name", "Analyst Name")
    #analysis_notes = c2.text_area("Analysis notes (for report)", height=90, placeholder="Short technical interpretation / approach...")
    st.divider()
    # --- 4. CONCLUSION ---
    st.subheader("4. Conclusions")
    summary = st.text_area("Final Remarks", height=100)

    st.divider()
    
    st.subheader("5. Report Sign-off")
    c1, c2 = st.columns(2)

    with c1:
        reported_by = st.text_input("Reported by (name)", "Reported by")
        reported_title = st.text_input("Reported by (title)", "Analyst")

    with c2:
        reviewed_by = st.text_input("Reviewed by (name)", "Reviewed by")
        reviewed_title = st.text_input("Reviewed by (title)", " Reviewer")


    st.divider()
    # ==========================================================================
    # 5. GENERATE PDF (Fixed "Blank Page" Issue)
    # ==========================================================================
    if st.button("🚀 Generate PDF Report", type="primary"):
        # A. Setup
        theme = Theme()
        meta = {
            "lab_name": lab_name, "lab_addr": lab_addr, "lab_no" :lab_no, "lab_mail":lab_mail,
            #"report_id": report_id, 
            "project_title": project_title,
            "requester": requester, # Added Requester
            "report_date": str(datetime.date.today()),
            "logo_path": save_upload(logo)
        }
        

        # B. Create Instance
        pdf = AnalyticalReportPDF(theme, meta)
        pdf.add_page()

        # Techniques included (based on checkboxes; stable order)
        technique_order = [
            "ICP-OES", "XRF", "C, S, N Analysis", "F Analysis",
            "Metallic Al Analysis", "Metallic Si Analysis",
            "SEM-EDX", "Optical", "Porosity",
            "PSD", "BET", "XRD", "TGA",
            "Al Grain Size", "Comparison Matrix"
        ]
        techniques_included = [k for k in technique_order if technique_flags.get(k)]
        #if not techniques_included:
          #  techniques_included = list(report_data.keys())

        

        

        # If you grouped chemical methods under one section, you still want them listed individually here:
        # (They exist as separate keys anyway, so the loop above includes them.)

        #pdf.add_techniques_table(tech_rows, title="Techniques included")


        # C. Info Block
        # C. Intro / Traceability block (first part of the report)
        pdf.section_header_keep(f"1. Project: {context}", first_block_mm=70.0)
        # --- Report Metadata block (QC-friendly) ---
        metadata_items = [
            #("Report ID", sample_id),
            ("Project/Report ID", sample_id),
            ("Requester", requester),
            ("Sample received", str(rx_date)),
            ("Project title", project_title),
            ("Report date", str(datetime.date.today())),
            ("Project", context),
            ("Environment", f"{temp}°C / {hum}% RH"),
            
            #("Lab", "Global R&D, Enschede"),
        ]
        pdf.add_kv_box(metadata_items, title="Report and Sample Information", cols=2)

        pdf.set_font(theme.font_main, "", 10)
        pdf.set_text_color(*theme.text_dark)

        # ---- Report header info moved into body (fixes overlap) ----
        #pdf.cell(40, 6, "Report ID:", 0, 0); pdf.cell(0, 6, report_id, 0, 1)
        #pdf.cell(40, 6, "Requester:", 0, 0); pdf.cell(0, 6, requester, 0, 1)
        #pdf.cell(40, 6, "Report date:", 0, 0); pdf.cell(0, 6, str(datetime.date.today()), 0, 1)

        pdf.ln(2)

        sample_photo_path = save_upload_as_jpg(sample_photo) if sample_photo else None

        if sample_photo_path and os.path.exists(sample_photo_path):
            pdf.subtitle("Sample photo (as received)")
            pdf.add_framed_image(sample_photo_path, caption="Sample as received")
            cleanup_file(sample_photo_path)

        # ---- sample condition/requester notes ----
        if sample_condition.strip():
            pdf.subtitle("Requester notes and Sample condition")
            pdf.set_font(theme.font_main, "", 10)
            pdf.set_text_color(*theme.text_dark)
            pdf.multi_cell(0, 6, sample_condition.strip())
            pdf.ln(1)

        # --- Techniques table ---
        pdf.subtitle("1.1 Techniques included")
        tech_rows = []
        for tech in techniques_included:
            d = report_data.get(tech, {})
            method_sop = (d.get("method_ref") or "").strip()
            if tech == "Comparison Matrix":
                tables = d.get("tables") or []
                used = []
                for b in tables:
                    tname = (b.get("technique") or "").strip()
                    if tname:
                        used.append(tname)
                used = sorted(set(used))
                if used:
                    method_sop = ", ".join(used)


            tech_rows.append({
                "Technique": tech,
                "Method/SOP": method_sop,
                "Output": technique_output_hint(tech),
            })
     
        pdf.add_techniques_table(tech_rows, title="")

        # ---- Sample traceability ----
        #pdf.cell(40, 6, "Sample ID:", 0, 0); pdf.cell(0, 6, sample_id, 0, 1)
        #pdf.cell(40, 6, "Received:", 0, 0); pdf.cell(0, 6, str(rx_date), 0, 1)
        #pdf.cell(40, 6, "Environment:", 0, 0); pdf.cell(0, 6, f"{temp}°C / {hum}% RH", 0, 1)

        


        # ✅ Techniques included (expected content)
        #pdf.ln(2)
        #pdf.set_font(theme.font_main, "B", 10)
        #pdf.cell(40, 6, "Techniques included:", 0, 0)

        #pdf.set_font(theme.font_main, "", 10)
        #pdf.multi_cell(0, 6, ", ".join(techniques_included) if techniques_included else "None selected")
        pdf.ln(2)

        # Group chemical methods under one PDF section
        chem_items = [(k, v) for k, v in report_data.items() if v.get("group") == "Chemical analysis"]
        non_chem_items = [(k, v) for k, v in report_data.items() if v.get("group") != "Chemical analysis"]

        pdf.section_header_keep("2. Analysis results", first_block_mm=35.0)

        # D. Loop through results
                # --- Non-chemical results ---
        result_index = 1

        # Non-chemical: show only items actually present in report_data, but in the technique_order order
        for tech in techniques_included:
            if tech not in report_data:
                continue
            data = report_data[tech]
            if data.get("group") == "Chemical analysis":
                continue

            # reserve space so subtitle doesn't orphan
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

            # meta
            if data.get("meta"):
                pdf.set_font(theme.font_mono, "", 8)
                pdf.set_text_color(*theme.text_gray)
                pdf.multi_cell(0, 5, str(data["meta"]))
                pdf.ln(1)
                pdf.set_text_color(*theme.text_dark)

            # ✅ operator notes ALWAYS (table or image)
            pdf_operator_block(pdf, theme, data.get("operator", ""), data.get("op_notes", ""))

            # table(s)
            if tech == "Comparison Matrix":
                tables = data.get("tables") or []
                for j, block in enumerate(tables, start=1):
                    df_cmp = block.get("table") if isinstance(block.get("table"), pd.DataFrame) else None
                    if df_cmp is None or df_cmp.empty:
                        continue

                    #method_used = (block.get("method_used") or "").strip()
                    #table_title = (block.get("title") or "").strip()

                    #ensure_space(pdf, 14.0 + estimate_table_height_mm(df_cmp))

                    #pdf.set_font(theme.font_main, "B", 10)
                    #pdf.set_text_color(*theme.text_dark)
                    #pdf.cell(0, 6, pdf_safe_text(f"Method used: {method_used or '-'}"), 0, 1, "L")

                    tech_used = (block.get("technique") or "").strip()
                    pdf.set_font(theme.font_main, "B", 10)
                    pdf.set_text_color(*theme.text_dark)
                    pdf.cell(0, 6, pdf_safe_text(f"{tech_used or 'Technique'} selected"), 0, 1, "L")

                    if table_title:
                        pdf.set_font(theme.font_main, "I", 9)
                        pdf.set_text_color(*theme.text_gray)
                        pdf.cell(0, 5, pdf_safe_text(table_title), 0, 1, "L")
                        pdf.set_text_color(*theme.text_dark)

                    pdf.add_zebra_table(df_cmp)
                    pdf.ln(1)

            else:
                if isinstance(data.get("table"), pd.DataFrame) and not data["table"].empty:
                    pdf.add_zebra_table(data["table"])


            # images
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
                        pdf.add_framed_image(tmp_path, cap)
                finally:
                    for p in tmp_paths:
                        cleanup_file(p)

        # --- Chemical analysis as one section ---
        if chem_items:
            ensure_space(pdf, 18.0)
            pdf.subtitle(f"2.{result_index} Chemical analysis")
            result_index += 1

            sub_i = 1
            for method_name, data in chem_items:
                df = data.get("table") if isinstance(data.get("table"), pd.DataFrame) else None

                # keep method subtitle + operator + start of table together
                keep_method_with_first_content(pdf, df=df)

                pdf.set_font(theme.font_main, "B", 10)
                pdf.set_text_color(*theme.text_dark)
                pdf.cell(0, 6, f"2.{result_index-1}.{sub_i} {method_name}", 0, 1, "L")
                sub_i += 1

                if data.get("meta"):
                    pdf.set_font(theme.font_mono, "", 8)
                    pdf.set_text_color(*theme.text_gray)
                    pdf.multi_cell(0, 5, str(data["meta"]))
                    pdf.set_text_color(*theme.text_dark)

                pdf_operator_block(pdf, theme, data.get("operator", ""), data.get("op_notes", ""), title="Method traceability")

                if df is not None:
                    pdf.add_zebra_table(df)




        # E. Summary
        pdf.section_header_keep("3. Conclusions / summary", first_block_mm=30.0)
        pdf.set_font(theme.font_main, "", 10)
        pdf.set_text_color(*theme.text_dark)
        pdf.multi_cell(0, 6, summary or "")


        pdf.section_header_keep("4. Sign-off", first_block_mm=25.0)
        pdf.add_signoff_two_columns(
            left_label="Reported by",
            left_name=reported_by,
            left_title=reported_title,
            right_label="Reviewed by",
            right_name=reviewed_by,
            right_title=reviewed_title
)


        
        # F. Output (THE FIX)
        try:
            # ENGINEERING FIX: Use bytes() conversion explicitly
            # fpdf2 output() returns bytearray by default in newer versions, 
            # but we force explicit bytes to be safe for Streamlit.     
            out = pdf.output(dest="S")
            if isinstance(out, str):
                # safest: ignore any last-moment unicode weirdness
                pdf_bytes = out.encode("latin-1", "ignore")
            else:
                pdf_bytes = bytes(out)

            cleanup_file(meta["logo_path"])

            st.success("✅ Report Generated!")
            st.download_button(
                label="📥 Download PDF",
                data=pdf_bytes,
                file_name=f"{sample_id}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
