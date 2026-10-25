"""Configuration, constants, path discovery, and date calculations for VoiceOps 2026.

STRICT COMPLIANCE RULES:
1. Reference Date: 2026-10-20 (20 October 2026) frozen reference date.
2. Calendar Days: Return eligibility windows (e.g. RET-2, RET-8).
3. Business Days: Monday to Saturday (skipping Sundays and holidays.csv dates).
4. CSV Loading: String types only (keep_default_na=False, dtype=str).
"""
import os
import csv
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional, Set
from dotenv import load_dotenv

# Load .env file if available
load_dotenv()

# ==============================================================================
# FIXED REFERENCE DATE (20 October 2026)
# ==============================================================================
REFERENCE_DATE = date(2026, 10, 20)
REFERENCE_DATE_STR = "2026-10-20"

# ==============================================================================
# PATH RESOLUTION & DISCOVERY
# ==============================================================================
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)

def find_dataset_dir() -> str:
    """Auto-detect the root directory of the Capstone dataset."""
    env_dir = os.environ.get("CAPSTONE_DATASET_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, "validate_dataset.py")):
        return os.path.abspath(env_dir)

    search_roots = [
        os.path.join(_PROJECT_ROOT, "..", "Capstone_Project_Dataset_FTTT_2026", "Capstone_Project_Dataset_FTTT_2026"),
        os.path.join(_PROJECT_ROOT, "Capstone_Project_Dataset_FTTT_2026", "Capstone_Project_Dataset_FTTT_2026"),
        os.path.join(_PROJECT_ROOT, "..", "Capstone_Project_Dataset_FTTT_2026"),
        os.path.join(_PROJECT_ROOT, "Capstone_Project_Dataset_FTTT_2026"),
        os.path.join(_PROJECT_ROOT, ".."),
        _PROJECT_ROOT,
    ]
    for candidate in search_roots:
        cand_norm = os.path.abspath(candidate)
        if os.path.exists(os.path.join(cand_norm, "validate_dataset.py")):
            return cand_norm

    # Fallback to current working directory tree
    for root, dirs, files in os.walk(os.path.dirname(_PROJECT_ROOT)):
        if "validate_dataset.py" in files and "kb" in dirs:
            return os.path.abspath(root)

    return os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "Capstone_Project_Dataset_FTTT_2026", "Capstone_Project_Dataset_FTTT_2026"))

DATASET_DIR = find_dataset_dir()
DATA_DIR = os.path.join(DATASET_DIR, "data")
KB_DIR = os.path.join(DATASET_DIR, "kb")
PRACTICE_DIR = os.path.join(DATASET_DIR, "practice")
TOOLS_DIR = os.path.join(DATASET_DIR, "tools")
SAMPLE_OUTPUTS_DIR = os.path.join(_PROJECT_ROOT, "sample_outputs")

# ==============================================================================
# HOLIDAYS & BUSINESS DAYS
# ==============================================================================
# Default warehouse holidays as defined in KB-00.3 / data/holidays.csv
DEFAULT_HOLIDAYS = {
    "2026-08-15": "Independence Day",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-19": "Warehouse closure (festival)",
    "2026-11-09": "Warehouse closure (festival)",
}

def load_holidays() -> Dict[str, str]:
    """Load holidays from data/holidays.csv or return default dictionary."""
    holidays = dict(DEFAULT_HOLIDAYS)
    holidays_file = os.path.join(DATA_DIR, "holidays.csv")
    if os.path.exists(holidays_file):
        with open(holidays_file, mode="r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                dt = row.get("date", "").strip()
                name = row.get("name", "").strip()
                if dt:
                    holidays[dt] = name
    return holidays

HOLIDAYS = load_holidays()

def is_business_day(d: date) -> bool:
    """Return True if date is Monday-Saturday and not a warehouse holiday.
    
    Sunday is weekday 6.
    """
    if d.weekday() == 6:  # Sunday
        return False
    d_str = d.strftime("%Y-%m-%d")
    if d_str in HOLIDAYS:
        return False
    return True

def add_business_days(start_date: date, num_days: int) -> date:
    """Add num_days business days (skipping Sundays and warehouse holidays).
    
    start_date is Day 0. The first business day after start_date is Day 1.
    """
    current = start_date
    counted = 0
    while counted < num_days:
        current += timedelta(days=1)
        if is_business_day(current):
            counted += 1
    return current

def add_calendar_days(start_date: date, num_days: int) -> date:
    """Add calendar days (used for return eligibility windows)."""
    return start_date + timedelta(days=num_days)

def parse_iso_date(d_str: str) -> Optional[date]:
    """Parse YYYY-MM-DD or return None."""
    if not d_str:
        return None
    clean = str(d_str).strip()[:10]
    try:
        return datetime.strptime(clean, "%Y-%m-%d").date()
    except Exception:
        return None

# ==============================================================================
# SPEECH & EAR FORMATTING HELPERS (Rule STY-3)
# ==============================================================================
MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def format_date_for_ear(d: date) -> str:
    """Format date for ear with month name and full year (e.g. '27 October 2026')."""
    return f"{d.day} {MONTH_NAMES[d.month]} {d.year}"

def format_order_id_for_ear(order_id: str) -> str:
    """Space 8-digit order number into two groups of 4 (e.g. '4663 7226')."""
    digits = "".join(ch for ch in str(order_id) if ch.isdigit())
    if len(digits) == 8:
        return f"{digits[:4]} {digits[4:]}"
    return digits

def format_inr_for_ear(amount: int) -> str:
    """Format currency in rupees."""
    return f"₹{amount}"

# ==============================================================================
# SAFE CSV DATA LOADER (Preserving IDs as strings)
# ==============================================================================
def read_csv_as_dicts(filepath: str) -> List[Dict[str, str]]:
    """Read a CSV file preserving all columns as exact strings with no NaN values."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: (v if v is not None else "") for k, v in row.items()})
        return rows

# ==============================================================================
# API KEYS & MODELS
# ==============================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "")

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GNANI_VOICE = "Kaveri"
DEFAULT_GNANI_MODEL = "timbre-v2.5"
