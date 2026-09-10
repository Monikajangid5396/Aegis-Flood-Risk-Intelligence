from pathlib import Path

import joblib
import pandas as pd
import requests
import difflib
import re
import pydeck as pdk
import streamlit as st


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Aegis | Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_DATA_PATH = BASE_DIR / "data" / "processed" / "aegis_risk_dashboard_data.csv"
PIN_DATA_PATH = BASE_DIR / "data" / "processed" / "rajasthan_pincode_directory.csv"
HISTORICAL_DATA_PATH = BASE_DIR / "data" / "processed" / "rajasthan_flood_weather_merged_1986_2025.csv"
MODEL_PATH = BASE_DIR / "models" / "ridge_flood_risk_model.joblib"


# ================================================================
# CONSTANTS
# ================================================================

RISK_ORDER = ["Low", "Moderate", "High", "Very High"]
RISK_SCORE = {"Low": 1, "Moderate": 2, "High": 3, "Very High": 4}

RISK_THRESHOLDS = {
    "Low": 0.9675,
    "Moderate": 3.35,
    "High": 6.6525,
}

RISK_UI = {
    "Low": {"icon": "🟢", "label": "Low Risk"},
    "Moderate": {"icon": "🟡", "label": "Moderate Risk"},
    "High": {"icon": "🟠", "label": "High Risk"},
    "Very High": {"icon": "🔴", "label": "Very High Risk"},
}

FEATURE_LABELS = {
    "avg_annual_rainfall_mm": "Annual Rainfall (mm)",
    "avg_temperature_c": "Temperature (°C)",
    "avg_humidity_percent": "Humidity (%)",
    "avg_wind_speed_mps": "Wind Speed (m/s)",
    "avg_pressure_kpa": "Pressure (kPa)",
    "max_daily_rainfall_mm": "Maximum Daily Rainfall (mm)",
}

DISTRICT_COORDINATES = {
    "AJMER": (26.452103, 74.638667),
    "ALWAR": (27.562461, 76.625004),
    "BANSWARA": (23.541091, 74.442501),
    "BARAN": (25.100000, 76.516667),
    "BARMER": (25.745717, 71.392112),
    "BHARATPUR": (27.217312, 77.490091),
    "BHILWARA": (25.347071, 74.640812),
    "BUNDI": (25.438547, 75.637350),
    "CHITTAURGARH": (24.889629, 74.624033),
    "DHAULPUR": (26.692864, 77.879677),
    "DUNGARPUR": (23.843059, 73.714657),
    "HANUMANGARH": (29.110000, 74.600000),
    "JAIPUR": (26.912434, 75.787270),
    "JALORE": (25.345577, 72.615595),
    "JHALAWAR": (24.416667, 76.250000),
    "JODHPUR": (26.448932, 73.006391),
    "KARAULI": (26.498306, 77.027550),
    "KOTA": (25.175117, 75.844116),
    "NAGAUR": (27.202011, 73.733940),
    "PALI": (25.772765, 73.323355),
    "PRATAPGARH": (24.032154, 74.781616),
    "RAJSAMAND": (25.071450, 73.879795),
    "SAWAI MADHOPUR": (26.023005, 76.344080),
    "SIROHI": (24.888383, 72.847939),
    "SRI GANGANAGAR": (29.920085, 73.874958),
    "TONK": (26.166377, 75.788240),
    "UDAIPUR": (24.585840, 73.713460),
}

ALIASES = {
    "RAJ SAMAND": "RAJSAMAND",
    "SRI GANGANAGAR": "GANGANAGAR",
    "GANGANAGAR": "SRI GANGANAGAR",
}


# ================================================================
# DATA LOADING
# ================================================================

@st.cache_data

def load_risk_data():
    if not RISK_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RISK_DATA_PATH)


@st.cache_data

def load_pin_data():
    if not PIN_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(PIN_DATA_PATH)


@st.cache_data

def load_historical_data():
    if not HISTORICAL_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(HISTORICAL_DATA_PATH)


@st.cache_resource

def load_prediction_model():
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


risk_data = load_risk_data()
pin_data = load_pin_data()
historical_data = load_historical_data()
model_package = load_prediction_model()


# ================================================================
# HELPERS
# ================================================================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return (
        str(value).strip().upper().replace("-", " ").replace(".", "")
        .replace("'", "").replace("(", "").replace(")", "")
    )


def normalize_pincode(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    return value.zfill(6)


@st.cache_data(ttl=3600)
def lookup_pincode(pincode):
    api_url = f"https://aniket-thapa.github.io/india-pincode-api/pincodes/{pincode}.json"
    try:
        response = requests.get(api_url, timeout=15)
        if response.status_code == 404:
            return None, "PIN code not found."
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, "No location information found."
        return data, None
    except requests.RequestException:
        return None, "PIN lookup service is temporarily unavailable."
    except ValueError:
        return None, "Invalid response received from PIN service."


def get_district_risk(district_name):
    target = normalize_text(district_name)
    if risk_data.empty or "district" not in risk_data.columns:
        return pd.DataFrame()

    copy = risk_data.copy()
    copy["_district_normalized"] = copy["district"].apply(normalize_text)
    matched = copy[copy["_district_normalized"] == target].copy()
    if not matched.empty:
        return matched

    alias = ALIASES.get(target)
    if alias:
        return copy[copy["_district_normalized"] == alias].copy()
    return pd.DataFrame()



def _subdistrict_mapping(data):
    if data.empty or "sub_district" not in data.columns:
        return {}

    mapping = {}
    for value in data["sub_district"].dropna().astype(str):
        original = value.strip()
        key = normalize_text(original)
        key = re.sub(r"[^A-Z0-9]", "", key)
        if key:
            mapping.setdefault(key, original)
    return mapping


def find_subdistrict_matches(query, data, limit=50, cutoff=0.55):
    """Generic typo-tolerant search for all available sub-districts."""
    if not query or data.empty or "sub_district" not in data.columns:
        return data.iloc[0:0].copy()

    cleaned = re.sub(r"[^A-Z0-9]", "", normalize_text(query))
    if not cleaned:
        return data.iloc[0:0].copy()

    mapping = _subdistrict_mapping(data)
    if not mapping:
        return data.iloc[0:0].copy()

    choices = list(mapping.keys())

    # Exact/partial first.
    direct = [key for key in choices if cleaned in key or key in cleaned]

    # Then spelling-tolerant matching.
    fuzzy = difflib.get_close_matches(
        cleaned,
        choices,
        n=limit,
        cutoff=cutoff,
    )

    ordered = []
    for key in direct + fuzzy:
        if key not in ordered:
            ordered.append(key)

    names = [mapping[key] for key in ordered[:limit]]
    return data[
        data["sub_district"].astype(str).str.strip().isin(names)
    ].copy()


def best_subdistrict_correction(query, data, cutoff=0.55):
    """Return the closest canonical spelling for a user's query."""
    if not query or data.empty or "sub_district" not in data.columns:
        return None

    cleaned = re.sub(r"[^A-Z0-9]", "", normalize_text(query))
    mapping = _subdistrict_mapping(data)

    if not cleaned or not mapping:
        return None

    match = difflib.get_close_matches(
        cleaned,
        list(mapping.keys()),
        n=1,
        cutoff=cutoff,
    )
    return mapping[match[0]] if match else None



def category_from_percent(value):
    if value <= RISK_THRESHOLDS["Low"]:
        return "Low"
    if value <= RISK_THRESHOLDS["Moderate"]:
        return "Moderate"
    if value <= RISK_THRESHOLDS["High"]:
        return "High"
    return "Very High"


def create_map_data(data):
    if data.empty:
        return pd.DataFrame()

    rows = []
    for district, group in data.groupby("district"):
        key = normalize_text(district)
        coords = DISTRICT_COORDINATES.get(key)
        if coords is None:
            continue

        max_score = int(group["risk_score"].max())
        category = RISK_ORDER[max_score - 1] if 1 <= max_score <= 4 else "Low"
        rows.append({
            "district": district,
            "latitude": coords[0],
            "longitude": coords[1],
            "risk_category": category,
            "very_high_risk": int((group["risk_category"] == "Very High").sum()),
            "high_risk": int((group["risk_category"] == "High").sum()),
            "max_flood_percent": float(group["flood_affected_percent"].max()),
            "avg_flood_percent": float(group["flood_affected_percent"].mean()),
            "risk_score": max_score,
            "color": {
                "Low": [40, 167, 69],
                "Moderate": [255, 193, 7],
                "High": [255, 140, 0],
                "Very High": [220, 53, 69],
            }[category],
        })
    return pd.DataFrame(rows)


def risk_message(category):
    return {
        "Low": {
            "title": "Normal Monitoring",
            "message": "The model estimate is in the relatively low-risk range.",
            "actions": [
                "Continue routine rainfall and weather monitoring.",
                "Keep local drainage systems maintained.",
                "Stay informed about official weather updates.",
            ],
        },
        "Moderate": {
            "title": "Preventive Preparedness",
            "message": "Preventive action is recommended because conditions may become more vulnerable to flooding.",
            "actions": [
                "Monitor rainfall and changing weather conditions closely.",
                "Inspect drainage and waterlogging-prone areas.",
                "Keep emergency contacts and basic resources ready.",
            ],
        },
        "High": {
            "title": "Enhanced Monitoring",
            "message": "Elevated flood risk is indicated; preparedness actions should be initiated.",
            "actions": [
                "Closely monitor rainfall and official weather information.",
                "Check drainage and known waterlogging-prone locations.",
                "Keep emergency response resources ready.",
                "Review local emergency response procedures.",
            ],
        },
        "Very High": {
            "title": "Emergency Preparedness",
            "message": "The model estimate is in the very-high-risk range. Enhanced preparedness and monitoring are recommended.",
            "actions": [
                "Activate enhanced flood-risk monitoring.",
                "Prepare evacuation and emergency response plans.",
                "Alert vulnerable and high-risk areas through appropriate channels.",
                "Keep emergency teams and resources ready.",
                "Follow instructions from local disaster-management authorities.",
            ],
        },
    }[category]


def render_risk_badge(category):
    icon = RISK_UI.get(category, {}).get("icon", "⚪")
    st.markdown(f"**{icon} {category} Risk**")


# ================================================================
# SAFETY CHECK
# ================================================================

if risk_data.empty:
    st.error("Aegis risk dataset could not be loaded. Check the processed data path.")
    st.stop()





# ================================================================
# AEGIS PREMIUM VISUAL SYSTEM — SINGLE CONSOLIDATED CSS
# High-contrast + 3D flood/water atmosphere + hover interactions
# ================================================================

st.markdown("""
<style>
/* ============================================================
   AEGIS — WORLD-CLASS FLOOD INTELLIGENCE UI
   Absolute text contrast + premium 3D water environment
   ============================================================ */

:root{
  --ink:#0b1b2b;
  --ink-2:#20364b;
  --muted:#52677b;
  --muted-2:#71859a;
  --white:#ffffff;
  --surface:#ffffff;
  --surface-glass:rgba(255,255,255,.965);
  --line:#d5e3eb;
  --teal:#087f86;
  --teal-dark:#075e68;
  --cyan:#079bc0;
  --navy:#06182b;
}

/* ============================================================
   1. BASE — LIGHT, STABLE, ALWAYS READABLE
   ============================================================ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main{
  background:#f2f8fb !important;
  color:var(--ink) !important;
}

[data-testid="stAppViewContainer"]{
  position:relative !important;
  overflow-x:hidden !important;
}

[data-testid="stHeader"]{
  background:rgba(242,248,251,.88) !important;
  backdrop-filter:blur(18px) saturate(140%) !important;
  border-bottom:1px solid rgba(213,227,235,.7) !important;
}

.block-container{
  max-width:1500px !important;
  padding:1.1rem 2rem 3rem !important;
  position:relative !important;
  z-index:20 !important;
}

/* ============================================================
   2. 3D FLOOD WORLD — BACKGROUND ONLY
   It can NEVER sit over text or controls.
   ============================================================ */
.aegis-scene{
  position:fixed !important;
  inset:0 !important;
  z-index:0 !important;
  pointer-events:none !important;
  overflow:hidden !important;
  background:
    radial-gradient(ellipse at 82% 10%,rgba(31,196,190,.17),transparent 25%),
    radial-gradient(ellipse at 8% 88%,rgba(18,154,196,.12),transparent 28%),
    radial-gradient(ellipse at 50% 52%,rgba(83,204,214,.055),transparent 42%),
    linear-gradient(145deg,#fbfdfe 0%,#edf7fa 48%,#f5fbfd 100%);
}

/* giant translucent water planets */
.aegis-orb{
  position:absolute;
  border-radius:50%;
  pointer-events:none;
  opacity:.72;
  filter:blur(.15px);
  box-shadow:
    inset -70px -60px 110px rgba(2,45,65,.10),
    inset 34px 25px 72px rgba(255,255,255,.84),
    0 38px 110px rgba(3,69,92,.11);
}
.aegis-orb.one{
  width:560px;height:560px;
  right:-185px;top:35px;
  background:
    radial-gradient(circle at 29% 20%,rgba(255,255,255,.88),transparent 12%),
    radial-gradient(circle at 40% 34%,rgba(93,234,212,.30),transparent 29%),
    radial-gradient(circle at 65% 67%,rgba(10,146,179,.17),transparent 55%),
    radial-gradient(circle at 50% 50%,rgba(154,232,236,.25),transparent 72%);
  animation:aegisOrb1 17s ease-in-out infinite alternate;
}
.aegis-orb.two{
  width:350px;height:350px;
  left:-145px;bottom:-20px;
  background:
    radial-gradient(circle at 31% 22%,rgba(255,255,255,.88),transparent 13%),
    radial-gradient(circle at 60% 60%,rgba(45,212,191,.20),transparent 52%),
    radial-gradient(circle at 45% 45%,rgba(56,189,248,.13),transparent 70%);
  animation:aegisOrb2 20s ease-in-out infinite alternate;
}
.aegis-orb.three{
  width:145px;height:145px;
  right:25%;top:31%;
  background:
    radial-gradient(circle at 30% 24%,rgba(255,255,255,.92),transparent 14%),
    radial-gradient(circle at 55% 58%,rgba(8,155,184,.14),transparent 65%);
  animation:aegisOrb3 12s ease-in-out infinite alternate;
}
@keyframes aegisOrb1{
  0%{transform:translate3d(0,0,0) rotate(0deg) scale(.95)}
  50%{transform:translate3d(-58px,34px,0) rotate(6deg) scale(1.05)}
  100%{transform:translate3d(25px,-32px,0) rotate(-5deg) scale(1)}
}
@keyframes aegisOrb2{
  0%{transform:translate3d(0,0,0) scale(.94)}
  100%{transform:translate3d(62px,-42px,0) scale(1.08)}
}
@keyframes aegisOrb3{
  0%{transform:translate3d(0,0,0) scale(.92)}
  100%{transform:translate3d(-34px,28px,0) scale(1.13)}
}

/* animated 3D terrain / flood grid */
.aegis-grid{
  position:absolute;
  left:-18%;right:-18%;bottom:-290px;
  height:680px;
  opacity:.20;
  background-image:
    linear-gradient(rgba(4,121,135,.38) 1px,transparent 1px),
    linear-gradient(90deg,rgba(4,121,135,.38) 1px,transparent 1px);
  background-size:52px 52px;
  transform:perspective(560px) rotateX(63deg) scale(1.42);
  transform-origin:center bottom;
  -webkit-mask-image:linear-gradient(to top,#000 0%,transparent 88%);
  mask-image:linear-gradient(to top,#000 0%,transparent 88%);
  animation:aegisGrid 16s linear infinite;
}
@keyframes aegisGrid{
  from{background-position:0 0,0 0}
  to{background-position:0 52px,52px 0}
}

/* water radar rings */
.aegis-ripple{
  position:absolute;
  width:430px;height:235px;
  right:5%;top:13%;
  border:1px solid rgba(4,121,135,.13);
  border-radius:50%;
  box-shadow:
    0 0 0 28px rgba(4,121,135,.035),
    0 0 0 58px rgba(4,121,135,.027),
    0 0 0 91px rgba(4,121,135,.020),
    0 0 0 128px rgba(4,121,135,.013);
  transform:rotate(-9deg);
  animation:aegisRipple 9s ease-in-out infinite alternate;
}
@keyframes aegisRipple{
  from{transform:translate3d(0,0,0) rotate(-9deg) scale(.94);opacity:.48}
  to{transform:translate3d(-45px,30px,0) rotate(-4deg) scale(1.08);opacity:.95}
}

/* layered water contour */
.aegis-wave{
  position:absolute;
  left:-7%;right:-7%;bottom:6%;
  height:150px;
  border-radius:50%;
  border-top:1px solid rgba(5,126,137,.12);
  box-shadow:
    0 -22px 0 -21px rgba(5,126,137,.085),
    0 -44px 0 -43px rgba(5,126,137,.060),
    0 -66px 0 -65px rgba(5,126,137,.042),
    0 -88px 0 -87px rgba(5,126,137,.028);
  animation:aegisWave 8s ease-in-out infinite alternate;
}
@keyframes aegisWave{
  from{transform:translateX(-30px) rotate(-3deg)}
  to{transform:translateX(38px) rotate(1deg)}
}

/* subtle rain particles */
.aegis-rain{
  position:absolute;
  inset:0;
  opacity:.10;
}
.aegis-rain i{
  position:absolute;
  top:-14%;
  width:1px;height:52px;
  border-radius:999px;
  background:linear-gradient(to bottom,transparent,rgba(8,145,178,.72),transparent);
  transform:rotate(12deg);
  animation:aegisRain linear infinite;
}
.aegis-rain i:nth-child(1){left:7%;animation-duration:8s}
.aegis-rain i:nth-child(2){left:18%;animation-duration:11s;animation-delay:-4s}
.aegis-rain i:nth-child(3){left:29%;animation-duration:9s;animation-delay:-2s}
.aegis-rain i:nth-child(4){left:41%;animation-duration:12s;animation-delay:-8s}
.aegis-rain i:nth-child(5){left:53%;animation-duration:8s;animation-delay:-5s}
.aegis-rain i:nth-child(6){left:65%;animation-duration:10s;animation-delay:-3s}
.aegis-rain i:nth-child(7){left:77%;animation-duration:9s;animation-delay:-6s}
.aegis-rain i:nth-child(8){left:89%;animation-duration:12s;animation-delay:-9s}
@keyframes aegisRain{
  from{transform:translate3d(0,-10vh,0) rotate(12deg)}
  to{transform:translate3d(-72px,125vh,0) rotate(12deg)}
}

/* ============================================================
   3. CONTENT LAYER
   ============================================================ */
.main .block-container{
  position:relative !important;
  z-index:20 !important;
}

/* Every normal text node gets a dark fallback */
.main [data-testid="stMarkdownContainer"],
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stMarkdownContainer"] span,
.main [data-testid="stMarkdownContainer"] li,
.main [data-testid="stCaptionContainer"],
.main [data-testid="stCaptionContainer"] p,
.main label,
.main h1,.main h2,.main h3,.main h4,.main h5,.main h6{
  color:var(--ink) !important;
}

.main [data-testid="stCaptionContainer"],
.main [data-testid="stCaptionContainer"] *{
  color:var(--muted) !important;
}

/* ============================================================
   4. SIDEBAR — FIXES THE EXACT BLACK-SELECT PROBLEM
   ============================================================ */
[data-testid="stSidebar"]{
  background:
    linear-gradient(180deg,#ffffff 0%,#f9fcfd 58%,#f1f8fa 100%) !important;
  border-right:1px solid #d4e3eb !important;
  color:var(--ink) !important;
  z-index:50 !important;
}

[data-testid="stSidebar"] *{
  color:var(--ink) !important;
}

/* sidebar headings/captions */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{
  color:#102033 !important;
}

/* ===== SELECT BOX: FORCE WHITE EVERYWHERE ===== */
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] [role="combobox"]{
  background:#ffffff !important;
  background-color:#ffffff !important;
  color:#102033 !important;
  border-color:#c8d7e1 !important;
  -webkit-text-fill-color:#102033 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] input{
  background:#ffffff !important;
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
  caret-color:#087f86 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] [data-baseweb="select"] div{
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] svg{
  fill:#334155 !important;
  color:#334155 !important;
}

/* opened dropdown menu */
[data-testid="stSidebar"] [role="listbox"],
[data-testid="stSidebar"] [role="option"],
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"]{
  background:#ffffff !important;
  color:#102033 !important;
}

[role="option"],
[role="option"] *{
  background:#ffffff !important;
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
}

[role="option"]:hover,
[role="option"][aria-selected="true"]{
  background:#e8f7f7 !important;
  color:#075985 !important;
}

/* sidebar search input */
[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] input{
  background:#ffffff !important;
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
  border-color:#c8d7e1 !important;
}

[data-testid="stSidebar"] input::placeholder{
  color:#7b8da0 !important;
  -webkit-text-fill-color:#7b8da0 !important;
  opacity:1 !important;
}

/* navigation hover */
[data-testid="stSidebar"] .stRadio label{
  color:#102033 !important;
  border-radius:12px !important;
  padding:9px 10px !important;
  font-weight:850 !important;
  transition:all .22s ease !important;
}
[data-testid="stSidebar"] .stRadio label:hover{
  background:#e7f7f7 !important;
  color:#075985 !important;
  transform:translateX(5px);
  box-shadow:0 9px 22px rgba(8,127,134,.11);
}

.nav-caption{
  color:#73879a !important;
  font-size:.64rem;
  font-weight:950;
  letter-spacing:.15em;
  text-transform:uppercase;
  margin:5px 0 8px;
}

.aegis-brand{
  display:flex;
  align-items:center;
  gap:12px;
  padding:4px 0 19px;
}
.brand-mark{
  width:48px;height:48px;
  border-radius:15px;
  display:flex;
  align-items:center;
  justify-content:center;
  background:linear-gradient(135deg,#075985,#0f766e);
  color:#ffffff !important;
  font-size:22px;
  box-shadow:0 13px 30px rgba(7,89,133,.23);
}
.brand-name{
  color:#0b1d31 !important;
  font-size:1.4rem;
  font-weight:950;
  line-height:1;
}
.brand-sub{
  color:#657a8e !important;
  font-size:.70rem;
  margin-top:4px;
}

/* ============================================================
   5. HERO — DARK SURFACE, WHITE TEXT
   ============================================================ */
.hero{
  position:relative !important;
  overflow:hidden !important;
  isolation:isolate;
  min-height:240px;
  border-radius:28px !important;
  padding:40px 44px !important;
  margin-bottom:16px !important;
  background:
    radial-gradient(circle at 75% 25%,rgba(45,212,191,.28),transparent 22%),
    radial-gradient(circle at 93% 84%,rgba(56,189,248,.20),transparent 30%),
    linear-gradient(135deg,#051526 0%,#082f49 48%,#087f86 100%) !important;
  box-shadow:0 30px 70px rgba(5,29,49,.23);
}
.hero *{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
.hero-kicker{
  position:relative;z-index:5;
  color:#d6f7f4 !important;
  -webkit-text-fill-color:#d6f7f4 !important;
  letter-spacing:.17em;
  font-size:.66rem;
  font-weight:950;
}
.hero h1{
  position:relative;z-index:5;
  color:#ffffff !important;
  font-size:2.95rem !important;
  font-weight:950 !important;
  letter-spacing:-.055em;
  margin:.36rem 0 .35rem !important;
}
.hero p{
  position:relative;z-index:5;
  color:#eaf8fb !important;
  -webkit-text-fill-color:#eaf8fb !important;
  max-width:900px;
  margin:0;
  font-size:1rem;
  line-height:1.58;
}
.hero-badge{
  position:relative;z-index:5;
  display:inline-block;
  margin-top:19px;
  color:#ffffff !important;
  padding:8px 14px;
  border:1px solid rgba(255,255,255,.28);
  border-radius:999px;
  background:rgba(255,255,255,.10);
  font-size:.73rem;
  backdrop-filter:blur(10px);
}
.hero::before{
  content:"";
  position:absolute;
  z-index:1;
  width:530px;height:175px;
  right:-90px;bottom:-58px;
  border-radius:50%;
  border:1px solid rgba(255,255,255,.17);
  box-shadow:
    0 -21px 0 -20px rgba(255,255,255,.12),
    0 -43px 0 -42px rgba(255,255,255,.08),
    0 -68px 0 -67px rgba(255,255,255,.05);
  animation:aegisHeroWave 7s ease-in-out infinite alternate;
}
.hero::after{
  content:"";
  position:absolute;
  z-index:1;
  width:360px;height:360px;
  right:-125px;top:-155px;
  border-radius:50%;
  border:1px solid rgba(255,255,255,.15);
  box-shadow:
    0 0 0 32px rgba(255,255,255,.026),
    0 0 0 66px rgba(255,255,255,.016);
  animation:aegisHeroRing 15s linear infinite;
}
@keyframes aegisHeroWave{
  from{transform:translateX(0) rotate(-7deg)}
  to{transform:translateX(-68px) rotate(-2deg)}
}
@keyframes aegisHeroRing{
  from{transform:rotate(0) scale(.94)}
  to{transform:rotate(360deg) scale(1.05)}
}
.hero-live-dot{
  color:#6ee7d5 !important;
  -webkit-text-fill-color:#6ee7d5 !important;
  animation:aegisPulse 1.6s ease-in-out infinite;
}
@keyframes aegisPulse{
  0%,100%{opacity:.45;transform:scale(.8)}
  50%{opacity:1;transform:scale(1.15)}
}

/* ============================================================
   6. CONTENT COMPONENTS
   ============================================================ */
.aegis-strip{
  display:flex;
  align-items:center;
  gap:7px;
  flex-wrap:wrap;
  margin:-1px 0 21px;
  padding:9px 12px;
  border:1px solid #d6e3ea;
  border-radius:13px;
  background:rgba(255,255,255,.96);
  box-shadow:0 8px 21px rgba(15,35,55,.055);
  backdrop-filter:blur(12px);
  color:#53687d !important;
  font-size:.68rem;
  font-weight:850;
}
.aegis-strip span{
  color:#53687d !important;
  padding:3px 9px;
  border-right:1px solid #e1e9ee;
}
.aegis-strip span:last-child{border-right:0}
.aegis-strip b{color:#087f86 !important}
.aegis-strip i{
  display:inline-block;
  width:7px;height:7px;
  margin-right:5px;
  border-radius:50%;
  background:#16a34a;
  box-shadow:0 0 0 4px rgba(22,163,74,.10);
}

.section-head{margin:27px 0 12px}
.section-title{color:#102033 !important;font-size:1.3rem;font-weight:950}
.section-desc{color:#53687d !important;font-size:.83rem;margin-top:3px}

.card{
  color:#102033 !important;
  border:1px solid #d6e3ea;
  border-radius:18px;
  padding:19px;
  background:rgba(255,255,255,.965);
  box-shadow:0 10px 29px rgba(15,35,55,.055);
  backdrop-filter:blur(12px);
  transition:transform .23s ease,box-shadow .23s ease,border-color .23s ease;
}
.card:hover{
  transform:translateY(-6px);
  border-color:#99cbd3 !important;
  box-shadow:0 21px 45px rgba(15,35,55,.13);
}
.card-title{color:#102033 !important;font-weight:900;margin-bottom:7px}
.card-muted{color:#53687d !important;font-size:.80rem;line-height:1.55}
.status-row{
  color:#304256 !important;
  display:flex;
  justify-content:space-between;
  padding:8px 0;
  border-bottom:1px solid #edf1f5;
  font-size:.79rem;
}
.status-row b{color:#087f86 !important}

/* metrics */
div[data-testid="stMetric"]{
  position:relative;
  overflow:hidden;
  background:#ffffff !important;
  border:1px solid #d6e3ea !important;
  border-radius:18px !important;
  padding:16px 18px !important;
  box-shadow:0 9px 28px rgba(15,35,55,.06) !important;
  transition:transform .23s ease,box-shadow .23s ease,border-color .23s ease !important;
}
div[data-testid="stMetric"]:hover{
  transform:translateY(-6px) !important;
  border-color:#99cbd3 !important;
  box-shadow:0 21px 43px rgba(15,35,55,.13) !important;
}
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *{
  color:#5b6f83 !important;
  -webkit-text-fill-color:#5b6f83 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] *{
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
  font-weight:950 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] *{
  color:#5b6f83 !important;
  -webkit-text-fill-color:#5b6f83 !important;
}

/* ============================================================
   7. ALL INPUTS — WHITE SURFACE + DARK TEXT
   ============================================================ */
[data-baseweb="input"]>div,
[data-baseweb="select"]>div{
  background:#ffffff !important;
  border:1px solid #c8d7e1 !important;
  border-radius:11px !important;
}
[data-baseweb="input"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input{
  background:#ffffff !important;
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
  caret-color:#087f86 !important;
}
[data-baseweb="select"] *,
[data-baseweb="select"] span{
  color:#102033 !important;
  -webkit-text-fill-color:#102033 !important;
}
input::placeholder{
  color:#8293a5 !important;
  -webkit-text-fill-color:#8293a5 !important;
  opacity:1 !important;
}
[data-testid="stNumberInput"] button{
  color:#334155 !important;
  background:#f7fafc !important;
}

/* ============================================================
   8. BUTTONS / TABS
   ============================================================ */
.stButton>button,
[data-testid="stFormSubmitButton"] button{
  min-height:44px !important;
  border-radius:12px !important;
  font-weight:900 !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  background:linear-gradient(135deg,#075985,#087f86) !important;
  border:1px solid #075985 !important;
  transition:transform .20s ease,box-shadow .20s ease,filter .20s ease !important;
}
.stButton>button *,
[data-testid="stFormSubmitButton"] button *{
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
}
.stButton>button:hover,
[data-testid="stFormSubmitButton"] button:hover{
  transform:translateY(-3px) scale(1.01) !important;
  filter:brightness(1.07);
  box-shadow:0 14px 30px rgba(7,89,133,.24) !important;
}

[data-baseweb="tab-list"]{
  gap:5px !important;
  border-bottom:1px solid #d8e5ed !important;
}
[data-baseweb="tab"]{
  color:#53687d !important;
  border-radius:10px 10px 0 0 !important;
  font-weight:850 !important;
  transition:all .18s ease !important;
}
[data-baseweb="tab"] *{
  color:#53687d !important;
  -webkit-text-fill-color:#53687d !important;
}
[data-baseweb="tab"]:hover{background:#e8f7f7 !important}
[aria-selected="true"][data-baseweb="tab"],
[aria-selected="true"][data-baseweb="tab"] *{
  color:#075985 !important;
  -webkit-text-fill-color:#075985 !important;
}

/* ============================================================
   9. RISK / TABLE / ALERT
   ============================================================ */
.risk-banner{
  color:#102033 !important;
  border-radius:18px;
  padding:19px 21px;
  margin:10px 0 17px;
  border:1px solid #d8e5ed;
  background:#ffffff !important;
  box-shadow:0 9px 27px rgba(15,35,55,.055);
  transition:transform .20s ease,box-shadow .20s ease;
}
.risk-banner:hover{
  transform:translateY(-4px);
  box-shadow:0 17px 35px rgba(15,35,55,.10);
}
.risk-banner strong{color:#102033 !important;font-size:1.08rem}
.risk-banner .card-muted{color:#53687d !important}
.risk-low{border-left:5px solid #16a34a}
.risk-moderate{border-left:5px solid #ca8a04}
.risk-high{border-left:5px solid #ea580c}
.risk-very-high{border-left:5px solid #dc2626}

[data-testid="stDataFrame"]{
  border:1px solid #d8e5ed !important;
  border-radius:15px !important;
  overflow:hidden !important;
  background:#ffffff !important;
  box-shadow:0 8px 23px rgba(15,35,55,.045);
}
[data-testid="stAlert"]{
  border-radius:13px !important;
}

/* footer */
.footer{
  text-align:center;
  color:#6f8194 !important;
  font-size:.75rem;
  padding:19px 0 4px;
}
.footer b{color:#40586c !important}

/* ============================================================
   10. RESPONSIVE + ACCESSIBILITY
   ============================================================ */
@media(max-width:850px){
  .block-container{
    padding-left:1rem !important;
    padding-right:1rem !important;
  }
  .hero{
    padding:28px !important;
    min-height:0;
  }
  .hero h1{font-size:2.15rem !important}
  .hero p{font-size:.9rem}
  .aegis-grid{bottom:-330px}
}
@media(prefers-reduced-motion:reduce){
  .aegis-orb,
  .aegis-grid,
  .aegis-ripple,
  .aegis-wave,
  .aegis-rain i,
  .hero::before,
  .hero::after,
  .hero-live-dot,
  .card,
  div[data-testid="stMetric"],
  .risk-banner{
    animation:none !important;
    transition:none !important;
  }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="aegis-scene" aria-hidden="true">
  <div class="aegis-orb one"></div>
  <div class="aegis-orb two"></div>
  <div class="aegis-orb three"></div>
  <div class="aegis-ripple"></div>
  <div class="aegis-wave"></div>
  <div class="aegis-grid"></div>
  <div class="aegis-rain">
    <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================

with st.sidebar:
    st.markdown("""
    <div class="aegis-brand">
        <div class="brand-mark">🌊</div>
        <div>
            <div class="brand-name">AEGIS</div>
            <div class="brand-sub">Flood Risk Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="nav-caption">Intelligence Modules</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "🏠 Command Center",
            "📍 Location Intelligence",
            "🤖 AI Risk Engine",
            "📈 Historical Intelligence",
            "🗺️ Risk Command Map",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="nav-caption">Global Filters</div>', unsafe_allow_html=True)

    districts = sorted(risk_data["district"].dropna().unique().tolist())
    selected_district = st.selectbox("District", ["All"] + districts)
    selected_risk = st.selectbox("Risk Category", ["All"] + RISK_ORDER)
    search_subdistrict = st.text_input(
        "Sub-District",
        placeholder="Search area..."
    )

    st.divider()

    st.markdown("""
    <div class="card">
        <div class="card-title">System Status</div>
        <div class="status-row"><span>● Risk Dataset</span><b>Ready</b></div>
        <div class="status-row"><span>● PIN Intelligence</span><b>Ready</b></div>
        <div class="status-row"><span>● AI Engine</span><b>Loaded</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"{risk_data['district'].nunique()} districts • {len(risk_data)} sub-district records")
    st.caption("Historical-model-based decision support")


# ================================================================
# FILTER DATA
# ================================================================

filtered_data = risk_data.copy()

if selected_district != "All":
    filtered_data = filtered_data[filtered_data["district"] == selected_district]

if selected_risk != "All":
    filtered_data = filtered_data[filtered_data["risk_category"] == selected_risk]

if search_subdistrict.strip():
    filtered_data = find_subdistrict_matches(
        search_subdistrict,
        filtered_data,
        limit=50,
        cutoff=0.55,
    )


# ================================================================
# HERO
# ================================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">Rajasthan • Climate & Disaster Risk Intelligence</div>
    <h1>🌊 Aegis</h1>
    <p>AI-powered flood risk intelligence for understanding historical risk,
    exploring locations, analysing weather-driven risk and supporting disaster preparedness.</p>
    <div class="hero-badge"><span class="hero-live-dot">●</span> Historical intelligence &nbsp; • &nbsp; AI-assisted analysis &nbsp; • &nbsp; Interactive geospatial view</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="aegis-strip">
    <span><i></i><b>AEGIS INTELLIGENCE ONLINE</b></span>
    <span>Historical Risk</span>
    <span>AI Decision Support</span>
    <span>Geospatial Monitoring</span>
</div>
""", unsafe_allow_html=True)

# ================================================================
# COMMAND CENTER
# ================================================================

if page == "🏠 Command Center":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Risk Command Center</div>
        <div class="section-desc">A high-level operational snapshot of Rajasthan's flood-risk landscape.</div>
    </div>
    """, unsafe_allow_html=True)

    total_subdistricts = len(risk_data)
    total_districts = risk_data["district"].nunique()
    low = int((risk_data["risk_category"] == "Low").sum())
    moderate = int((risk_data["risk_category"] == "Moderate").sum())
    high = int((risk_data["risk_category"] == "High").sum())
    very_high = int((risk_data["risk_category"] == "Very High").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monitored Records", total_subdistricts)
    c2.metric("Districts Covered", total_districts)
    c3.metric("High + Very High", high + very_high)
    c4.metric("Very High Areas", very_high)

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Risk Landscape</div>
        <div class="section-desc">Distribution of classified sub-district risk levels.</div>
    </div>
    """, unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("🟢 Low", low)
    r2.metric("🟡 Moderate", moderate)
    r3.metric("🟠 High", high)
    r4.metric("🔴 Very High", very_high)

    left, right = st.columns([1.15, .85])

    with left:
        st.markdown("#### Risk Distribution")
        counts = risk_data["risk_category"].value_counts().reindex(RISK_ORDER, fill_value=0)
        st.bar_chart(counts, height=310)

    with right:
        st.markdown("#### Critical Areas")
        top = risk_data.sort_values("flood_affected_percent", ascending=False).head(8)
        st.dataframe(
            top[["district", "sub_district", "flood_affected_percent", "risk_category"]]
            .rename(columns={
                "district":"District",
                "sub_district":"Sub-District",
                "flood_affected_percent":"Flood %",
                "risk_category":"Risk"
            }),
            width="stretch",
            hide_index=True,
            height=310,
        )

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Filtered Intelligence</div>
        <div class="section-desc">Use the sidebar filters to narrow the operational view.</div>
    </div>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    f1.metric("Matching Records", len(filtered_data))
    f2.metric(
        "High / Very High Matches",
        int(filtered_data["risk_category"].isin(["High", "Very High"]).sum())
    )
    f3.metric(
        "Maximum Flood-Affected",
        f"{filtered_data['flood_affected_percent'].max():.2f}%" if not filtered_data.empty else "—"
    )

    if search_subdistrict.strip():
        corrected = best_subdistrict_correction(search_subdistrict, risk_data)
        if corrected and normalize_text(corrected) != normalize_text(search_subdistrict):
            st.success(
                f"Spelling corrected automatically: **{search_subdistrict.strip()}** → **{corrected}**"
            )

    if not filtered_data.empty:
        st.dataframe(
            filtered_data[
                ["risk_rank","district","sub_district","flood_affected_percent","risk_category","risk_score"]
            ].rename(columns={
                "risk_rank":"Rank",
                "district":"District",
                "sub_district":"Sub-District",
                "flood_affected_percent":"Flood %",
                "risk_category":"Risk",
                "risk_score":"Score"
            }),
            width="stretch",
            hide_index=True,
            height=360,
        )
    else:
        st.info("No records match the current filters.")


# ================================================================
# LOCATION INTELLIGENCE
# ================================================================

elif page == "📍 Location Intelligence":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Location Intelligence</div>
        <div class="section-desc">Search a PIN or explore any district and its available flood-risk profile.</div>
    </div>
    """, unsafe_allow_html=True)

    pin_tab, district_tab = st.tabs(["📍 PIN Intelligence", "🧭 District Explorer"])

    with pin_tab:
        search_col, info_col = st.columns([1.25, .75])

        with search_col:
            st.markdown("#### Search by PIN")
            pin = st.text_input(
                "6-digit Rajasthan PIN Code",
                placeholder="Enter PIN code, e.g. 341501",
                key="location_pin",
            )

        with info_col:
            st.markdown("""
            <div class="card">
                <div class="card-title">Location Resolution</div>
                <div class="card-muted">PIN → district → available Aegis risk intelligence. The PIN is not treated as a household-level flood prediction.</div>
            </div>
            """, unsafe_allow_html=True)

        if pin.strip():
            pincode = normalize_pincode(pin)

            if not pincode.isdigit() or len(pincode) != 6:
                st.warning("Please enter a valid 6-digit PIN code.")
            else:
                pin_info, pin_error = lookup_pincode(pincode)

                if pin_error:
                    st.warning(pin_error)
                else:
                    api_district = str(pin_info.get("district", "")).strip()
                    api_state = str(pin_info.get("state", "")).strip()

                    if api_state and normalize_text(api_state) != "RAJASTHAN":
                        st.warning(f"PIN {pincode} belongs to {api_state}, not Rajasthan.")
                    else:
                        st.success(f"PIN {pincode} resolved successfully.")

                        p1, p2, p3 = st.columns(3)
                        p1.metric("PIN Code", pincode)
                        p2.metric("District", api_district.title() or "—")
                        p3.metric("State", api_state.title() if api_state else "Rajasthan")

                        if not pin_data.empty and "pincode" in pin_data.columns:
                            local = pin_data.copy()
                            local["_pin"] = local["pincode"].apply(normalize_pincode)
                            matches = local[local["_pin"] == pincode]

                            if not matches.empty and "office_name" in matches.columns:
                                areas = (
                                    matches["office_name"]
                                    .dropna()
                                    .astype(str)
                                    .str.strip()
                                    .unique()
                                    .tolist()
                                )
                                if areas:
                                    st.info("Associated areas/offices: " + ", ".join(areas))

                        district_risk = get_district_risk(api_district)

                        if district_risk.empty:
                            st.info("The PIN was resolved, but this district is not available in the current Aegis risk dataset.")
                        else:
                            pin_category = RISK_ORDER[int(district_risk["risk_score"].max()) - 1]

                            st.markdown(
                                f'<div class="risk-banner risk-{pin_category.lower().replace(" ","-")}"><strong>{RISK_UI[pin_category]["icon"]} {api_district.title()} — {pin_category} Risk</strong><br><span class="card-muted">District-level Aegis profile associated with this PIN.</span></div>',
                                unsafe_allow_html=True,
                            )

                            q1, q2, q3, q4 = st.columns(4)
                            q1.metric("Sub-Districts", len(district_risk))
                            q2.metric("Very High", int((district_risk["risk_category"] == "Very High").sum()))
                            q3.metric("High", int((district_risk["risk_category"] == "High").sum()))
                            q4.metric("Max Flood-Affected", f"{district_risk['flood_affected_percent'].max():.2f}%")

                            st.markdown("#### Risk Areas")
                            st.dataframe(
                                district_risk[
                                    ["risk_rank","sub_district","flood_affected_percent","risk_category","risk_score"]
                                ].sort_values(
                                    "flood_affected_percent",
                                    ascending=False
                                ).rename(columns={
                                    "risk_rank":"Rank",
                                    "sub_district":"Sub-District",
                                    "flood_affected_percent":"Flood %",
                                    "risk_category":"Risk",
                                    "risk_score":"Score"
                                }),
                                width="stretch",
                                hide_index=True,
                            )

                            st.markdown("#### District Risk Distribution")
                            st.bar_chart(
                                district_risk["risk_category"]
                                .value_counts()
                                .reindex(RISK_ORDER, fill_value=0),
                                height=280,
                            )

                            st.caption(
                                "PIN lookup identifies the district; it does not represent an exact household/PIN-level flood prediction."
                            )

    with district_tab:
        st.markdown("#### Explore a District")

        location_query = st.text_input(
            "Search any sub-district",
            placeholder="Try: Rupangath, Jaipurr, Udaipuur...",
            key="location_subdistrict_search",
        )

        if location_query.strip():
            location_matches = find_subdistrict_matches(
                location_query,
                risk_data,
                limit=25,
                cutoff=0.55,
            )

            corrected = best_subdistrict_correction(location_query, risk_data)
            if corrected and normalize_text(corrected) != normalize_text(location_query):
                st.success(
                    f"Spelling corrected automatically: **{location_query.strip()}** → **{corrected}**"
                )

            if location_matches.empty:
                st.warning("No matching sub-district found. Try a shorter spelling.")
            else:
                st.dataframe(
                    location_matches[
                        ["district","sub_district","flood_affected_percent","risk_category","risk_score"]
                    ].sort_values(
                        "flood_affected_percent",
                        ascending=False,
                    ).rename(columns={
                        "district":"District",
                        "sub_district":"Sub-District",
                        "flood_affected_percent":"Flood %",
                        "risk_category":"Risk",
                        "risk_score":"Score",
                    }),
                    width="stretch",
                    hide_index=True,
                )

        st.markdown("#### Explore a District")

        manual_district = st.selectbox(
            "Select district",
            districts,
            index=districts.index(selected_district) if selected_district in districts else 0,
            key="manual_district_explorer",
        )

        manual = get_district_risk(manual_district)

        if manual.empty:
            st.info("No Aegis risk records are available for this district.")
        else:
            category = RISK_ORDER[int(manual["risk_score"].max()) - 1]

            st.markdown(
                f'<div class="risk-banner risk-{category.lower().replace(" ","-")}"><strong>{RISK_UI[category]["icon"]} {manual_district} — {category} Risk Profile</strong><br><span class="card-muted">Highest risk category present among available sub-district records.</span></div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Sub-Districts", len(manual))
            m2.metric("Very High", int((manual["risk_category"] == "Very High").sum()))
            m3.metric("High", int((manual["risk_category"] == "High").sum()))
            m4.metric("Max Flood-Affected", f"{manual['flood_affected_percent'].max():.2f}%")

            st.markdown("#### Risk Areas")
            st.dataframe(
                manual[
                    ["risk_rank","sub_district","flood_affected_percent","risk_category","risk_score"]
                ].sort_values(
                    "flood_affected_percent",
                    ascending=False
                ).rename(columns={
                    "risk_rank":"Rank",
                    "sub_district":"Sub-District",
                    "flood_affected_percent":"Flood %",
                    "risk_category":"Risk",
                    "risk_score":"Score"
                }),
                width="stretch",
                hide_index=True,
            )


# ================================================================
# AI RISK ENGINE
# ================================================================

elif page == "🤖 AI Risk Engine":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">AI Risk Engine</div>
        <div class="section-desc">Estimate historical flood-affected percentage from weather conditions using the trained Ridge Regression model.</div>
    </div>
    """, unsafe_allow_html=True)

    st.info("AI-assisted historical-model estimate — not a real-time operational flood forecast or exact PIN-level prediction.")

    if model_package is None:
        st.error("Prediction model not found. Run `python .\\src\\train_model.py` first.")
    else:
        prediction_model = model_package["model"]
        prediction_features = model_package["features"]
        defaults = risk_data[prediction_features].median()

        st.markdown("#### Weather Conditions")

        with st.form("prediction_form"):
            a, b = st.columns(2)
            with a:
                rainfall = st.number_input("Annual Rainfall (mm)", min_value=0.0, value=float(defaults["avg_annual_rainfall_mm"]), step=10.0)
                temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=float(defaults["avg_temperature_c"]), step=0.5)
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=float(defaults["avg_humidity_percent"]), step=1.0)
            with b:
                wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=float(defaults["avg_wind_speed_mps"]), step=0.1)
                pressure = st.number_input("Pressure (kPa)", min_value=0.0, value=float(defaults["avg_pressure_kpa"]), step=0.1)
                max_daily_rainfall = st.number_input("Maximum Daily Rainfall (mm)", min_value=0.0, value=float(defaults["max_daily_rainfall_mm"]), step=5.0)

            submitted = st.form_submit_button("🔮 Analyze Flood Risk", type="primary", width="stretch")

        if submitted:
            input_values = {
                "avg_annual_rainfall_mm": rainfall,
                "avg_temperature_c": temperature,
                "avg_humidity_percent": humidity,
                "avg_wind_speed_mps": wind_speed,
                "avg_pressure_kpa": pressure,
                "max_daily_rainfall_mm": max_daily_rainfall,
            }
            input_data = pd.DataFrame(
                [[input_values[f] for f in prediction_features]],
                columns=prediction_features,
            )

            try:
                predicted_percent = float(prediction_model.predict(input_data)[0])
                predicted_percent = max(0.0, min(100.0, predicted_percent))
                category = category_from_percent(predicted_percent)
                info = risk_message(category)

                st.markdown(f"""
                <div class="risk-banner risk-{category.lower().replace(" ","-")}">
                    <strong>{RISK_UI[category]["icon"]} {category.upper()} RISK</strong><br>
                    <span class="card-muted">{info["message"]}</span>
                </div>
                """, unsafe_allow_html=True)

                r1, r2, r3 = st.columns(3)
                r1.metric("Estimated Flood-Affected", f"{predicted_percent:.2f}%")
                r2.metric("Risk Category", category)
                r3.metric("Risk Score", RISK_SCORE[category])

                st.markdown("#### Recommended Response")

                action_cols = st.columns(min(3, max(1, len(info["actions"]))))
                for i, action in enumerate(info["actions"]):
                    with action_cols[i % len(action_cols)]:
                        st.markdown(
                            f'<div class="card"><div class="card-title">Action {i+1}</div><div class="card-muted">✅ {action}</div></div>',
                            unsafe_allow_html=True,
                        )

                st.caption("Recommendations are AI-assisted decision-support suggestions and do not replace official warnings or disaster-management instructions.")

                st.divider()
                st.markdown("#### 🧠 Explainable AI — Risk Drivers")
                st.caption("Contributions are calculated from the actual coefficients of the trained Ridge Regression model.")

                coefficients = prediction_model.coef_
                explanation = pd.DataFrame({
                    "Weather Factor": [FEATURE_LABELS.get(f, f) for f in prediction_features],
                    "Feature": prediction_features,
                    "Model Coefficient": coefficients,
                    "Input Value": [input_values[f] for f in prediction_features],
                })
                explanation["Contribution"] = explanation["Model Coefficient"] * explanation["Input Value"]
                explanation["Absolute Contribution"] = explanation["Contribution"].abs()
                explanation = explanation.sort_values("Absolute Contribution", ascending=False).reset_index(drop=True)

                pos, neg = st.columns(2)

                with pos:
                    st.markdown("**⬆️ Factors increasing the estimate**")
                    positive = explanation[explanation["Contribution"] > 0].head(3)
                    if positive.empty:
                        st.caption("No positive contributions for these inputs.")
                    else:
                        for _, row in positive.iterrows():
                            st.write(f"• **{row['Weather Factor']}** → +{row['Contribution']:.2f}")

                with neg:
                    st.markdown("**⬇️ Factors reducing the estimate**")
                    negative = explanation[explanation["Contribution"] < 0].head(3)
                    if negative.empty:
                        st.caption("No negative contributions for these inputs.")
                    else:
                        for _, row in negative.iterrows():
                            st.write(f"• **{row['Weather Factor']}** → {row['Contribution']:.2f}")

                with st.expander("View detailed model contributions"):
                    st.dataframe(
                        explanation[["Weather Factor","Model Coefficient","Input Value","Contribution"]],
                        width="stretch",
                        hide_index=True,
                    )

                with st.expander("View prediction inputs"):
                    st.dataframe(input_data.rename(columns=FEATURE_LABELS), width="stretch", hide_index=True)

            except Exception as error:
                st.error(f"Prediction could not be generated. Model error: {error}")


# ================================================================
# HISTORICAL INTELLIGENCE
# ================================================================

elif page == "📈 Historical Intelligence":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Historical Intelligence</div>
        <div class="section-desc">Explore the historical weather and flood-affected data supporting Aegis risk intelligence.</div>
    </div>
    """, unsafe_allow_html=True)

    data = historical_data if not historical_data.empty else risk_data

    if data.empty:
        st.warning("Historical dataset is not available.")
    else:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Records", len(data))
        h2.metric("Districts", data["district"].nunique() if "district" in data.columns else "—")
        h3.metric("Period", "1986–2025")
        h4.metric("Core Metric", "Flood-Affected %")

        st.markdown("#### Flood History")
        if "flood_affected_percent" in data.columns:
            st.line_chart(data["flood_affected_percent"].dropna().reset_index(drop=True), height=300)

        st.markdown("#### Weather Intelligence")

        weather_cols = [
            c for c in [
                "avg_annual_rainfall_mm",
                "avg_temperature_c",
                "avg_humidity_percent",
                "avg_wind_speed_mps",
                "avg_pressure_kpa",
                "max_daily_rainfall_mm",
            ] if c in data.columns
        ]

        tabs = st.tabs([FEATURE_LABELS.get(c, c) for c in weather_cols])

        for tab, col in zip(tabs, weather_cols):
            with tab:
                st.line_chart(data[col].dropna().reset_index(drop=True), height=300)

        st.markdown("#### Historical Data Explorer")
        st.dataframe(data.head(150), width="stretch", hide_index=True, height=430)

        st.warning("Historical test results showed substantial prediction error. The model should therefore be presented as historical risk intelligence / decision support rather than a highly accurate operational forecast.")


# ================================================================
# RISK COMMAND MAP
# ================================================================

elif page == "🗺️ Risk Command Map":

    st.markdown("""
    <div class="section-head">
        <div class="section-title">Risk Command Map</div>
        <div class="section-desc">Interactive district-level view of the highest risk category present in each district.</div>
    </div>
    """, unsafe_allow_html=True)

    map_data = create_map_data(risk_data)

    if map_data.empty:
        st.warning("Map coordinates are not available.")
    else:
        focus = selected_district if selected_district != "All" else None
        coords = DISTRICT_COORDINATES.get(normalize_text(focus)) if focus else None

        center_lat, center_lon, zoom = (
            (coords[0], coords[1], 8) if coords else (26.5, 74.5, 6)
        )

        # Selected district summary above the map.
        if focus:
            focus_row = map_data[map_data["district"].astype(str).apply(normalize_text) == normalize_text(focus)]
            if not focus_row.empty:
                fr = focus_row.iloc[0]
                st.markdown(
                    f'<div class="risk-banner risk-{fr["risk_category"].lower().replace(" ","-")}"><strong>{RISK_UI[fr["risk_category"]]["icon"]} {fr["district"]} — {fr["risk_category"]}</strong><br><span class="card-muted">Max flood-affected: {fr["max_flood_percent"]:.2f}% • Very High sub-districts: {fr["very_high_risk"]}</span></div>',
                    unsafe_allow_html=True,
                )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position=["longitude", "latitude"],
            get_fill_color=["color[0]", "color[1]", "color[2]", 200],
            get_line_color=[255, 255, 255, 230],
            get_radius=18000,
            radius_min_pixels=7,
            radius_max_pixels=30,
            pickable=True,
            stroked=True,
            filled=True,
        )

        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
            ),
            tooltip={
                "html": """
                    <b>{district}</b><br/>
                    Risk: {risk_category}<br/>
                    Very High Areas: {very_high_risk}<br/>
                    High Areas: {high_risk}<br/>
                    Maximum Flood-Affected: {max_flood_percent}%<br/>
                    Average Flood-Affected: {avg_flood_percent}%<br/>
                    Maximum Risk Score: {risk_score}
                """,
                "style": {
                    "backgroundColor": "#0f172a",
                    "color": "white",
                    "fontSize": "12px",
                },
            },
        )

        st.pydeck_chart(deck, width="stretch", height=640)

        l1, l2, l3, l4 = st.columns(4)
        l1.markdown("🟢 **Low**")
        l2.markdown("🟡 **Moderate**")
        l3.markdown("🟠 **High**")
        l4.markdown("🔴 **Very High**")

        st.markdown("#### Highest-Risk Districts")
        map_summary = (
            map_data.sort_values(
                ["risk_score", "max_flood_percent"],
                ascending=[False, False]
            )
            [["district","risk_category","very_high_risk","high_risk","max_flood_percent"]]
            .rename(columns={
                "district":"District",
                "risk_category":"Risk",
                "very_high_risk":"Very High Areas",
                "high_risk":"High Areas",
                "max_flood_percent":"Max Flood %"
            })
        )
        st.dataframe(map_summary, width="stretch", hide_index=True, height=330)

        st.caption("Map markers use district headquarters / representative coordinates for visualization; they are not exact sub-district coordinates.")


# ================================================================
# FOOTER
# ================================================================

st.divider()

st.markdown("""
<div class="footer">
    <b>Aegis</b> • Flood Risk Intelligence Platform<br>
    Historical-model-based decision support • Rajasthan
</div>
""", unsafe_allow_html=True)
