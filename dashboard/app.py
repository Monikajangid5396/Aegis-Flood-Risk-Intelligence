from pathlib import Path

import joblib
import pandas as pd
import requests
import pydeck as pdk
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Aegis | Flood Risk Intelligence",
    page_icon="🌊",
    layout="wide"
)


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RISK_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "aegis_risk_dashboard_data.csv"
)

PIN_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "rajasthan_pincode_directory.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ridge_flood_risk_model.joblib"
)


# ==================================================
# LOAD RISK DATA
# ==================================================

@st.cache_data
def load_risk_data():
    return pd.read_csv(RISK_DATA_PATH)


# ==================================================
# LOAD PIN DATA
# ==================================================

@st.cache_data
def load_pin_data():

    if PIN_DATA_PATH.exists():
        return pd.read_csv(PIN_DATA_PATH)

    return pd.DataFrame()


# ==================================================
# LOAD AI MODEL
# ==================================================

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
model_package = load_prediction_model()


# ==================================================
# NORMALIZE TEXT
# ==================================================

def normalize_text(value):

    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace("-", " ")
        .replace(".", "")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


# ==================================================
# NORMALIZE PIN
# ==================================================

def normalize_pincode(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    return value.zfill(6)


# ==================================================
# DYNAMIC PIN LOOKUP
# ==================================================

@st.cache_data(ttl=3600)
def lookup_pincode(pincode):

    api_url = (
        "https://aniket-thapa.github.io/"
        f"india-pincode-api/pincodes/{pincode}.json"
    )

    try:

        response = requests.get(
            api_url,
            timeout=15
        )

        if response.status_code == 404:
            return None, "PIN code not found."

        response.raise_for_status()

        data = response.json()

        if not data:
            return None, "No location information found."

        return data, None

    except requests.RequestException:

        return None, (
            "PIN lookup service is temporarily unavailable."
        )

    except ValueError:

        return None, (
            "Invalid response received from PIN service."
        )


# ==================================================
# MATCH DISTRICT WITH AEGIS DATA
# ==================================================

def get_district_risk(district_name):

    district_normalized = normalize_text(
        district_name
    )

    risk_copy = risk_data.copy()

    risk_copy["_district_normalized"] = (
        risk_copy["district"]
        .apply(normalize_text)
    )

    matched = risk_copy[
        risk_copy["_district_normalized"]
        == district_normalized
    ].copy()

    if not matched.empty:
        return matched

    aliases = {
        "RAJ SAMAND": "RAJSAMAND",
        "SRI GANGANAGAR": "GANGANAGAR",
        "GANGANAGAR": "SRI GANGANAGAR",
    }

    alias = aliases.get(district_normalized)

    if alias:

        matched = risk_copy[
            risk_copy["_district_normalized"]
            == alias
        ].copy()

    return matched


# ==================================================
# DISTRICT COORDINATES
# ==================================================
#
# These represent district headquarters / primary
# district locations for visualization.
#
# They are NOT sub-district coordinates.
# ==================================================

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


# ==================================================
# CREATE MAP DATA
# ==================================================

def create_map_data(data):

    map_rows = []

    for district, group in data.groupby("district"):

        district_key = normalize_text(district)

        coordinates = DISTRICT_COORDINATES.get(
            district_key
        )

        if coordinates is None:
            continue

        latitude, longitude = coordinates

        very_high = (
            group["risk_category"]
            == "Very High"
        ).sum()

        high = (
            group["risk_category"]
            == "High"
        ).sum()

        maximum_flood = (
            group["flood_affected_percent"]
            .max()
        )

        average_flood = (
            group["flood_affected_percent"]
            .mean()
        )

        maximum_score = (
            group["risk_score"]
            .max()
        )

        # Color is based on the highest risk category
        # present among the district's sub-districts.

        if maximum_score >= 4:

            color = [220, 53, 69]
            map_risk = "Very High"

        elif maximum_score >= 3:

            color = [255, 140, 0]
            map_risk = "High"

        elif maximum_score >= 2:

            color = [255, 193, 7]
            map_risk = "Moderate"

        else:

            color = [40, 167, 69]
            map_risk = "Low"

        map_rows.append(
            {
                "district": district,
                "latitude": latitude,
                "longitude": longitude,
                "risk_category": map_risk,
                "very_high_risk": int(very_high),
                "high_risk": int(high),
                "max_flood_percent": float(
                    maximum_flood
                ),
                "avg_flood_percent": float(
                    average_flood
                ),
                "risk_score": int(
                    maximum_score
                ),
                "color": color,
            }
        )

    return pd.DataFrame(map_rows)


# ==================================================
# HEADER
# ==================================================

st.title("🌊 Aegis")

st.subheader(
    "AI-Powered Flood Risk Intelligence Platform"
)

st.caption(
    "Historical flood-risk intelligence, dynamic PIN-based "
    "location lookup, interactive mapping, and AI-assisted "
    "weather-based risk estimation."
)

st.divider()


# ==================================================
# OVERALL RISK OVERVIEW
# ==================================================

st.success(
    "Flood-risk and PIN datasets loaded successfully!"
)

st.subheader("📊 Overall Risk Overview")


total_subdistricts = len(risk_data)

total_districts = (
    risk_data["district"]
    .nunique()
)

low_risk_count = (
    risk_data["risk_category"]
    == "Low"
).sum()

moderate_risk_count = (
    risk_data["risk_category"]
    == "Moderate"
).sum()

high_risk_count = (
    risk_data["risk_category"]
    == "High"
).sum()

very_high_risk_count = (
    risk_data["risk_category"]
    == "Very High"
).sum()


# ==================================================
# MAIN SUMMARY CARDS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Sub-Districts",
        total_subdistricts
    )

with col2:

    st.metric(
        "Districts",
        total_districts
    )

with col3:

    st.metric(
        "High + Very High Risk",
        high_risk_count
        + very_high_risk_count
    )

with col4:

    st.metric(
        "Very High Risk",
        very_high_risk_count
    )


# ==================================================
# RISK CATEGORY CARDS
# ==================================================

risk_col1, risk_col2, risk_col3, risk_col4 = (
    st.columns(4)
)

with risk_col1:

    st.metric(
        "🟢 Low",
        low_risk_count
    )

with risk_col2:

    st.metric(
        "🟡 Moderate",
        moderate_risk_count
    )

with risk_col3:

    st.metric(
        "🟠 High",
        high_risk_count
    )

with risk_col4:

    st.metric(
        "🔴 Very High",
        very_high_risk_count
    )


# ==================================================
# RISK FILTERS
# ==================================================

st.subheader("🔎 Risk Filters")

filter_col1, filter_col2, filter_col3 = (
    st.columns(3)
)

with filter_col1:

    selected_district = st.selectbox(
        "Select District",
        ["All"]
        + sorted(
            risk_data["district"]
            .dropna()
            .unique()
            .tolist()
        )
    )

with filter_col2:

    selected_risk = st.selectbox(
        "Select Risk Category",
        [
            "All",
            "Low",
            "Moderate",
            "High",
            "Very High"
        ]
    )

with filter_col3:

    search_subdistrict = st.text_input(
        "Search Sub-District",
        placeholder="e.g. Chitalwana"
    )


# ==================================================
# APPLY FILTERS
# ==================================================

filtered_data = risk_data.copy()

if selected_district != "All":

    filtered_data = filtered_data[
        filtered_data["district"]
        == selected_district
    ]

if selected_risk != "All":

    filtered_data = filtered_data[
        filtered_data["risk_category"]
        == selected_risk
    ]

if search_subdistrict.strip():

    filtered_data = filtered_data[
        filtered_data["sub_district"]
        .str.contains(
            search_subdistrict.strip(),
            case=False,
            na=False
        )
    ]


# ==================================================
# PIN CODE SEARCH
# ==================================================

st.divider()

st.subheader("📍 Flood Risk by PIN Code")

st.caption(
    "A PIN code is used to identify its district. "
    "The displayed risk profile is based on the available "
    "Aegis sub-district dataset."
)

search_pincode = st.text_input(
    "Enter 6-digit Rajasthan PIN Code",
    placeholder="e.g. 341501"
)

# District selected manually can also control the map.
# A valid PIN lookup takes priority when both are provided.
pin_district_for_map = (
    selected_district if selected_district != "All" else None
)


if search_pincode.strip():

    pincode = normalize_pincode(
        search_pincode
    )

    if not pincode.isdigit() or len(pincode) != 6:

        st.warning(
            "Please enter a valid 6-digit PIN code."
        )

    else:

        pin_info, pin_error = lookup_pincode(
            pincode
        )

        if pin_error:

            st.warning(pin_error)

        else:

            api_district = str(
                pin_info.get(
                    "district",
                    ""
                )
            ).strip()

            api_state = str(
                pin_info.get(
                    "state",
                    ""
                )
            ).strip()

            if (
                api_state
                and normalize_text(api_state)
                != "RAJASTHAN"
            ):

                st.warning(
                    f"PIN {pincode} belongs to "
                    f"{api_state}, not Rajasthan."
                )

            else:

                pin_district_for_map = api_district

                st.success(
                    f"PIN {pincode} located successfully."
                )

                location_col1, location_col2 = (
                    st.columns(2)
                )

                with location_col1:

                    st.metric(
                        "PIN Code",
                        pincode
                    )

                with location_col2:

                    st.metric(
                        "District",
                        api_district.title()
                    )

                if api_state:

                    st.caption(
                        f"State: {api_state.title()}"
                    )

                # ------------------------------------------
                # ASSOCIATED AREAS
                # ------------------------------------------

                if not pin_data.empty:

                    local_pin_data = pin_data.copy()

                    local_pin_data[
                        "_normalized_pin"
                    ] = (
                        local_pin_data["pincode"]
                        .apply(normalize_pincode)
                    )

                    local_matches = (
                        local_pin_data[
                            local_pin_data[
                                "_normalized_pin"
                            ]
                            == pincode
                        ]
                    )

                    if (
                        not local_matches.empty
                        and "office_name"
                        in local_matches.columns
                    ):

                        areas = (
                            local_matches[
                                "office_name"
                            ]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .unique()
                            .tolist()
                        )

                        if areas:

                            st.write(
                                "**Associated Areas:** "
                                + ", ".join(areas)
                            )

                # ------------------------------------------
                # DISTRICT RISK
                # ------------------------------------------

                district_risk = (
                    get_district_risk(
                        api_district
                    )
                )

                if district_risk.empty:

                    st.info(
                        "The PIN location was found, "
                        "but this district is not available "
                        "in the current Aegis flood-risk dataset."
                    )

                else:

                    st.subheader(
                        "🌊 Flood Risk Profile"
                    )

                    pin_total_subdistricts = (
                        len(district_risk)
                    )

                    pin_very_high_count = (
                        district_risk[
                            "risk_category"
                        ]
                        == "Very High"
                    ).sum()

                    pin_high_count = (
                        district_risk[
                            "risk_category"
                        ]
                        == "High"
                    ).sum()

                    pin_max_flood = (
                        district_risk[
                            "flood_affected_percent"
                        ]
                        .max()
                    )

                    risk_col1, risk_col2, risk_col3, risk_col4 = (
                        st.columns(4)
                    )

                    with risk_col1:

                        st.metric(
                            "Sub-Districts",
                            pin_total_subdistricts
                        )

                    with risk_col2:

                        st.metric(
                            "Very High Risk",
                            pin_very_high_count
                        )

                    with risk_col3:

                        st.metric(
                            "High Risk",
                            pin_high_count
                        )

                    with risk_col4:

                        st.metric(
                            "Maximum Flood-Affected",
                            f"{pin_max_flood:.2f}%"
                        )

                    st.write(
                        "**Flood-risk areas in this district:**"
                    )

                    pin_risk_columns = [
                        "risk_rank",
                        "sub_district",
                        "flood_affected_percent",
                        "risk_category",
                        "risk_score"
                    ]

                    district_risk_display = (
                        district_risk[
                            pin_risk_columns
                        ]
                        .sort_values(
                            "flood_affected_percent",
                            ascending=False
                        )
                    )

                    st.dataframe(
                        district_risk_display,
                        width="stretch",
                        hide_index=True
                    )

                    st.write(
                        "**Risk Category Distribution**"
                    )

                    pin_risk_counts = (
                        district_risk[
                            "risk_category"
                        ]
                        .value_counts()
                        .reindex(
                            [
                                "Low",
                                "Moderate",
                                "High",
                                "Very High"
                            ],
                            fill_value=0
                        )
                    )

                    st.bar_chart(
                        pin_risk_counts
                    )


# ==================================================
# MANUAL DISTRICT RISK PROFILE
# ==================================================

if selected_district != "All" and not search_pincode.strip():

    manual_district_risk = get_district_risk(
        selected_district
    )

    if not manual_district_risk.empty:

        st.divider()
        st.subheader(
            f"🌊 {selected_district} Flood Risk Profile"
        )

        manual_total = len(manual_district_risk)

        manual_very_high = (
            manual_district_risk["risk_category"]
            == "Very High"
        ).sum()

        manual_high = (
            manual_district_risk["risk_category"]
            == "High"
        ).sum()

        manual_max_flood = (
            manual_district_risk["flood_affected_percent"]
            .max()
        )

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)

        with mcol1:
            st.metric("Sub-Districts", manual_total)

        with mcol2:
            st.metric("Very High Risk", manual_very_high)

        with mcol3:
            st.metric("High Risk", manual_high)

        with mcol4:
            st.metric(
                "Maximum Flood-Affected",
                f"{manual_max_flood:.2f}%"
            )

        st.write("**Flood-risk areas in this district:**")

        manual_display_columns = [
            "risk_rank",
            "sub_district",
            "flood_affected_percent",
            "risk_category",
            "risk_score"
        ]

        st.dataframe(
            manual_district_risk[
                manual_display_columns
            ].sort_values(
                "flood_affected_percent",
                ascending=False
            ),
            width="stretch",
            hide_index=True
        )


# ==================================================
# AI FLOOD RISK PREDICTION
# ==================================================

st.divider()

st.subheader("🤖 AI Flood Risk Prediction")

st.caption(
    "Enter weather conditions to estimate the historical "
    "flood-affected percentage using the trained Aegis "
    "Ridge Regression model."
)

st.info(
    "This module provides an AI-assisted historical-model "
    "risk estimate. It is not an exact real-time flood forecast "
    "or a PIN-level prediction."
)


if model_package is None:

    st.error(
        "Prediction model not found. "
        "Please run: python .\\src\\train_model.py"
    )

else:

    prediction_model = model_package["model"]
    prediction_features = model_package["features"]

    # Use dataset medians as sensible initial values.
    prediction_defaults = (
        risk_data[prediction_features]
        .median()
    )

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:

        rainfall = st.number_input(
            "Average Annual Rainfall (mm)",
            min_value=0.0,
            value=float(
                prediction_defaults[
                    "avg_annual_rainfall_mm"
                ]
            ),
            step=10.0
        )

        temperature = st.number_input(
            "Average Temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=float(
                prediction_defaults[
                    "avg_temperature_c"
                ]
            ),
            step=0.5
        )

        humidity = st.number_input(
            "Average Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(
                prediction_defaults[
                    "avg_humidity_percent"
                ]
            ),
            step=1.0
        )

    with pred_col2:

        wind_speed = st.number_input(
            "Average Wind Speed (m/s)",
            min_value=0.0,
            value=float(
                prediction_defaults[
                    "avg_wind_speed_mps"
                ]
            ),
            step=0.1
        )

        pressure = st.number_input(
            "Average Pressure (kPa)",
            min_value=0.0,
            value=float(
                prediction_defaults[
                    "avg_pressure_kpa"
                ]
            ),
            step=0.1
        )

        max_daily_rainfall = st.number_input(
            "Maximum Daily Rainfall (mm)",
            min_value=0.0,
            value=float(
                prediction_defaults[
                    "max_daily_rainfall_mm"
                ]
            ),
            step=5.0
        )

    predict_button = st.button(
        "🔮 Predict Flood Risk",
        type="primary"
    )

    if predict_button:

        input_values = {
            "avg_annual_rainfall_mm": rainfall,
            "avg_temperature_c": temperature,
            "avg_humidity_percent": humidity,
            "avg_wind_speed_mps": wind_speed,
            "avg_pressure_kpa": pressure,
            "max_daily_rainfall_mm": max_daily_rainfall,
        }

        input_data = pd.DataFrame(
            [[
                input_values[feature]
                for feature in prediction_features
            ]],
            columns=prediction_features
        )

        try:

            predicted_percent = float(
                prediction_model.predict(
                    input_data
                )[0]
            )

            # Flood-affected percentage cannot be negative
            # or exceed 100%.

            predicted_percent = max(
                0.0,
                min(
                    100.0,
                    predicted_percent
                )
            )

            # Same historical thresholds used by Aegis.
            if predicted_percent <= 0.9675:

                predicted_category = "Low"

            elif predicted_percent <= 3.35:

                predicted_category = "Moderate"

            elif predicted_percent <= 6.6525:

                predicted_category = "High"

            else:

                predicted_category = "Very High"


            st.success(
                "AI risk estimate generated successfully."
            )

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Estimated Flood-Affected Area",
                    f"{predicted_percent:.2f}%"
                )

            with result_col2:

                st.metric(
                    "Estimated Risk Category",
                    predicted_category
                )
            
            # ============================================================
            # AI RISK RECOMMENDATIONS & ALERT PANEL
            # ============================================================

            st.subheader("🚨 AI Risk Recommendations & Alert Panel")

            risk_recommendations = {
                "Low": {
                    "icon": "🟢",
                    "title": "Low Risk – Normal Monitoring",
                    "message": (
                        "Current model estimate indicates relatively low "
                        "flood risk."
                    ),
                    "actions": [
                        "Continue routine monitoring of rainfall conditions",
                        "Keep local drainage systems maintained",
                        "Stay informed about weather updates"
                    ]
                },
                "Moderate": {
                    "icon": "🟡",
                    "title": "Moderate Risk – Preventive Preparedness",
                    "message": (
                        "Preventive action is recommended as flood risk "
                        "may increase."
                    ),
                    "actions": [
                        "Monitor rainfall and weather conditions closely",
                        "Inspect drainage and waterlogging-prone areas",
                        "Keep emergency contacts and resources ready"
                    ]
                },
                "High": {
                    "icon": "🟠",
                    "title": "High Risk – Enhanced Monitoring",
                    "message": (
                        "Elevated flood risk detected. Preparedness actions "
                        "should be initiated."
                    ),
                    "actions": [
                        "Closely monitor rainfall and changing weather conditions",
                        "Check drainage and waterlogging-prone areas",
                        "Keep emergency response resources ready",
                        "Review local emergency response procedures"
                    ]
                },
                "Very High": {
                    "icon": "🔴",
                    "title": "Very High Risk – Emergency Preparedness",
                    "message": (
                        "Very high flood risk detected. Immediate preparedness "
                        "and enhanced monitoring are recommended."
                    ),
                    "actions": [
                        "Activate enhanced flood-risk monitoring",
                        "Prepare evacuation and emergency response plans",
                        "Alert vulnerable and high-risk areas",
                        "Keep emergency response teams and resources ready",
                        "Follow instructions from local disaster-management authorities"
                    ]
                }
            }

            recommendation = risk_recommendations.get(
                predicted_category,
                risk_recommendations["Moderate"]
            )

            if predicted_category == "Very High":
                st.error(
                    f"{recommendation['icon']} **{recommendation['title']}**"
                )
            elif predicted_category == "High":
                st.warning(
                    f"{recommendation['icon']} **{recommendation['title']}**"
                )
            elif predicted_category == "Moderate":
                st.info(
                    f"{recommendation['icon']} **{recommendation['title']}**"
                )
            else:
                st.success(
                    f"{recommendation['icon']} **{recommendation['title']}**"
                )

            st.write(recommendation["message"])

            st.markdown("### 📋 Recommended Actions")

            for action in recommendation["actions"]:
                st.markdown(f"- ✅ {action}")

            st.caption(
                "⚠️ These are AI-assisted decision-support suggestions "
                "based on the model-estimated risk category. They do not "
                "replace official warnings or instructions from "
                "disaster-management authorities."
            )

            # Explainable AI - risk drivers
            st.subheader("🧠 Why This Risk Level?")
            st.caption(
                "The explanation below is based on the actual "
                "coefficients of the trained Ridge Regression model."
            )
            model_coefficients = prediction_model.coef_

            explanation_data = pd.DataFrame({
                "Weather Factor": prediction_features,
                "Model Coefficient": model_coefficients,
                "Input Value": [
                    input_values[feature]
                    for feature in prediction_features
                ]
            })
            explanation_data["Contribution"] = (
                explanation_data["Model Coefficient"]
                * explanation_data["Input Value"]
            )

            # Sort by absolute contribution
            explanation_data["Absolute Contribution"] = (
                explanation_data["Contribution"].abs()
            )
            
            explanation_data = (
                explanation_data
                .sort_values(
                    "Absolute Contribution",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            # ------------------------------------------
            # TOP DRIVERS
            # ------------------------------------------

            positive_drivers = (
                explanation_data[
                    explanation_data["Contribution"] > 0
                ]
            )
            negative_drivers = (
                explanation_data[
                    explanation_data["Contribution"] < 0
                ]
            )

            driver_col1, driver_col2 = st.columns(2)

            with driver_col1:

                st.write("**⬆️ Factors increasing the estimate**")

                if positive_drivers.empty:

                    st.caption(
                        "No positive model contributions "
                        "for these inputs."
                    )

                else:

                    for _, row in positive_drivers.head(3).iterrows():

                        st.write(
                            f"• **{row['Weather Factor']}** "
                            f"→ positive contribution "
                            f"({row['Contribution']:.2f})"
                        )

            with driver_col2:

                st.write("**⬇️ Factors reducing the estimate**")

                if negative_drivers.empty:

                    st.caption(
                        "No negative model contributions "
                        "for these inputs."
                    )

                else:

                    for _, row in negative_drivers.head(3).iterrows():

                        st.write(
                            f"• **{row['Weather Factor']}** "
                            f"→ negative contribution "
                            f"({row['Contribution']:.2f})"
                        )

            # ------------------------------------------
            # MODEL COEFFICIENT TABLE
            # ------------------------------------------

            with st.expander("View model factor contributions"):

                st.dataframe(
                    explanation_data[
                        [
                            "Weather Factor",
                            "Model Coefficient",
                            "Input Value",
                            "Contribution"
                        ]
                    ],
                    width="stretch",
                    hide_index=True
                )

            # ------------------------------------------
            # PREDICTION INTERPRETATION
            # ------------------------------------------

            if predicted_category == "Low":

                st.success(
                    "The historical model places these "
                    "weather conditions in the Low-risk range."
                )

            elif predicted_category == "Moderate":

                st.warning(
                    "The historical model places these "
                    "weather conditions in the Moderate-risk range."
                )

            elif predicted_category == "High":

                st.warning(
                    "The historical model places these "
                    "weather conditions in the High-risk range. "
                    "Closer monitoring is recommended."
                )

            else:

                st.error(
                    "The historical model places these "
                    "weather conditions in the Very High-risk range. "
                    "Enhanced preparedness and monitoring are recommended."
                )

            with st.expander("View prediction inputs"):

                st.dataframe(
                    input_data,
                    width="stretch",
                    hide_index=True
                )

        except Exception as error:

            st.error(
                "Prediction could not be generated. "
                f"Model error: {error}"
            )


# ==================================================
# FILTERED RISK RESULTS
# ==================================================

st.divider()

st.subheader("📊 Filtered Risk Areas")

st.metric(
    "Matching Sub-Districts",
    len(filtered_data)
)

if filtered_data.empty:

    st.warning(
        "No sub-districts match the selected filters."
    )

else:

    risk_display_columns = [
        "risk_rank",
        "district",
        "sub_district",
        "flood_affected_percent",
        "risk_category",
        "risk_score"
    ]

    st.dataframe(
        filtered_data[
            risk_display_columns
        ],
        width="stretch",
        hide_index=True
    )


# ==================================================
# RISK CATEGORY DISTRIBUTION
# ==================================================

st.subheader(
    "📈 Risk Category Distribution"
)

risk_counts = (
    filtered_data["risk_category"]
    .value_counts()
    .reindex(
        [
            "Low",
            "Moderate",
            "High",
            "Very High"
        ],
        fill_value=0
    )
)

st.bar_chart(
    risk_counts
)


# ==================================================
# TOP 15 HIGHEST-RISK SUB-DISTRICTS
# ==================================================

st.divider()

st.subheader(
    "🚨 Top 15 Highest-Risk Sub-Districts"
)

top_risk_data = (
    risk_data
    .sort_values(
        "flood_affected_percent",
        ascending=False
    )
    .head(15)
    .copy()
)

top_chart_data = (
    top_risk_data[
        [
            "sub_district",
            "flood_affected_percent"
        ]
    ]
    .set_index(
        "sub_district"
    )
)

st.bar_chart(
    top_chart_data
)

st.write(
    "**Highest flood-affected areas:**"
)

st.dataframe(
    top_risk_data[
        [
            "risk_rank",
            "district",
            "sub_district",
            "flood_affected_percent",
            "risk_category",
            "risk_score"
        ]
    ],
    width="stretch",
    hide_index=True
)


# ==================================================
# DISTRICT-WISE VERY HIGH RISK
# ==================================================

st.divider()

st.subheader(
    "🔥 District-wise Very High Risk Areas"
)

very_high_districts = (
    risk_data[
        risk_data["risk_category"]
        == "Very High"
    ]
    .groupby("district")
    .size()
    .sort_values(
        ascending=False
    )
)

st.bar_chart(
    very_high_districts
)

very_high_summary = (
    very_high_districts
    .reset_index(
        name="Very High Risk Sub-Districts"
    )
)

st.dataframe(
    very_high_summary,
    width="stretch",
    hide_index=True
)


# ==================================================
# INTERACTIVE FLOOD RISK MAP
# ==================================================

st.divider()

st.subheader(
    "🗺️ Rajasthan Flood Risk Map"
)

st.caption(
    "Map markers represent district-level locations. "
    "Each marker summarizes the highest risk category "
    "present among that district's Aegis sub-districts."
)

map_data = create_map_data(
    risk_data
)

if map_data.empty:

    st.warning(
        "Map coordinates are not available."
    )

else:

    # ----------------------------------------------
    # MAP CENTER
    # ----------------------------------------------

    if pin_district_for_map:

        focus_key = normalize_text(
            pin_district_for_map
        )

        focus_coordinates = (
            DISTRICT_COORDINATES.get(
                focus_key
            )
        )

    else:

        focus_coordinates = None

    if focus_coordinates:

        center_lat = focus_coordinates[0]
        center_lon = focus_coordinates[1]
        zoom_level = 8

    else:

        center_lat = 26.5
        center_lon = 74.5
        zoom_level = 6

    # ----------------------------------------------
    # PYDECK MAP
    # ----------------------------------------------

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_data,
        get_position=[
            "longitude",
            "latitude"
        ],
        get_fill_color=[
            "color[0]",
            "color[1]",
            "color[2]",
            190
        ],
        get_line_color=[
            255,
            255,
            255,
            220
        ],
        get_radius=18000,
        radius_min_pixels=7,
        radius_max_pixels=30,
        pickable=True,
        stroked=True,
        filled=True
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom_level,
        pitch=0
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
            "html": """
            <b>{district}</b><br/>
            Risk Level: {risk_category}<br/>
            Very High Sub-Districts:
            {very_high_risk}<br/>
            High Sub-Districts:
            {high_risk}<br/>
            Maximum Flood-Affected:
            {max_flood_percent}%<br/>
            Average Flood-Affected:
            {avg_flood_percent}%<br/>
            Maximum Risk Score:
            {risk_score}
            """,
            "style": {
                "backgroundColor": "#111827",
                "color": "white"
            }
        }
    )

    st.pydeck_chart(
        deck,
        width="stretch",
        height=600
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Aegis | Flood Risk Intelligence Platform • "
    "Historical-model-based decision support"
)