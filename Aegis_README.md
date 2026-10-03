# Aegis --- AI-Powered Flood Risk Intelligence Platform

Aegis is a Streamlit-based flood-risk intelligence platform designed for
Rajasthan. It combines historical flood-risk records, weather variables,
PIN/location intelligence, an ML-based risk engine, and district-level
geospatial visualization into one dashboard.

> **Important:** Aegis is designed as historical-model-based decision
> support. It is not presented as an official real-time flood warning
> system or an exact household/PIN-level flood forecast.

------------------------------------------------------------------------

## 1. Core Objectives

Aegis is designed to help users:

-   Explore historical flood-risk patterns across Rajasthan.
-   Filter risk information by district, risk category, and
    sub-district.
-   Search locations using PIN intelligence.
-   Estimate historical flood-affected percentage from weather inputs
    using a trained Ridge Regression model.
-   Inspect model-driven risk categories and explainable weather-factor
    contributions.
-   Explore historical weather/flood data.
-   Visualize district-level risk on an interactive map.
-   Use AWS S3 as the primary data/model source with local fallback
    support.

------------------------------------------------------------------------

## 2. Main Dashboard Modules

### Command Center

Provides a high-level operational view including:

-   Monitored records
-   Districts covered
-   High + Very High areas
-   Very High areas
-   Risk distribution
-   Critical areas
-   Filtered intelligence

The sidebar filters drive the dashboard view.

### Location Intelligence

Provides:

-   6-digit PIN lookup
-   PIN → district resolution
-   Associated local area/office information when available
-   District-level Aegis risk profile
-   Sub-district risk table
-   District risk distribution
-   Fuzzy sub-district search and spelling correction

The application explicitly treats PIN lookup as district/location
resolution rather than household-level flood prediction.

### AI Risk Engine

Uses six weather variables:

1.  Annual Rainfall
2.  Temperature
3.  Humidity
4.  Wind Speed
5.  Pressure
6.  Maximum Daily Rainfall

The trained Ridge Regression model estimates flood-affected percentage.
The estimate is converted into:

-   Low
-   Moderate
-   High
-   Very High

The module also displays:

-   Estimated flood-affected percentage
-   Risk category
-   Risk score
-   Recommended response
-   Explainable model contributions
-   Prediction inputs

### Historical Intelligence

Provides access to the historical data supporting Aegis risk
intelligence, including:

-   Flood-affected data
-   Weather variables
-   Historical data explorer
-   1986--2025 historical period shown by the current application

### Risk Command Map

Provides a district-level interactive geospatial view.

The current application uses representative/district-headquarter
coordinates for visualization. These coordinates should not be
interpreted as exact sub-district coordinates.

------------------------------------------------------------------------

## 3. Technology Stack

### Application

-   Python
-   Streamlit
-   Pandas
-   Plotly
-   PyDeck

### Machine Learning

-   Scikit-learn
-   Ridge Regression
-   Joblib

### Cloud / Data

-   AWS S3
-   Boto3

### Supporting Services / Libraries

-   Requests
-   Difflib
-   Regular Expressions
-   Pathlib

------------------------------------------------------------------------

## 4. AWS S3 Architecture

Aegis uses an **S3-first architecture**.

### S3 Bucket

``` text
aegis-flood-risk-intelligence-2026
```

### Region

``` text
ap-south-1
```

### S3 Objects

``` text
processed-data/aegis_risk_dashboard_data.csv
external-data/rajasthan_pincode_directory.csv
processed-data/rajasthan_flood_weather_merged_1986_2025.csv
models/ridge_flood_risk_model.joblib
```

### Data flow

``` text
                 ┌──────────────────────┐
                 │      AWS S3 Bucket    │
                 │ aegis-flood-risk-...  │
                 └──────────┬───────────┘
                            │
                    Primary data source
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Aegis Streamlit    │
                 │      Dashboard       │
                 └──────────┬───────────┘
                            │
            ┌───────────────┼────────────────┐
            ▼               ▼                ▼
      Risk Analytics   Location/PIN      AI Risk Engine
            │               │                │
            └───────────────┼────────────────┘
                            ▼
                    User-facing insights

        S3 unavailable / object unavailable
                            │
                            ▼
                    Local project files
                    (fallback mechanism)
```

The current application independently checks the four S3 sources and
displays their status in the sidebar.

------------------------------------------------------------------------

## 5. Local Project Structure

The project is organized around data, models, source code, dashboard,
and cloud configuration.

``` text
Aegis/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   └── ridge_flood_risk_model.joblib
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── dashboard/
│   └── app.py
│
├── aws/
│
├── snowflake/
│
├── requirements.txt
└── README.md
```

The exact contents of auxiliary folders may vary as the project evolves.

------------------------------------------------------------------------

## 6. Local Setup

### 1. Open the project

``` powershell
cd "C:\Users\Monika\OneDrive\Desktop\Aegis\Aegis"
```

### 2. Activate the virtual environment

``` powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install requirements

``` powershell
pip install -r requirements.txt
```

### 4. Run the dashboard

``` powershell
streamlit run .\dashboard\app.py
```

The application normally opens at:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

## 7. AWS Credentials

The application uses Boto3's AWS credential resolution rather than
hard-coding credentials into the dashboard.

Depending on the deployment environment, credentials can be supplied
through an appropriate AWS credential mechanism or Streamlit secrets.

**Never commit AWS access keys or secret keys to GitHub.**

------------------------------------------------------------------------

## 8. Data Loading Strategy

For each major source, Aegis follows this pattern:

``` text
Try S3
   │
   ├── Success → load from S3
   │
   └── Failure → use local fallback
```

The four monitored sources are:

-   Risk dataset
-   PIN dataset
-   Historical dataset
-   Prediction model

The current dashboard exposes this state through the **System Status**
card.

------------------------------------------------------------------------

## 9. Risk Classification

The application uses four risk categories:

  Category      Score
  ----------- -------
  Low               1
  Moderate          2
  High              3
  Very High         4

The current application converts the model's predicted flood-affected
percentage into these categories using the configured thresholds in
`app.py`.

------------------------------------------------------------------------

## 10. Current Dashboard Validation

The completed functional testing covered:

-   S3 data-source status
-   Sidebar scrolling
-   System Status visibility
-   District filtering
-   Risk Category filtering
-   Sub-District search
-   Fuzzy matching
-   Automatic spelling correction
-   Risk distribution
-   Critical-area table
-   Location Intelligence
-   PIN/location flow
-   AI Risk Engine
-   Historical Intelligence
-   Risk Command Map

The tested dashboard displayed 27 districts and 162 sub-district records
when the global filters were set to `All`.

------------------------------------------------------------------------

## 11. Important Limitations

Aegis should be described accurately during demonstrations and
presentations.

### Historical decision support

The AI Risk Engine is based on a trained historical model. It should not
be presented as an official real-time flood-warning system.

### PIN resolution

PIN intelligence resolves a location to a district and connects it with
available Aegis risk information. It is not an exact household-level
flood prediction.

### Map coordinates

The current map uses district headquarters / representative district
coordinates. A marker therefore represents a district-level
visualization rather than an exact sub-district location.

### External warning authority

Emergency decisions should follow official disaster-management and
weather authorities. Aegis is a decision-support and intelligence
platform.

------------------------------------------------------------------------

## 12. Demonstration Flow

For a project presentation or live demo:

1.  Open **Command Center**.
2.  Show the overall Rajasthan risk distribution.
3.  Demonstrate **District** filtering.
4.  Demonstrate **Risk Category** filtering.
5.  Demonstrate fuzzy **Sub-District** search.
6.  Open **Location Intelligence** and demonstrate PIN resolution.
7.  Open **AI Risk Engine** and enter weather values.
8.  Show the estimated flood-affected percentage and risk category.
9.  Show the explainable model contributions.
10. Open **Historical Intelligence**.
11. Open **Risk Command Map**.
12. Show the sidebar **System Status** and explain the S3-first
    architecture.

------------------------------------------------------------------------

## 13. Project Positioning

### One-line description

**Aegis is an AI-powered flood-risk intelligence platform that combines
historical flood data, weather-driven machine learning, location
intelligence, AWS cloud data storage, and interactive geospatial
analytics for Rajasthan.**

### Short presentation description

Aegis integrates historical flood-risk information with weather
variables and location intelligence to provide an interactive
decision-support dashboard for Rajasthan. Its AI Risk Engine uses a
trained Ridge Regression model to estimate historical flood-affected
percentage from six weather signals, while AWS S3 provides centralized
cloud-based data and model storage with local fallback support.

------------------------------------------------------------------------

## 14. Future Scope

Potential future extensions include:

-   Real-time weather ingestion
-   River-level and reservoir monitoring
-   Satellite-based flood extent analysis
-   Live rainfall alerts
-   Time-series forecasting
-   More advanced ML/deep-learning models
-   Automated district-level alerts
-   Historical event timelines
-   Higher-resolution geospatial layers
-   Disaster-response resource optimization
-   Production deployment with authentication and monitoring

------------------------------------------------------------------------

## 15. Project Status

**Current status: Functional dashboard completed and tested.**

The next project phase can focus on:

-   Final documentation
-   Architecture diagrams
-   Project report
-   Presentation/PPT
-   GitHub cleanup
-   Deployment preparation
-   Demo script
-   Resume/project description
