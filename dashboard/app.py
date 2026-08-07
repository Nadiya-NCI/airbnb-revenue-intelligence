# Airbnb Revenue Intelligence Dashboard

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Airbnb Revenue Intelligence Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# Professional visual theme
# ---------------------------------------------------------

st.markdown(
    """
<style>

/* =========================================================
Page
========================================================= */

.block-container{
    padding-top:2rem;
    padding-bottom:2.5rem;
    max-width:1500px;
}


/* =========================================================
Sidebar
========================================================= */

section[data-testid="stSidebar"]{
    background:#F8FAFC;
    border-right:1px solid #E5E7EB;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3{
    color:#1E3A8A;
}


/* =========================================================
Headings
========================================================= */

h1{
    color:#1E3A8A;
    font-size:2.6rem;
    font-weight:700;
}

h2{
    color:#1E40AF;
    margin-top:1.2rem;
    margin-bottom:0.8rem;
}

h3{
    color:#1E40AF;
    margin-top:1rem;
}


/* =========================================================
Metric Cards
========================================================= */

div[data-testid="metric-container"]{
    background:#FFFFFF;
    border:1px solid #E5E7EB;
    border-radius:14px;
    padding:18px;
    box-shadow:0 2px 8px rgba(0,0,0,.05);
    transition:all .2s ease;
}

div[data-testid="metric-container"]:hover{
    transform:translateY(-2px);
    box-shadow:0 8px 20px rgba(37,99,235,.12);
}


/* Metric label */

div[data-testid="stMetricLabel"]{
    color:#64748B;
    font-size:14px;
    font-weight:600;
}


/* Metric value */

div[data-testid="stMetricValue"]{
    color:#1E3A8A;
    font-weight:700;
}


/* =========================================================
Buttons
========================================================= */

div.stButton > button{
    border-radius:10px;
    height:48px;
    font-size:16px;
    font-weight:600;
    transition:all .2s ease;
    border:none;
}

div.stButton > button:hover{
    transform:translateY(-1px);
    box-shadow:0 4px 14px rgba(37,99,235,.25);
}


/* =========================================================
Input Widgets
========================================================= */

div[data-baseweb="select"]{
    border-radius:10px;
}


/* =========================================================
Selected Amenity Tags
========================================================= */

span[data-baseweb="tag"]{
    background:#EFF6FF !important;
    color:#2563EB !important;
    border:1px solid #DBEAFE !important;
    border-radius:8px !important;
    font-weight:600 !important;
    padding:2px 8px !important;
}


/* Small X button inside amenity tag */

span[data-baseweb="tag"] svg{
    color:#2563EB !important;
}


/* =========================================================
Plotly Charts
========================================================= */

div[data-testid="stPlotlyChart"]{
    background:#FFFFFF;
    border-radius:14px;
    border:1px solid #E5E7EB;
    padding:8px;
    box-shadow:0 2px 6px rgba(0,0,0,.04);
}


/* =========================================================
Alert Boxes
========================================================= */

div[data-testid="stAlert"]{
    border-radius:12px;
    border:none;
}


/* =========================================================
Business Insight Cards
========================================================= */

.insight-card{
    padding:18px 20px;
    border-radius:12px;
    margin-bottom:16px;
    border:1px solid #E5E7EB;
    min-height:125px;
    box-shadow:0 2px 6px rgba(0,0,0,.04);
    transition:all .2s ease;
}

.insight-card:hover{
    transform:translateY(-2px);
    box-shadow:0 6px 16px rgba(37,99,235,.10);
}

.insight-title{
    font-size:16px;
    font-weight:700;
    color:#1E3A8A;
    margin-bottom:8px;
}

.insight-text{
    font-size:14px;
    line-height:1.55;
    color:#475569;
}


/* Pricing strategy card */

.insight-blue{
    background:#EFF6FF;
    border-left:4px solid #2563EB;
}


/* Listing maturity card */

.insight-teal{
    background:#F0FDFA;
    border-left:4px solid #14B8A6;
}


/* Local market card */

.insight-lightblue{
    background:#F0F9FF;
    border-left:4px solid #60A5FA;
}


/* Model limitations card */

.insight-amber{
    background:#FFFBEB;
    border-left:4px solid #F59E0B;
}


/* =========================================================
Horizontal Lines
========================================================= */

hr{
    margin-top:2.2rem;
    margin-bottom:2.2rem;
}


/* =========================================================
Captions
========================================================= */

div[data-testid="stCaptionContainer"]{
    color:#64748B;
    font-size:0.92rem;
}


/* =========================================================
General Paragraph Styling
========================================================= */

p{
    line-height:1.55;
}


/* =========================================================
Small Responsive Improvement
========================================================= */

@media (max-width:900px){

    .block-container{
        padding-left:1rem;
        padding-right:1rem;
    }

    h1{
        font-size:2rem;
    }

    .insight-card{
        min-height:auto;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Colour palette
# ---------------------------------------------------------

COLOUR_REVENUE = "#2563EB"      # Primary blue
COLOUR_PROPERTY = "#60A5FA"     # Light blue
COLOUR_CITY = "#14B8A6"         # Teal
COLOUR_ADR = "#F59E0B"          # Amber
COLOUR_FEATURES = "#1E40AF"     # Dark blue


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "airbnb_cleaned.csv.gz"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "airbnb_revenue_model_compressed_xz9.joblib"
)

FEATURE_IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "tables"
    / "table_23_random_forest_feature_importance.csv"
)


# ---------------------------------------------------------
# Load project artefacts
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(
        DATA_PATH,
        dtype={
            "Zipcode": "string"
        },
        low_memory=False,
        parse_dates=[
            "Created Date",
            "Last Scraped Date"
        ]
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_importance():
    return pd.read_csv(
        FEATURE_IMPORTANCE_PATH
    )


airbnb = load_data()
model = load_model()
feature_importance = load_feature_importance()


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def display_label(value):
    """
    Convert dataset labels into cleaner display labels.
    Example: entire_home -> Entire Home
    """

    return str(value).replace("_", " ").title()


def option_index(options, value):
    """
    Return a safe default index for Streamlit select boxes.
    """

    try:
        return options.index(value)
    except ValueError:
        return 0


# ---------------------------------------------------------
# Select a realistic default listing
# ---------------------------------------------------------

required_default_columns = [
    "Property Type",
    "Listing Type",
    "Bedrooms",
    "Bathrooms",
    "Max Guests",
    "City",
    "Cancellation Policy",
    "Has Wifi",
    "Has Kitchen",
    "Has Parking",
    "Has Pool",
    "Has Air Conditioning",
    "Has Washer",
    "Airbnb Superhost",
    "Listing Age (Days)",
    "Host Listing Count",
    "Average Daily Rate (USD)",
    "Annual Revenue LTM (USD)"
]

complete_listings = (
    airbnb[
        required_default_columns
    ]
    .dropna()
    .copy()
)

dataset_median_revenue = airbnb[
    "Annual Revenue LTM (USD)"
].median()

complete_listings["Distance_From_Median"] = (
    complete_listings[
        "Annual Revenue LTM (USD)"
    ]
    .sub(dataset_median_revenue)
    .abs()
)

default_listing = (
    complete_listings
    .sort_values("Distance_From_Median")
    .iloc[0]
)


# ---------------------------------------------------------
# Dashboard title
# ---------------------------------------------------------

st.title("Airbnb Revenue Intelligence Dashboard")

st.caption(
    f"Interactive dashboard for analysing **{len(airbnb):,} Airbnb listings** "
    f"across **{airbnb['City'].nunique()} global cities** and estimating "
    f"annual revenue using the final tuned Random Forest model."
)

st.markdown("---")


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

st.sidebar.title("Filters")

city_options = sorted(
    airbnb["City"]
    .dropna()
    .unique()
    .tolist()
)

property_options = sorted(
    airbnb["Property Type"]
    .dropna()
    .unique()
    .tolist()
)

listing_type_options = sorted(
    airbnb["Listing Type"]
    .dropna()
    .unique()
    .tolist()
)

selected_city = st.sidebar.selectbox(
    "City",
    ["All"] + city_options
)

selected_property = st.sidebar.selectbox(
    "Property Type",
    ["All"] + property_options
)

selected_listing = st.sidebar.selectbox(
    "Listing Type",
    ["All"] + listing_type_options,
    format_func=lambda value: (
        display_label(value)
        if value != "All"
        else "All"
    )
)


# ---------------------------------------------------------
# Apply sidebar filters
# ---------------------------------------------------------

filtered_airbnb = airbnb.copy()

if selected_city != "All":
    filtered_airbnb = filtered_airbnb[
        filtered_airbnb["City"] == selected_city
    ]

if selected_property != "All":
    filtered_airbnb = filtered_airbnb[
        filtered_airbnb["Property Type"] == selected_property
    ]

if selected_listing != "All":
    filtered_airbnb = filtered_airbnb[
        filtered_airbnb["Listing Type"] == selected_listing
    ]


# ---------------------------------------------------------
# Market overview
# ---------------------------------------------------------

st.markdown(
    "High-level summary of the selected Airbnb market."
    )

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="🏠 Listings",
        value=f"{len(filtered_airbnb):,}"
    )

with col2:
    average_revenue = (
        filtered_airbnb[
            "Annual Revenue LTM (USD)"
        ].mean()
        if not filtered_airbnb.empty
        else np.nan
    )

    st.metric(
        label="💰 Annual Revenue",
        value=(
            f"${average_revenue:,.0f}"
            if pd.notna(average_revenue)
            else "N/A"
        )
    )

with col3:
    average_adr = (
        filtered_airbnb[
            "Average Daily Rate (USD)"
        ].mean()
        if not filtered_airbnb.empty
        else np.nan
    )

    st.metric(
        label="🏷 Average ADR",
        value=(
            f"${average_adr:,.0f}"
            if pd.notna(average_adr)
            else "N/A"
        )
    )

with col4:
    superhost_rate = (
        (
            filtered_airbnb[
                "Airbnb Superhost"
            ] == "t"
        ).mean() * 100
        if not filtered_airbnb.empty
        else np.nan
    )

    st.metric(
        label="⭐ Superhosts",
        value=(
            f"{superhost_rate:.1f}%"
            if pd.notna(superhost_rate)
            else "N/A"
        )
    )

with col5:

    median_revenue = (
        filtered_airbnb[
            "Annual Revenue LTM (USD)"
        ].median()
        if not filtered_airbnb.empty
        else np.nan
    )

    st.metric(
        label="📈 Median Revenue",
        value=(
            f"${median_revenue:,.0f}"
            if pd.notna(median_revenue)
            else "N/A"
        )
    )

st.caption(
    f"Current filters return **{len(filtered_airbnb):,} listings** "
    f"from **{filtered_airbnb['City'].nunique()} cities**."
)

if filtered_airbnb.empty:
    st.warning(
        "No listings match the selected filters. "
        "Change one or more sidebar selections."
    )


# ---------------------------------------------------------
# Market explorer
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## Market Explorer")

if filtered_airbnb.empty:
    st.warning(
        "No data is available for the selected filters."
    )

else:

    chart_col1, chart_col2 = st.columns(2)

    # -----------------------------------------------------
    # Revenue distribution
    # -----------------------------------------------------

    with chart_col1:

        revenue_cap = (
            filtered_airbnb[
                "Annual Revenue LTM (USD)"
            ]
            .quantile(0.99)
        )

        revenue_chart_data = filtered_airbnb[
            filtered_airbnb[
                "Annual Revenue LTM (USD)"
            ] <= revenue_cap
        ]

        revenue_figure = px.histogram(
            revenue_chart_data,
            x="Annual Revenue LTM (USD)",
            nbins=40,
            title="Revenue Distribution",
            color_discrete_sequence=[COLOUR_REVENUE]
        )

        revenue_figure.update_traces(
            opacity=0.85
        )

        revenue_figure.update_layout(
            template="simple_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#334155"),
            title_font=dict(
                size=20,
                color="#1E3A8A"
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            ),
            xaxis_title="Annual Revenue (USD)",
            yaxis_title="Listings",
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        revenue_figure.update_xaxes(
            tickprefix="$",
            separatethousands=True
        )

        st.plotly_chart(
            revenue_figure,
            use_container_width=True
        )

        st.caption(
            "Values above the 99th percentile are hidden for readability."
        )

    # -----------------------------------------------------
    # Property type comparison
    # -----------------------------------------------------

    with chart_col2:

        property_revenue = (
            filtered_airbnb
            .groupby(
                "Property Type",
                as_index=False
            )
            .agg(
                Average_Revenue=(
                    "Annual Revenue LTM (USD)",
                    "mean"
                ),
                Listings=(
                    "Annual Revenue LTM (USD)",
                    "size"
                )
            )
            .query("Listings >= 30")
            .sort_values(
                "Average_Revenue",
                ascending=False
            )
            .head(10)
        )

        if property_revenue.empty:

            st.info(
                "Not enough listings are available to compare property types."
            )

        else:

            property_figure = px.bar(
                property_revenue,
                x="Average_Revenue",
                y="Property Type",
                orientation="h",
                title="Property Types by Revenue",
                custom_data=["Listings"],
                color_discrete_sequence=[COLOUR_PROPERTY]
            )

            property_figure.update_traces(
                opacity=0.90,
                marker_line_width=0,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Average Revenue: $%{x:,.0f}<br>"
                    "Listings: %{customdata[0]:,}"
                    "<extra></extra>"
                )
            )

            property_figure.update_layout(
                template="simple_white",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#334155"),
                title_font=dict(
                    size=20,
                    color="#1E3A8A"
                ),
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=14
                ),
                xaxis_title="Average Annual Revenue (USD)",
                yaxis_title="",
                yaxis=dict(
                    categoryorder="total ascending"
                ),
                showlegend=False,
                margin=dict(
                    l=20,
                    r=20,
                    t=70,
                    b=20
                )
            )

            property_figure.update_xaxes(
                tickprefix="$",
                separatethousands=True
            )

            st.plotly_chart(
                property_figure,
                use_container_width=True
            )

            st.caption(
                "Property types with fewer than 30 listings are excluded."
            )

    chart_col3, chart_col4 = st.columns(2)

    # -----------------------------------------------------
    # Revenue by city
    # -----------------------------------------------------

    with chart_col3:

        city_revenue = (
            filtered_airbnb
            .groupby(
                "City",
                as_index=False
            )
            .agg(
                Average_Revenue=(
                    "Annual Revenue LTM (USD)",
                    "mean"
                ),
                Listings=(
                    "Annual Revenue LTM (USD)",
                    "size"
                )
            )
            .sort_values(
                "Average_Revenue",
                ascending=False
            )
        )

        city_figure = px.bar(
            city_revenue,
            x="City",
            y="Average_Revenue",
            title="Revenue by City",
            custom_data=["Listings"],
            color_discrete_sequence=[COLOUR_CITY]
        )

        city_figure.update_traces(
            opacity=0.90,
            marker_line_width=0,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Average Revenue: $%{y:,.0f}<br>"
                "Listings: %{customdata[0]:,}"
                "<extra></extra>"
            )
        )

        city_figure.update_layout(
            template="simple_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#334155"),
            title_font=dict(
                size=20,
                color="#1E3A8A"
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            ),
            xaxis_title="City",
            yaxis_title="Average Annual Revenue (USD)",
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        city_figure.update_xaxes(
            categoryorder="total descending"
        )

        city_figure.update_yaxes(
            tickprefix="$",
            separatethousands=True
        )

        st.plotly_chart(
            city_figure,
            use_container_width=True
        )

        st.caption(
            "Calculated using the selected filters."
        )

    # -----------------------------------------------------
    # ADR distribution
    # -----------------------------------------------------

    with chart_col4:

        adr_cap = (
            filtered_airbnb[
                "Average Daily Rate (USD)"
            ]
            .quantile(0.99)
        )

        adr_chart_data = filtered_airbnb[
            filtered_airbnb[
                "Average Daily Rate (USD)"
            ] <= adr_cap
        ]

        adr_figure = px.histogram(
            adr_chart_data,
            x="Average Daily Rate (USD)",
            nbins=40,
            title="ADR Distribution",
            color_discrete_sequence=[COLOUR_ADR]
        )

        adr_figure.update_traces(
            opacity=0.85
        )

        adr_figure.update_layout(
            template="simple_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#334155"),
            title_font=dict(
                size=20,
                color="#1E3A8A"
            ),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14
            ),
            xaxis_title="Average Daily Rate (USD)",
            yaxis_title="Listings",
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=70,
                b=20
            )
        )

        adr_figure.update_xaxes(
            tickprefix="$",
            separatethousands=True
        )

        st.plotly_chart(
            adr_figure,
            use_container_width=True
        )

        st.caption(
            "Values above the 99th percentile are hidden for readability."
        )


# ---------------------------------------------------------
# Revenue estimator
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## 💰 Revenue Estimator")

st.caption(
    "Adjust the listing characteristics below to estimate annual Airbnb revenue "
    "using the final tuned Random Forest model."
)

st.info(
    "The estimator opens with a real Airbnb listing from the dataset. "
    "You can modify any values to explore different investment scenarios."
)

# ---------------------------------------------------------
# Prediction options
# ---------------------------------------------------------

prediction_city_options = city_options
prediction_property_options = property_options
prediction_listing_options = listing_type_options

prediction_policy_options = sorted(
    airbnb[
        "Cancellation Policy"
    ]
    .dropna()
    .unique()
    .tolist()
)

# ---------------------------------------------------------
# Prediction inputs
# ---------------------------------------------------------

input_col1, input_col2, input_col3 = st.columns(3)

# =========================================================
# Location & Property
# =========================================================

with input_col1:

    st.markdown("#### 📍 Location & Property")

    prediction_city = st.selectbox(
        "City",
        prediction_city_options,
        index=option_index(
            prediction_city_options,
            default_listing["City"]
        ),
        key="prediction_city"
    )

    prediction_property = st.selectbox(
        "Property Type",
        prediction_property_options,
        index=option_index(
            prediction_property_options,
            default_listing["Property Type"]
        ),
        key="prediction_property"
    )

    prediction_listing = st.selectbox(
        "Listing Type",
        prediction_listing_options,
        index=option_index(
            prediction_listing_options,
            default_listing["Listing Type"]
        ),
        format_func=display_label,
        key="prediction_listing"
    )

    prediction_policy = st.selectbox(
        "Cancellation Policy",
        prediction_policy_options,
        index=option_index(
            prediction_policy_options,
            default_listing["Cancellation Policy"]
        ),
        format_func=display_label,
        key="prediction_policy"
    )

# =========================================================
# Property Details
# =========================================================

with input_col2:

    st.markdown("#### 🏠 Property Details")

    prediction_bedrooms = st.number_input(
        "Bedrooms",
        min_value=0,
        max_value=20,
        value=int(
            default_listing["Bedrooms"]
        ),
        step=1
    )

    bathroom_options = [
        x / 2
        for x in range(0, 41)
    ]

    prediction_bathrooms = st.selectbox(
        "Bathrooms",
        bathroom_options,
        index=option_index(
            bathroom_options,
            float(
                default_listing["Bathrooms"]
            )
        ),
        format_func=lambda value: f"{value:g}"
    )

    prediction_guests = st.number_input(
        "Maximum Guests",
        min_value=1,
        max_value=40,
        value=int(
            default_listing["Max Guests"]
        ),
        step=1
    )

    prediction_adr = st.number_input(
        "Average Daily Rate (USD)",
        min_value=1.0,
        max_value=5000.0,
        value=float(
            default_listing[
                "Average Daily Rate (USD)"
            ]
        ),
        step=10.0,
        format="%.0f"
    )

# =========================================================
# Host & Amenities
# =========================================================

with input_col3:

    st.markdown("#### ⭐ Host & Amenities")

    prediction_listing_age = st.number_input(
        "Listing Age (Days)",
        min_value=0,
        max_value=10000,
        value=int(
            default_listing[
                "Listing Age (Days)"
            ]
        ),
        step=30
    )

    prediction_host_count = st.number_input(
        "Host Listing Count",
        min_value=0,
        max_value=10000,
        value=int(
            default_listing[
                "Host Listing Count"
            ]
        ),
        step=1
    )

    superhost_options = [
        "f",
        "t"
    ]

    prediction_superhost = st.selectbox(
        "Airbnb Superhost",
        superhost_options,
        index=option_index(
            superhost_options,
            default_listing[
                "Airbnb Superhost"
            ]
        ),
        format_func=lambda value: (
            "Yes"
            if value == "t"
            else "No"
        )
    )

    default_amenities = []

    if default_listing["Has Wifi"] == 1:
        default_amenities.append("Wifi")

    if default_listing["Has Kitchen"] == 1:
        default_amenities.append("Kitchen")

    if default_listing["Has Parking"] == 1:
        default_amenities.append("Parking")

    if default_listing["Has Pool"] == 1:
        default_amenities.append("Pool")

    if default_listing["Has Air Conditioning"] == 1:
        default_amenities.append(
            "Air Conditioning"
        )

    if default_listing["Has Washer"] == 1:
        default_amenities.append("Washer")

    prediction_amenities = st.multiselect(
        "Amenities",
        [
            "Wifi",
            "Kitchen",
            "Parking",
            "Pool",
            "Air Conditioning",
            "Washer"
        ],
        default=default_amenities
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Build prediction record
# ---------------------------------------------------------

prediction_record = pd.DataFrame({
    "Property Type": [prediction_property],
    "Listing Type": [prediction_listing],
    "Bedrooms": [prediction_bedrooms],
    "Bathrooms": [prediction_bathrooms],
    "Max Guests": [prediction_guests],
    "City": [prediction_city],
    "Cancellation Policy": [prediction_policy],
    "Has Wifi": [int("Wifi" in prediction_amenities)],
    "Has Kitchen": [int("Kitchen" in prediction_amenities)],
    "Has Parking": [int("Parking" in prediction_amenities)],
    "Has Pool": [int("Pool" in prediction_amenities)],
    "Has Air Conditioning": [int("Air Conditioning" in prediction_amenities)],
    "Has Washer": [int("Washer" in prediction_amenities)],
    "Airbnb Superhost": [prediction_superhost],
    "Listing Age (Days)": [prediction_listing_age],
    "Host Listing Count": [prediction_host_count],
    "Log ADR": [np.log1p(prediction_adr)]
})


# ---------------------------------------------------------
# Prediction button
# ---------------------------------------------------------

st.markdown("---")

left, centre, right = st.columns([2, 3, 2])

with centre:
    predict_clicked = st.button(
        "Estimate Annual Revenue",
        type="primary",
        use_container_width=True
    )


# ---------------------------------------------------------
# Generate prediction
# ---------------------------------------------------------

if predict_clicked:

    predicted_log_revenue = model.predict(
        prediction_record
    )[0]

    predicted_revenue = max(
        np.expm1(predicted_log_revenue),
        0
    )

    # -----------------------------------------------------
    # Comparable listings
    # -----------------------------------------------------

    comparable_market = airbnb[
        (airbnb["City"] == prediction_city)
        &
        (airbnb["Property Type"] == prediction_property)
        &
        (airbnb["Listing Type"] == prediction_listing)
    ]

    comparison_label = "Similar Listings"

    if len(comparable_market) < 30:

        comparable_market = airbnb[
            (airbnb["City"] == prediction_city)
            &
            (airbnb["Listing Type"] == prediction_listing)
        ]

        comparison_label = (
            f"{prediction_city} "
            f"{display_label(prediction_listing)} Listings"
        )

    if len(comparable_market) < 30:

        comparable_market = airbnb[
            airbnb["City"] == prediction_city
        ]

        comparison_label = (
            f"{prediction_city} Listings"
        )

    benchmark_revenue = (
        comparable_market[
            "Annual Revenue LTM (USD)"
        ]
        .median()
    )

    difference = (
        predicted_revenue
        - benchmark_revenue
    )

    difference_percentage = (
        difference
        / benchmark_revenue
        * 100
        if benchmark_revenue > 0
        else np.nan
    )

    # -----------------------------------------------------
    # Prediction results
    # -----------------------------------------------------

    st.markdown("### Estimated Performance")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "💰 Estimated Revenue",
            f"${predicted_revenue:,.0f}"
        )

    with result_col2:

        st.metric(
            "📊 Comparable Median",
            f"${benchmark_revenue:,.0f}"
        )

    with result_col3:

        st.metric(
            "Difference",
            f"{difference_percentage:+.1f}%"
        )

    st.caption(
        f"Benchmark calculated from **{len(comparable_market):,} listings** "
        f"within **{comparison_label}**."
    )

    # -----------------------------------------------------
    # Interpretation
    # -----------------------------------------------------

    if predicted_revenue >= benchmark_revenue * 1.20:

        st.success(
            "Estimated revenue is above the comparable market median."
        )

    elif predicted_revenue >= benchmark_revenue * 0.80:

        st.info(
            "Estimated revenue is broadly aligned with comparable listings."
        )

    else:

        st.warning(
            "Estimated revenue is below the comparable market median."
        )

    st.caption(
        "Revenue estimates are generated by the tuned Random Forest model. "
        "They should be interpreted as decision-support estimates rather "
        "than guaranteed financial outcomes."
    )
# ---------------------------------------------------------
# Model performance
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## 📋 Model Performance")

model_col1, model_col2, model_col3, model_col4 = (
    st.columns(4)
)

with model_col1:
    st.metric(
        label="Model",
        value="Tuned Random Forest"
    )

with model_col2:
    st.metric(
        label="Test R²",
        value="0.504"
    )

with model_col3:
    st.metric(
        label="Cross-Validated R²",
        value="0.502"
    )

with model_col4:
    st.metric(
        label="Test MAE",
        value="$9,828"
    )

st.caption(
    "R² values are measured on the log-transformed annual revenue target. "
    "MAE is reported in original US-dollar terms."
)


# ---------------------------------------------------------
# Key Revenue Drivers
# ---------------------------------------------------------

st.markdown("---")
st.markdown("## 💡 Key Revenue Drivers")

st.caption(
    "The chart below shows the variables that contributed most strongly "
    "to predictions made by the final tuned Random Forest model."
)

# ---------------------------------------------------------
# Prepare feature-importance data
# ---------------------------------------------------------

feature_importance_plot = feature_importance.copy()

# Cleaner feature names for dashboard display
feature_importance_plot["Feature"] = (
    feature_importance_plot["Feature"]
    .replace({
        "Log ADR": "Average Daily Rate"
    })
)

# Convert importance to percentage
feature_importance_plot["Importance (%)"] = (
    feature_importance_plot["Importance"] * 100
)

# Select top ten drivers
top_features = (
    feature_importance_plot
    .sort_values(
        "Importance (%)",
        ascending=False
    )
    .head(10)
    .copy()
)

# ---------------------------------------------------------
# Feature importance chart
# ---------------------------------------------------------

feature_figure = px.bar(
    top_features.sort_values(
        "Importance (%)",
        ascending=True
    ),
    x="Importance (%)",
    y="Feature",
    orientation="h",
    title="Top Revenue Drivers",
    text="Importance (%)",
    color_discrete_sequence=[
        COLOUR_FEATURES
    ]
)

feature_figure.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
    cliponaxis=False,
    marker_line_width=0,
    opacity=0.90,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Feature Importance: %{x:.1f}%"
        "<extra></extra>"
    )
)

feature_figure.update_layout(
    template="simple_white",
    plot_bgcolor="white",
    paper_bgcolor="white",

    font=dict(
        color="#334155"
    ),

    title_font=dict(
        size=20,
        color="#1E3A8A"
    ),

    xaxis_title="Feature Importance (%)",
    yaxis_title="",

    showlegend=False,

    hoverlabel=dict(
        bgcolor="white",
        font_size=14
    ),

    bargap=0.35,

    margin=dict(
        l=30,
        r=70,
        t=70,
        b=30
    )
)

feature_figure.update_xaxes(
    ticksuffix="%",
    tickfont=dict(
        size=12
    ),
    gridcolor="#E5E7EB",
    range=[
        0,
        top_features["Importance (%)"].max() * 1.12
    ]
)

feature_figure.update_yaxes(
    tickfont=dict(
        size=13
    )
)

st.plotly_chart(
    feature_figure,
    use_container_width=True
)

st.caption(
    "Feature importance indicates how strongly each variable contributes "
    "to the model's predictions. Higher values represent greater influence "
    "on predicted annual revenue and do not imply a causal relationship."
)

# ---------------------------------------------------------
# Business insights
# ---------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Business Insights")

st.caption(
    "Key findings from the exploratory analysis and final predictive model."
)

insight_col1, insight_col2 = st.columns(2)

with insight_col1:

    st.markdown(
        """
        <div class="insight-card insight-blue">
            <div class="insight-title">
                💰 Pricing Strategy
            </div>
            <div class="insight-text">
                Average Daily Rate is the strongest predictor of annual
                revenue. Pricing should therefore be benchmarked against
                comparable listings within the same local market.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="insight-card insight-teal">
            <div class="insight-title">
                🏠 Listing Maturity
            </div>
            <div class="insight-text">
                Listing age is the second most important predictor.
                Established listings may benefit from accumulated visibility,
                reviews and operating experience.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with insight_col2:

    st.markdown(
        """
        <div class="insight-card insight-lightblue">
            <div class="insight-title">
                🌍 Local Market Conditions
            </div>
            <div class="insight-text">
                City, property type and cancellation policy all contribute
                to predicted revenue. Revenue expectations should therefore
                reflect local demand and listing characteristics.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="insight-card insight-amber">
            <div class="insight-title">
                📊 Model Limitations
            </div>
            <div class="insight-text">
                The final model explains approximately half of the variation
                in log-transformed annual revenue. Predictions should be used
                for decision support rather than treated as guaranteed
                financial outcomes.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )