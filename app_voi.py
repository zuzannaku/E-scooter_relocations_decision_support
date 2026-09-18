import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pydeck as pdk
import json
import json
import geopandas as gpd

from pathlib import Path
from sklearn.cluster import DBSCAN


####### Page setup

st.set_page_config(
    page_title="E-Scooter Relocation Decision Support",
    layout="wide"
)

####### File paths

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "xgb_relocation_model.pkl"
FEATURES_PATH = BASE_DIR / "models" / "model_features.pkl"
DATA_PATH = BASE_DIR / "data" / "test_idling_processed.parquet"
SERVICE_AREA_PATH = BASE_DIR / "data" / "service_area_voi.parquet"


####### Load model

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features


####### Load data

@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)

    df["starttime"] = pd.to_datetime(df["starttime"])
    df["endtime"] = pd.to_datetime(df["endtime"])

    return df


####### Load service area

@st.cache_data
def load_service_area():
    df_area = pd.read_parquet(SERVICE_AREA_PATH)

    df_area["geometry"] = gpd.GeoSeries.from_wkt(
        df_area["geom_service_area"]
    )

    gdf = gpd.GeoDataFrame(
        df_area,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return json.loads(gdf.to_json())

####### Load files

model, model_features = load_model()
df = load_data()
service_area = load_service_area()

####### Header

st.title("E-Scooter Relocation Decision Support")

st.write(
    """
    Replay a historical instance from Munich and examine which of the idle scooters
    could have been flagged by the model for relocation.

    The model calculates the likelihood of whether a currently idle scooter will
    remain unused for more than 8 hours. These scooters are then flagged and
    clustered together to create operational relocation zones.
    """
)


####### Default snapshot

DEFAULT_SNAPSHOT = pd.Timestamp("2025-03-15 12:00:00")

min_date = df["starttime"].dt.date.min()
max_date = df["starttime"].dt.date.max()


####### Historical replay

st.sidebar.header("Historical Replay")

selected_date = st.sidebar.date_input(
    "Select date",
    value=DEFAULT_SNAPSHOT.date(),
    min_value=min_date,
    max_value=max_date
)

selected_hour = st.sidebar.slider(
    "Select hour",
    min_value=0,
    max_value=23,
    value=DEFAULT_SNAPSHOT.hour,
    step=1
)

snapshot_time = (
    pd.Timestamp(selected_date)
    + pd.Timedelta(hours=selected_hour)
)

st.sidebar.caption(
    f"Selected snapshot: "
    f"{snapshot_time.strftime('%d %B %Y, %H:%M')}"
)


####### Relocation settings

st.sidebar.divider()
st.sidebar.header("Relocation Settings")

probability_threshold = st.sidebar.slider(
    "Minimum relocation probability",
    min_value=0.0,
    max_value=1.0,
    value=0.60,
    step=0.05
)

cluster_radius_m = st.sidebar.slider(
    "Cluster radius",
    min_value=10,
    max_value=200,
    value=50,
    step=10,
    format="%d m"
)

min_scooters = st.sidebar.slider(
    "Minimum scooters per cluster",
    min_value=5,
    max_value=20,
    value=6,
    step=1
)


####### Historical snapshot

snapshot = df[
    (df["starttime"] <= snapshot_time)
    &
    (df["endtime"] > snapshot_time)
].copy()

if snapshot.empty:
    st.warning(
        "No idle scooters were found at this historical moment. "
        "Try another date or hour."
    )
    st.stop()


####### Model predictions

X_snapshot = snapshot[model_features].copy()

snapshot["relocation_probability"] = (
    model.predict_proba(X_snapshot)[:, 1]
)

relocation_candidates = snapshot[
    snapshot["relocation_probability"] >= probability_threshold
].copy()


####### Metrics for summary tavle

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "Idle Scooters",
    f"{len(snapshot):,}"
)

metric2.metric(
    "Relocation Candidates",
    f"{len(relocation_candidates):,}"
)

candidate_share = (
    len(relocation_candidates)
    / len(snapshot)
    * 100
)

metric3.metric(
    "Candidate Share",
    f"{candidate_share:.1f}%"
)

cluster_count_placeholder = metric4.empty()


####### Check candidate count

if len(relocation_candidates) < min_scooters:

    cluster_count_placeholder.metric(
        "Priority Zones",
        "0"
    )

    st.info(
        "There are not enough relocation candidates "
        "to create a spatial cluster."
    )

    st.stop()


####### DBSCAN clustering

coords = np.radians(
    relocation_candidates[
        ["lat", "lon"]
    ].astype(float).values
)

EARTH_RADIUS_KM = 6371.0

dbscan = DBSCAN(
    eps=(cluster_radius_m / 1000.0) / EARTH_RADIUS_KM,
    min_samples=min_scooters,
    metric="haversine"
)

relocation_candidates["cluster"] = (
    dbscan.fit_predict(coords)
)

clustered = relocation_candidates[
    relocation_candidates["cluster"] != -1
].copy()


####### Cluster summary

if clustered.empty:

    cluster_summary = pd.DataFrame(
        columns=[
            "cluster",
            "scooter_count",
            "avg_relocation_probability",
            "center_lat",
            "center_lon",
            "priority_score"
        ]
    )

else:

    cluster_summary = (
        clustered
        .groupby("cluster")
        .agg(
            scooter_count=("id", "count"),

            avg_relocation_probability=(
                "relocation_probability",
                "mean"
            ),

            center_lat=("lat", "mean"),
            center_lon=("lon", "mean")
        )
        .reset_index()
    )

    cluster_summary["priority_score"] = (
        cluster_summary["scooter_count"]
        *
        cluster_summary["avg_relocation_probability"]
    )

    cluster_summary = (
        cluster_summary
        .sort_values(
            "priority_score",
            ascending=False
        )
        .reset_index(drop=True)
    )


cluster_count_placeholder.metric(
    "Priority Zones",
    f"{len(cluster_summary):,}"
)


####### Map data

idle_map = snapshot[
    ["id", "lat", "lon"]
].copy()

candidate_map = relocation_candidates[
    [
        "id",
        "lat",
        "lon",
        "relocation_probability",
        "cluster"
    ]
].copy()

idle_map["lat"] = idle_map["lat"].astype(float)
idle_map["lon"] = idle_map["lon"].astype(float)

candidate_map["lat"] = candidate_map["lat"].astype(float)
candidate_map["lon"] = candidate_map["lon"].astype(float)

candidate_map["relocation_probability"] = (
    candidate_map["relocation_probability"].astype(float)
)

candidate_map["cluster"] = (
    candidate_map["cluster"].astype(int)
)


if not cluster_summary.empty:

    cluster_map = cluster_summary[
        [
            "cluster",
            "scooter_count",
            "avg_relocation_probability",
            "center_lat",
            "center_lon",
            "priority_score"
        ]
    ].copy()

    cluster_map["cluster"] = (
        cluster_map["cluster"].astype(int)
    )

    cluster_map["scooter_count"] = (
        cluster_map["scooter_count"].astype(int)
    )

    cluster_map["avg_relocation_probability"] = (
        cluster_map["avg_relocation_probability"].astype(float)
    )

    cluster_map["center_lat"] = (
        cluster_map["center_lat"].astype(float)
    )

    cluster_map["center_lon"] = (
        cluster_map["center_lon"].astype(float)
    )

    cluster_map["priority_score"] = (
        cluster_map["priority_score"].astype(float)
    )

else:

    cluster_map = pd.DataFrame()


####### Scooter layers

idle_scooters_layer = pdk.Layer(
    "ScatterplotLayer",
    data=idle_map,
    get_position="[lon, lat]",
    get_radius=10,
    get_fill_color=[95, 95, 95, 150],
    pickable=False,
    radius_min_pixels=1,
    radius_max_pixels=3
)

candidate_layer = pdk.Layer(
    "ScatterplotLayer",
    data=candidate_map,
    get_position="[lon, lat]",
    get_radius=16,
    get_fill_color=[190, 65, 55, 220],
    pickable=False,
    radius_min_pixels=2,
    radius_max_pixels=5
)


####### Service area

map_layers = []

if service_area is not None:

    service_area_layer = pdk.Layer(
        "GeoJsonLayer",
        data=service_area,
        stroked=True,
        filled=False,
        get_line_color=[70, 70, 70, 190],
        line_width_min_pixels=2,
        pickable=False
    )

    map_layers.append(service_area_layer)

map_layers.append(idle_scooters_layer)
map_layers.append(candidate_layer)


####### Relocation zones

if not cluster_map.empty:

    cluster_zone_layer = pdk.Layer(
        "ScatterplotLayer",
        data=cluster_map,
        get_position="[center_lon, center_lat]",
        get_radius=40,
        radius_units="meters",
        filled=True,
        stroked=True,
        get_fill_color=[210, 65, 55, 25],
        get_line_color=[180, 45, 40, 170],
        line_width_min_pixels=1,
        pickable=True
    )

    map_layers.append(cluster_zone_layer)


####### Map view

view_state = pdk.ViewState(
    latitude=float(snapshot["lat"].mean()),
    longitude=float(snapshot["lon"].mean()),
    zoom=11.2,
    pitch=0,
    bearing=0
)


####### Tooltip

tooltip = {
    "html": """
        <b>Priority Zone {cluster}</b><br/>
        Scooters: {scooter_count}<br/>
        Average relocation probability:
        {avg_relocation_probability}<br/>
        Priority score:
        {priority_score}
    """,

    "style": {
        "backgroundColor": "white",
        "color": "black"
    }
}


####### Map

deck = pdk.Deck(
    layers=map_layers,
    initial_view_state=view_state,
    map_style="light",
    tooltip=tooltip
)


st.divider()
st.subheader("Relocation Map")

map_col, info_col = st.columns(
    [1.45, 0.75],
    gap="large"
)


with map_col:

    st.pydeck_chart(
        deck,
        use_container_width=True,
        height=700
    )


####### Priority zones

with info_col:

    st.markdown("### Priority Zones")

    if cluster_summary.empty:

        st.info(
            "No spatial clusters were identified "
            "with the current settings."
        )

    else:

        display_clusters = cluster_summary[
            [
                "cluster",
                "scooter_count",
                "avg_relocation_probability",
                "priority_score"
            ]
        ].copy()

        display_clusters[
            "avg_relocation_probability"
        ] = (
            display_clusters[
                "avg_relocation_probability"
            ].round(3)
        )

        display_clusters[
            "priority_score"
        ] = (
            display_clusters[
                "priority_score"
            ].round(2)
        )

        display_clusters = display_clusters.rename(
            columns={
                "cluster": "Zone",
                "scooter_count": "Scooters",
                "avg_relocation_probability":
                    "Avg. Relocation Probability",
                "priority_score": "Priority"
            }
        )

        st.dataframe(
            display_clusters,
            use_container_width=True,
            hide_index=True
        )


####### Legend

st.markdown(
    """
    **Map legend**  
    • **Grey dots** — currently idle scooters  
    • **Red dots** — scooters identified as relocation candidates  
    • **Red outlined circles** — spatial relocation zones created from nearby candidates
    """
)