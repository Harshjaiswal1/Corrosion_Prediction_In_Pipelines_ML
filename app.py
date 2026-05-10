
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── optional SHAP import (graceful fallback if not installed) ──────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Corrosion Prediction Dashboard",
    layout="wide",
)

st.title("Corrosion Prediction Dashboard")
st.write(
    """
    Predicts **Thickness Loss (mm)**, **Material Loss (%)**, **Corrosion Impact (%)**,
    and **Pipe Condition** — plus **SHAP explanations** and **Remaining Useful Life**.
    """
)

# ============================================================
# CONSTANTS
# ============================================================

# Real-world grade mapping per material type.
# Non-metal pipes in this dataset were still encoded with the steel grade
# labels during training, so we keep the same options but show a note.
MATERIAL_GRADE_MAP = {
    "Carbon Steel": [
        "ASTM A333 Grade 6",
        "ASTM A106 Grade B",
        "API 5L X52",
        "API 5L X42",
        "API 5L X65",
    ],
    "Stainless Steel": [
        "ASTM A333 Grade 6",
        "ASTM A106 Grade B",
        "API 5L X52",
        "API 5L X42",
        "API 5L X65",
    ],
    # Non-metals: model was trained with steel-grade labels even for these
    # materials, so we still expose those options but warn the user.
    "PVC":        ["ASTM A333 Grade 6", "ASTM A106 Grade B", "API 5L X52",
                   "API 5L X42", "API 5L X65"],
    "HDPE":       ["ASTM A333 Grade 6", "ASTM A106 Grade B", "API 5L X52",
                   "API 5L X42", "API 5L X65"],
    "Fiberglass": ["ASTM A333 Grade 6", "ASTM A106 Grade B", "API 5L X52",
                   "API 5L X42", "API 5L X65"],
}

NON_METAL_MATERIALS = {"PVC", "HDPE", "Fiberglass"}

# Critical threshold for RUL: percentage of original wall thickness lost
RUL_CRITICAL_LOSS_FRACTION = 0.30   # 30 % of original thickness → "end of life"

# Path to the training dataset for real correlation heatmap
TRAINING_DATA_PATH = "market_pipe_thickness_loss_calibrated_ph_flow.csv"

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_models(model_dir: str = "saved_models"):
    models = {}
    model_files = {
        "Thickness_Loss_mm":      "best_model_Thickness_Loss_mm.pkl",
        "Material_Loss_Percent":  "best_model_Material_Loss_Percent.pkl",
        "Corrosion_Impact_Percent": "best_model_Corrosion_Impact_Percent.pkl",
        "Condition":              "best_model_condition_encoded.pkl",
    }
    for key, fname in model_files.items():
        path = os.path.join(model_dir, fname)
        if os.path.exists(path):
            models[key] = joblib.load(path)
        else:
            st.error(f"Model file not found: {path}")

    encoder_path = os.path.join(model_dir, "condition_label_encoder.pkl")
    le_condition = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    if le_condition is None:
        st.error(f"Label encoder not found: {encoder_path}")

    return models, le_condition


models, le_condition = load_models()

# ============================================================
# SIDEBAR — INPUT PARAMETERS
# ============================================================
st.sidebar.header("Input Parameters")

pipe_size_mm = st.sidebar.number_input(
    "Pipe Size (mm)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0
)
thickness_mm = st.sidebar.number_input(
    "Wall Thickness (mm)", min_value=0.1, max_value=200.0, value=8.0, step=0.5
)
max_pressure_psi = st.sidebar.number_input(
    "Max Pressure (psi)", min_value=0.0, max_value=20000.0, value=150.0, step=10.0
)
temperature_c = st.sidebar.number_input(
    "Temperature (°C)", min_value=-50.0, max_value=400.0, value=40.0, step=1.0
)
time_years = st.sidebar.number_input(
    "Exposure Time (years)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
)
ph = st.sidebar.number_input(
    "pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1
)
flowrate_cmh = st.sidebar.number_input(
    "Flowrate (m³/h)", min_value=0.0, max_value=100000.0, value=50.0, step=1.0
)

st.sidebar.markdown("---")

# ── MATERIAL → dynamic GRADE options ──────────────────────────────────────────
material = st.sidebar.selectbox(
    "Material",
    options=list(MATERIAL_GRADE_MAP.keys()),
)

# Grade list updates automatically based on material selection
available_grades = MATERIAL_GRADE_MAP[material]
grade = st.sidebar.selectbox("Grade", options=available_grades)

# Warn when a non-metal pipe is paired with a steel-specific grade
if material in NON_METAL_MATERIALS:
    st.sidebar.warning(
        f"⚠️ **{material}** is a non-metal pipe. The grade labels shown are "
        "steel-specific (as used during model training). Predictions remain valid, "
        "but interpret the grade field as a material-class encoding rather than "
        "a metallurgical standard."
    )

# ============================================================
# BUILD INPUT DATAFRAME
# ============================================================
input_data = pd.DataFrame([{
    "Pipe_Size_mm":        pipe_size_mm,
    "Thickness_mm":        thickness_mm,
    "Max_Pressure_psi":    max_pressure_psi,
    "Temperature_C":       temperature_c,
    "Time_Years":          time_years,
    "pH":                  ph,
    "flowrate_cmh":        flowrate_cmh,
    "Material":            material,
    "Grade":               grade,
}])

st.write("### Current Input Data")
st.dataframe(input_data, use_container_width=True)

# ============================================================
# HELPER — safe single prediction
# ============================================================
def safe_predict(model_key: str, data: pd.DataFrame):
    if model_key not in models:
        return None
    try:
        return models[model_key].predict(data)[0]
    except Exception as e:
        st.error(f"Prediction error ({model_key}): {e}")
        return None


# ============================================================
# HELPER — decode condition label
# ============================================================
def decode_condition(raw_pred):
    if raw_pred is None or le_condition is None:
        return None
    try:
        return le_condition.inverse_transform([int(round(raw_pred))])[0]
    except Exception as e:
        st.error(f"Condition decode error: {e}")
        return None


# ============================================================
# RECOMMENDATIONS
# ============================================================
def generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label, original_thickness):
    recs = []
    if tl_pred is not None and original_thickness > 0:
        loss_fraction = tl_pred / original_thickness
        if loss_fraction < 0.10:
            recs.append(f"✅ Thickness loss is low ({tl_pred:.2f} mm): Continue routine inspection intervals.")
        elif loss_fraction < 0.25:
            recs.append(f"🟡 Moderate thickness loss ({tl_pred:.2f} mm): Schedule a detailed inspection soon.")
        else:
            recs.append(f"🔴 High thickness loss ({tl_pred:.2f} mm): Plan immediate inspection — evaluate repair or replacement.")

    if ml_pred is not None:
        if ml_pred < 10:
            recs.append(f"✅ Material loss acceptable ({ml_pred:.2f}%): Maintain current operating conditions.")
        elif ml_pred < 30:
            recs.append(f"🟡 Moderate material loss ({ml_pred:.2f}%): Review coatings, inhibitors, or cathodic protection.")
        else:
            recs.append(f"🔴 Severe material loss ({ml_pred:.2f}%): Corrective action and potential pipe replacement required.")

    if ci_pred is not None:
        if ci_pred < 20:
            recs.append(f"✅ Corrosion impact low ({ci_pred:.2f}%): Risk currently manageable.")
        elif ci_pred < 50:
            recs.append(f"🟡 Corrosion impact moderate ({ci_pred:.2f}%): Reassess inspection frequency.")
        else:
            recs.append(f"🔴 Corrosion impact high ({ci_pred:.2f}%): Prioritise for maintenance planning immediately.")

    if cond_label is not None:
        cl = str(cond_label).lower()
        if any(x in cl for x in ["severe", "critical", "poor", "bad"]):
            recs.append(f"🚨 Condition **{cond_label}**: Immediate maintenance/repair recommended.")
        elif any(x in cl for x in ["fair", "moderate"]):
            recs.append(f"🟡 Condition **{cond_label}**: Increase monitoring and plan preventive maintenance.")
        else:
            recs.append(f"✅ Condition **{cond_label}**: Maintain current inspection schedule.")

    return recs or ["⚠️ No recommendations generated — verify input values and model outputs."]


# ============================================================
# SHAP EXPLANATION
# ============================================================
def show_shap_explanation(model_key: str, data: pd.DataFrame, title: str):
    """
    Renders a SHAP waterfall plot for a single prediction.
    Works with tree-based models (RandomForest, XGBoost, LightGBM, etc.)
    and correctly handles scikit-learn Pipelines.
    """
    if not SHAP_AVAILABLE:
        st.info("Install **shap** (`pip install shap`) to enable explainability plots.")
        return
    if model_key not in models:
        return

    model = models[model_key]
    try:
        # If the model is a Pipeline, extract preprocessor and estimator
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            preprocessor = model.named_steps["preprocessor"]
            estimator_name = list(model.named_steps.keys())[-1]
            estimator = model.named_steps[estimator_name]
            
            # Transform the data
            data_transformed = preprocessor.transform(data)
            
            # Convert sparse matrices to dense arrays for SHAP
            if hasattr(data_transformed, "toarray"):
                data_transformed = data_transformed.toarray()
            
            # Extract feature names if available
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = None
                
            explainer = shap.Explainer(estimator)
            shap_values = explainer(data_transformed)
            
            # Add feature names to the explanation
            if feature_names is not None:
                shap_values.feature_names = list(feature_names)
        else:
            explainer = shap.Explainer(model)
            shap_values = explainer(data)

        # ── Waterfall — why THIS prediction? ─────────────────────────────────
        st.markdown(f"##### 🔍 {title} — Why this prediction?")
        fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        # ── Bar — global feature importance ──────────────────────────────────
        st.markdown(f"##### 📊 {title} — Feature Importance (this prediction)")
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated for {title}: {e}")


# ============================================================
# REMAINING USEFUL LIFE  (RUL)
# ============================================================
def compute_rul(input_row: pd.DataFrame, original_thickness: float) -> dict:
    """
    Iterates time from current exposure to 50 years in 0.5-year steps,
    predicts Thickness Loss at each step, and finds the year at which
    cumulative thickness loss exceeds RUL_CRITICAL_LOSS_FRACTION of the
    original wall thickness.

    Returns a dict with 'rul_years', 'critical_year', 'timeline_df'.
    """
    if "Thickness_Loss_mm" not in models:
        return None

    critical_loss_mm = original_thickness * RUL_CRITICAL_LOSS_FRACTION
    current_time     = float(input_row["Time_Years"].iloc[0])
    time_steps       = np.arange(current_time, max(current_time + 10.0, 51.0), 0.5)

    times, losses = [], []
    for t in time_steps:
        row = input_row.copy()
        row["Time_Years"] = t
        pred = safe_predict("Thickness_Loss_mm", row)
        if pred is not None:
            times.append(t)
            losses.append(float(pred))

    if not times:
        return None

    timeline_df = pd.DataFrame({"Time_Years": times, "Thickness_Loss_mm": losses})

    # Find first time thickness loss ≥ critical threshold
    critical_mask = timeline_df["Thickness_Loss_mm"] >= critical_loss_mm
    if critical_mask.any():
        critical_year = float(timeline_df.loc[critical_mask.idxmax(), "Time_Years"])
        rul_years     = max(critical_year - current_time, 0.0)
    else:
        critical_year = None
        rul_years     = None   # pipe survives past 50-year window

    return {
        "rul_years":     rul_years,
        "critical_year": critical_year,
        "timeline_df":   timeline_df,
        "critical_loss_mm": critical_loss_mm,
        "current_time":  current_time,
    }


def show_rul(input_row: pd.DataFrame, original_thickness: float, current_cond_label: str = None):
    st.markdown("---")
    st.subheader("Remaining Useful Life (RUL)")
    st.write(
        f"The RUL estimates **how many more years** before the pipe reaches a "
        f"critical state, defined as losing **{RUL_CRITICAL_LOSS_FRACTION*100:.0f}%** "
        f"of its original wall thickness "
        f"(**{original_thickness * RUL_CRITICAL_LOSS_FRACTION:.2f} mm**)."
    )

    with st.spinner("Computing RUL over time horizon…"):
        result = compute_rul(input_row, original_thickness)

    if result is None:
        st.warning("RUL could not be computed — Thickness Loss model unavailable.")
        return

    # ── Metric cards ─────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Exposure",
            f"{result['current_time']:.1f} yrs",
        )
    with col2:
        if result["rul_years"] is not None:
            color_label = "🔴" if result["rul_years"] < 5 else ("🟡" if result["rul_years"] < 15 else "🟢")
            st.metric(
                "Remaining Useful Life",
                f"{color_label} {result['rul_years']:.1f} yrs",
            )
        else:
            st.metric("Remaining Useful Life", "🟢 >50 yrs (safe)")
    with col3:
        if result["critical_year"] is not None:
            st.metric("Predicted End-of-Life", f"Year {result['critical_year']:.1f}")
        else:
            st.metric("Predicted End-of-Life", "Beyond 50 yrs")

    # ── Timeline chart ────────────────────────────────────────────────────────
    df_tl = result["timeline_df"]
    critical_loss_mm = result["critical_loss_mm"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_tl["Time_Years"], df_tl["Thickness_Loss_mm"],
            color="#1f77b4", linewidth=2, label="Predicted Thickness Loss")
    ax.axhline(critical_loss_mm, color="red", linestyle="--", linewidth=1.5,
               label=f"Critical threshold ({critical_loss_mm:.2f} mm)")

    if result["critical_year"] is not None:
        ax.axvline(result["critical_year"], color="red", linestyle=":", linewidth=1.5,
                   label=f"End-of-life (Year {result['critical_year']:.1f})")
        ax.axvline(result["current_time"], color="orange", linestyle="-.", linewidth=1.5,
                   label=f"Current time ({result['current_time']:.1f} yrs)")

    ax.fill_between(df_tl["Time_Years"], df_tl["Thickness_Loss_mm"], critical_loss_mm,
                    where=(df_tl["Thickness_Loss_mm"] >= critical_loss_mm),
                    alpha=0.15, color="red", label="Critical zone")

    ax.set_xlabel("Exposure Time (years)")
    ax.set_ylabel("Predicted Thickness Loss (mm)")
    ax.set_title("Pipe Thickness Loss Over Time — RUL Projection")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    if current_cond_label is not None and any(x in str(current_cond_label).lower() for x in ["severe", "critical", "poor", "bad"]):
        st.error(f"🚨 **Override Warning:** The Classification model flags this pipe as **{current_cond_label}**. Immediate repair is recommended, overriding the Regression RUL projection.")
    elif result["rul_years"] is not None and result["rul_years"] < 5:
        st.error(f"⚠️ **Critical Warning:** This pipe is projected to reach end-of-life "
                 f"in just **{result['rul_years']:.1f} years**. Immediate action recommended.")
    elif result["rul_years"] is not None and result["rul_years"] < 15:
        st.warning(f"🟡 **Moderate Warning:** End-of-life projected in **{result['rul_years']:.1f} years**. "
                   "Schedule preventive maintenance.")
    else:
        st.success("🟢 Pipe is projected to remain within safe operating limits for the foreseeable future.")


# ============================================================
# REAL CORRELATION HEATMAP  (uses actual training data)
# ============================================================
def show_real_correlation_heatmap():
    """
    Loads the real training CSV and displays a Pearson correlation heatmap
    for the numeric columns.  This reflects true data relationships, unlike
    the previous approach of sampling random inputs and correlating model outputs.
    """
    st.markdown("---")
    st.subheader("Feature Correlation Heatmap (Training Data)")
    st.write(
        "This heatmap shows the **actual Pearson correlations** between features "
        "in the training dataset — not model-generated samples. "
        "Stronger colours indicate stronger linear relationships."
    )

    if not os.path.exists(TRAINING_DATA_PATH):
        st.warning(
            f"Training data file `{TRAINING_DATA_PATH}` not found. "
            "Place it in the same directory as `app.py` to enable this chart."
        )
        return

    try:
        df = pd.read_csv(TRAINING_DATA_PATH)
    except Exception as e:
        st.error(f"Could not load training data: {e}")
        return

    # Keep numeric columns only; drop the 'source' flag column if present
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "source" in numeric_cols:
        numeric_cols.remove("source")

    corr = df[numeric_cols].corr()

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle → cleaner
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Pearson Correlation — Numeric Features (Training Data)", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    # Highlight the top 5 most correlated pairs (excluding self-correlation)
    corr_pairs = (
        corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
            .stack()
            .reset_index()
    )
    corr_pairs.columns = ["Feature A", "Feature B", "Correlation"]
    corr_pairs["Abs"] = corr_pairs["Correlation"].abs()
    top5 = corr_pairs.nlargest(5, "Abs").drop(columns="Abs").reset_index(drop=True)
    top5["Correlation"] = top5["Correlation"].map("{:.3f}".format)

    st.write("**Top 5 correlated feature pairs:**")
    st.table(top5)


# ============================================================
# MAIN PREDICTION FLOW
# ============================================================
if st.button("Predict Corrosion Metrics & Condition", type="primary"):
    if not models:
        st.error("No models loaded — check the `saved_models/` folder.")
        st.stop()

    st.write("## Prediction Results")

    # ── Run predictions ───────────────────────────────────────────────────────
    tl_raw    = safe_predict("Thickness_Loss_mm",      input_data)
    ml_raw    = safe_predict("Material_Loss_Percent",  input_data)
    ci_raw    = safe_predict("Corrosion_Impact_Percent", input_data)
    
    # Derive Condition from Material Loss to prevent ML model contradictions
    if ml_raw is not None:
        if ml_raw < 10:
            cond_label = "Normal / Good"
        elif ml_raw < 30:
            cond_label = "Fair / Moderate"
        else:
            cond_label = "Severe / Critical"
    else:
        cond_raw   = safe_predict("Condition", input_data)
        cond_label = decode_condition(cond_raw)

    # ── Summary table ─────────────────────────────────────────────────────────
    results = []
    if tl_raw    is not None: results.append({"Metric": "Thickness Loss (mm)",    "Value": f"{tl_raw:.3f}"})
    if ml_raw    is not None: results.append({"Metric": "Material Loss (%)",      "Value": f"{ml_raw:.3f}"})
    if ci_raw    is not None: results.append({"Metric": "Corrosion Impact (%)",   "Value": f"{ci_raw:.3f}"})
    if cond_label is not None: results.append({"Metric": "Condition",             "Value": str(cond_label)})

    if results:
        st.subheader("Prediction Summary")
        st.table(pd.DataFrame(results))
    else:
        st.warning("No prediction results available.")

    # ── Recommendations ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Maintenance & Operational Recommendations")
    for rec in generate_recommendations(tl_raw, ml_raw, ci_raw, cond_label, thickness_mm):
        st.markdown(f"- {rec}")

    # ── SHAP Explanations ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("SHAP Explainability — Why Did the Model Predict This?")
    st.write(
        "SHAP (SHapley Additive exPlanations) shows the **contribution of each input "
        "feature** to this specific prediction. Red bars push the prediction **higher**; "
        "blue bars push it **lower**. The base value is the model's average prediction."
    )

    if not SHAP_AVAILABLE:
        st.info("Run `pip install shap` and restart the app to enable these plots.")
    else:
        shap_tabs = st.tabs([
            "Thickness Loss", "Material Loss", "Corrosion Impact"
        ])
        with shap_tabs[0]:
            show_shap_explanation("Thickness_Loss_mm",      input_data, "Thickness Loss (mm)")
        with shap_tabs[1]:
            show_shap_explanation("Material_Loss_Percent",  input_data, "Material Loss (%)")
        with shap_tabs[2]:
            show_shap_explanation("Corrosion_Impact_Percent", input_data, "Corrosion Impact (%)")

    # ── Remaining Useful Life ─────────────────────────────────────────────────
    show_rul(input_data, original_thickness=thickness_mm, current_cond_label=cond_label)

    # ── Real Correlation Heatmap ──────────────────────────────────────────────
    show_real_correlation_heatmap()

    st.success("✅ Analysis complete.")

else:
    st.info(
        "Set the parameters in the sidebar, then click "
        "**'Predict Corrosion Metrics & Condition'** to run the full analysis."
    )
