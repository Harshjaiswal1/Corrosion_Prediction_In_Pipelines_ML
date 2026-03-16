
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# import matplotlib.pyplot as plt  

# # ============================================
# # 1. Page config
# # ============================================
# st.set_page_config(
#     page_title="Corrosion Prediction Dashboard",
#     layout="centered"
# )

# st.title("Corrosion Prediction Dashboard")
# st.write(
#     """
#     This app predicts **Thickness Loss (mm)**, **Material Loss (%)**, 
#     **Corrosion Impact (%)**, and **Condition** of a pipe segment based on 
#     its material, geometry, operating conditions, and exposure time.
#     """
# )

# # ============================================
# # 2. Load models & encoder
# # ============================================
# @st.cache_resource
# def load_models(model_dir: str = "saved_models"):
#     models = {}

#     # Regression models
#     model_files = {
#         "Thickness_Loss_mm": "best_model_Thickness_Loss_mm.pkl",
#         "Material_Loss_Percent": "best_model_Material_Loss_Percent.pkl",
#         "Corrosion_Impact_Percent": "best_model_Corrosion_Impact_Percent.pkl",
#         "Condition": "best_model_condition_encoded.pkl",  # classifier
#     }

#     for key, fname in model_files.items():
#         path = os.path.join(model_dir, fname)
#         if not os.path.exists(path):
#             st.error(f"Model file not found: {path}")
#         else:
#             models[key] = joblib.load(path)

#     # Label encoder for Condition
#     encoder_path = os.path.join(model_dir, "condition_label_encoder.pkl")
#     if not os.path.exists(encoder_path):
#         st.error(f"Label encoder file not found: {encoder_path}")
#         le_condition = None
#     else:
#         le_condition = joblib.load(encoder_path)

#     return models, le_condition


# models, le_condition = load_models()

# # ============================================
# # 3. Define input fields (must match training features)
# # Numerical inputs:
# #   Pipe_Size_mm, Thickness_mm, Max_Pressure_psi, Temperature_C,
# #   Time_Years, pH, flowrate_cmh
# # Categorical:
# #   Material, Grade
# # ============================================

# st.sidebar.header("Input Parameters")

# # You can adjust default values / ranges according to your dataset
# pipe_size_mm = st.sidebar.number_input(
#     "Pipe Size (mm)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0
# )

# thickness_mm = st.sidebar.number_input(
#     "Wall Thickness (mm)", min_value=0.0, max_value=200.0, value=8.0, step=0.5
# )

# max_pressure_psi = st.sidebar.number_input(
#     "Max Pressure (psi)", min_value=0.0, max_value=20000.0, value=150.0, step=10.0
# )

# temperature_c = st.sidebar.number_input(
#     "Temperature (°C)", min_value=-50.0, max_value=400.0, value=40.0, step=1.0
# )

# time_years = st.sidebar.number_input(
#     "Exposure Time (years)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
# )

# ph = st.sidebar.number_input(
#     "pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1
# )

# flowrate_cmh = st.sidebar.number_input(
#     "Flowrate (m³/h)", min_value=0.0, max_value=100000.0, value=50.0, step=1.0
# )

# st.sidebar.markdown("---")

# # Categorical features – adjust options to match your dataset values if needed
# material = st.sidebar.selectbox(
#     "Material",
#     options=[
#         "Carbon Steel",
#         "PVC",
#         "HDPE",
#         "Fiberglass",
#         "Stainless Steel",
#     ],
# )

# grade = st.sidebar.selectbox(
#     "Grade",
#     options=[
#         "ASTM A333 Grade 6",
#         "ASTM A106 Grade B",
#         "API 5L X52",
#         "API 5L X42",
#         "API 5L X65",
#     ],
# )

# # Build the input DataFrame with EXACT same column names as training
# input_data = pd.DataFrame(
#     [
#         {
#             "Pipe_Size_mm": pipe_size_mm,
#             "Thickness_mm": thickness_mm,
#             "Max_Pressure_psi": max_pressure_psi,
#             "Temperature_C": temperature_c,
#             "Time_Years": time_years,
#             "pH": ph,
#             "flowrate_cmh": flowrate_cmh,
#             "Material": material,
#             "Grade": grade,
#         }
#     ]
# )

# st.write("### Current Input Data")
# st.dataframe(input_data)

# # ============================================
# # Recommendations based on predictions
# # ============================================

# def generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label):
#     recs = []

#     # Thickness loss based recommendations
#     if tl_pred is not None:
#         if tl_pred < 0.5:
#             recs.append(
#                 f"Thickness loss is low ({tl_pred:.2f} mm): Continue with routine inspection intervals."
#             )
#         elif tl_pred < 2.0:
#             recs.append(
#                 f"Moderate thickness loss detected ({tl_pred:.2f} mm): Consider scheduling a detailed inspection in the near term."
#             )
#         else:
#             recs.append(
#                 f"High thickness loss detected ({tl_pred:.2f} mm): Plan immediate inspection and evaluate for repair or replacement."
#             )

#     # Material loss based recommendations
#     if ml_pred is not None:
#         if ml_pred < 10:
#             recs.append(
#                 f"Material loss is within acceptable range ({ml_pred:.2f}%): Keep current operating conditions under control."
#             )
#         elif ml_pred < 30:
#             recs.append(
#                 f"Material loss is moderate ({ml_pred:.2f}%): Review corrosion control measures (coating, inhibitors, etc.)."
#             )
#         else:
#             recs.append(
#                 f"Severe material loss observed ({ml_pred:.2f}%): Strongly consider corrective actions and potential pipe replacement."
#             )

#     # Corrosion impact based recommendations
#     if ci_pred is not None:
#         if ci_pred < 20:
#             recs.append(
#                 f"Overall corrosion impact is low ({ci_pred:.2f}%): Risk is currently manageable."
#             )
#         elif ci_pred < 50:
#             recs.append(
#                 f"Corrosion impact is moderate ({ci_pred:.2f}%): Reassess inspection frequency and mitigation strategies."
#             )
#         else:
#             recs.append(
#                 f"Corrosion impact is high ({ci_pred:.2f}%): Treat this as a priority for maintenance planning."
#             )

#     # Condition label based recommendations
#     if cond_label is not None:
#         cond_lower = str(cond_label).lower()
#         if any(x in cond_lower for x in ["severe", "critical", "poor", "bad"]):
#             recs.append(
#                 f"Predicted condition is **{cond_label}**: Immediate maintenance/repair is recommended."
#             )
#         elif any(x in cond_lower for x in ["fair", "moderate"]):
#             recs.append(
#                 f"Predicted condition is **{cond_label}**: Increase monitoring and plan preventive maintenance."
#             )
#         else:
#             recs.append(
#                 f"Predicted condition is **{cond_label}**: Maintain current inspection schedule and operating practices."
#             )

#     if not recs:
#         recs.append("No specific recommendations could be generated. Please verify the input values and model outputs.")

#     return recs


# # ============================================
# # Correlation matrix generation
# # ============================================

# def generate_correlation_matrix(models, base_input, n_samples: int = 200):
#     """
#     Generate a synthetic dataset by sampling across numeric input ranges,
#     predicting outputs, and computing correlation between inputs & outputs.
#     """

#     numeric_ranges = {
#         "Pipe_Size_mm": (0.0, 5000.0),
#         "Thickness_mm": (0.0, 200.0),
#         "Max_Pressure_psi": (0.0, 20000.0),
#         "Temperature_C": (-50.0, 400.0),
#         "Time_Years": (0.0, 100.0),
#         "pH": (0.0, 14.0),
#         "flowrate_cmh": (0.0, 100000.0),
#     }

#     rows = []

#     # Fix categorical values to current selection for sensitivity wrt numerics
#     material_val = base_input["Material"].iloc[0]
#     grade_val = base_input["Grade"].iloc[0]

#     for _ in range(n_samples):
#         row = {}

#         # Sample numeric inputs
#         sampled_features = {}
#         for col, (low, high) in numeric_ranges.items():
#             val = np.random.uniform(low, high)
#             sampled_features[col] = val
#             row[col] = val

#         # Build DataFrame row for prediction
#         sample_df = pd.DataFrame(
#             [
#                 {
#                     **sampled_features,
#                     "Material": material_val,
#                     "Grade": grade_val,
#                 }
#             ]
#         )

#         # Predict outputs if models exist
#         if "Thickness_Loss_mm" in models:
#             try:
#                 row["Thickness_Loss_mm"] = models["Thickness_Loss_mm"].predict(sample_df)[0]
#             except Exception:
#                 pass

#         if "Material_Loss_Percent" in models:
#             try:
#                 row["Material_Loss_Percent"] = models["Material_Loss_Percent"].predict(sample_df)[0]
#             except Exception:
#                 pass

#         if "Corrosion_Impact_Percent" in models:
#             try:
#                 row["Corrosion_Impact_Percent"] = models["Corrosion_Impact_Percent"].predict(sample_df)[0]
#             except Exception:
#                 pass

#         rows.append(row)

#     df = pd.DataFrame(rows)
#     corr = df.corr(numeric_only=True)

#     st.write("### Correlation Matrix (Inputs & Predicted Outputs)")
#     st.dataframe(corr.round(3))

#     # Heatmap
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

#     ax.set_xticks(range(len(corr.columns)))
#     ax.set_xticklabels(corr.columns, rotation=45, ha="right")
#     ax.set_yticks(range(len(corr.index)))
#     ax.set_yticklabels(corr.index)

#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label("Correlation", rotation=270, labelpad=15)

#     plt.tight_layout()
#     st.pyplot(fig)
    


# # ============================================
# # Prediction
# # ============================================
# if st.button("Predict Corrosion Metrics & Condition"):
#     if not models:
#         st.error("Models did not load correctly. Please check `saved_models/` folder.")
#     else:
#         st.write("## Prediction Results")

#         # Store predictions for recommendations
#         tl_pred = None
#         ml_pred = None
#         ci_pred = None
#         cond_label = None

#         # 1. Predict Thickness Loss (mm)
#         if "Thickness_Loss_mm" in models:
#             try:
#                 tl_model = models["Thickness_Loss_mm"]
#                 tl_pred = tl_model.predict(input_data)[0]
#             except Exception as e:
#                 st.error(f"Error while predicting Thickness Loss: {e}")
#         else:
#             st.warning("Thickness Loss model not available.")

#         # 2. Predict Material Loss (%)
#         if "Material_Loss_Percent" in models:
#             try:
#                 ml_model = models["Material_Loss_Percent"]
#                 ml_pred = ml_model.predict(input_data)[0]
#             except Exception as e:
#                 st.error(f"Error while predicting Material Loss: {e}")
#         else:
#             st.warning("Material Loss model not available.")

#         # 3. Predict Corrosion Impact (%)
#         if "Corrosion_Impact_Percent" in models:
#             try:
#                 ci_model = models["Corrosion_Impact_Percent"]
#                 ci_pred = ci_model.predict(input_data)[0]
#             except Exception as e:
#                 st.error(f"Error while predicting Corrosion Impact: {e}")
#         else:
#             st.warning("Corrosion Impact model not available.")

#         # 4. Predict Condition (Classification)
#         if "Condition" in models and le_condition is not None:
#             try:
#                 cond_model = models["Condition"]
#                 cond_encoded = cond_model.predict(input_data)[0]

#                 # Make sure it's an int index for label encoder
#                 cond_encoded_int = int(round(cond_encoded))
#                 cond_label = le_condition.inverse_transform([cond_encoded_int])[0]

#             except Exception as e:
#                 st.error(f"Error while predicting Condition: {e}")
#         else:
#             st.warning("Condition model or label encoder not available.")

#         # ============================================
#         # Show predictions in tabular format
#         # ============================================
#         results = []

#         if tl_pred is not None:
#             results.append({"Metric": "Thickness Loss (mm)", "Value": f"{tl_pred:.3f}"})
#         if ml_pred is not None:
#             results.append({"Metric": "Material Loss (%)", "Value": f"{ml_pred:.3f}"})
#         if ci_pred is not None:
#             results.append({"Metric": "Corrosion Impact (%)", "Value": f"{ci_pred:.3f}"})
#         if cond_label is not None:
#             results.append({"Metric": "Condition", "Value": str(cond_label)})

#         if results:
#             st.subheader("Prediction Summary")
#             res_df = pd.DataFrame(results)
#             st.table(res_df)
#         else:
#             st.warning("No prediction results available to display.")
        

#         # ============================================
#         # Recommendations
#         # ============================================
#         st.markdown("## Maintenance & Operational Recommendations")
#         recs = generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label)
#         for rec in recs:
#             st.markdown(f"- {rec}")

#         # ============================================
#         # Correlation Matrix & Heatmap
#         # ============================================
#         generate_correlation_matrix(models, input_data, n_samples=200)

#         st.success("✅ Prediction complete.")
# else:
#     st.info(
#         "Set the parameters in the sidebar and click **'Predict Corrosion Metrics & Condition'**."
#     )


# # ====================================================================================================
# ## ----- code section that has color background change based on condition -----
# # TIP : use light background in streamlit app to see the effect clearly
# # ====================================================================================================


# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib
# # import os
# # import matplotlib.pyplot as plt  # for correlation heatmap

# # # ============================================
# # # 1. Page config
# # # ============================================
# # st.set_page_config(
# #     page_title="Corrosion Prediction Dashboard",
# #     layout="centered"
# # )

# # st.title("Corrosion Prediction Dashboard")
# # st.write(
# #     """
# #     This app predicts **Thickness Loss (mm)**, **Material Loss (%)**, 
# #     **Corrosion Impact (%)**, and **Condition** of a pipe segment based on 
# #     its material, geometry, operating conditions, and exposure time.
# #     """
# # )

# # # ============================================
# # # 2. Load models & encoder
# # # ============================================
# # @st.cache_resource
# # def load_models(model_dir: str = "saved_models"):
# #     models = {}

# #     # Regression & classification models
# #     model_files = {
# #         "Thickness_Loss_mm": "best_model_Thickness_Loss_mm.pkl",
# #         "Material_Loss_Percent": "best_model_Material_Loss_Percent.pkl",
# #         "Corrosion_Impact_Percent": "best_model_Corrosion_Impact_Percent.pkl",
# #         "Condition": "best_model_condition_encoded.pkl",  # classifier
# #     }

# #     for key, fname in model_files.items():
# #         path = os.path.join(model_dir, fname)
# #         if not os.path.exists(path):
# #             st.error(f"Model file not found: {path}")
# #         else:
# #             models[key] = joblib.load(path)

# #     # Label encoder for Condition
# #     encoder_path = os.path.join(model_dir, "condition_label_encoder.pkl")
# #     if not os.path.exists(encoder_path):
# #         st.error(f"Label encoder file not found: {encoder_path}")
# #         le_condition = None
# #     else:
# #         le_condition = joblib.load(encoder_path)

# #     return models, le_condition


# # models, le_condition = load_models()

# # # ============================================
# # # 3. Define input fields (must match training features)
# # # ============================================

# # st.sidebar.header("Input Parameters")

# # pipe_size_mm = st.sidebar.number_input(
# #     "Pipe Size (mm)", min_value=0.0, max_value=5000.0, value=300.0, step=10.0
# # )

# # thickness_mm = st.sidebar.number_input(
# #     "Wall Thickness (mm)", min_value=0.0, max_value=200.0, value=8.0, step=0.5
# # )

# # max_pressure_psi = st.sidebar.number_input(
# #     "Max Pressure (psi)", min_value=0.0, max_value=20000.0, value=150.0, step=10.0
# # )

# # temperature_c = st.sidebar.number_input(
# #     "Temperature (°C)", min_value=-50.0, max_value=400.0, value=40.0, step=1.0
# # )

# # time_years = st.sidebar.number_input(
# #     "Exposure Time (years)", min_value=0.0, max_value=100.0, value=5.0, step=0.5
# # )

# # ph = st.sidebar.number_input(
# #     "pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1
# # )

# # flowrate_cmh = st.sidebar.number_input(
# #     "Flowrate (m³/h)", min_value=0.0, max_value=100000.0, value=50.0, step=1.0
# # )

# # st.sidebar.markdown("---")

# # # Categorical features – adjust options to match your dataset values if needed
# # material = st.sidebar.selectbox(
# #     "Material",
# #     options=[
# #         "Carbon Steel",
# #         "PVC",
# #         "HDPE",
# #         "Fiberglass",
# #         "Stainless Steel",
# #     ],
# # )

# # grade = st.sidebar.selectbox(
# #     "Grade",
# #     options=[
# #         "ASTM A333 Grade 6",
# #         "ASTM A106 Grade B",
# #         "API 5L X52",
# #         "API 5L X42",
# #         "API 5L X65",
# #     ],
# # )

# # # Build the input DataFrame with EXACT same column names as training
# # input_data = pd.DataFrame(
# #     [
# #         {
# #             "Pipe_Size_mm": pipe_size_mm,
# #             "Thickness_mm": thickness_mm,
# #             "Max_Pressure_psi": max_pressure_psi,
# #             "Temperature_C": temperature_c,
# #             "Time_Years": time_years,
# #             "pH": ph,
# #             "flowrate_cmh": flowrate_cmh,
# #             "Material": material,
# #             "Grade": grade,
# #         }
# #     ]
# # )

# # st.write("### Current Input Data")
# # st.dataframe(input_data)

# # # ============================================
# # # Helper: Set background color based on condition
# # # ============================================
# # def set_background_by_condition(cond_label: str | None):
# #     if cond_label is None:
# #         return

# #     cond_lower = cond_label.lower().strip()
# #     # Expecting: "critical", "moderate", "normal"
# #     if cond_lower == "critical":
# #         color = "#ffe5e5"  # light red
# #     elif cond_lower == "moderate":
# #         color = "#fff4e0"  # light orange
# #     elif cond_lower == "normal":
# #         color = "#e5ffe5"  # light green
# #     else:
# #         return  # unknown label, do nothing

# #     st.markdown(
# #         f"""
# #         <style>
# #         .stApp {{
# #             background-color: {color};
# #         }}
# #         </style>
# #         """,
# #         unsafe_allow_html=True,
# #     )

# # # ============================================
# # # Helper: Recommendations based on predictions
# # # ============================================
# # def generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label):
# #     recs = []

# #     # Thickness loss
# #     if tl_pred is not None:
# #         if tl_pred < 0.5:
# #             recs.append(
# #                 f"Thickness loss is low ({tl_pred:.2f} mm). Continue with routine inspection intervals."
# #             )
# #         elif tl_pred < 2.0:
# #             recs.append(
# #                 f"Moderate thickness loss detected ({tl_pred:.2f} mm). Consider scheduling a detailed inspection in the near term."
# #             )
# #         else:
# #             recs.append(
# #                 f"High thickness loss detected ({tl_pred:.2f} mm). Plan immediate inspection and evaluate for repair or replacement."
# #             )

# #     # Material loss
# #     if ml_pred is not None:
# #         if ml_pred < 10:
# #             recs.append(
# #                 f"Material loss is within acceptable range ({ml_pred:.2f}%). Keep current operating conditions under control."
# #             )
# #         elif ml_pred < 30:
# #             recs.append(
# #                 f"Material loss is moderate ({ml_pred:.2f}%). Review corrosion control measures (coating, inhibitors, etc.)."
# #             )
# #         else:
# #             recs.append(
# #                 f"Severe material loss observed ({ml_pred:.2f}%). Strongly consider corrective actions and potential pipe replacement."
# #             )

# #     # Corrosion impact
# #     if ci_pred is not None:
# #         if ci_pred < 20:
# #             recs.append(
# #                 f"Overall corrosion impact is low ({ci_pred:.2f}%). Risk is currently manageable."
# #             )
# #         elif ci_pred < 50:
# #             recs.append(
# #                 f"Corrosion impact is moderate ({ci_pred:.2f}%). Reassess inspection frequency and mitigation strategies."
# #             )
# #         else:
# #             recs.append(
# #                 f"Corrosion impact is high ({ci_pred:.2f}%). Treat this as a priority for maintenance planning."
# #             )

# #     # Condition-based (critical, moderate, normal)
# #     if cond_label is not None:
# #         cond_lower = cond_label.lower().strip()
# #         if cond_lower == "critical":
# #             recs.append(
# #                 f"Predicted condition is **{cond_label}**. Immediate maintenance/repair is strongly recommended."
# #             )
# #         elif cond_lower == "moderate":
# #             recs.append(
# #                 f"Predicted condition is **{cond_label}**. Increase monitoring and plan preventive maintenance soon."
# #             )
# #         elif cond_lower == "normal":
# #             recs.append(
# #                 f"Predicted condition is **{cond_label}**. Maintain current inspection schedule and good operating practices."
# #             )
# #         else:
# #             recs.append(
# #                 f"Predicted condition is **{cond_label}**. Please verify the condition label mapping."
# #             )

# #     if not recs:
# #         recs.append("No specific recommendations could be generated. Please verify the input values and model outputs.")

# #     return recs

# # # ============================================
# # # Helper: Correlation matrix generation
# # # ============================================
# # def generate_correlation_matrix(models, base_input, n_samples: int = 200):
# #     """
# #     Generate a synthetic dataset by sampling across numeric input ranges,
# #     predicting outputs, and computing correlation between inputs & outputs.
# #     """

# #     numeric_ranges = {
# #         "Pipe_Size_mm": (0.0, 5000.0),
# #         "Thickness_mm": (0.0, 200.0),
# #         "Max_Pressure_psi": (0.0, 20000.0),
# #         "Temperature_C": (-50.0, 400.0),
# #         "Time_Years": (0.0, 100.0),
# #         "pH": (0.0, 14.0),
# #         "flowrate_cmh": (0.0, 100000.0),
# #     }

# #     rows = []

# #     # Fix categorical values to current selection for sensitivity wrt numerics
# #     material_val = base_input["Material"].iloc[0]
# #     grade_val = base_input["Grade"].iloc[0]

# #     for _ in range(n_samples):
# #         row = {}

# #         # Sample numeric inputs
# #         sampled_features = {}
# #         for col, (low, high) in numeric_ranges.items():
# #             val = np.random.uniform(low, high)
# #             sampled_features[col] = val
# #             row[col] = val

# #         # Build DataFrame row for prediction
# #         sample_df = pd.DataFrame(
# #             [
# #                 {
# #                     **sampled_features,
# #                     "Material": material_val,
# #                     "Grade": grade_val,
# #                 }
# #             ]
# #         )

# #         # Predict outputs if models exist
# #         if "Thickness_Loss_mm" in models:
# #             try:
# #                 row["Thickness_Loss_mm"] = models["Thickness_Loss_mm"].predict(sample_df)[0]
# #             except Exception:
# #                 pass

# #         if "Material_Loss_Percent" in models:
# #             try:
# #                 row["Material_Loss_Percent"] = models["Material_Loss_Percent"].predict(sample_df)[0]
# #             except Exception:
# #                 pass

# #         if "Corrosion_Impact_Percent" in models:
# #             try:
# #                 row["Corrosion_Impact_Percent"] = models["Corrosion_Impact_Percent"].predict(sample_df)[0]
# #             except Exception:
# #                 pass

# #         rows.append(row)

# #     df = pd.DataFrame(rows)
# #     corr = df.corr(numeric_only=True)

# #     st.write("### Correlation Matrix (Inputs & Predicted Outputs)")
# #     st.dataframe(corr.round(3))

# #     # Heatmap
# #     fig, ax = plt.subplots(figsize=(8, 6))
# #     im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")

# #     ax.set_xticks(range(len(corr.columns)))
# #     ax.set_xticklabels(corr.columns, rotation=45, ha="right")
# #     ax.set_yticks(range(len(corr.index)))
# #     ax.set_yticklabels(corr.index)

# #     cbar = fig.colorbar(im, ax=ax)
# #     cbar.set_label("Correlation", rotation=270, labelpad=15)

# #     plt.tight_layout()
# #     st.pyplot(fig)

# # # ============================================
# # # 4. Prediction
# # # ============================================
# # if st.button("Predict Corrosion Metrics & Condition"):
# #     if not models:
# #         st.error("Models did not load correctly. Please check `saved_models/` folder.")
# #     else:
# #         st.write("## Prediction Results")

# #         # Store predictions
# #         tl_pred = None
# #         ml_pred = None
# #         ci_pred = None
# #         cond_label = None

# #         # 1. Predict Thickness Loss (mm)
# #         if "Thickness_Loss_mm" in models:
# #             try:
# #                 tl_model = models["Thickness_Loss_mm"]
# #                 tl_pred = tl_model.predict(input_data)[0]
# #             except Exception as e:
# #                 st.error(f"Error while predicting Thickness Loss: {e}")
# #         else:
# #             st.warning("Thickness Loss model not available.")

# #         # 2. Predict Material Loss (%)
# #         if "Material_Loss_Percent" in models:
# #             try:
# #                 ml_model = models["Material_Loss_Percent"]
# #                 ml_pred = ml_model.predict(input_data)[0]
# #             except Exception as e:
# #                 st.error(f"Error while predicting Material Loss: {e}")
# #         else:
# #             st.warning("Material Loss model not available.")

# #         # 3. Predict Corrosion Impact (%)
# #         if "Corrosion_Impact_Percent" in models:
# #             try:
# #                 ci_model = models["Corrosion_Impact_Percent"]
# #                 ci_pred = ci_model.predict(input_data)[0]
# #             except Exception as e:
# #                 st.error(f"Error while predicting Corrosion Impact: {e}")
# #         else:
# #             st.warning("Corrosion Impact model not available.")

# #         # 4. Predict Condition (Classification)
# #         if "Condition" in models and le_condition is not None:
# #             try:
# #                 cond_model = models["Condition"]
# #                 cond_encoded = cond_model.predict(input_data)[0]

# #                 # Make sure it's an int index for label encoder
# #                 cond_encoded_int = int(round(cond_encoded))
# #                 cond_label = le_condition.inverse_transform([cond_encoded_int])[0]
# #             except Exception as e:
# #                 st.error(f"Error while predicting Condition: {e}")
# #         else:
# #             st.warning("Condition model or label encoder not available.")

# #         # ============================================
# #         # 4.1 Show predictions in tabular format
# #         # ============================================
# #         result_dict = {}

# #         if tl_pred is not None:
# #             result_dict["Thickness Loss (mm)"] = f"{tl_pred:.3f}"
# #         if ml_pred is not None:
# #             result_dict["Material Loss (%)"] = f"{ml_pred:.3f}"
# #         if ci_pred is not None:
# #             result_dict["Corrosion Impact (%)"] = f"{ci_pred:.3f}"
# #         if cond_label is not None:
# #             result_dict["Condition"] = str(cond_label)

# #         if result_dict:
# #             res_df = pd.DataFrame(
# #                 {
# #                     "Metric": list(result_dict.keys()),
# #                     "Value": list(result_dict.values()),
# #                 }
# #             )
# #             st.subheader("Prediction Summary (Tabular)")
# #             st.table(res_df)
# #         else:
# #             st.warning("No prediction results available to display.")

# #         # ============================================
# #         # 4.2 Change background color based on condition
# #         # ============================================
# #         set_background_by_condition(cond_label)

# #         # ============================================
# #         # 5. Recommendations
# #         # ============================================
# #         st.markdown("## Maintenance & Operational Recommendations")
# #         recs = generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label)
# #         for rec in recs:
# #             st.markdown(f"- {rec}")

# #         # ============================================
# #         # 6. Correlation Matrix & Heatmap
# #         # ============================================
# #         generate_correlation_matrix(models, input_data, n_samples=200)

# #         st.success("✅ Prediction complete.")
# # else:
# #     st.info(
# #         "Set the parameters in the sidebar and click **'Predict Corrosion Metrics & Condition'**."
# #     )

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

st.title("🔧 Corrosion Prediction Dashboard")
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
st.sidebar.header("⚙️ Input Parameters")

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

st.write("### 📋 Current Input Data")
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
def generate_recommendations(tl_pred, ml_pred, ci_pred, cond_label):
    recs = []
    if tl_pred is not None:
        if tl_pred < 0.5:
            recs.append(f"✅ Thickness loss is low ({tl_pred:.2f} mm): Continue routine inspection intervals.")
        elif tl_pred < 2.0:
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
    """
    if not SHAP_AVAILABLE:
        st.info("Install **shap** (`pip install shap`) to enable explainability plots.")
        return
    if model_key not in models:
        return

    model = models[model_key]
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(data)

        # ── Waterfall — why THIS prediction? ─────────────────────────────────
        st.markdown(f"##### 🔍 {title} — Why this prediction?")
        fig_wf, ax_wf = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

        # ── Bar — global feature importance ──────────────────────────────────
        st.markdown(f"##### 📊 {title} — Feature Importance (this prediction)")
        shap.plots.bar(shap_values, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)

    except Exception as e:
        st.warning(f"SHAP plot could not be generated for {title}: {e}")


# ============================================================
# REMAINING USEFUL LIFE  (RUL)
# ============================================================
def compute_rul(input_row: pd.DataFrame, original_thickness: float) -> dict:
    """
    Iterates time from current exposure to 100 years in 0.5-year steps,
    predicts Thickness Loss at each step, and finds the year at which
    cumulative thickness loss exceeds RUL_CRITICAL_LOSS_FRACTION of the
    original wall thickness.

    Returns a dict with 'rul_years', 'critical_year', 'timeline_df'.
    """
    if "Thickness_Loss_mm" not in models:
        return None

    critical_loss_mm = original_thickness * RUL_CRITICAL_LOSS_FRACTION
    current_time     = float(input_row["Time_Years"].iloc[0])
    time_steps       = np.arange(current_time, 101.0, 0.5)

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
        rul_years     = None   # pipe survives past 100-year window

    return {
        "rul_years":     rul_years,
        "critical_year": critical_year,
        "timeline_df":   timeline_df,
        "critical_loss_mm": critical_loss_mm,
        "current_time":  current_time,
    }


def show_rul(input_row: pd.DataFrame, original_thickness: float):
    st.markdown("---")
    st.subheader("⏳ Remaining Useful Life (RUL)")
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
            st.metric("Remaining Useful Life", "🟢 >100 yrs (safe)")
    with col3:
        if result["critical_year"] is not None:
            st.metric("Predicted End-of-Life", f"Year {result['critical_year']:.1f}")
        else:
            st.metric("Predicted End-of-Life", "Beyond 100 yrs")

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

    if result["rul_years"] is not None and result["rul_years"] < 5:
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
    st.subheader("📊 Feature Correlation Heatmap (Training Data)")
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
if st.button("🔍 Predict Corrosion Metrics & Condition", type="primary"):
    if not models:
        st.error("No models loaded — check the `saved_models/` folder.")
        st.stop()

    st.write("## 📈 Prediction Results")

    # ── Run predictions ───────────────────────────────────────────────────────
    tl_raw    = safe_predict("Thickness_Loss_mm",      input_data)
    ml_raw    = safe_predict("Material_Loss_Percent",  input_data)
    ci_raw    = safe_predict("Corrosion_Impact_Percent", input_data)
    cond_raw  = safe_predict("Condition",              input_data)
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
    st.markdown("## 🛠️ Maintenance & Operational Recommendations")
    for rec in generate_recommendations(tl_raw, ml_raw, ci_raw, cond_label):
        st.markdown(f"- {rec}")

    # ── SHAP Explanations ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🧠 SHAP Explainability — Why Did the Model Predict This?")
    st.write(
        "SHAP (SHapley Additive exPlanations) shows the **contribution of each input "
        "feature** to this specific prediction. Red bars push the prediction **higher**; "
        "blue bars push it **lower**. The base value is the model's average prediction."
    )

    if not SHAP_AVAILABLE:
        st.info("👉 Run `pip install shap` and restart the app to enable these plots.")
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
    show_rul(input_data, original_thickness=thickness_mm)

    # ── Real Correlation Heatmap ──────────────────────────────────────────────
    show_real_correlation_heatmap()

    st.success("✅ Analysis complete.")

else:
    st.info(
        "⬅️ Set the parameters in the sidebar, then click "
        "**'Predict Corrosion Metrics & Condition'** to run the full analysis."
    )
