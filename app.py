import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# ======= Define feature lists EXACTLY as in training =========

numeric_features = [
    'puiss_admin_98', 'typ_boite_nb_rapp', 'hybride', 
    'conso_urb', 'conso_exurb', 'masse_ordma_min'
]
categorical_features = [
    'cod_cbr', 'lib_mrq', 'lib_mod_doss', 'lib_mod',
    'dscom', 'champ_v9', 'Carrosserie', 'gamme'
]
all_features = numeric_features + categorical_features

# ======= Load Preprocessor & Models ==========

preprocessor = joblib.load('preprocessor.pkl')
models = {}

for model_name in ['Lasso', 'Ridge', 'RandomForest', 'XGBoost']:
    if model_name == 'XGBoost':
        booster = xgb.Booster()
        booster.load_model(f'xgb_model_{model_name}.json')
        xgb_model = xgb.XGBRegressor()
        xgb_model._Booster = booster
        models[model_name] = xgb_model
    else:
        models[model_name] = joblib.load(f'model_{model_name}.pkl')

# ======= Define a prediction helper to cast types correctly ==========

def prepare_input(df):
    """Ensure correct dtypes before transforming."""
    # Cast numerics
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Cast categoricals
    for col in categorical_features:
        df[col] = df[col].astype(str)
    # Ensure column order and presence
    df = df[all_features]
    return df

def make_pipeline(preprocessor, model):
    # Returns a function that preprocesses and predicts given X
    def predict(X):
        # Prepare X
        X_clean = prepare_input(X.copy())
        # Transform and predict
        X_prep = preprocessor.transform(X_clean)
        return model.predict(X_prep)
    return predict

predictors = {name: make_pipeline(preprocessor, mdl) for name, mdl in models.items()}

# ======= Bonus/Malus Calculator ==========

def bonus_malus_2014(co2):
    if co2 <= 100:
        return "Bonus: 1000€"
    elif co2 <= 130:
        return "No bonus or malus"
    elif co2 <= 160:
        return "Malus: 200€"
    elif co2 <= 190:
        return "Malus: 1000€"
    else:
        return "Malus: 2000€"

# ======= Streamlit UI ==========

st.title("Car CO₂ Emissions Predictor & 2014 Bonus/Malus Calculator")
selected_model = st.selectbox("Choose prediction model", options=list(models.keys()))

# Input widgets (customize as needed)
power = st.number_input('Power (puiss_admin_98)', min_value=20, max_value=400, value=100)
fuel_type = st.selectbox('Fuel Type (cod_cbr)', options=['FE', 'GO', 'GP/ES', 'ES', 'GN'])
gearbox = st.selectbox('Gearbox type (typ_boite_nb_rapp)', options=[4, 5, 6, 7])

# Prepare input dataframe
input_dict = {
    'puiss_admin_98': [power],
    'typ_boite_nb_rapp': [gearbox],
    'hybride': [0],
    'conso_urb': [5],
    'conso_exurb': [4],
    'masse_ordma_min': [1200],
    'cod_cbr': [fuel_type],
    'lib_mrq': ['default'],
    'lib_mod_doss': ['default'],
    'lib_mod': ['default'],
    'dscom': ['default'],
    'champ_v9': ['default'],
    'Carrosserie': ['default'],
    'gamme': ['default']
}
input_df = pd.DataFrame(input_dict)
input_df = input_df[all_features]  # Ensure column order

# Prediction button
if st.button("Predict CO₂ & Bonus/Malus"):
    try:
        # See the data passed for sanity check (optional debug)
        # st.write("Prepared input:", input_df)
        pred_func = predictors[selected_model]
        co2_pred = pred_func(input_df)[0]
        st.success(f"Predicted CO₂ emissions by {selected_model}: {co2_pred:.2f} g/km")
        bm = bonus_malus_2014(co2_pred)
        st.info(f"Bonus/Malus status (2014 rules): {bm}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.write("Input DataFrame:", input_df)
