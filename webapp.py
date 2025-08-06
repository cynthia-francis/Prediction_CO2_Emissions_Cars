'''import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import xgboost as xgb
import dash_bootstrap_components as dbc

# ========== Feature Lists ==========
features = [
    'cod_cbr', 'hybride', 'puiss_admin_98', 'conso_urb', 'conso_exurb',
    'gearbox_type', 'num_gears', 'masse_ordma_avg', 'Carrosserie', 'gamme'
]

numeric_features = ['puiss_admin_98', 'conso_urb', 'conso_exurb', 'masse_ordma_avg', 'num_gears']
categorical_features = list(set(features) - set(numeric_features))

# ========== Load Models ==========
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


def prepare_input(df):
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in categorical_features:
        df[col] = df[col].astype(str)
    df = df[features]
    return df


def make_pipeline(preprocessor, model):
    def predict(X):
        X_clean = prepare_input(X.copy())
        X_prep = preprocessor.transform(X_clean)
        return model.predict(X_prep)
    return predict


predictors = {name: make_pipeline(preprocessor, mdl) for name, mdl in models.items()}


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


# ========== Dash App ==========
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = dbc.Container([
    html.H2("Car CO₂ Emissions Predictor & 2014 Bonus/Malus Calculator", className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Model"),
            dcc.Dropdown(id='model-select', options=[{'label': k, 'value': k} for k in models], value='Lasso')
        ]),
        dbc.Col([
            dbc.Label("Power (puiss_admin_98)"),
            dcc.Input(id='power', type='number', min=20, max=400, value=100)
        ]),
        dbc.Col([
            dbc.Label("Urban Consumption (conso_urb)"),
            dcc.Input(id='conso_urb', type='number', min=0, max=50, value=5.0)
        ]),
        dbc.Col([
            dbc.Label("Extra-urban Consumption (conso_exurb)"),
            dcc.Input(id='conso_exurb', type='number', min=0, max=50, value=4.0)
        ]),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Mass (masse_ordma_avg)"),
            dcc.Input(id='mass', type='number', min=500, max=3000, value=1200)
        ]),
        dbc.Col([
            dbc.Label("Number of Gears"),
            dcc.Dropdown(id='num_gears', options=[{'label': i, 'value': i} for i in [0, 4, 5, 6, 7, 8, 9]], value=6)
        ]),
        dbc.Col([
            dbc.Label("Fuel Type"),
            dcc.Dropdown(id='fuel_type', options=[{'label': ft, 'value': ft} for ft in
                                                  ['ES', 'GO', 'ES/GP', 'GP/ES', 'EH', 'GH', 'ES/GN', 'GN/ES', 'FE', 'GN', 'GL']], value='ES')
        ]),
        dbc.Col([
            dbc.Label("Hybrid"),
            dcc.Dropdown(id='hybride', options=[{'label': 'oui', 'value': 'oui'}, {'label': 'non', 'value': 'non'}], value='non')
        ])
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Gearbox Type"),
            dcc.Dropdown(id='gearbox_type', options=[{'label': g, 'value': g} for g in ['M', 'A', 'D', 'V', 'S']], value='M')
        ]),
        dbc.Col([
            dbc.Label("Body Type"),
            dcc.Dropdown(id='carrosserie', options=[{'label': c, 'value': c} for c in [
                'BERLINE', 'BREAK', 'COUPE', 'CABRIOLET', 'TS TERRAINS/CHEMINS',
                'COMBISPACE', 'MINISPACE', 'MONOSPACE COMPACT', 'MONOSPACE', 'MINIBUS', 'COMBISPCACE']], value='BERLINE')
        ]),
        dbc.Col([
            dbc.Label("Gamme"),
            dcc.Dropdown(id='gamme', options=[{'label': g, 'value': g} for g in
                                              ['MOY-SUPER', 'LUXE', 'MOY-INFER', 'INFERIEURE', 'SUPERIEURE', 'ECONOMIQUE']], value='MOY-SUPER')
        ])
    ], className="mb-3"),

    html.Button("Predict CO₂ & Bonus/Malus", id='predict-btn', className='btn btn-primary'),

    html.Br(), html.Br(),
    html.Div(id='prediction-output')
], fluid=True)


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('model-select', 'value'),
    State('power', 'value'),
    State('conso_urb', 'value'),
    State('conso_exurb', 'value'),
    State('mass', 'value'),
    State('num_gears', 'value'),
    State('fuel_type', 'value'),
    State('hybride', 'value'),
    State('gearbox_type', 'value'),
    State('carrosserie', 'value'),
    State('gamme', 'value'),
)
def predict(n_clicks, model_name, power, conso_urb, conso_exurb, mass,
            num_gears, fuel_type, hybride, gearbox_type, carrosserie, gamme):
    if not n_clicks:
        return ""

    input_dict = {
        'puiss_admin_98': [power],
        'conso_urb': [conso_urb],
        'conso_exurb': [conso_exurb],
        'masse_ordma_avg': [mass],
        'num_gears': [num_gears],
        'cod_cbr': [fuel_type],
        'hybride': [hybride],
        'gearbox_type': [gearbox_type],
        'Carrosserie': [carrosserie],
        'gamme': [gamme]
    }

    try:
        df = pd.DataFrame(input_dict)
        df = df[features]
        pred_func = predictors[model_name]
        co2 = pred_func(df)[0]
        bm = bonus_malus_2014(co2)
        return dbc.Alert(f"{model_name} predicts CO₂: {co2:.2f} g/km — {bm}", color="success")
    except Exception as e:
        return dbc.Alert(f"Prediction error: {e}", color="danger")


if __name__ == '__main__':
    app.run(debug=True)
'''


import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import joblib
import xgboost as xgb
import dash_bootstrap_components as dbc
import pandas as pd



model_summary_df = pd.read_csv("model_results.csv")


# ========== Feature Lists ==========
features = [
    'cod_cbr', 'hybride', 'puiss_admin_98', 'conso_urb', 'conso_exurb',
    'gearbox_type', 'num_gears', 'masse_ordma_avg', 'Carrosserie', 'gamme'
]

numeric_features = ['puiss_admin_98', 'conso_urb', 'conso_exurb', 'masse_ordma_avg', 'num_gears']
categorical_features = list(set(features) - set(numeric_features))

display_names = {
    'model-select': 'Prediction Model',
    'puiss_admin_98': 'Power (HP)',
    'conso_urb': 'Urban Consumption (L/100km)',
    'conso_exurb': 'Extra-urban Consumption (L/100km)',
    'masse_ordma_avg': 'Mass (kg)',
    'num_gears': 'Number of Gears',
    'cod_cbr': 'Fuel Type',
    'hybride': 'Hybrid',
    'gearbox_type': 'Gearbox Type',
    'Carrosserie': 'Body Type',
    'gamme': 'Gamme',
}



# ========== Load Models ==========
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


def prepare_input(df):
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in categorical_features:
        df[col] = df[col].astype(str)
    df = df[features]
    return df


def make_pipeline(preprocessor, model):
    def predict(X):
        X_clean = prepare_input(X.copy())
        X_prep = preprocessor.transform(X_clean)
        return model.predict(X_prep)
    return predict


predictors = {name: make_pipeline(preprocessor, mdl) for name, mdl in models.items()}


import pandas as pd

# Load bonus-malus data
bonus_malus_df = pd.read_csv("bonus_malus_france_2014_combined.csv")

# Define the function
def bonus_malus_2014(co2):
    for _, row in bonus_malus_df.iterrows():
        if row["co2_min"] <= co2 <= row["co2_max"]:
            note = row["note"].lower()
            amount = int(row["amount_eur"])
            if "bonus" in note:
                return "bonus", f"🌿 Bonus: {amount}€"
            elif "neutral" in note:
                return "neutral", "⚖️ No bonus or malus"
            else:
                return "malus", f"💸 Malus: {amount}€"
    return "unknown", "❓ No matching range"


# ========== Dash App ==========
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


fuel_type_labels = {
    'ES': 'Petrol (ES)', 'GO': 'Diesel (GO)', 'ES/GP': 'Petrol/LPG (ES/GP)', 
    'GP/ES': 'LPG/Petrol (GP/ES)', 'EH': 'Hybrid Electric (EH)', 
    'GH': 'Gasoline Hybrid (GH)', 'ES/GN': 'Petrol/Natural Gas (ES/GN)', 
    'GN/ES': 'Natural Gas/Petrol (GN/ES)', 'FE': 'Electric (FE)', 
    'GN': 'Natural Gas (GN)', 'GL': 'LPG (GL)'
}
hybrid_labels = {
    'oui': 'Yes',
    'non': 'No'
}

app.layout = dbc.Container([
    html.H2("🚗 Car CO₂ Emissions Predictor & 2014 Bonus/Malus", className="text-center my-2"),
    html.H5("👤 Author: Cynthia Francis", className="text-center mb-4 text-muted"),


    dbc.Row([
        dbc.Col([
            dbc.Label("Prediction Model"),
            dcc.Dropdown(id='model-select', options=[{'label': k, 'value': k} for k in models], value='Lasso')
        ], md=3),
        dbc.Col([
            dbc.Label("Power (HP)"),
            dcc.Input(id='power', type='number', min=20, max=400, value=100, className="form-control")
        ], md=3),
        dbc.Col([
            dbc.Label("Urban Consumption (L/100km)"),
            dcc.Input(id='conso_urb', type='number', min=0, max=50, value=5.0, className="form-control")
        ], md=3),
        dbc.Col([
            dbc.Label("Extra-urban Consumption (L/100km)"),
            dcc.Input(id='conso_exurb', type='number', min=0, max=50, value=4.0, className="form-control")
        ], md=3),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Mass (kg)"),
            dcc.Input(id='mass', type='number', min=500, max=3000, value=1200, className="form-control")
        ], md=3),
        dbc.Col([
            dbc.Label("Number of Gears"),
            dcc.Dropdown(id='num_gears', options=[{'label': i, 'value': i} for i in [0, 4, 5, 6, 7, 8, 9]], value=6)
        ], md=3),
        dbc.Col([
            dbc.Label("Fuel Type"),
            dcc.Dropdown(id='fuel_type',options=[{'label': label, 'value': code} for code, label in fuel_type_labels.items()],
                                value='ES')], md=3),
        dbc.Col([
            dbc.Label("Hybrid"),
            dcc.Dropdown(id='hybride', options=[{'label': 'Yes', 'value': 'oui'}, {'label': 'No', 'value': 'non'}], value='non')
        ], md=3)
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Label("Gearbox Type"),
            dcc.Dropdown(id='gearbox_type', options=[{'label': g, 'value': g} for g in ['M', 'A', 'D', 'V', 'S']], value='M')
        ], md=4),
        dbc.Col([
            dbc.Label("Body Type"),
            dcc.Dropdown(id='carrosserie', options=[{'label': c, 'value': c} for c in [
                'BERLINE', 'BREAK', 'COUPE', 'CABRIOLET', 'TS TERRAINS/CHEMINS',
                'COMBISPACE', 'MINISPACE', 'MONOSPACE COMPACT', 'MONOSPACE', 'MINIBUS', 'COMBISPCACE']], value='BERLINE')
        ], md=4),
        dbc.Col([
            dbc.Label("Gamme"),
            dcc.Dropdown(id='gamme', options=[{'label': g, 'value': g} for g in
                                              ['MOY-SUPER', 'LUXE', 'MOY-INFER', 'INFERIEURE', 'SUPERIEURE', 'ECONOMIQUE']], value='MOY-SUPER')
        ], md=4)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.Button("Predict CO₂ & Bonus/Malus", id='predict-btn', className='btn btn-primary w-100')
        ])
    ]),

    html.Br(),
    html.Div(id='prediction-output'),
    html.Br(),
    html.Div(id='input-table'),  
    html.Hr(),
    html.Div(id='summary-table'),  
    html.Hr(),
    html.Footer("© 2025 Cynthia Francis", className="text-center text-muted mb-2"),

], fluid=True)


@app.callback(
    Output('prediction-output', 'children'),
    Output('input-table', 'children'),
    Output('summary-table', 'children'), 
    Input('predict-btn', 'n_clicks'),
    State('model-select', 'value'),
    State('power', 'value'),
    State('conso_urb', 'value'),
    State('conso_exurb', 'value'),
    State('mass', 'value'),
    State('num_gears', 'value'),
    State('fuel_type', 'value'),
    State('hybride', 'value'),
    State('gearbox_type', 'value'),
    State('carrosserie', 'value'),
    State('gamme', 'value'),
)
def predict(n_clicks, model_name, power, conso_urb, conso_exurb, mass,
            num_gears, fuel_type, hybride, gearbox_type, carrosserie, gamme):

    if not n_clicks:
            return "", "", ""

    input_dict = {
        'puiss_admin_98': [power],
        'conso_urb': [conso_urb],
        'conso_exurb': [conso_exurb],
        'masse_ordma_avg': [mass],
        'num_gears': [num_gears],
        'cod_cbr': [fuel_type],
        'hybride': [hybride],
        'gearbox_type': [gearbox_type],
        'Carrosserie': [carrosserie],
        'gamme': [gamme]
    }

    try:
        df = pd.DataFrame(input_dict)
        df = df[features]
        pred_func = predictors[model_name]
        co2 = pred_func(df)[0]
        status, msg = bonus_malus_2014(co2)

        color = "success" if status == "bonus" else "warning" if status == "neutral" else "danger"

        # Format input as table
        input_table = dbc.Table.from_dataframe(
            df.T.reset_index().rename(columns={'index': 'Feature', 0: 'Value'}),
            bordered=True, hover=True, striped=True, size="sm", responsive=True
        )
        
        
        # Replace raw feature names with display names
        df_display = df.T.reset_index().rename(columns={'index': 'Feature', 0: 'Value'})
        df_display['Feature'] = df_display['Feature'].map(display_names).fillna(df_display['Feature'])
        
        # Replace feature names with display names (as before)
        df_display = df.T.reset_index().rename(columns={'index': 'Feature', 0: 'Value'})
        df_display['Feature'] = df_display['Feature'].map(display_names).fillna(df_display['Feature'])

        # Map specific feature values to their friendly names
        def map_value(row):
            feature = row['Feature']
            value = row['Value']
            if feature == 'Fuel Type':
                return fuel_type_labels.get(value, value)
            elif feature == 'Hybrid':
                return hybrid_labels.get(value, value)
            else:
                return value

        df_display['Value'] = df_display.apply(map_value, axis=1)

        # Format input table as a Dash DataTable (or dbc.Table)
        input_table = dbc.Table.from_dataframe(
            df_display,
            bordered=True, hover=True, striped=True, size="sm", responsive=True
        )
        # Format model summary as a Dash DataTable (or dbc.Table)
        summary_table = dbc.Table.from_dataframe(
            model_summary_df,
            bordered=True, hover=True, striped=True, responsive=True,
            style={"fontSize": "0.8rem", "tableLayout": "fixed"}
        )

        return (
            dbc.Alert(f"{model_name} predicts CO₂ emissions: {co2:.2f} g/km — {msg}", color=color),
            html.Div([
                html.H5("🔍 Input Summary", className="mt-4"),
                input_table
            ]),
            html.Div([
                html.H5("📊 Model Performance Summary", className="mt-4"),
                summary_table
            ])
        )

    except Exception as e:
        return dbc.Alert(f"Prediction error: {e}", color="danger"), ""

if __name__ == '__main__':
    app.run(debug=True)

