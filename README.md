# 🚗 CO₂ Emissions Prediction & Bonus-Malus Mapping (France, 2014)

This project uses official 2014 vehicle homologation data from ADEME and UTAC to train machine learning models that predict the CO₂ emissions of cars based on their technical specifications. It also connects these predictions to the official French **bonus-malus écologique** policy from 2014, quantifying the financial incentives or penalties that would apply based on emissions levels. An interactive Dash web app allows users to input car features, get real-time CO₂ predictions, and see the corresponding bonus or malus


---


## 📊 Dataset


The data was provided by **UTAC (Union Technique de l’Automobile, du motocycle et du Cycle)** to **ADEME** and includes:


- Fuel type, hybrid status, number of gears 
- Administrative horsepower 
- Gearbox type, urban & extra-urban fuel consumption 
- Vehicle mass 
- CO₂ Emissions 
- Car body type and market segment (gamme) 


**Source:** [data.gouv.fr – Emissions de CO₂ et de polluants des véhicules commercialisés en France](https://www.data.gouv.fr/fr/datasets/emissions-de-co2-et-de-polluants-des-vehicules-commercialises-en-france/) 
**License:** [Licence Ouverte / Open License (Etalab)](https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf)


---


## 🧠 Objective


1. **Preprocess and explore** the 2014 vehicle data.
2. **Train machine learning models** Lasso, Ridge, Random Forest, XGBoost) to predict CO₂ emissions.
3. **Evaluate performance** using cross-validation, R², RMSE, etc.
4. **Map emissions predictions** to the French **bonus-malus écologique** policy.
5. **Build a web app** where users can:
  - Input car specs
  - View the CO₂ prediction
  - See the corresponding bonus or malus
  - Preview their inputs and model performance summaries
---


## 🛠️ Tech Stack


- Python (Jupyter, Dash) 
- Data analysis: Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost
- Web App: Dash, HTML components, Bootstrap


---


## 🗺️ Bonus-Malus (2014)


The bonus-malus ecological policy assigns financial incentives or penalties depending on the CO₂ emissions. This dataset was constructed using information from the Le Particulier (Le Figaro) article on the 2014 French ecological bonus-malus system. The values and emission ranges are based on official government data and decrees from 2014, which defined the financial incentives and penalties tied to vehicle CO₂ emissions in grams per kilometer. This table combines both the bonus amounts for low-emission vehicles and the malus penalties for higher-emission vehicles, reflecting the official framework applied in France during that year.


Example mapping:


| CO₂ (g/km)     | Bonus / Malus (€)    | Category             |
|----------------|----------------------|----------------------|
| 0–20           | +6 300               | Bonus (Electric)     |
| 21–60          | +4 000               | Bonus (Plug-in)      |
| 91–130         | 0                    | Neutral              |
| 131–135        | –150                 | Malus                |
| 191–200        | –6 500               | Malus                |
| > 200          | –8 000               | Malus                |


The full grid is in [`data/bonus_malus_france_2014_combined.csv`](./data/bonus_malus_france_2014_combined.csv).


---




## 💻 Web App Features


Built using [Dash](https://dash.plotly.com), the app offers:


- Input form for vehicle specs 
-Real-time prediction of CO₂ emissions 
- Dynamic bonus/malus display 
- Summary table of selected inputs 
- Model performance summary after prediction


---


 ## 🚀 Run the App Locally


1. **Clone the repo**
```bash
git clone https://github.com/cynthia-francis/Prediction_CO2_Emissions_Cars
.git
cd co2-emissions-bonus-malu
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Run the web app**
```bash
python app/webapp.py
```
Then go to: http://127.0.0.1:8050




N:B: For Mac users, make sure Python can find the libomp library at runtime and then run Step 3 in case of error:
```bash
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"
```




## 📁 Folder Structure




```text
Prediction_CO2_Emissions_Cars/
├── app/
│   └── webapp.py
├── data/
│   ├── bonus_malus_france_2014_combined.csv
│   ├── carlab-annuaire-variable.xlsx
│   ├── carlab-mars-2014-complete.zip
│   ├── mars-2014-complete.csv
│
├── src/
│   ├──car_emissions_modeling.ipynb
│   ├── co2_emission_predictor.pkl
│   ├── model_Lasso.pkl
│   ├── model_RandomForest.pkl
│   └── model_results.csv
│   ├── model_Ridge.pkl
│   └── preprocessor.pkl
│   ├── xgb_model_XGBoost.json
│
├── CO2_Emissions_Prediction_Report.pdf
├── README.md
├── requirements.txt
├── .env
└── .gitignore
```
















