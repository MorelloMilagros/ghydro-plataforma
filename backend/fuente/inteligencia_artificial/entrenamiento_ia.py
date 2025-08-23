import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv("PG_USER")
db_pass = os.getenv("PG_PASS")
db_host = os.getenv("PG_HOST")
db_port = os.getenv("PG_PORT")
db_name = os.getenv("PG_DB")

print("Conectando a la base de datos PostgreSQL...")
engine_string = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(engine_string)
print("Conexión exitosa.")


print("Cargando datos históricos del laboratorio...")
query = "SELECT fecha_toma, parametro, valor FROM resultados_laboratorio ORDER BY fecha_toma ASC"
df_historico = pd.read_sql(query, engine, index_col='fecha_toma', parse_dates=True)
print(f"Se cargaron {len(df_historico)} registros.")

df_pivot = df_historico.pivot_table(index='fecha_toma', columns='parametro', values='valor')
df_pivot.fillna(method='ffill', inplace=True)
df_pivot.dropna(inplace=True)
print("Datos preparados para el entrenamiento:")
print(df_pivot.head())


print("\nEntrenando modelo de Detección de Anomalías (Isolation Forest)...")
modelo_anomalias = IsolationForest(contamination='auto', random_state=42)
modelo_anomalias.fit(df_pivot)

joblib.dump(modelo_anomalias, 'modelo_anomalias.pkl')
print("✅ Modelo de Detección de Anomalías entrenado y guardado en 'modelo_anomalias.pkl'")


parametro_a_predecir = 'ph'
print(f"\nEntrenando modelo Predictivo para el parámetro '{parametro_a_predecir}' (SARIMA)...")

serie_ph = df_pivot[parametro_a_predecir]

modelo_predictivo = sm.tsa.SARIMAX(serie_ph, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
resultado_modelo = modelo_predictivo.fit(disp=False)

joblib.dump(resultado_modelo, 'modelo_predictivo_ph.pkl')
print(f"✅ Modelo Predictivo para '{parametro_a_predecir}' entrenado y guardado en 'modelo_predictivo_ph.pkl'")