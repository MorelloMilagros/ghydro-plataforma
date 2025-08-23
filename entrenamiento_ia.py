import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
import joblib

# --- 1. CONEXIÓN A LA BASE DE DATOS DE POSTGRESQL ---
# Usa las credenciales que te pasó Cami
db_user = "postgres"
db_pass = "Catalina1234"
db_host = "localhost"
db_port = "5432"
db_name = "ghydro"

print("Conectando a la base de datos PostgreSQL...")
# Creamos la cadena de conexión
engine_string = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
engine = create_engine(engine_string)
print("Conexión exitosa.")

# --- 2. CARGA Y PREPARACIÓN DE DATOS HISTÓRICOS ---
# Asumimos que la tabla se llama 'resultados_laboratorio'
print("Cargando datos históricos del laboratorio...")
query = "SELECT fecha_toma, parametro, valor FROM resultados_laboratorio ORDER BY fecha_toma ASC"
df_historico = pd.read_sql(query, engine, index_col='fecha_toma', parse_dates=True)
print(f"Se cargaron {len(df_historico)} registros.")

# Para los modelos, necesitamos los datos en columnas. Hacemos una "tabla dinámica".
df_pivot = df_historico.pivot_table(index='fecha_toma', columns='parametro', values='valor')
# Llenamos valores faltantes con el método 'forward fill' (arrastra el último valor conocido)
df_pivot.fillna(method='ffill', inplace=True)
df_pivot.dropna(inplace=True) # Eliminamos filas que aún puedan tener NAs
print("Datos preparados para el entrenamiento:")
print(df_pivot.head())


# --- 3. TAREA 1: ENTRENAR MODELO DE DETECCIÓN DE ANOMALÍAS ---
print("\nEntrenando modelo de Detección de Anomalías (Isolation Forest)...")
# Usamos Isolation Forest, un modelo robusto y eficiente para esta tarea
modelo_anomalias = IsolationForest(contamination='auto', random_state=42)
modelo_anomalias.fit(df_pivot)

# Guardamos el modelo entrenado en un archivo
joblib.dump(modelo_anomalias, 'modelo_anomalias.pkl')
print("✅ Modelo de Detección de Anomalías entrenado y guardado en 'modelo_anomalias.pkl'")


# --- 4. TAREA 2: ENTRENAR MODELO PREDICTIVO (SERIES DE TIEMPO) ---
# Vamos a predecir un solo parámetro como ejemplo. Elegimos 'ph'.
parametro_a_predecir = 'ph'
print(f"\nEntrenando modelo Predictivo para el parámetro '{parametro_a_predecir}' (SARIMA)...")

# Preparamos la serie de tiempo para el pH
serie_ph = df_pivot[parametro_a_predecir]

# Usamos un modelo SARIMA, un estándar para predicciones de series de tiempo
# (los parámetros (1,1,1) son un punto de partida común)
modelo_predictivo = sm.tsa.SARIMAX(serie_ph, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
resultado_modelo = modelo_predictivo.fit(disp=False)

# Guardamos el modelo entrenado
joblib.dump(resultado_modelo, 'modelo_predictivo_ph.pkl')
print(f"✅ Modelo Predictivo para '{parametro_a_predecir}' entrenado y guardado en 'modelo_predictivo_ph.pkl'")