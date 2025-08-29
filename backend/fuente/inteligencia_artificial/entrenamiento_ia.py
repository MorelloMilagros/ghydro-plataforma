import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import joblib
import os
from dotenv import load_dotenv

# === CONFIGURACIÓN DE CARPETAS ===
import os
from pathlib import Path

# Obtener directorio actual del script
SCRIPT_DIR = Path(__file__).parent

# Usar la carpeta que ya tienes
MODELOS_DIR = SCRIPT_DIR / "modelos_entrenados"
MODELOS_DIR.mkdir(exist_ok=True)  # Crear si no existe

print(f"Guardando modelos en: {MODELOS_DIR}")

#Conexión a la base de datos PostgreSQL
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

#print("Cargando datos históricos del laboratorio...")
#query = "SELECT fecha_toma, parametro, valor FROM resultados_laboratorio ORDER BY fecha_toma ASC"
#df_historico = pd.read_sql(query, engine, index_col='fecha_toma', parse_dates=True)
#print(f"Se cargaron {len(df_historico)} registros.")

#Cargando datos históricos de mediciones
print("Cargando datos históricos del laboratorio...")
query = "SELECT fecha, ph, conductividad, turbidez, salinidad FROM mediciones ORDER BY fecha ASC"
df_historico = pd.read_sql(query, engine, index_col='fecha', parse_dates=True)
print(f"Se cargaron {len(df_historico)} registros.")

#df_pivot = df_historico.pivot_table(index='fecha_toma', columns='parametro', values='valor')
#df_pivot.fillna(method='ffill', inplace=True)
#df_pivot.dropna(inplace=True)
#print("Datos preparados para el entrenamiento:")
#print(df_pivot.head())

df_preparado = df_historico.copy()
df_preparado.fillna(method='ffill', inplace=True)
df_preparado.dropna(inplace=True)

print("Datos preparados para el entrenamiento:")
print(df_preparado.head())

#=== ENTRENAMIENTO DEL MODELO DE DETECCIÓN DE ANOMALÍAS ===
print("\nEntrenando modelo de Detección de Anomalías (Isolation Forest)...")
modelo_anomalias = IsolationForest(contamination='auto', random_state=42)
modelo_anomalias.fit(df_preparado)

#=== EVALUACIÓN DEL MODELO DE DETECCIÓN DE ANOMALÍAS ===
predicciones = modelo_anomalias.predict(df_preparado)
scores_anomalias = modelo_anomalias.decision_function(df_preparado)

# 1. PORCENTAJE DE ANOMALÍAS DETECTADAS
n_anomalias = sum(predicciones == -1)
porcentaje_anomalias = (n_anomalias / len(predicciones)) * 100

# 2. ESTADÍSTICAS DE SCORES
scores = modelo_anomalias.decision_function(df_preparado)
print(f"Score promedio: {scores.mean():.4f}")
print(f"Desviación estándar: {scores.std():.4f}")

# 3. DISTRIBUCIÓN DE SCORES (Percentiles)
percentiles = np.percentile(scores, [5, 25, 50, 75, 95, 99])


# Guardar el modelo de detección de anomalías - CORREGIDO
# Guardar el modelo de detección de anomalías en la carpeta definida
path_modelo_anomalias = MODELOS_DIR / 'modelo_anomalias.pkl'
joblib.dump(modelo_anomalias, path_modelo_anomalias)
print(f"Modelo de Detección de Anomalías guardado en: {path_modelo_anomalias}")


# === ENTRENAMIENTO DE MODELOS PREDICTIVOS (SARIMA) ===
# SARIMA es UNIVARIADO - debemos entrenar un modelo por cada parámetro
parametros = ['ph', 'conductividad', 'turbidez', 'salinidad']  # Lista de parámetros

for parametro in parametros:
    print(f"\nEntrenando modelo Predictivo para '{parametro}' (SARIMA)...")
    
    try:
        # Obtener la serie temporal UNIVARIADA
        serie_tiempo = df_preparado[parametro]  # Esto es una Series, no DataFrame
        
        # Configuración SARIMA (p,d,q)(P,D,Q,s)
        modelo_predictivo = sm.tsa.SARIMAX(
            serie_tiempo, 
            order=(1, 1, 1), 
            seasonal_order=(1, 1, 1, 12)
        )
        resultado_modelo = modelo_predictivo.fit(disp=False)
        
        # Función auxiliar para thresholds por parámetro
        def obtener_threshold(parametro):
            """Devuelve el threshold de MAE aceptable para cada parámetro"""
            thresholds = {
                'ph': 0.5,               # MAE < 0.5 unidades de pH
                'conductividad': 200,    # MAE < 200 μS/cm
                'turbidez': 300,         # MAE < 300 NTU
                'salinidad': 0.5         # MAE < 0.5 ppm 
            }
            return thresholds.get(parametro, 1.0)  
        
        # Evaluación del modelo
        predicciones_sarima = resultado_modelo.predict(start=0, end=len(serie_tiempo)-1, dynamic=False)
        mae = mean_absolute_error(serie_tiempo, predicciones_sarima)
        mse = mean_squared_error(serie_tiempo, predicciones_sarima)
        rmse = np.sqrt(mse)
        
        print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")
        
        # Guardar el modelo predictivo si cumple criterios de calidad
        if mae < obtener_threshold(parametro):  
            nombre_archivo = MODELOS_DIR / f'modelo_predictivo_{parametro}.pkl'
            joblib.dump(resultado_modelo, nombre_archivo)
            print(f"Modelo Predictivo para '{parametro}' guardado en '{nombre_archivo}'")
        else:
            print(f"Modelo para '{parametro}' NO guardado - error muy alto (MAE: {mae:.4f})")
            
    except Exception as e:
        print(f"Error entrenando modelo para {parametro}: {str(e)}")
        continue

