import pandas as pd
import joblib
import pathlib # <- Importamos la librería para manejar rutas de archivos


# Obtenemos la ruta de la carpeta donde se encuentra ESTE MISMO SCRIPT
ruta_actual = pathlib.Path(__file__).parent.resolve()

# Construimos la ruta completa a los archivos de los modelos
ruta_modelo_anomalias = ruta_actual / "modelos_entrenados" / "modelo_anomalias.pkl"
ruta_modelo_predictivo = ruta_actual / "modelos_entrenados" / "modelo_predictivo_ph.pkl"

print("Cargando modelos de IA entrenados...")
# Cargamos los modelos usando la ruta completa y segura
modelo_anomalias = joblib.load(ruta_modelo_anomalias)
modelo_predictivo_ph = joblib.load(ruta_modelo_predictivo)
print("Modelos cargados correctamente.")


def analizar_nuevos_datos(nuevos_datos):
    """
    Esta función recibe los datos de un sensor en tiempo real y los analiza
    con los modelos de IA cargados.
    
    :param nuevos_datos: Un diccionario con los datos, ej: {'ph': 6.5, 'temperatura': 22.3, ...}
    :return: Un diccionario con los resultados del análisis.
    """
    print(f"\nAnalizando nuevos datos: {nuevos_datos}")
    
    # --- Detección de Anomalías ---
    df_nuevos_datos = pd.DataFrame([nuevos_datos])
    columnas_modelo = modelo_anomalias.feature_names_in_
    df_nuevos_datos = df_nuevos_datos.reindex(columns=columnas_modelo)
    
    prediccion_anomalia = modelo_anomalias.predict(df_nuevos_datos)
    es_anomalia = True if prediccion_anomalia[0] == -1 else False
    
    # --- Predicción a Futuro ---
    prediccion_futura_ph = modelo_predictivo_ph.forecast(steps=1)
    
    # --- Devolver Resultados ---
    resultados = {
        "es_anomalia": es_anomalia,
        "valor_anomalia": prediccion_anomalia[0],
        "proxima_prediccion_ph": round(prediccion_futura_ph.iloc[0], 2)
    }
    
    print(f"Resultados del análisis: {resultados}")
    return resultados
