# Librerías necesarias
import os
import json
import sys
import paho.mqtt.client as mqtt
from dotenv import load_dotenv

ruta_actual = os.path.dirname(os.path.abspath(__file__))
ruta_fuente = os.path.abspath(os.path.join(ruta_actual, '..'))
sys.path.append(ruta_fuente)

from inteligencia_artificial.analizador_ia import analizar_nuevos_datos

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

load_dotenv()

INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")

MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "ghydro/datos/pozo_demo"

influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)


def on_message(client, userdata, msg):
    print(f"Mensaje recibido: {msg.payload.decode()}")
    try:
        datos = json.loads(msg.payload.decode())

        point_temp = Point("mediciones_pozo").tag("pozo_id", "demo_01").field("temperatura", float(datos.get("temperatura")))
        point_ph = Point("mediciones_pozo").tag("pozo_id", "demo_01").field("ph", float(datos.get("ph")))
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=[point_temp, point_ph])
        print("-> Datos guardados correctamente en InfluxDB.")

        print("Iniciando análisis con IA...")
        resultados_ia = analizar_nuevos_datos(datos)
        print(f"-> Resultados del análisis de IA: {resultados_ia}")

    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

def on_connect(client, userdata, flags, rc):
    print(f"Conectado al Broker MQTT (código {rc})")
    client.subscribe(MQTT_TOPIC)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Intentando conectar al broker en {MQTT_BROKER}...")
client.connect(MQTT_BROKER, 1883, 60)

client.loop_forever()




