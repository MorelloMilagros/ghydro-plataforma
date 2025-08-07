import os
import json
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS


load_dotenv() # carga las credenciales desde el archivo .env

# Lee las credenciales de InfluxDB del entorno
INFLUX_URL = os.getenv("INFLUXDB_URL")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")

# Configuración del Broker y Topic de MQTT
MQTT_BROKER = "broker.hivemq.com"
MQTT_TOPIC = "ghydro/datos/pozo_demo"

#  conecta a la base de datos 
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api(write_options=SYNCHRONOUS)

# lógica de conexión y manejo de mensajes MQTT

# función se activa cada vez que llega un mensaje del sensor.
def on_message(client, userdata, msg):
    print(f"Mensaje recibido: {msg.payload.decode()}")
    try:
        datos = json.loads(msg.payload.decode()) # el mensaje se espera en formato JSON.
        # Prepara los puntos de datos para InfluxDB
        point_temp = Point("mediciones_pozo").tag("pozo_id", "demo_01").field("temperatura", float(datos.get("temperatura")))
        point_ph = Point("mediciones_pozo").tag("pozo_id", "demo_01").field("ph", float(datos.get("ph")))
        # Escribe los puntos en la base de datos
        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=[point_temp, point_ph])
        print("-> Datos guardados correctamente en InfluxDB.")

    except Exception as e:
        print(f"Error al procesar el mensaje: {e}")

# Esta se activa al conectarse al broker.
def on_connect(client, userdata, flags, rc):
    print(f"Conectado al Broker MQTT (código {rc})")
    client.subscribe(MQTT_TOPIC)

# ejecución principal
client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

print(f"Intentando conectar al broker en {MQTT_BROKER}...")
client.connect(MQTT_BROKER, 1883, 60) #1888 es el puerto por defecto para MQTT, 60 es el tiempo de espera en segundos, MQTT_BROKER es el nombre del broker.

# inicializa un bucle infinito para mantener el script escuchando por mensajes.
client.loop_forever()
