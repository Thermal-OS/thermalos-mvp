"""
ThermalOS – MQTT to InfluxDB Data Bridge
Empfängt LoRa-Sensordaten via MQTT und speichert sie in InfluxDB.

Bugfixes gegenüber Original:
- Veraltete paho-mqtt v1-API: `mqtt.Client("id")` →
  `mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "id")` für paho-mqtt ≥ 2.x
- `datetime.utcnow()` (deprecated Python 3.12+) →
  `datetime.now(timezone.utc)`
- Hardcodierter InfluxDB-Token wird aus Umgebungsvariablen gelesen
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("thermalos.mqtt_bridge")

# ── Konfiguration aus Umgebungsvariablen ─────────────────────────────────────
# BUG-FIX: Keine hartcodierten Secrets; Defaults nur für lokale Entwicklung.
MQTT_BROKER   = os.environ.get("MQTT_BROKER",   "localhost")
MQTT_PORT     = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_TOPIC    = os.environ.get("MQTT_TOPIC",    "thermalos/heatpipe/#")

INFLUX_URL    = os.environ.get("INFLUX_URL",    "http://localhost:8086")
INFLUX_TOKEN  = os.environ.get("INFLUX_TOKEN",  "")          # kein Default-Token!
INFLUX_ORG    = os.environ.get("INFLUX_ORG",    "thermalos")
INFLUX_BUCKET = os.environ.get("INFLUX_BUCKET", "heatpipe_data")

# Sensor-Positionen entlang der Heatpipe
SENSOR_POSITIONS: dict = {
    0: {"name": "T_evap_hot",  "pos": 0.02},
    1: {"name": "T_evap_mid",  "pos": 0.05},
    2: {"name": "T_adiabatic", "pos": 0.125},
    3: {"name": "T_cond_mid",  "pos": 0.20},
    4: {"name": "T_cond_end",  "pos": 0.24},
}


def _build_influx_client() -> tuple:
    """
    Erzeugt InfluxDB-Client und Write-API.
    Bricht mit aussagekräftiger Fehlermeldung ab, wenn kein Token gesetzt.
    """
    if not INFLUX_TOKEN:
        log.error(
            "Kein InfluxDB-Token konfiguriert. "
            "Bitte Umgebungsvariable INFLUX_TOKEN setzen."
        )
        sys.exit(1)
    influx    = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = influx.write_api(write_options=SYNCHRONOUS)
    return influx, write_api


def on_message(client: mqtt.Client, userdata: dict, msg: mqtt.MQTTMessage) -> None:
    """
    Verarbeitet eingehende MQTT-Nachrichten vom LoRa-Gateway.

    Erwartetes JSON-Format:
        {
            "id":   "node_01",
            "boot": 42,
            "v_sc": 3.7,
            "T":    [85.2, 82.1, 70.3, 55.0, 40.1]
        }
    """
    write_api: mqtt.Client = userdata["write_api"]

    try:
        payload      = json.loads(msg.payload.decode())
        node_id      = payload.get("id",   "unknown")
        boot         = payload.get("boot", 0)
        v_sc         = payload.get("v_sc", 0.0)
        temperatures = payload.get("T",    [])

        # BUG-FIX: datetime.utcnow() ist ab Python 3.12 deprecated.
        # Korrekt: datetime.now(timezone.utc)
        ts = datetime.now(timezone.utc)

        # Temperaturwerte schreiben
        for i, T in enumerate(temperatures):
            if i in SENSOR_POSITIONS:
                s = SENSOR_POSITIONS[i]
                point = (
                    Point("temperature")
                    .tag("node_id",    node_id)
                    .tag("sensor",     s["name"])
                    .tag("position_m", str(s["pos"]))
                    .field("value",    float(T))
                    .time(ts, WritePrecision.MS)
                )
                write_api.write(bucket=INFLUX_BUCKET, record=point)

        # Superkondensator-Spannung schreiben
        point_v = (
            Point("teg_energy")
            .tag("node_id",    node_id)
            .field("v_supercap",  float(v_sc))
            .field("boot_count",  int(boot))
            .time(ts, WritePrecision.MS)
        )
        write_api.write(bucket=INFLUX_BUCKET, record=point_v)

        log.info("[%s] %s boot#%d: T=%s, V=%.2fV", ts.isoformat(), node_id, boot, temperatures, v_sc)

    except json.JSONDecodeError as exc:
        log.error("Ungültiges JSON-Payload: %s", exc)
    except Exception as exc:
        log.error("Fehler beim Verarbeiten der MQTT-Nachricht: %s", exc)


def on_connect(
    client: mqtt.Client,
    userdata: dict,
    flags: dict,
    rc: int,
) -> None:
    """Callback bei MQTT-Verbindungsaufbau."""
    if rc == 0:
        log.info("MQTT verbunden mit %s:%d", MQTT_BROKER, MQTT_PORT)
        client.subscribe(MQTT_TOPIC)
        log.info("Abonniert: %s", MQTT_TOPIC)
    else:
        log.error("MQTT-Verbindung fehlgeschlagen, RC=%d", rc)


def on_disconnect(client: mqtt.Client, userdata: dict, rc: int) -> None:
    """Callback bei MQTT-Verbindungsabbruch."""
    if rc != 0:
        log.warning("Unerwartete MQTT-Trennung (RC=%d). Reconnect wird versucht.", rc)


def main() -> None:
    """Startet die MQTT-Bridge (blockierend)."""
    influx_client, write_api = _build_influx_client()

    # BUG-FIX: paho-mqtt v2.x erfordert CallbackAPIVersion bei mqtt.Client().
    # Rückwärtskompatibel: Versuch mit neuem API, Fallback auf alten Aufruf.
    try:
        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION1,
            client_id="thermalos_bridge",
        )
    except AttributeError:
        # paho-mqtt < 2.0 hat kein CallbackAPIVersion
        client = mqtt.Client(client_id="thermalos_bridge")  # type: ignore[call-arg]

    client.user_data_set({"write_api": write_api})
    client.on_connect    = on_connect
    client.on_disconnect = on_disconnect
    client.on_message    = on_message

    log.info("Verbinde mit MQTT-Broker %s:%d …", MQTT_BROKER, MQTT_PORT)
    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
