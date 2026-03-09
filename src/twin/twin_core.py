"""
ThermalOS – Digital Twin Core
Orchestriert: Sensordaten → Physikmodell → KI-Korrektur → Anomalieerkennung

Bugfixes gegenüber Original:
- Kalman-Filter: Kalibrierungsfaktor wird jetzt korrekt auf R_th_model
  angewendet (additive Korrektur statt multiplikative Innovation auf Q)
- update() akzeptiert optionalen timestamp-Parameter
"""

import time
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field


@dataclass
class TwinState:
    """Zustand des Digital Twins zu einem Zeitpunkt."""

    timestamp:          float
    T_measured:         dict   # sensor_id → °C
    T_predicted:        dict   # sensor_id → °C
    residuals:          dict   # sensor_id → ΔT [K]
    R_th_measured:      float  # gemessener thermischer Widerstand [K/W]
    R_th_model:         float  # Modell-Widerstand (unkalibriert) [K/W]
    R_th_calibrated:    float  # Modell-Widerstand nach Kalman-Kalibierung [K/W]
    health_score:       float  # 0 – 1
    anomaly_detected:   bool
    anomaly_cause:      Optional[str]
    teg_power_mW:       float
    calibration_factor: float  # Kalman-Kalibrierungsfaktor


class DigitalTwin:
    """
    Heatpipe Digital Twin – synchronisiert Physikmodell mit Sensordaten.

    Kalman-Filter-Kalibrierung:
        Der Kalibrierungsfaktor `calibration_factor` (≈ 1.0) wird so
        angepasst, dass `R_th_model * calibration_factor ≈ R_th_measured`.

        Innovation:
            ν = R_th_measured − R_th_model · calibration_factor

        Update:
            calibration_factor ← calibration_factor + K · ν / R_th_model

        Die Division durch R_th_model stellt sicher, dass der Faktor
        dimensionslos bleibt und physikalisch als Skalierungsgröße wirkt.

    Hinweis: Die ursprüngliche Implementierung wendete die Innovation auf Q
    an (`K * innovation * 0.1`) und multiplizierte damit Q bei der nächsten
    Berechnung. Das entsprach nicht der Intention, den Widerstand des Modells
    zu kalibrieren, und enthielt den arbiträren Dämpfungsfaktor 0.1.
    """

    def __init__(self, physics_model, anomaly_detector) -> None:
        self.physics              = physics_model
        self.anomaly              = anomaly_detector
        self.calibration_factor: float = 1.0
        self.state_history: List[TwinState] = []

        # Kalman-Filter Zustandsparameter
        self._kalman_P: float = 0.1    # Schätzfehler-Kovarianz
        self._kalman_Q: float = 0.001  # Prozessrauschen
        self._kalman_R: float = 0.05   # Messrauschen

    def update(
        self,
        sensor_readings: list,
        Q_input: float,
        T_source: float,
        timestamp: Optional[float] = None,
    ) -> TwinState:
        """
        Ein Update-Zyklus des Digital Twin.

        Ablauf:
            1. Physikmodell berechnen
            2. Mit Sensordaten vergleichen
            3. Kalman-Update des Kalibrierungsfaktors
            4. Anomalieerkennung

        Args:
            sensor_readings: Liste von SensorReading-Objekten
            Q_input:         Eingangswärmeleistung [W]
            T_source:        Quelltemperatur [°C]
            timestamp:       Optionaler Unix-Zeitstempel; time.time() wenn None

        Returns:
            TwinState mit vollständigem Zustand des Twins
        """
        ts = timestamp if timestamp is not None else time.time()

        # 1. Physikmodell (mit aktuellem Kalibrierungsfaktor auf Q)
        result = self.physics.temperature_profile(
            Q_input * self.calibration_factor, T_source
        )

        # 2. Sensor-Messwerte extrahieren
        T_meas: dict   = {}
        teg_power: float = 0.0
        for r in sensor_readings:
            if getattr(r, "unit", None) == "°C":
                T_meas[r.sensor_id] = r.value
            elif getattr(r, "sensor_id", None) == "TEG_power":
                teg_power = r.value

        # Modellvorhersage an Sensorpositionen interpolieren
        T_pred: dict = {}
        for r in sensor_readings:
            if getattr(r, "unit", None) == "°C" and getattr(r, "position_m", 0) > 0:
                T_pred[r.sensor_id] = float(
                    np.interp(r.position_m, result["x"], result["T"])
                )

        # 3. Residuen
        residuals = {
            k: T_meas.get(k, 0.0) - T_pred.get(k, 0.0)
            for k in T_pred
        }

        # Gemessener R_th
        T_e = T_meas.get("T_evap_hot", result["T_evap"])
        T_c = T_meas.get("T_cond_end",  result["T_cond"])
        R_th_meas  = (T_e - T_c) / max(Q_input, 0.1)
        R_th_model = result["R_total"]

        # 4. Kalman-Update für Kalibrierungsfaktor
        # BUG-FIX: Innovation bezieht sich auf den Modell-Widerstand (skaliert).
        # Korrektur wird als dimensionsloser Faktor (nicht als Q-Offset) angewendet.
        innovation      = R_th_meas - R_th_model * self.calibration_factor
        self._kalman_P += self._kalman_Q
        K               = self._kalman_P / (self._kalman_P + self._kalman_R)
        # Additive Anpassung: ν / R_th_model hält calibration_factor dimensionslos
        if abs(R_th_model) > 1e-9:
            self.calibration_factor += K * innovation / R_th_model
        self._kalman_P *= (1.0 - K)
        self.calibration_factor = float(np.clip(self.calibration_factor, 0.5, 2.0))

        R_th_calibrated = R_th_model * self.calibration_factor

        # 5. Anomalieerkennung
        anom = self.anomaly.add_measurement(
            T_evap=T_e,
            T_cond=T_c,
            Q=Q_input,
            T_model_evap=result["T_evap"],
            T_model_cond=result["T_cond"],
            timestamp=ts,
        )

        state = TwinState(
            timestamp=ts,
            T_measured=T_meas,
            T_predicted=T_pred,
            residuals=residuals,
            R_th_measured=R_th_meas,
            R_th_model=R_th_model,
            R_th_calibrated=R_th_calibrated,
            health_score=self.anomaly.get_health_score(),
            anomaly_detected=anom.is_anomaly,
            anomaly_cause=anom.probable_cause,
            teg_power_mW=teg_power,
            calibration_factor=self.calibration_factor,
        )
        self.state_history.append(state)
        return state
