"""
ThermalOS – Sensor Data Simulator
Erzeugt realistische Sensordaten basierend auf dem Versuchsaufbau:
    - 5× DS18B20 Temperatursensoren entlang der Heatpipe
    - TEG SP1848-27145 Spannung/Strom
    - ESP32 LoRa Status

Änderungen gegenüber Original:
- drift_accumulator wird auf ±max_drift [°C] begrenzt (kein unbegrenztes Anwachsen)
- max_drift ist als SensorConfig-Parameter konfigurierbar
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SensorReading:
    """Ein einzelner Messwert eines Sensors."""

    timestamp:  float
    sensor_id:  str
    value:      float
    unit:       str
    position_m: float = 0.0


@dataclass
class SensorConfig:
    """Konfigurationsparameter des Sensor-Simulators."""

    positions: List[float] = field(
        default_factory=lambda: [0.02, 0.05, 0.125, 0.20, 0.24]
    )
    labels: List[str] = field(
        default_factory=lambda: [
            "T_evap_hot", "T_evap_mid", "T_adiabatic", "T_cond_mid", "T_cond_end"
        ]
    )
    noise_std:       float = 0.3     # °C – typisches DS18B20-Rauschen
    drift_rate:      float = 0.001   # °C/Messung – Langzeitdrift je Sensor
    max_drift:       float = 2.0     # °C – maximaler Drift (Clamp-Grenze)
    anomaly_prob:    float = 0.02    # Wahrscheinlichkeit einer spontanen Anomalie
    sample_interval: float = 60.0   # Sekunden zwischen Messungen


class HeatpipeSensorSimulator:
    """
    Simuliert Sensordaten für den Digital-Twin-Mockup.

    Eigenschaften:
        - Gausssches Messrauschen (DS18B20-typisch: ±0.5 °C)
        - Langsame Sensordrift (begrenzt durch max_drift)
        - Zufällig injizierte Dryout-Anomalien
        - TEG-Spannung und -Leistung aus Temperaturdifferenz
    """

    def __init__(self, cfg: SensorConfig = SensorConfig()) -> None:
        self.cfg = cfg
        self.step: int = 0
        # BUG-FIX: drift_accumulator wird auf ±max_drift geclampet
        self.drift_accumulator: np.ndarray = np.zeros(len(cfg.positions))
        self._anomaly_active: bool = False
        self._anomaly_start: int   = 0
        self._base_profile: Optional[tuple] = None

    def set_base_profile(self, x_model: np.ndarray, T_model: np.ndarray) -> None:
        """Setzt das Basistemperaturprofil aus dem Physikmodell."""
        self._base_profile = (x_model, T_model)

    def _interpolate_temp(self, pos: float) -> float:
        """Interpoliert die Temperatur an Position pos [m] aus dem Basisprofi."""
        if self._base_profile is None:
            return 60.0
        x, T = self._base_profile
        return float(np.interp(pos, x, T))

    def generate_reading(self, inject_anomaly: bool = False) -> List[SensorReading]:
        """
        Erzeugt einen Satz Sensordaten (1 Messzyklus).

        Args:
            inject_anomaly: Erzwingt eine Anomalie in diesem Zyklus.

        Returns:
            Liste von SensorReading-Objekten (Temperatur + TEG).
        """
        ts = time.time()
        self.step += 1
        readings: List[SensorReading] = []

        # Anomalie simulieren (Dryout)
        if inject_anomaly or (np.random.random() < self.cfg.anomaly_prob):
            self._anomaly_active = True
            self._anomaly_start  = self.step

        anomaly_offset = np.zeros(len(self.cfg.positions))
        if self._anomaly_active:
            duration = self.step - self._anomaly_start
            anomaly_offset[0] = min(duration * 0.5, 15.0)   # Verdampfer steigt
            anomaly_offset[1] = min(duration * 0.3, 10.0)
            if duration > 30:
                self._anomaly_active = False

        for i, (pos, label) in enumerate(zip(self.cfg.positions, self.cfg.labels)):
            T_base  = self._interpolate_temp(pos)
            noise   = np.random.normal(0.0, self.cfg.noise_std)

            # BUG-FIX: Drift akkumulieren und auf ±max_drift clampen
            self.drift_accumulator[i] += self.cfg.drift_rate
            self.drift_accumulator[i] = float(
                np.clip(
                    self.drift_accumulator[i],
                    -self.cfg.max_drift,
                    self.cfg.max_drift,
                )
            )

            T_measured = T_base + noise + self.drift_accumulator[i] + anomaly_offset[i]
            readings.append(SensorReading(ts, label, round(T_measured, 2), "°C", pos))

        # TEG-Lesungen aus gemessener Temperaturdifferenz
        T_hot  = readings[0].value
        T_cold = readings[-1].value
        dT     = max(T_hot - T_cold, 0.0)
        V_teg  = 0.045 * dT + np.random.normal(0.0, 0.02)
        P_teg  = V_teg ** 2 / (4.0 * 2.3) * 1000.0  # mW

        readings.append(SensorReading(ts, "TEG_voltage", round(V_teg, 3), "V"))
        readings.append(SensorReading(ts, "TEG_power",   round(max(P_teg, 0.0), 2), "mW"))

        return readings

    def generate_timeseries(
        self,
        n_steps: int = 120,
        anomaly_at: Optional[int] = None,
    ) -> List[List[SensorReading]]:
        """
        Erzeugt eine Zeitreihe von n_steps Messungen.

        Args:
            n_steps:    Anzahl der Messzyklen.
            anomaly_at: Schritt-Index, bei dem eine Anomalie injiziert wird (optional).

        Returns:
            Liste von Listen mit SensorReading-Objekten.
        """
        series: List[List[SensorReading]] = []
        for i in range(n_steps):
            inject = (anomaly_at is not None and i == anomaly_at)
            series.append(self.generate_reading(inject_anomaly=inject))
        return series
