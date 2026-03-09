"""
ThermalOS – Anomaly Detection Module
Residuenbasierte + Isolation-Forest-Erkennung für Heatpipe-Health-Monitoring

Änderungen gegenüber Original:
- Residuenschwellwerte (residual_high, residual_low) sind jetzt konfigurierbar
  via __init__-Parameter statt fest verdrahteter Konstanten 0.3 / 0.5
- Isolation Forest wird beim ersten Aufruf mit ausreichend Daten gefittet
  und danach nicht mehr neu trainiert (Verhalten ist dokumentiert)
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class AnomalyResult:
    """Ergebnis einer Anomalie-Detektion für einen einzelnen Messpunkt."""

    timestamp:      float
    is_anomaly:     bool
    anomaly_score:  float              # Isolation-Forest-Score: −1 (Anomalie) bis 0 (normal)
    residual:       float              # Abweichung Modell vs. Messung [K/W]
    probable_cause: Optional[str] = None
    confidence:     float = 0.0       # 0.0 – 1.0


ANOMALY_CAUSES: dict = {
    "high_positive_residual": {
        "cause":    "Möglicher Dryout: Verdampfertemperatur ungewöhnlich hoch",
        "severity": "HIGH",
    },
    "high_negative_residual": {
        "cause":    "Mögliche NCG-Bildung: Kondensator blockiert",
        "severity": "MEDIUM",
    },
    "drift": {
        "cause":    "Langfristige Degradation des thermischen Widerstands",
        "severity": "LOW",
    },
    "oscillation": {
        "cause":    "Instabiler Betrieb: Temperaturoszillationen detektiert",
        "severity": "MEDIUM",
    },
}


class HeatpipeAnomalyDetector:
    """
    Zweistufige Anomalieerkennung für Heatpipes.

    Stufe 1 – Residuenanalyse:
        Vergleich gemessener vs. modellierter thermischer Widerstand R_th.
        Konfigurierbarer Schwellwert (residual_high / residual_low).

    Stufe 2 – Isolation Forest:
        Univariater Isolation Forest auf Feature-Vektoren
        [T_evap, T_cond, Q, R_th_gemessen, Residuum].
        Wird einmalig gefittet, sobald ≥ window_size/2 Datenpunkte vorliegen.

    Hinweis zur Isolation-Forest-Aktualisierung:
        Der Isolation Forest wird nach dem ersten Fit **nicht** auf neuen Daten
        neu trainiert. Das ist für Demo-Szenarien ausreichend, würde aber in
        Produktionssystemen durch periodisches Retraining ergänzt werden.
    """

    def __init__(
        self,
        window_size:    int   = 60,
        contamination:  float = 0.05,
        residual_high:  float = 0.3,   # Schwellwert für positives Residuum [K/W]
        residual_low:   float = 0.3,   # Schwellwert für negatives Residuum [K/W]
        iso_threshold:  float = -0.5,  # Isolation-Forest-Score-Grenze für Anomalie
    ) -> None:
        """
        Args:
            window_size:   Fenstergröße des Rollpuffers (Anzahl Messungen).
            contamination: Erwarteter Anomalie-Anteil für Isolation Forest.
            residual_high: Schwellwert für positives R_th-Residuum [K/W].
                           Werte > residual_high → "high_positive_residual".
            residual_low:  Schwellwert für negatives R_th-Residuum [K/W].
                           Werte < −residual_low → "high_negative_residual".
            iso_threshold: Score-Grenze: Werte < iso_threshold gelten als Anomalie.
        """
        self.window_size   = window_size
        self.contamination = contamination
        self.residual_high = residual_high
        self.residual_low  = residual_low
        self.iso_threshold = iso_threshold

        self.history            = deque(maxlen=window_size)
        self.model_predictions  = deque(maxlen=window_size)
        self.iso_forest         = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        self.baseline_R_th: Optional[float] = None
        self.is_fitted: bool = False

    def set_baseline(self, R_th_baseline: float) -> None:
        """Setzt den Baseline-Widerstand für Drift-Erkennung."""
        self.baseline_R_th = R_th_baseline

    def add_measurement(
        self,
        T_evap:       float,
        T_cond:       float,
        Q:            float,
        T_model_evap: float,
        T_model_cond: float,
        timestamp:    Optional[float] = None,
    ) -> AnomalyResult:
        """
        Verarbeitet eine neue Messung und gibt eine Anomalie-Bewertung zurück.

        Args:
            T_evap:       Gemessene Verdampfertemperatur [°C]
            T_cond:       Gemessene Kondensatortemperatur [°C]
            Q:            Aktuelle Wärmeleistung [W]
            T_model_evap: Modell-Vorhersage Verdampfertemperatur [°C]
            T_model_cond: Modell-Vorhersage Kondensatortemperatur [°C]
            timestamp:    Unix-Zeitstempel; aktuell wenn nicht angegeben.

        Returns:
            AnomalyResult mit Bewertung und Ursachendiagnose.
        """
        ts = timestamp if timestamp is not None else time.time()

        R_th_measured = (T_evap - T_cond) / max(Q, 0.1)
        R_th_model    = (T_model_evap - T_model_cond) / max(Q, 0.1)
        residual      = R_th_measured - R_th_model

        # Feature-Vektor: [T_evap, T_cond, Q, R_th_gemessen, Residuum]
        features = np.array([T_evap, T_cond, Q, R_th_measured, residual])
        self.history.append(features)

        # Zu wenige Datenpunkte → kein Urteil
        if len(self.history) < 10:
            return AnomalyResult(ts, False, 0.5, residual)

        X = np.array(list(self.history))

        # Isolation Forest einmalig fitten
        if not self.is_fitted and len(self.history) >= self.window_size // 2:
            self.iso_forest.fit(X)
            self.is_fitted = True

        if self.is_fitted:
            score      = float(self.iso_forest.score_samples(features.reshape(1, -1))[0])
            is_anomaly = score < self.iso_threshold
        else:
            score      = 0.5
            is_anomaly = abs(residual) > self.residual_high

        # Ursachenbestimmung
        cause:      Optional[str] = None
        confidence: float         = 0.0

        if is_anomaly:
            if residual > self.residual_high:
                cause      = ANOMALY_CAUSES["high_positive_residual"]["cause"]
                confidence = min(abs(residual) / (self.residual_high * 3.33), 0.99)
            elif residual < -self.residual_low:
                cause      = ANOMALY_CAUSES["high_negative_residual"]["cause"]
                confidence = min(abs(residual) / (self.residual_low * 3.33), 0.99)
            else:
                # Oszillation oder Drift prüfen
                recent_T = [h[0] for h in list(self.history)[-20:]]
                std_T    = float(np.std(recent_T))
                if std_T > 3.0:
                    cause      = ANOMALY_CAUSES["oscillation"]["cause"]
                    confidence = min(std_T / 10.0, 0.95)
                elif (
                    self.baseline_R_th is not None
                    and R_th_measured > self.baseline_R_th * 1.3
                ):
                    cause      = ANOMALY_CAUSES["drift"]["cause"]
                    confidence = min((R_th_measured / self.baseline_R_th - 1.0) * 2.0, 0.95)

        return AnomalyResult(ts, is_anomaly, score, residual, cause, confidence)

    def get_health_score(self) -> float:
        """
        Gibt einen Gesundheitsscore zurück: 0.0 (kritisch) bis 1.0 (perfekt).

        Basiert auf dem mittleren absoluten R_th-Residuum im Rollpuffer.
        """
        if len(self.history) < 5:
            return 1.0
        X = np.array(list(self.history))
        residuals      = X[:, 4]
        mean_abs_res   = float(np.mean(np.abs(residuals)))
        # Normierung: Residuum von 0.5 K/W entspricht Health-Score 0
        return float(max(0.0, min(1.0, 1.0 - mean_abs_res * 2.0)))
