"""
ThermalOS – AI Surrogate Model
Physics-informed Neural Network Surrogate für Heatpipe-Thermalvorhersage

Bugfixes gegenüber Original:
- Kaputten absoluten Import `from thermalos_mvp.src.physics...` durch
  sys.path-gesicherten relativen Import ersetzt
- `generate_demo_comparison()` implementiert ehrlichen Vergleich zwischen
  Physics Engine und einem inline-trainierten Mini-Surrogate
"""

import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Importpfad absichern ──────────────────────────────────────────────────────
# Damit das Modul sowohl aus dem Projektroot als auch aus src/ai/ heraus
# importiert werden kann, wird der Projektroot zum sys.path hinzugefügt.
_THIS_DIR   = Path(__file__).resolve().parent          # src/ai/
_SRC_DIR    = _THIS_DIR.parent                         # src/
_ROOT_DIR   = _SRC_DIR.parent                          # Projektroot

for _p in (_ROOT_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# BUG-FIX: Import-Pfad war `thermalos_mvp.src.physics.heatpipe_model` (falsch).
# Korrekt: relativer Import aus dem Projektroot.
from src.physics.heatpipe_model import (  # noqa: E402
    HeatpipeConfig,
    HeatpipeModel,
)


class HeatpipeSurrogate:
    """
    ML-Surrogatmodell, trainiert auf Daten der Physics Engine.

    Ersetzt die physikalische Simulation durch eine Vorhersage in <1 ms.
    Das Modell kann mit `train()` inline auf synthetischen Daten trainiert
    oder mit `load()` aus einem vorberechneten Checkpoint geladen werden.

    Eingabe-Features: [Q, T_source, length, diameter, orientation_deg]
    Ausgabe-Targets:  [T_cond, R_total, Q_max]
    """

    def __init__(self) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.is_trained: bool = False
        self.metrics: dict = {}

    # ── Trainingsdaten ────────────────────────────────────────────────────────

    def generate_training_data(self, n_samples: int = 5000) -> tuple:
        """
        Erzeugt synthetische Trainingsdaten aus der Physics Engine.

        Zufällige Stichproben über den typischen Parameterraum:
            Q ∈ [5, 100] W
            T_src ∈ [40, 120] °C
            L ∈ [0.1, 0.5] m
            D ∈ [0.006, 0.02] m
            orient ∈ [-90, 90]°

        Returns:
            (X, y): Feature-Matrix [n×5] und Target-Matrix [n×3]
        """
        X: list = []
        y: list = []
        rng = np.random.default_rng(seed=0)

        for _ in range(n_samples):
            Q      = rng.uniform(5.0, 100.0)
            T_src  = rng.uniform(40.0, 120.0)
            L      = rng.uniform(0.1, 0.5)
            D      = rng.uniform(0.006, 0.02)
            orient = rng.uniform(-90.0, 90.0)

            cfg = HeatpipeConfig(length=L, diameter=D, orientation_deg=orient)
            mdl = HeatpipeModel(cfg)
            try:
                result = mdl.temperature_profile(Q, T_src)
                X.append([Q, T_src, L, D, orient])
                y.append([result["T_cond"], result["R_total"], result["Q_max"]])
            except Exception:
                continue

        return np.array(X, dtype=float), np.array(y, dtype=float)

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        n_samples: int = 5000,
    ) -> dict:
        """
        Trainiert das Surrogatmodell.

        Args:
            X:         Vorberechnetes Feature-Array (optional).
            y:         Vorberechnetes Target-Array (optional).
            n_samples: Anzahl Trainingspunkte falls X/y nicht übergeben.

        Returns:
            dict mit RMSE, R² und Trainingszeit.
        """
        if X is None or y is None:
            X, y = self.generate_training_data(n_samples)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        X_train_s = self.scaler_X.fit_transform(X_train)
        y_train_s = self.scaler_y.fit_transform(y_train)
        X_test_s  = self.scaler_X.transform(X_test)

        t0 = time.time()
        self.model.fit(X_train_s, y_train_s)
        train_time = time.time() - t0

        y_pred_s = self.model.predict(X_test_s)
        y_pred   = self.scaler_y.inverse_transform(y_pred_s)

        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))
        ss_res = np.sum((y_test - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_test - np.mean(y_test, axis=0)) ** 2, axis=0)
        r2     = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

        self.is_trained = True
        self.metrics = {
            "rmse_T_cond":   float(rmse[0]),
            "rmse_R_total":  float(rmse[1]),
            "rmse_Q_max":    float(rmse[2]),
            "r2_T_cond":     float(r2[0]),
            "r2_R_total":    float(r2[1]),
            "r2_Q_max":      float(r2[2]),
            "train_time_s":  train_time,
            "n_train":       len(X_train),
            "n_test":        len(X_test),
        }
        return self.metrics

    # ── Inferenz ──────────────────────────────────────────────────────────────

    def predict(
        self,
        Q: float,
        T_source: float,
        length: float,
        diameter: float,
        orientation: float = 0.0,
    ) -> dict:
        """
        Schnelle Vorhersage (<1 ms) für eine Heatpipe-Konfiguration.

        Args:
            Q:           Wärmeleistung [W]
            T_source:    Quelltemperatur [°C]
            length:      Gesamtlänge [m]
            diameter:    Außendurchmesser [m]
            orientation: Neigungswinkel [°]

        Returns:
            dict mit T_cond, R_total, Q_max, inference_ms
        """
        if not self.is_trained:
            raise RuntimeError(
                "Surrogatmodell noch nicht trainiert. Bitte train() aufrufen."
            )
        X   = np.array([[Q, T_source, length, diameter, orientation]], dtype=float)
        X_s = self.scaler_X.transform(X)

        t0 = time.time()
        y_s = self.model.predict(X_s)
        inference_time = time.time() - t0

        y = self.scaler_y.inverse_transform(y_s)[0]
        return {
            "T_cond":       float(y[0]),
            "R_total":      float(y[1]),
            "Q_max":        float(y[2]),
            "inference_ms": inference_time * 1000.0,
        }

    # ── Demo-Vergleich ────────────────────────────────────────────────────────

    @staticmethod
    def generate_demo_comparison(
        n_train: int = 300,
        n_eval: int = 50,
    ) -> dict:
        """
        Trainiert ein Mini-Surrogate inline und vergleicht es ehrlich
        mit der Physics Engine über einen Evaluationsdatensatz.

        Dies ist ein echter Vergleich – kein zufälliges Rauschen.
        Der Surrogate wird auf n_train Punkten trainiert und auf n_eval
        ungesehenen Punkten evaluiert.

        Args:
            n_train: Anzahl Trainingspunkte (weniger = schneller, aber
                     schlechtere Genauigkeit)
            n_eval:  Anzahl Evaluationspunkte

        Returns:
            dict mit:
                Q_values:          Wärmeleistungen der Evaluationspunkte [W]
                physics_T_cond:    Physics-Engine-Vorhersage T_cond [°C]
                surrogate_T_cond:  Surrogate-Vorhersage T_cond [°C]
                physics_R_total:   Physics-Engine R_total [K/W]
                surrogate_R_total: Surrogate R_total [K/W]
                rmse_T_cond:       RMSE auf Evaluationsmenge [°C]
                rmse_R_total:      RMSE auf Evaluationsmenge [K/W]
                r2_T_cond:         R² auf Evaluationsmenge
                r2_R_total:        R² auf Evaluationsmenge
                speedup_factor:    Inferenz-Beschleunigung vs. Physics
                train_time_s:      Trainingszeit [s]
                n_train:           Tatsächliche Anzahl Trainingspunkte
        """
        surrogate = HeatpipeSurrogate()
        # Kleineres MLP für Demo-Training
        surrogate.model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

        # ── Trainingsdaten erzeugen ──────────────────────────────────────────
        X_train_raw, y_train_raw = surrogate.generate_training_data(n_samples=n_train)

        # ── Training ────────────────────────────────────────────────────────
        t_train_start = time.time()
        X_s = surrogate.scaler_X.fit_transform(X_train_raw)
        y_s = surrogate.scaler_y.fit_transform(y_train_raw)
        surrogate.model.fit(X_s, y_s)
        train_time = time.time() - t_train_start
        surrogate.is_trained = True

        # ── Evaluationsdaten (ungesehen) ────────────────────────────────────
        rng   = np.random.default_rng(seed=99)
        Q_arr = rng.uniform(10.0, 80.0, size=n_eval)
        T_src = 80.0
        L     = 0.25
        D     = 0.01

        physics_T_cond:    list = []
        physics_R:         list = []
        surrogate_T_cond:  list = []
        surrogate_R:       list = []

        t_phys_total  = 0.0
        t_surr_total  = 0.0

        cfg = HeatpipeConfig(length=L, diameter=D)
        mdl = HeatpipeModel(cfg)

        for Q in Q_arr:
            # Physics
            t0 = time.time()
            res = mdl.temperature_profile(float(Q), T_src)
            t_phys_total += time.time() - t0
            physics_T_cond.append(res["T_cond"])
            physics_R.append(res["R_total"])

            # Surrogate
            t0 = time.time()
            pred = surrogate.predict(float(Q), T_src, L, D, 0.0)
            t_surr_total += time.time() - t0
            surrogate_T_cond.append(pred["T_cond"])
            surrogate_R.append(pred["R_total"])

        phy_T  = np.array(physics_T_cond)
        phy_R  = np.array(physics_R)
        sur_T  = np.array(surrogate_T_cond)
        sur_R  = np.array(surrogate_R)

        def _r2(actual: np.ndarray, predicted: np.ndarray) -> float:
            ss_res = np.sum((actual - predicted) ** 2)
            ss_tot = np.sum((actual - np.mean(actual)) ** 2)
            return float(1.0 - ss_res / max(ss_tot, 1e-12))

        speedup = t_phys_total / max(t_surr_total, 1e-9)

        return {
            "Q_values":           Q_arr.tolist(),
            "physics_T_cond":     phy_T.tolist(),
            "surrogate_T_cond":   sur_T.tolist(),
            "physics_R_total":    phy_R.tolist(),
            "surrogate_R_total":  sur_R.tolist(),
            "rmse_T_cond":        float(np.sqrt(np.mean((phy_T - sur_T) ** 2))),
            "rmse_R_total":       float(np.sqrt(np.mean((phy_R - sur_R) ** 2))),
            "r2_T_cond":          _r2(phy_T, sur_T),
            "r2_R_total":         _r2(phy_R, sur_R),
            "speedup_factor":     speedup,
            "train_time_s":       train_time,
            "n_train":            len(X_train_raw),
        }

    # ── Persistenz ────────────────────────────────────────────────────────────

    def save(self, path: str = "model_surrogate.pkl") -> None:
        """Speichert Modell, Scaler und Metriken als Pickle-Datei."""
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model":     self.model,
                    "scaler_X":  self.scaler_X,
                    "scaler_y":  self.scaler_y,
                    "metrics":   self.metrics,
                },
                fh,
            )

    def load(self, path: str = "model_surrogate.pkl") -> None:
        """Lädt Modell, Scaler und Metriken aus einer Pickle-Datei."""
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self.model     = data["model"]
        self.scaler_X  = data["scaler_X"]
        self.scaler_y  = data["scaler_y"]
        self.metrics   = data["metrics"]
        self.is_trained = True
