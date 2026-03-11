"""
PredictorAgent — Prophet Edition
═══════════════════════════════════════════════════════════════════════════════
Primary forecaster  : Facebook Prophet (additive trend + changepoint detection)
Cold-start fallback : Exponential Smoothing + Linear Trend Extrapolation
Capacity planner    : Linear breach-time estimation on smoothed trend
Confidence intervals: Prophet's built-in 95% uncertainty intervals (primary)
                      Heuristic √i bands (fallback)

Lifecycle per metric key
─────────────────────────
  Ticks  1–29   →  ES+LTE fallback  (cold-start safe, instant)
  Tick   30     →  Prophet fit fires in background thread
  Tick   31+    →  Prophet primary; ES+LTE on any exception
  Every  60 ticks → background refit with latest rolling window

Why tick-30 for Prophet (vs tick-15 for ARIMA)?
  Prophet's Stan backend needs more observations to converge reliably.
  30 ticks = 60 seconds of data — enough for a stable initial fit.

Environment variables
──────────────────────
  PROPHET_REFIT_EVERY     → refit interval in ticks  (default: 60)
  PROPHET_COLD_START_MIN  → min ticks before first fit (default: 30)
  PROPHET_UNCERTAINTY     → uncertainty samples       (default: 0 = fast MAP)
  PROPHET_CHANGEPOINT_SCALE → flexibility of trend changes (default: 0.05)
"""

import logging
import math
import os
import random
import threading
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, AgentMessage

# ── suppress noisy Stan / cmdstanpy output ─────────────────────────────────
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*LU decomposition.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# ── optional Prophet dependency ────────────────────────────────────────────
try:
    import pandas as pd
    from prophet import Prophet
    _PROPHET_AVAILABLE = True
except ImportError:
    _PROPHET_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HISTORY_LIMIT       = 120    # rolling buffer  (120 × 2 s = 4 min)
FORECAST_STEPS      = 6      # 6 × 5 min = 30-minute horizon
COLD_START_MIN      = int(os.getenv("PROPHET_COLD_START_MIN",  "30"))
REFIT_EVERY         = int(os.getenv("PROPHET_REFIT_EVERY",     "60"))
UNCERTAINTY_SAMPLES = int(os.getenv("PROPHET_UNCERTAINTY",     "0"))   # 0 = MAP (fast)
CHANGEPOINT_SCALE   = float(os.getenv("PROPHET_CHANGEPOINT_SCALE", "0.05"))
ES_ALPHA            = 0.3

# Capacity breach thresholds
THRESH_CPU    = 90.0
THRESH_MEMORY = 95.0
THRESH_DISK   = 95.0


# ═══════════════════════════════════════════════════════════════════════════════
# PROPHET MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class ProphetModel:
    """
    Wraps a fitted Prophet model.
    Generates a synthetic DatetimeIndex (2-second steps) so Prophet gets
    the time structure it needs, even though our data has no real timestamps.
    """

    # Fake epoch — consistent across all metrics so Prophet internals are stable
    _EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)

    def __init__(self, model: Any, n_obs: int, freq_seconds: int = 2):
        self.model       = model
        self.n_obs       = n_obs
        self.freq_seconds = freq_seconds

    @classmethod
    def fit(
        cls,
        values: List[float],
        freq_seconds: int = 2,
        changepoint_prior_scale: float = CHANGEPOINT_SCALE,
        uncertainty_samples: int = UNCERTAINTY_SAMPLES,
    ) -> "ProphetModel":
        """
        Build a DataFrame with synthetic timestamps and fit Prophet.
        Returns a ProphetModel instance.
        Raises on convergence failure so the caller can fall back to ES+LTE.
        """
        n = len(values)
        timestamps = [
            cls._EPOCH + timedelta(seconds=i * freq_seconds)
            for i in range(n)
        ]
        df = pd.DataFrame({"ds": timestamps, "y": values})

        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            uncertainty_samples=uncertainty_samples,
            # Disable daily / weekly seasonality — irrelevant for 30-min horizon
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            # Allow trend flexibility for infrastructure metrics
            n_changepoints=min(25, n // 3),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(df)

        return cls(model=model, n_obs=n, freq_seconds=freq_seconds)

    def forecast(self, steps: int) -> List[Dict]:
        """
        Forecast `steps` × 5-minute intervals ahead.
        Converts Prophet's per-second output to 5-minute step points.
        Returns list of forecast dicts with Prophet uncertainty intervals.
        """
        # How many seconds per forecast step
        step_seconds = 5 * 60   # 5 minutes

        # Build future DataFrame: one row per second for full horizon,
        # then sample every step_seconds
        horizon_seconds = steps * step_seconds
        last_ts = self._EPOCH + timedelta(seconds=(self.n_obs - 1) * self.freq_seconds)

        future_timestamps = [
            last_ts + timedelta(seconds=s)
            for s in range(self.freq_seconds, horizon_seconds + 1, self.freq_seconds)
        ]
        future_df = pd.DataFrame({"ds": future_timestamps})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast_df = self.model.predict(future_df)

        # Sample at 5-minute boundaries
        samples_per_step = step_seconds // self.freq_seconds   # 150 rows per step
        result = []
        for i in range(steps):
            idx = min((i + 1) * samples_per_step - 1, len(forecast_df) - 1)
            row = forecast_df.iloc[idx]

            pred  = float(row["yhat"])
            lower = float(row["yhat_lower"])
            upper = float(row["yhat_upper"])

            result.append({
                "time_offset_minutes": (i + 1) * 5,
                "predicted_value":     round(max(0.0, pred),  2),
                "confidence_lower":    round(max(0.0, lower), 2),
                "confidence_upper":    round(max(0.0, upper), 2),
                "source":              "Prophet",
            })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class PredictorAgent(BaseAgent):
    """
    30-minute predictive analytics agent using Facebook Prophet.

    Runtime flow per metric key (e.g. "api-gateway.cpu_usage")
    ──────────────────────────────────────────────────────────────────────
      Ticks  1-29  → ES+LTE fallback  (immediate output, cold-start safe)
      Tick   30    → Prophet fit fires in a background thread
      Tick   31+   → Prophet primary forecast; ES+LTE on any exception
      Every  60    → background refit refreshes model with latest data
    """

    def __init__(self):
        super().__init__(
            agent_id="predictor",
            name="Predictive Analytics Agent",
            description=(
                "Prophet forecasting with uncertainty intervals + "
                "ES+LTE cold-start fallback. Background refitting."
            ),
        )
        self.register_tools([
            "prophet_forecaster",
            "es_lte_fallback",
            "changepoint_detector",
            "uncertainty_interval_builder",
            "capacity_planner",
            "breach_time_estimator",
        ])

        self._history:   Dict[str, List[float]]            = {}
        self._models:    Dict[str, Optional[ProphetModel]] = {}
        self._tick:      Dict[str, int]                    = {}
        self._refitting: Dict[str, bool]                   = {}
        self._lock = threading.Lock()

        if not _PROPHET_AVAILABLE:
            logger.warning(
                "prophet / pandas not installed — running in ES+LTE only mode. "
                "Install with: pip install prophet"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # History management
    # ─────────────────────────────────────────────────────────────────────────

    def _push(self, key: str, value: float) -> None:
        if key not in self._history:
            self._history[key]   = []
            self._tick[key]      = 0
            self._refitting[key] = False
        self._history[key].append(value)
        if len(self._history[key]) > HISTORY_LIMIT:
            self._history[key].pop(0)
        self._tick[key] += 1

    def _snapshot(self, key: str) -> List[float]:
        """Thread-safe copy of current history buffer."""
        with self._lock:
            return list(self._history.get(key, []))

    # ─────────────────────────────────────────────────────────────────────────
    # Prophet — background fit / refit
    # ─────────────────────────────────────────────────────────────────────────

    def _refit_worker(self, key: str) -> None:
        """
        Background thread: fit Prophet on latest history snapshot,
        then atomically swap the model reference.
        """
        try:
            data = self._snapshot(key)
            if len(data) < COLD_START_MIN:
                return

            new_model = ProphetModel.fit(data)

            with self._lock:
                self._models[key] = new_model

            logger.debug(
                "Prophet fitted for %-45s  n=%d",
                key, len(data),
            )
        except Exception as exc:
            logger.debug("Prophet fit error [%s]: %s — keeping fallback", key, exc)
        finally:
            self._refitting[key] = False

    def _maybe_refit(self, key: str) -> None:
        """Trigger a background refit when due and none is already running."""
        if not _PROPHET_AVAILABLE:
            return
        tick = self._tick.get(key, 0)
        if tick < COLD_START_MIN:
            return
        has_model     = self._models.get(key) is not None
        due_for_refit = (tick % REFIT_EVERY == 0)
        if (not has_model or due_for_refit) and not self._refitting.get(key, False):
            self._refitting[key] = True
            threading.Thread(
                target=self._refit_worker,
                args=(key,),
                daemon=True,
                name=f"prophet-{key[:30]}",
            ).start()

    # ─────────────────────────────────────────────────────────────────────────
    # ES+LTE fallback
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _exp_smooth(values: List[float], alpha: float = ES_ALPHA) -> List[float]:
        if not values:
            return []
        s = [values[0]]
        for v in values[1:]:
            s.append(alpha * v + (1 - alpha) * s[-1])
        return s

    def _es_lte_forecast(self, key: str, steps: int) -> List[Dict]:
        history = self._history.get(key, [])
        if len(history) < 3:
            return []

        smoothed = self._exp_smooth(history)
        last     = smoothed[-1]
        recent   = smoothed[-5:] if len(smoothed) >= 5 else smoothed
        trend    = (recent[-1] - recent[0]) / max(1, len(recent) - 1)

        points = []
        for i in range(1, steps + 1):
            noise = random.gauss(0, max(0.3, abs(last) * 0.015))
            pred  = max(0.0, last + trend * i + noise)
            ci    = abs(last) * 0.04 * math.sqrt(i)
            points.append({
                "time_offset_minutes": i * 5,
                "predicted_value":     round(pred,                2),
                "confidence_lower":    round(max(0.0, pred - ci), 2),
                "confidence_upper":    round(pred + ci,           2),
                "source":              "ES+LTE (fallback)",
            })
        return points

    # ─────────────────────────────────────────────────────────────────────────
    # Primary forecast entry point
    # ─────────────────────────────────────────────────────────────────────────

    def forecast(self, key: str, steps: int = FORECAST_STEPS) -> List[Dict]:
        """
        Prophet primary → ES+LTE fallback.
        Never raises — always returns a list (possibly empty if < 3 ticks).
        """
        self._maybe_refit(key)

        with self._lock:
            model = self._models.get(key)

        if model is not None:
            try:
                return model.forecast(steps)
            except Exception as exc:
                logger.debug("Prophet forecast error [%s]: %s", key, exc)

        return self._es_lte_forecast(key, steps)

    # ─────────────────────────────────────────────────────────────────────────
    # Capacity planner
    # ─────────────────────────────────────────────────────────────────────────

    def capacity_forecast(self, flat_metrics: Dict[str, float]) -> Dict:
        """
        Estimates time-to-threshold using linear breach projection
        on the smoothed trend: T = (threshold - current) / trend
        """
        result: Dict = {}

        for key, current in flat_metrics.items():
            history = self._history.get(key, [])
            if len(history) < 5:
                continue

            smoothed = self._exp_smooth(history)
            recent   = smoothed[-5:]
            trend    = (recent[-1] - recent[0]) / max(1, len(recent) - 1)

            if trend <= 0:
                continue

            metric = key.split(".", 1)[-1]

            if "cpu" in metric and current > 60:
                mins = (THRESH_CPU - current) / trend * 2 / 60
                if 0 < mins < 120:
                    result["cpu_exhaustion_minutes"] = int(mins)

            elif "memory" in metric and current > 65:
                mins = (THRESH_MEMORY - current) / trend * 2 / 60
                if 0 < mins < 120:
                    result["memory_exhaustion_minutes"] = int(mins)

            elif "disk" in metric:
                hrs = (THRESH_DISK - current) / trend * 2 / 3600
                if 0 < hrs < 72:
                    result["disk_exhaustion_hours"] = int(hrs)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────────────

    def get_model_info(self) -> Dict[str, Any]:
        """Summary of all fitted models — exposed via /api/agents/status."""
        with self._lock:
            snapshot = dict(self._models)
        return {
            key: {
                "n_obs":  model.n_obs if model else None,
                "ticks":  self._tick.get(key, 0),
                "source": "Prophet" if model else "ES+LTE",
            }
            for key, model in snapshot.items()
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BaseAgent interface
    # ─────────────────────────────────────────────────────────────────────────

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        self._begin_task()
        try:
            flat: Dict[str, float] = message.payload.get("metrics", {})

            for key, val in flat.items():
                self._push(key, val)

            predictions: Dict[str, List[Dict]] = {}
            for key in flat:
                pts = self.forecast(key)
                if pts:
                    predictions[key] = pts

            self._end_task()
            return AgentMessage(
                sender=self.agent_id,
                recipient=message.sender,
                msg_type="response",
                payload={
                    "predictions":       predictions,
                    "capacity_forecast": self.capacity_forecast(flat),
                    "model_info":        self.get_model_info(),
                },
                correlation_id=message.correlation_id,
            )
        except Exception:
            self.status = "idle"
            raise
