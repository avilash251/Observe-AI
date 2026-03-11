"""
PredictorAgent — ARIMA + Prometheus Edition
═══════════════════════════════════════════════════════════════════════════════
Primary forecaster  : ARIMA (statsmodels) with auto order selection
Cold-start fallback : Exponential Smoothing + Linear Trend Extrapolation
Data source         : Prometheus range_query API (backfill on startup)
Capacity planner    : Linear breach-time estimation on smoothed trend
Confidence intervals: Statsmodels 95% CI (ARIMA) / heuristic √i (fallback)

Lifecycle per metric key
─────────────────────────
  Startup           → backfill_from_prometheus() pre-loads 4-min history
                      ARIMA fits immediately — zero cold start
  Ticks 1-14*       → ES+LTE fallback  (* only if Prometheus unavailable)
  Tick  15+         → ARIMA primary; ES+LTE on any exception
  Every 30 ticks    → background refit with latest rolling window

Environment variables
──────────────────────
  PROMETHEUS_URL    → default: http://localhost:9090
  PROMETHEUS_STEP   → scrape step in seconds, default: 2
  PROMETHEUS_LOOKBACK_MINUTES → history window to backfill, default: 4
"""

import asyncio
import logging
import math
import os
import random
import threading
import warnings
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

from agents.base_agent import BaseAgent, AgentMessage

# ── optional ARIMA dependency ──────────────────────────────────────────────
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
    _ARIMA_AVAILABLE = True
except ImportError:
    _ARIMA_AVAILABLE = False

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PROMETHEUS_URL       = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
PROMETHEUS_STEP      = int(os.getenv("PROMETHEUS_STEP", "2"))          # seconds
LOOKBACK_MINUTES     = int(os.getenv("PROMETHEUS_LOOKBACK_MINUTES", "4"))

HISTORY_LIMIT        = 120     # rolling buffer size  (120 × 2 s = 4 min)
FORECAST_STEPS       = 6       # 6 × 5 min = 30-minute horizon
COLD_START_MIN       = 15      # ticks needed before ARIMA attempted
REFIT_EVERY          = 30      # refit interval in ticks (~60 s)
ES_ALPHA             = 0.3     # smoothing factor for ES+LTE fallback
PROMETHEUS_TIMEOUT   = 10.0    # HTTP timeout in seconds

# Candidate ARIMA orders tried during auto-selection — cheapest first
CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 1),
    (1, 1, 2),
    (2, 1, 2),
]

# Capacity breach thresholds (%)
THRESH_CPU    = 90.0
THRESH_MEMORY = 95.0
THRESH_DISK   = 95.0

# ── Default PromQL queries per ObservaAI metric key ────────────────────────
# Override by passing custom_queries dict to backfill_all_from_prometheus()
#
# Key format : "<service_id>.<metric_name>"
# Service IDs: api-gateway | auth-service | user-service |
#              payment-service | notification-service | fraud-detection
#
# These are sensible defaults for a standard Kubernetes / Docker deployment.
# Adjust label selectors to match your actual Prometheus label set.

DEFAULT_PROM_QUERIES: Dict[str, str] = {
    # ── api-gateway ──────────────────────────────────────────────────────
    "api-gateway.cpu_usage": (
        'rate(process_cpu_seconds_total{job="api-gateway"}[30s]) * 100'
    ),
    "api-gateway.memory_usage": (
        'process_resident_memory_bytes{job="api-gateway"} '
        '/ process_virtual_memory_bytes{job="api-gateway"} * 100'
    ),
    "api-gateway.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="api-gateway"}[1m])) * 1000'
    ),
    "api-gateway.error_rate": (
        'rate(http_requests_total{job="api-gateway",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="api-gateway"}[1m]) * 100'
    ),
    "api-gateway.request_rate": (
        'rate(http_requests_total{job="api-gateway"}[30s])'
    ),
    "api-gateway.queue_depth": (
        'queue_depth_current{job="api-gateway"}'
    ),

    # ── auth-service ─────────────────────────────────────────────────────
    "auth-service.cpu_usage": (
        'rate(process_cpu_seconds_total{job="auth-service"}[30s]) * 100'
    ),
    "auth-service.memory_usage": (
        'process_resident_memory_bytes{job="auth-service"} '
        '/ process_virtual_memory_bytes{job="auth-service"} * 100'
    ),
    "auth-service.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="auth-service"}[1m])) * 1000'
    ),
    "auth-service.error_rate": (
        'rate(http_requests_total{job="auth-service",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="auth-service"}[1m]) * 100'
    ),
    "auth-service.request_rate": (
        'rate(http_requests_total{job="auth-service"}[30s])'
    ),
    "auth-service.queue_depth": (
        'queue_depth_current{job="auth-service"}'
    ),

    # ── user-service ─────────────────────────────────────────────────────
    "user-service.cpu_usage": (
        'rate(process_cpu_seconds_total{job="user-service"}[30s]) * 100'
    ),
    "user-service.memory_usage": (
        'process_resident_memory_bytes{job="user-service"} '
        '/ process_virtual_memory_bytes{job="user-service"} * 100'
    ),
    "user-service.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="user-service"}[1m])) * 1000'
    ),
    "user-service.error_rate": (
        'rate(http_requests_total{job="user-service",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="user-service"}[1m]) * 100'
    ),
    "user-service.request_rate": (
        'rate(http_requests_total{job="user-service"}[30s])'
    ),
    "user-service.queue_depth": (
        'queue_depth_current{job="user-service"}'
    ),

    # ── payment-service ──────────────────────────────────────────────────
    "payment-service.cpu_usage": (
        'rate(process_cpu_seconds_total{job="payment-service"}[30s]) * 100'
    ),
    "payment-service.memory_usage": (
        'process_resident_memory_bytes{job="payment-service"} '
        '/ process_virtual_memory_bytes{job="payment-service"} * 100'
    ),
    "payment-service.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="payment-service"}[1m])) * 1000'
    ),
    "payment-service.error_rate": (
        'rate(http_requests_total{job="payment-service",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="payment-service"}[1m]) * 100'
    ),
    "payment-service.request_rate": (
        'rate(http_requests_total{job="payment-service"}[30s])'
    ),
    "payment-service.queue_depth": (
        'queue_depth_current{job="payment-service"}'
    ),

    # ── notification-service ─────────────────────────────────────────────
    "notification-service.cpu_usage": (
        'rate(process_cpu_seconds_total{job="notification-service"}[30s]) * 100'
    ),
    "notification-service.memory_usage": (
        'process_resident_memory_bytes{job="notification-service"} '
        '/ process_virtual_memory_bytes{job="notification-service"} * 100'
    ),
    "notification-service.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="notification-service"}[1m])) * 1000'
    ),
    "notification-service.error_rate": (
        'rate(http_requests_total{job="notification-service",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="notification-service"}[1m]) * 100'
    ),
    "notification-service.request_rate": (
        'rate(http_requests_total{job="notification-service"}[30s])'
    ),
    "notification-service.queue_depth": (
        'queue_depth_current{job="notification-service"}'
    ),

    # ── fraud-detection ──────────────────────────────────────────────────
    "fraud-detection.cpu_usage": (
        'rate(process_cpu_seconds_total{job="fraud-detection"}[30s]) * 100'
    ),
    "fraud-detection.memory_usage": (
        'process_resident_memory_bytes{job="fraud-detection"} '
        '/ process_virtual_memory_bytes{job="fraud-detection"} * 100'
    ),
    "fraud-detection.response_time_p99": (
        'histogram_quantile(0.99, '
        'rate(http_request_duration_seconds_bucket{job="fraud-detection"}[1m])) * 1000'
    ),
    "fraud-detection.error_rate": (
        'rate(http_requests_total{job="fraud-detection",status=~"5.."}[1m]) '
        '/ rate(http_requests_total{job="fraud-detection"}[1m]) * 100'
    ),
    "fraud-detection.request_rate": (
        'rate(http_requests_total{job="fraud-detection"}[30s])'
    ),
    "fraud-detection.queue_depth": (
        'queue_depth_current{job="fraud-detection"}'
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# ARIMA MODEL WRAPPER
# ═══════════════════════════════════════════════════════════════════════════════

class ARIMAModel:
    """Wraps a fitted statsmodels ARIMA result."""

    def __init__(self, order: Tuple[int, int, int], fit_result: Any):
        self.order      = order
        self.fit_result = fit_result
        self.aic        = fit_result.aic
        self.n_obs      = fit_result.nobs

    def forecast(self, steps: int) -> List[Dict]:
        """Returns forecast list with 95% CI from statsmodels. Values clipped >= 0."""
        fc  = self.fit_result.get_forecast(steps=steps)
        pts = fc.predicted_mean
        ci  = fc.conf_int(alpha=0.05)

        result = []
        for i in range(steps):
            pred  = float(pts.iloc[i])
            lower = float(ci.iloc[i, 0])
            upper = float(ci.iloc[i, 1])
            result.append({
                "time_offset_minutes": (i + 1) * 5,
                "predicted_value":     round(max(0.0, pred),  2),
                "confidence_lower":    round(max(0.0, lower), 2),
                "confidence_upper":    round(max(0.0, upper), 2),
                "source":              f"ARIMA{self.order}",
            })
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class PredictorAgent(BaseAgent):
    """
    30-minute predictive analytics agent with Prometheus data source.

    Startup flow
    ─────────────
      1. main.py calls await predictor.backfill_all_from_prometheus()
      2. For every metric key, 4-min history is loaded from Prometheus
      3. ARIMA is fitted in background threads immediately
      4. By the time the first WebSocket client connects, forecasts are ready

    Runtime flow
    ─────────────
      • Every 2s tick: _push() new value → _maybe_refit() checks schedule
      • forecast() → ARIMA if model ready, else ES+LTE fallback
      • capacity_forecast() → linear breach-time projection
    """

    def __init__(self):
        super().__init__(
            agent_id="predictor",
            name="Predictive Analytics Agent",
            description=(
                "ARIMA forecasting with Prometheus backfill + 95% CI. "
                "ES+LTE cold-start fallback. Auto order selection."
            ),
        )
        self.register_tools([
            "arima_forecaster",
            "prometheus_backfill",
            "auto_order_selector",
            "es_lte_fallback",
            "confidence_interval_builder",
            "capacity_planner",
            "breach_time_estimator",
        ])

        self._history:   Dict[str, List[float]]          = {}
        self._models:    Dict[str, Optional[ARIMAModel]] = {}
        self._tick:      Dict[str, int]                  = {}
        self._refitting: Dict[str, bool]                 = {}
        self._lock = threading.Lock()

        # Backfill stats exposed via model_info
        self._backfill_stats: Dict[str, int] = {}

        if not _ARIMA_AVAILABLE:
            logger.warning(
                "statsmodels not found — running in ES+LTE only mode. "
                "pip install statsmodels"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # PROMETHEUS BACKFILL
    # ─────────────────────────────────────────────────────────────────────────

    async def backfill_from_prometheus(
        self,
        promql: str,
        key: str,
        lookback_minutes: int = LOOKBACK_MINUTES,
        step_seconds: int = PROMETHEUS_STEP,
    ) -> int:
        """
        Query Prometheus range_query API and load historical values into the
        rolling buffer for `key`.

        Returns the number of samples successfully loaded.
        Logs a warning (does not raise) on any HTTP / parse error so that
        startup continues even when Prometheus is unreachable.

        Args:
            promql           : PromQL expression to query
            key              : Internal metric key e.g. "api-gateway.cpu_usage"
            lookback_minutes : How far back to fetch (default: 4 min)
            step_seconds     : Prometheus resolution step (default: 2 s)
        """
        end   = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)

        params = {
            "query": promql,
            "start": start.timestamp(),
            "end":   end.timestamp(),
            "step":  f"{step_seconds}s",
        }

        try:
            async with httpx.AsyncClient(timeout=PROMETHEUS_TIMEOUT) as client:
                resp = await client.get(
                    f"{PROMETHEUS_URL}/api/v1/query_range",
                    params=params,
                )
                resp.raise_for_status()
                payload = resp.json()
        except httpx.ConnectError:
            logger.warning(
                "Prometheus unreachable at %s — skipping backfill for %s",
                PROMETHEUS_URL, key,
            )
            return 0
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Prometheus HTTP %s for key=%s query=%s",
                exc.response.status_code, key, promql[:60],
            )
            return 0
        except Exception as exc:
            logger.warning("Prometheus backfill error [%s]: %s", key, exc)
            return 0

        # ── Parse response ────────────────────────────────────────────────
        results = payload.get("data", {}).get("result", [])
        if not results:
            logger.debug("Prometheus returned no data for key=%s query=%s", key, promql[:60])
            return 0

        # Use first matching series (most queries return a single series)
        values: List[Tuple[float, str]] = results[0].get("values", [])
        loaded = 0
        for _ts, val_str in values:
            try:
                self._push(key, float(val_str))
                loaded += 1
            except (ValueError, TypeError):
                continue   # skip NaN / stale markers

        self._backfill_stats[key] = loaded
        logger.info("Prometheus backfill: loaded %3d samples for %s", loaded, key)

        # Trigger immediate ARIMA fit if we loaded enough data
        if loaded >= COLD_START_MIN:
            self._maybe_refit(key)

        return loaded

    async def backfill_all_from_prometheus(
        self,
        custom_queries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, int]:
        """
        Backfill all metrics concurrently using DEFAULT_PROM_QUERIES
        (or custom_queries if provided).

        Returns dict of {key: samples_loaded}.

        Call this once from main.py lifespan before the server starts
        accepting WebSocket connections:

            predictor = orchestrator.agents["predictor"]
            stats = await predictor.backfill_all_from_prometheus()
            logger.info("Backfill complete: %s", stats)
        """
        queries = custom_queries or DEFAULT_PROM_QUERIES

        tasks = [
            self.backfill_from_prometheus(promql, key)
            for key, promql in queries.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summary: Dict[str, int] = {}
        for key, result in zip(queries.keys(), results):
            if isinstance(result, Exception):
                logger.warning("Backfill task failed for %s: %s", key, result)
                summary[key] = 0
            else:
                summary[key] = result   # type: ignore[assignment]

        total = sum(summary.values())
        logger.info(
            "Prometheus backfill complete — %d metrics, %d total samples",
            len([v for v in summary.values() if v > 0]),
            total,
        )
        return summary

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
        with self._lock:
            return list(self._history.get(key, []))

    # ─────────────────────────────────────────────────────────────────────────
    # ARIMA — auto order selection + background refit
    # ─────────────────────────────────────────────────────────────────────────

    def _select_best_order(
        self, data: List[float]
    ) -> Tuple[Optional[Tuple], Optional[Any]]:
        best_aic, best_order, best_fit = float("inf"), None, None
        for order in CANDIDATE_ORDERS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(data, order=order).fit(
                        method_kwargs={"warn_convergence": False}
                    )
                if fit.aic < best_aic:
                    best_aic, best_order, best_fit = fit.aic, order, fit
            except Exception:
                continue
        return best_order, best_fit

    def _refit_worker(self, key: str) -> None:
        try:
            data = self._snapshot(key)
            if len(data) < COLD_START_MIN:
                return
            order, fit = self._select_best_order(data)
            if fit is None:
                return
            with self._lock:
                self._models[key] = ARIMAModel(order, fit)
            logger.debug(
                "ARIMA%s fitted for %-45s  AIC=%7.1f  n=%d",
                order, key, fit.aic, len(data),
            )
        except Exception as exc:
            logger.debug("ARIMA refit error [%s]: %s", key, exc)
        finally:
            self._refitting[key] = False

    def _maybe_refit(self, key: str) -> None:
        if not _ARIMA_AVAILABLE:
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
                name=f"arima-{key[:30]}",
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
        """ARIMA primary → ES+LTE fallback. Never raises."""
        self._maybe_refit(key)
        with self._lock:
            model = self._models.get(key)
        if model is not None:
            try:
                return model.forecast(steps)
            except Exception as exc:
                logger.debug("ARIMA forecast error [%s]: %s", key, exc)
        return self._es_lte_forecast(key, steps)

    # ─────────────────────────────────────────────────────────────────────────
    # Capacity planner
    # ─────────────────────────────────────────────────────────────────────────

    def capacity_forecast(self, flat_metrics: Dict[str, float]) -> Dict:
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
        with self._lock:
            snapshot = dict(self._models)
        return {
            key: {
                "order":            model.order if model else None,
                "aic":              round(model.aic, 2) if model else None,
                "n_obs":            model.n_obs if model else None,
                "ticks":            self._tick.get(key, 0),
                "source":           f"ARIMA{model.order}" if model else "ES+LTE",
                "backfill_samples": self._backfill_stats.get(key, 0),
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
