# Monitoring

The stack ships three services: the inference server, Prometheus, and Grafana. All three start together with `make up`.

---

## Start the monitoring stack

```bash
make up

# Check all three are running
docker compose ps
```

| Service | URL | Default credentials |
|---|---|---|
| Inference server | http://localhost:8080 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / rescuevision |

---

## Prometheus metrics

The server exposes `GET /metrics` in Prometheus text format. Scraped every 15 seconds.

### Metrics reference

| Metric | Type | Labels | Description |
|---|---|---|---|
| `rescuevision_requests_total` | Counter | method, path, status_code | Total HTTP requests |
| `rescuevision_request_duration_seconds` | Histogram | method, path | End-to-end HTTP duration |
| `rescuevision_inference_latency_seconds` | Histogram | — | Pure ONNX Runtime inference time |
| `rescuevision_detections_per_request` | Histogram | — | Detections returned per predict call |
| `rescuevision_model_info` | Info | version, format | Currently loaded model metadata |
| `rescuevision_startup_timestamp_seconds` | Gauge | — | Unix timestamp when model became ready |

### Useful PromQL queries

**Request rate (last 1 min):**
```promql
sum(rate(rescuevision_requests_total[1m]))
```

**Error rate:**
```promql
sum(rate(rescuevision_requests_total{status_code=~"5.."}[1m]))
/ sum(rate(rescuevision_requests_total[1m]))
```

**p95 inference latency:**
```promql
histogram_quantile(0.95,
  sum(rate(rescuevision_inference_latency_seconds_bucket[5m])) by (le)
)
```

**Average detections per request:**
```promql
rate(rescuevision_detections_per_request_sum[5m])
/ rate(rescuevision_detections_per_request_count[5m])
```

---

## Grafana dashboard

The `RescueVision — Inference Observability` dashboard is auto-provisioned at startup. No manual import needed.

**Panels:**

| Panel | What it shows |
|---|---|
| Request Rate | req/s over time |
| Error Rate | 5xx rate |
| p95 Inference Latency | ONNX-only latency |
| Avg Detections / Request | Distribution over time |
| Latency Percentiles | p50 / p95 / p99 of inference time |
| End-to-End Request Duration | Full HTTP round trip for `/predict` |
| Request Rate by Status Code | 200 vs 4xx vs 5xx over time |
| Detections Histogram | How many detections per request |

---

## Scrape config

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: rescuevision-serve
    static_configs:
      - targets: ["rescuevision-serve:8080"]
    metrics_path: /metrics
```

To scrape a different host (e.g. a remote server), change `targets`.

---

## Grafana provisioning

Auto-provisioning files at startup:

```
monitoring/grafana/
├── provisioning/
│   ├── datasources/prometheus.yml    → points Grafana at Prometheus
│   └── dashboards/provider.yml       → tells Grafana where to load dashboards from
└── dashboards/
    └── rescuevision.json             → the actual dashboard definition
```

To add a new dashboard: save it as JSON from Grafana → place in `monitoring/grafana/dashboards/` → restart the grafana container.
