# Runbook: Metrics & Monitoring Setup

**Time to Resolution:** Setup: 30-60 minutes, Troubleshooting: 5-15 minutes
**Severity:** Medium (enhances observability)
**Related:** [Log Debugging](LOG_DEBUGGING.md), [Audio Backpressure](AUDIO_BACKPRESSURE.md)

---

## Overview

This runbook covers monitoring and metrics setup for the M2 Realtime Duplex Voice Demo system using Prometheus and Grafana.

**Monitoring Stack:**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Application metrics**: Custom TTS metrics
- **System metrics**: Container and host metrics

---

## Quick Start

### Docker Compose Setup

**Add to docker-compose.yml:**
```yaml
services:
  # ... existing services ...

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=7d'
    networks:
      - tts-network

  grafana:
    image: grafana/grafana:latest
    container_name: grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - tts-network
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

**Start monitoring stack:**
```bash
docker compose up prometheus grafana -d

# Verify
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health  # Grafana
```

---

## Prometheus Configuration

### Create Prometheus Config

**File: `monitoring/prometheus.yml`**
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'tts-demo'
    environment: 'development'

# Scrape configurations
scrape_configs:
  # Orchestrator metrics
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8081']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # TTS Worker metrics
  - job_name: 'tts-worker'
    static_configs:
      - targets: ['tts-worker:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s

  # Redis metrics (if redis_exporter enabled)
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Docker container metrics (if cAdvisor enabled)
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

# Alerting rules
rule_files:
  - 'alerts.yml'

# Alertmanager configuration (optional)
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

---

## Application Metrics

### Orchestrator Metrics

**Exposed at:** `http://localhost:8081/metrics`

**Key metrics:**

**Connection metrics:**
```promql
# Active WebSocket connections
tts_websocket_connections_active

# Total connections (counter)
tts_websocket_connections_total

# Connection errors
tts_websocket_connection_errors_total
```

**Session metrics:**
```promql
# Active TTS sessions
tts_sessions_active

# Total sessions
tts_sessions_total

# Session duration (histogram)
tts_session_duration_seconds

# Sessions by model
tts_sessions_total{model_id="cosyvoice2-en-base"}
```

**Frame delivery metrics:**
```promql
# Frame queue depth (gauge)
tts_frame_queue_depth{session_id="abc123"}

# Frames sent
tts_frames_sent_total

# Frames dropped
tts_frames_dropped_total

# Frame send latency (histogram)
tts_frame_send_duration_seconds
```

**gRPC client metrics:**
```promql
# gRPC request duration
tts_grpc_request_duration_seconds{method="Synthesize"}

# gRPC request errors
tts_grpc_request_errors_total{method="Synthesize"}
```

---

### Worker Metrics

**Exposed at:** `http://localhost:9090/metrics`

**Key metrics:**

**Synthesis metrics:**
```promql
# Synthesis requests
tts_synthesis_requests_total

# Synthesis duration (histogram)
tts_synthesis_duration_seconds

# Real-time factor (RTF)
tts_synthesis_rtf{model_id="cosyvoice2-en-base"}

# First audio latency (FAL)
tts_first_audio_latency_seconds
```

**Model management metrics:**
```promql
# Resident models (gauge)
tts_models_resident_count

# Model load time (histogram)
tts_model_load_duration_seconds

# Model unload events
tts_model_unload_total

# Model eviction events
tts_model_eviction_total
```

**Resource metrics:**
```promql
# GPU memory usage (bytes)
tts_gpu_memory_used_bytes

# GPU utilization (percent)
tts_gpu_utilization_percent

# CPU usage
tts_cpu_usage_percent

# Process memory (resident set size)
process_resident_memory_bytes
```

**Audio processing metrics:**
```promql
# Frame generation count
tts_audio_frames_generated_total

# Audio normalization events
tts_audio_normalization_total

# Loudness measurements (LUFS)
tts_audio_loudness_lufs
```

---

## Grafana Setup

### Initial Login

1. Open Grafana: http://localhost:3000
2. Login: `admin` / `admin` (change on first login)
3. Add Prometheus data source

---

### Configure Data Source

**File: `monitoring/grafana/datasources/prometheus.yml`**
```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
```

**Or via UI:**
1. Configuration → Data Sources → Add data source
2. Select "Prometheus"
3. URL: `http://prometheus:9090`
4. Save & Test

---

### TTS Dashboard

**File: `monitoring/grafana/dashboards/tts-dashboard.json`**

**Key panels:**

**1. Active Sessions**
```promql
tts_sessions_active
```

**2. Session Rate (sessions/min)**
```promql
rate(tts_sessions_total[1m]) * 60
```

**3. Frame Queue Depth**
```promql
tts_frame_queue_depth
```

**4. Frame Drop Rate**
```promql
rate(tts_frames_dropped_total[1m])
```

**5. Average RTF**
```promql
avg(tts_synthesis_rtf)
```

**6. p95 First Audio Latency**
```promql
histogram_quantile(0.95, rate(tts_first_audio_latency_seconds_bucket[5m]))
```

**7. Model Load Time**
```promql
histogram_quantile(0.95, rate(tts_model_load_duration_seconds_bucket[5m]))
```

**8. GPU Memory Usage**
```promql
tts_gpu_memory_used_bytes / 1024 / 1024 / 1024  # Convert to GB
```

**9. Error Rate**
```promql
rate(tts_grpc_request_errors_total[1m])
```

**10. Connection Count**
```promql
tts_websocket_connections_active
```

---

### Create Dashboard

**Manual creation:**
1. Dashboards → New Dashboard → Add panel
2. Query: Enter PromQL query
3. Visualization: Select type (graph, gauge, stat, etc.)
4. Panel title and settings
5. Save dashboard

**Import from JSON:**
```bash
# Place dashboard JSON in monitoring/grafana/dashboards/
# Grafana auto-provisions on startup (with provisioning config)
```

**Provisioning config: `monitoring/grafana/dashboards/dashboard.yml`**
```yaml
apiVersion: 1

providers:
  - name: 'TTS Dashboards'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
```

---

## Alert Rules

### Prometheus Alerts

**File: `monitoring/alerts.yml`**
```yaml
groups:
  - name: tts_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(tts_grpc_request_errors_total[1m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High gRPC error rate"
          description: "Error rate: {{ $value }} errors/sec"

      # Frame drops
      - alert: FramesDropping
        expr: rate(tts_frames_dropped_total[1m]) > 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Audio frames being dropped"
          description: "Drop rate: {{ $value }} frames/sec"

      # High frame queue
      - alert: HighFrameQueue
        expr: tts_frame_queue_depth > 40
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Frame queue depth > 80%"
          description: "Queue depth: {{ $value }}/50"

      # High RTF (too slow)
      - alert: HighRTF
        expr: avg(tts_synthesis_rtf) > 0.8
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "RTF approaching real-time (slow synthesis)"
          description: "RTF: {{ $value }} (target: < 0.5)"

      # High first audio latency
      - alert: HighFirstAudioLatency
        expr: histogram_quantile(0.95, rate(tts_first_audio_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "p95 First Audio Latency > 500ms"
          description: "p95 FAL: {{ $value }}s (target: < 0.3s)"

      # Model load failures
      - alert: ModelLoadFailures
        expr: rate(tts_model_load_errors_total[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Model loading failures detected"
          description: "Model load error rate: {{ $value }}/min"

      # No active worker
      - alert: NoActiveWorker
        expr: up{job="tts-worker"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "TTS worker down"
          description: "No metrics from TTS worker for 1 minute"

      # High GPU memory usage
      - alert: HighGPUMemory
        expr: (tts_gpu_memory_used_bytes / tts_gpu_memory_total_bytes) > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory usage > 90%"
          description: "GPU memory: {{ $value | humanizePercentage }}"

      # Session queue backup
      - alert: SessionQueueBackup
        expr: tts_session_queue_depth > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Session queue backing up"
          description: "Queue depth: {{ $value }}"
```

**Load alerts:**
```bash
# Reload Prometheus config
curl -X POST http://localhost:9090/-/reload

# Or restart Prometheus
docker compose restart prometheus

# Check alerts
curl http://localhost:9090/api/v1/rules | jq .
```

---

## Alertmanager (Optional)

### Configuration

**File: `monitoring/alertmanager.yml`**
```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'job']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'

receivers:
  - name: 'default'
    email_configs:
      - to: 'alerts@example.com'
        from: 'prometheus@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: 'prometheus@example.com'
        auth_password: 'password'

  - name: 'slack'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/YOUR/WEBHOOK/URL'
        channel: '#tts-alerts'
        title: 'TTS Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}\n{{ end }}'

  - name: 'pagerduty'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

**Add to docker-compose.yml:**
```yaml
alertmanager:
  image: prom/alertmanager:latest
  container_name: alertmanager
  ports:
    - "9093:9093"
  volumes:
    - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
  networks:
    - tts-network
```

---

## System Metrics

### cAdvisor (Container Metrics)

**Add to docker-compose.yml:**
```yaml
cadvisor:
  image: gcr.io/cadvisor/cadvisor:latest
  container_name: cadvisor
  ports:
    - "8080:8080"
  volumes:
    - /:/rootfs:ro
    - /var/run:/var/run:ro
    - /sys:/sys:ro
    - /var/lib/docker/:/var/lib/docker:ro
  networks:
    - tts-network
  privileged: true
```

**Useful queries:**
```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total{name="orchestrator"}[1m]) * 100

# Container memory usage
container_memory_usage_bytes{name="orchestrator"} / 1024 / 1024

# Container network I/O
rate(container_network_transmit_bytes_total{name="orchestrator"}[1m])
```

---

### Node Exporter (Host Metrics)

**Add to docker-compose.yml:**
```yaml
node-exporter:
  image: prom/node-exporter:latest
  container_name: node-exporter
  ports:
    - "9100:9100"
  volumes:
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
    - /:/rootfs:ro
  command:
    - '--path.procfs=/host/proc'
    - '--path.sysfs=/host/sys'
    - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
  networks:
    - tts-network
```

**Add to Prometheus config:**
```yaml
scrape_configs:
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

**Useful queries:**
```promql
# CPU usage
100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)

# Memory usage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Disk usage
(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100

# Network traffic
rate(node_network_transmit_bytes_total[1m])
```

---

## Useful PromQL Queries

### Performance Metrics

**Average session duration:**
```promql
avg(rate(tts_session_duration_seconds_sum[5m]) / rate(tts_session_duration_seconds_count[5m]))
```

**p95/p99 latency:**
```promql
histogram_quantile(0.95, rate(tts_first_audio_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(tts_first_audio_latency_seconds_bucket[5m]))
```

**Throughput (frames/sec):**
```promql
rate(tts_frames_sent_total[1m])
```

**Error percentage:**
```promql
(rate(tts_grpc_request_errors_total[5m]) / rate(tts_grpc_requests_total[5m])) * 100
```

### Resource Utilization

**Top models by usage:**
```promql
topk(5, sum by (model_id) (rate(tts_synthesis_requests_total[5m])))
```

**Memory trend:**
```promql
deriv(process_resident_memory_bytes[5m])
```

**GPU utilization over time:**
```promql
avg_over_time(tts_gpu_utilization_percent[1h])
```

### Capacity Planning

**Max concurrent sessions (last 24h):**
```promql
max_over_time(tts_sessions_active[24h])
```

**Frame queue max depth:**
```promql
max(tts_frame_queue_depth)
```

**Model load frequency:**
```promql
sum by (model_id) (rate(tts_model_load_total[1h]))
```

---

## Troubleshooting

### No Metrics Showing

**Check metrics endpoint:**
```bash
curl http://localhost:8081/metrics
curl http://localhost:9090/metrics  # Worker
```

**Expected output:**
```
# HELP tts_sessions_active Active TTS sessions
# TYPE tts_sessions_active gauge
tts_sessions_active 2
...
```

**If empty or error:**
```bash
# Check if metrics enabled in config
grep -A 5 "metrics:" configs/worker.yaml

# Should show:
# metrics:
#   enabled: true
#   prometheus_port: 9090
```

---

### Prometheus Can't Scrape Target

**Check Prometheus targets:**
```bash
curl http://localhost:9090/api/v1/targets | jq .
```

**Look for state: "DOWN"**

**Debug:**
```bash
# Check network connectivity
docker exec prometheus ping orchestrator

# Check DNS resolution
docker exec prometheus nslookup orchestrator

# Check firewall
docker exec prometheus wget -O- http://orchestrator:8081/metrics
```

**Fix network issues:**
```yaml
# Ensure all services on same network
networks:
  - tts-network
```

---

### Grafana Not Showing Data

**Check data source:**
1. Grafana → Configuration → Data Sources → Prometheus
2. Click "Test" button
3. Should show: "Data source is working"

**If failing:**
- Check Prometheus URL: `http://prometheus:9090`
- Verify Prometheus is running: `docker ps | grep prometheus`
- Check network: `docker exec grafana ping prometheus`

**Check query syntax:**
```promql
# Test query in Prometheus UI first
# http://localhost:9090/graph
# Enter query, click "Execute"
```

---

### High Cardinality Issues

**Symptom:** Prometheus slow or high memory usage

**Cause:** Too many unique label combinations (e.g., session_id in labels)

**Check cardinality:**
```bash
# Top metrics by cardinality
curl http://localhost:9090/api/v1/status/tsdb | jq '.data.seriesCountByMetricName | to_entries | sort_by(.value) | reverse | .[0:10]'
```

**Fix:** Use labels sparingly
```python
# Bad: High cardinality
metrics.counter('requests', labels={'session_id': session_id})

# Good: Use session_id only for tracing, not metrics
metrics.counter('requests', labels={'model_id': model_id})
```

---

## Best Practices

### Metric Naming

**Convention:** `component_subsystem_unit`

Examples:
- `tts_synthesis_duration_seconds` (histogram)
- `tts_sessions_active` (gauge)
- `tts_frames_sent_total` (counter)
- `tts_gpu_memory_used_bytes` (gauge)

### Label Design

**Good labels (low cardinality):**
- `model_id` (finite set: cosyvoice2-en-base, xtts-v2, ...)
- `error_type` (finite set: connection_error, timeout, ...)
- `status_code` (finite set: 200, 404, 500, ...)

**Bad labels (high cardinality):**
- `session_id` (unbounded)
- `user_id` (unbounded)
- `timestamp` (unbounded)

### Dashboard Organization

**Recommended structure:**

1. **Overview Dashboard:**
   - Active sessions
   - Request rate
   - Error rate
   - p95 latency

2. **Performance Dashboard:**
   - RTF
   - FAL
   - Frame jitter
   - Queue depth

3. **Resource Dashboard:**
   - CPU usage
   - Memory usage
   - GPU utilization
   - Disk I/O

4. **Model Dashboard:**
   - Resident models
   - Load time
   - Eviction rate
   - Usage by model

---

## Production Recommendations

### Prometheus

- **Retention:** 7-30 days (adjust based on disk space)
- **Scrape interval:** 10-15s (balance between granularity and load)
- **Remote write:** For long-term storage (e.g., Thanos, Cortex)
- **High availability:** Run 2+ Prometheus instances

### Grafana

- **Authentication:** Enable and enforce strong passwords
- **HTTPS:** Use reverse proxy (nginx/traefik) with TLS
- **Backup:** Regular dashboard JSON exports
- **Organization:** Separate dashboards by team/function

### Alerting

- **Alert fatigue:** Tune thresholds to minimize false positives
- **Runbooks:** Link alerts to resolution runbooks
- **On-call rotation:** Integrate with PagerDuty/Opsgenie
- **Escalation:** Define severity levels and escalation paths

---

## Related Runbooks

- **[Log Debugging](LOG_DEBUGGING.md)** - Complementary log-based diagnostics
- **[Audio Backpressure](AUDIO_BACKPRESSURE.md)** - Frame queue metrics
- **[Advanced Troubleshooting](ADVANCED_TROUBLESHOOTING.md)** - Deep diagnostics

---

## Further Help

**Quick checks:**
```bash
# Verify Prometheus
curl http://localhost:9090/-/healthy

# Verify Grafana
curl http://localhost:3000/api/health

# Check targets
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastError}'

# Test query
curl -G http://localhost:9090/api/v1/query --data-urlencode 'query=up'
```

**Useful resources:**
- Prometheus docs: https://prometheus.io/docs/
- Grafana docs: https://grafana.com/docs/
- PromQL cheat sheet: https://promlabs.com/promql-cheat-sheet/
- Example dashboards: https://grafana.com/grafana/dashboards/
