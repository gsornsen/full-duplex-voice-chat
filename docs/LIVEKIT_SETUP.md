# LiveKit Setup Guide

## Quick Start

### Local Development with Docker Compose

1. **Start all services:**
   ```bash
   docker compose up -d
   ```

2. **Verify services are healthy:**
   ```bash
   docker ps
   ```

   Expected output should show all services as `(healthy)`:
   - `livekit-server`
   - `orchestrator`
   - `tts-worker-0`
   - `redis-tts`

3. **Check LiveKit logs:**
   ```bash
   docker logs livekit-server
   ```

   Should see:
   ```
   INFO  Starting TURN server
   INFO  starting LiveKit server  {"portHttp": 7880, ...}
   ```

4. **Access web client:**
   ```bash
   cd src/client/web
   cp .env.example .env.local
   # Edit .env.local if needed (defaults are correct for local development)
   pnpm install
   pnpm dev
   ```

### Stopping Services

```bash
docker compose down
```

## Configuration Details

### Development Credentials

**API Key:** `devkey`
**API Secret:** `devsecret1234567890abcdefghijklmn` (32 characters minimum)

These credentials are pre-configured in:
- `configs/livekit.yaml`
- `configs/orchestrator.yaml`
- `src/client/web/.env.example`
- `docker-compose.yml`

### Port Mapping

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| LiveKit | 7880 | TCP | WebRTC/WebSocket + Health |
| LiveKit | 7881 | TCP | RTC TCP |
| LiveKit | 7882 | UDP | TURN |
| LiveKit | 50000-50099 | UDP | RTC/ICE (subset) |
| Orchestrator | 8080 | TCP | WebSocket API |
| Orchestrator | 8081 | TCP | Health Check |
| TTS Worker | 7001 | TCP | gRPC |
| TTS Worker | 9090 | TCP | Metrics |
| Redis | 6380 | TCP | Service Discovery |

### Network Architecture

```
Browser Client
    |
    | WebRTC/WS (port 7880)
    v
LiveKit Server (container: livekit-server)
    |
    | LiveKit SDK
    v
Orchestrator (container: orchestrator)
    |
    | gRPC (port 7001)
    v
TTS Worker (container: tts-worker-0)
    |
    | Redis (port 6380)
    v
Service Discovery (container: redis-tts)
```

## TURN Server Configuration

TURN (Traversal Using Relays around NAT) is enabled for NAT traversal in WebRTC connections.

### Current Settings (Development)
```yaml
turn:
  enabled: true
  domain: localhost
  udp_port: 7882
  external_tls: false
```

### When to Use TURN

TURN is automatically used when direct peer-to-peer connection fails due to:
- Symmetric NATs
- Restrictive firewalls
- Network address translation issues

### Testing TURN

```bash
# Check TURN server is listening
docker exec livekit-server netstat -uln | grep 7882

# View TURN logs
docker logs livekit-server | grep TURN
```

## Health Checks

### LiveKit Health Check
```bash
curl http://localhost:7880/
# Expected: OK
```

### Orchestrator Health Check
```bash
curl http://localhost:8081/health
# Expected: {"status": "healthy"}
```

### TTS Worker Health Check
```bash
nc -zv localhost 7001
# Expected: Connection to localhost 7001 port [tcp/*] succeeded!
```

### Redis Health Check
```bash
docker exec redis-tts redis-cli ping
# Expected: PONG
```

## Troubleshooting

### LiveKit Won't Start

**Symptom:** `ERROR: secret is too short`
**Solution:** Verify API secret is at least 32 characters in `configs/livekit.yaml`

**Symptom:** `invalid TURN ports`
**Solution:** Check TURN configuration includes `domain` and `udp_port` fields

### Health Check Failing

**Symptom:** Container shows `(unhealthy)` status
**Solution:**
1. Check logs: `docker logs <container-name>`
2. Verify port is accessible: `docker exec <container-name> wget -O- http://localhost:<port>/`
3. Increase `start_period` in health check configuration

### WebRTC Connection Issues

**Symptom:** Unable to establish WebRTC connection from browser
**Solution:**
1. Verify LiveKit is healthy: `docker ps`
2. Check browser console for ICE connection errors
3. Verify firewall allows UDP traffic on ports 7882 and 50000-50099
4. Check TURN server logs: `docker logs livekit-server | grep TURN`

### Port Conflicts

**Symptom:** `bind: address already in use`
**Solution:**
1. Stop conflicting services: `docker compose down`
2. Find process using port: `lsof -i :<port>` or `netstat -tuln | grep <port>`
3. Change port in `docker-compose.yml` if needed

## Production Deployment

### Security Checklist

- [ ] Generate secure API keys: `docker run --rm livekit/livekit-server generate-keys`
- [ ] Enable TLS: Set `turn.external_tls: true`
- [ ] Use proper domain name instead of `localhost`
- [ ] Configure reverse proxy (nginx/traefik) for HTTPS
- [ ] Update all configuration files with production credentials
- [ ] Enable firewall rules for required ports only
- [ ] Set up monitoring and alerting
- [ ] Configure log aggregation
- [ ] Enable rate limiting
- [ ] Review and apply security best practices

### Generating Production Keys

```bash
docker run --rm livekit/livekit-server generate-keys
```

Example output:
```
API Key: APIxxxxxxxxx
API Secret: SECRETxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Update these in:
1. `configs/livekit.yaml` - `keys` section
2. `configs/orchestrator.yaml` - `livekit.api_key` and `livekit.api_secret`
3. Web client `.env.local` - `LIVEKIT_API_KEY` and `LIVEKIT_API_SECRET`
4. Orchestrator environment variables in deployment configuration

### TLS Configuration

For production, enable TLS in `configs/livekit.yaml`:

```yaml
turn:
  enabled: true
  domain: your-domain.com
  external_tls: true
  tls_port: 443
  cert_file: /path/to/cert.pem
  key_file: /path/to/key.pem
```

### Multi-Region Deployment

For global deployment with low latency:

```yaml
region: us-west-2  # or eu-west-1, ap-southeast-1, etc.
```

Configure multiple LiveKit servers in different regions and use a global load balancer.

## Development Workflow

### Rebuilding Services

```bash
# Rebuild specific service
docker compose build orchestrator

# Rebuild and restart
docker compose up --build orchestrator -d

# Rebuild all services
docker compose build

# Full restart with rebuild
docker compose up --build -d
```

### Viewing Logs

```bash
# Follow logs for all services
docker compose logs -f

# Follow logs for specific service
docker compose logs -f livekit

# View last 50 lines
docker logs livekit-server --tail 50

# View logs with timestamps
docker logs livekit-server -t
```

### Testing WebRTC Connection

1. Start services:
   ```bash
   docker compose up -d
   ```

2. Check LiveKit is healthy:
   ```bash
   docker ps | grep livekit
   ```

3. Open browser DevTools and navigate to web client

4. Check WebRTC stats in browser console:
   ```javascript
   // Get RTCPeerConnection stats
   const pc = /* your RTCPeerConnection instance */;
   const stats = await pc.getStats();
   stats.forEach(report => console.log(report));
   ```

## Performance Tuning

### Resource Limits

Add resource limits in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
    reservations:
      cpus: '1'
      memory: 2G
```

### Logging Level

Adjust log verbosity in `configs/livekit.yaml`:

```yaml
logging:
  level: debug  # Options: debug, info, warn, error
```

### Connection Limits

Configure in `configs/livekit.yaml`:

```yaml
room:
  max_participants: 10  # Adjust based on server capacity
```

## References

- [LiveKit Official Documentation](https://docs.livekit.io/)
- [Self-Hosting Guide](https://docs.livekit.io/home/self-hosting/deployment/)
- [TURN Configuration](https://docs.livekit.io/home/self-hosting/turn/)
- [WebRTC Troubleshooting](https://webrtc.github.io/samples/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)

## Support

For issues specific to this project:
1. Check logs: `docker compose logs`
2. Review configuration files in `configs/`
3. Consult `LIVEKIT_CONFIG_FIX.md` for resolved issues
4. Check LiveKit server health: `curl http://localhost:7880/`

For LiveKit-specific issues:
- [LiveKit Community Slack](https://livekit.io/slack)
- [GitHub Issues](https://github.com/livekit/livekit-server/issues)
