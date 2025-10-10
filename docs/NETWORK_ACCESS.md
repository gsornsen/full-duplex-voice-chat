# Network Access Configuration

This guide covers accessing the Full-Duplex Voice Chat web client from devices on your local network.

## Overview

By default, the web client runs at `http://localhost:3000` and is only accessible from the host machine. To access from other devices (phones, tablets, other computers on the same network), you need HTTPS because browsers require secure context for camera/microphone access.

## Why HTTPS is Required

Browsers enforce strict security policies for accessing sensitive hardware like cameras and microphones:

- **HTTP is blocked** except for `localhost`
- **HTTPS is required** for remote access
- **WebSocket must be WSS** (WebSocket Secure) for HTTPS pages

Without HTTPS, you'll see:
- "Permission denied" for camera/microphone
- WebSocket connection failures
- Mixed content warnings

## Quick Setup

### 1. One-Command Configuration

```bash
# Auto-detect IP and configure everything
just update-ip
```

This will:
- Detect your host IP (e.g., 172.24.35.45)
- Update Caddyfile with your IP
- Create `.env.local` with correct LiveKit WSS URL
- Show next steps

### 2. Start Services

```bash
# Terminal 1: Start Docker stack
docker compose up --build

# Terminal 2: Start Next.js web client
cd src/client/web
pnpm dev
```

### 3. Access from Another Device

1. Find your IP: `just show-ip`
2. Open browser on another device
3. Navigate to `https://YOUR_IP` (e.g., `https://172.24.35.45`)
4. Accept the self-signed certificate warning
5. Grant camera/microphone permissions

## Architecture

```
┌─────────────────────┐
│   Remote Device     │
│    (Browser)        │
└──────────┬──────────┘
           │
           │ https://172.24.35.45 (Web UI)
           │ wss://172.24.35.45:7443 (LiveKit)
           │
           ▼
┌─────────────────────┐
│   Caddy Proxy       │
│   (In Docker)       │
└──────────┬──────────┘
           │
           ├──────────► Next.js Web Client (Host :3000)
           │
           └──────────► LiveKit Server (Docker :7880)
                              │
                              ▼
                        Orchestrator → TTS Worker
```

## Components

### Caddy Reverse Proxy
- **Image:** `caddy:2-alpine`
- **Purpose:** Provides HTTPS/WSS with self-signed certificates
- **Ports:**
  - 443 → Web client (proxies to host:3000)
  - 7443 → LiveKit (proxies to livekit:7880)

### Configuration Files

**Caddyfile**
- Location: `/Caddyfile`
- Defines reverse proxy rules
- Configures TLS (self-signed certificates)
- Sets up WebSocket upgrades

**Web Client Environment**
- Location: `/src/client/web/.env.local`
- Sets `LIVEKIT_URL=wss://YOUR_IP:7443`
- Contains LiveKit API keys

## Firewall Configuration

You must allow incoming connections on these ports:

### Linux (ufw)
```bash
sudo ufw allow 443/tcp
sudo ufw allow 7443/tcp
sudo ufw status
```

### Linux (firewalld)
```bash
sudo firewall-cmd --permanent --add-port=443/tcp
sudo firewall-cmd --permanent --add-port=7443/tcp
sudo firewall-cmd --reload
```

### Windows Firewall
1. Open Windows Defender Firewall
2. Click "Advanced settings"
3. Add Inbound Rules for:
   - TCP port 443
   - TCP port 7443

### macOS
macOS typically allows local network connections by default. If using third-party firewall software, add rules for ports 443 and 7443.

## Accepting Self-Signed Certificates

Since we use self-signed certificates for local development, you'll see a warning.

### Chrome/Edge
1. You'll see "Your connection is not private"
2. Click **Advanced**
3. Click **Proceed to [IP] (unsafe)**

**Pro tip:** Type `thisisunsafe` (no text field needed) to bypass

### Firefox
1. You'll see "Warning: Potential Security Risk Ahead"
2. Click **Advanced**
3. Click **Accept the Risk and Continue**

### Safari (macOS/iOS)
1. Click **Show Details**
2. Click/Tap **visit this website**
3. Confirm

**iOS Note:** For full trust, go to:
Settings → General → About → Certificate Trust Settings

## Network IP Changes

If your host IP changes (e.g., DHCP reassignment):

```bash
# Reconfigure for new IP
just update-ip

# Restart Caddy
just caddy-restart

# Restart Next.js to pick up new .env.local
cd src/client/web
pnpm dev
```

## Helpful Commands

```bash
# Show current host IP
just show-ip

# Update configuration for current IP
just update-ip

# Manually specify IP
just update-ip IP="192.168.1.100"

# View Caddy logs
just caddy-logs

# Restart Caddy
just caddy-restart

# Test HTTPS endpoints
just test-https
```

## Troubleshooting

### Can't connect from remote device

**Check 1: Verify IP address**
```bash
just show-ip
# Make sure you're using this IP in the browser
```

**Check 2: Verify services are running**
```bash
docker compose ps
# All services should show "healthy" or "running"
```

**Check 3: Test from host first**
```bash
# Test web client
curl -k https://172.24.35.45

# Test LiveKit
curl -k https://172.24.35.45:7443
```

**Check 4: Verify firewall**
```bash
# Linux
sudo ufw status

# Check if ports 443 and 7443 are allowed
```

### Certificate errors persist

**Solution 1: Clear browser data**
- Clear cookies and cached data
- Close and reopen browser

**Solution 2: Regenerate certificates**
```bash
# Restart Caddy to regenerate certificates
docker compose restart caddy
```

**Solution 3: Try incognito/private mode**
- Opens fresh session without cached security decisions

### WebSocket connection fails

**Check 1: Verify .env.local**
```bash
cat src/client/web/.env.local
# Should show: LIVEKIT_URL=wss://YOUR_IP:7443
```

**Check 2: Test WebSocket from browser console**
On the HTTPS page, open browser console:
```javascript
new WebSocket('wss://172.24.35.45:7443')
// Should see "WebSocket is already in CLOSING or CLOSED state"
// or connection attempt (not "failed immediately")
```

**Check 3: Check Caddy logs**
```bash
just caddy-logs
# Look for WebSocket upgrade requests
```

### Camera/microphone permissions denied

**Cause:** Browser requires HTTPS for getUserMedia API

**Solution:**
1. Verify you're using `https://` not `http://`
2. Accept the certificate warning first
3. Refresh the page
4. Check browser site permissions (may need to reset)

**iOS Specific:**
- Go to Settings → Safari → Camera/Microphone
- Ensure not blocked globally
- May need to trust certificate in Settings

### Next.js not reachable from Caddy

**Check 1: Verify Next.js is running**
```bash
curl http://localhost:3000
# Should return HTML content
```

**Check 2: Test host.docker.internal from Caddy**
```bash
docker compose exec caddy ping -c 2 host.docker.internal
# Should succeed
```

**Check 3: Check Caddy can reach Next.js**
```bash
docker compose exec caddy wget -O- http://host.docker.internal:3000
# Should return HTML
```

## Network Topologies

### Same WiFi Network
- Easiest setup
- Host and remote device on same router
- Use host IP directly (e.g., 192.168.1.100)

### Wired + Wireless
- Host on Ethernet, device on WiFi (or vice versa)
- Works if both connect to same router
- Use host IP from the shared network

### Multiple Network Interfaces
If host has multiple IPs (Ethernet + WiFi):
```bash
# List all IPs
hostname -I

# Configure Caddy for all IPs
just update-ip IP="192.168.1.100"
# Then manually edit Caddyfile to add second IP:
# https://192.168.1.100, https://10.0.0.50, https://localhost {
```

## Port Reference

| Port | Service | Protocol | Exposed To | Purpose |
|------|---------|----------|------------|---------|
| 443  | Caddy   | HTTPS    | Network    | Web client UI |
| 7443 | Caddy   | WSS      | Network    | LiveKit WebRTC |
| 7880 | LiveKit | WS       | Localhost  | Direct WebSocket |
| 3000 | Next.js | HTTP     | Localhost  | Dev server |
| 80   | Caddy   | HTTP     | Network    | Optional redirect |

## Security Notes

### For Local Development

This setup is designed for **local network access only**:
- Self-signed certificates (not trusted by default)
- Dev mode API keys (not production-ready)
- No rate limiting or DDoS protection
- Logs may contain sensitive data

**Do NOT expose to the public internet without:**
1. Real TLS certificates (Let's Encrypt)
2. Strong authentication
3. Rate limiting
4. Security hardening
5. Log sanitization

### For Production

See `docs/HTTPS_SETUP.md` for production deployment guidance:
- Use a real domain name
- Configure Let's Encrypt for automatic certificates
- Generate strong LiveKit API keys
- Set up monitoring and alerting
- Configure firewall rules properly
- Review security best practices

## Advanced Topics

### Custom Local Domain

If you control your local DNS (Pi-hole, router, etc.):

1. Add DNS entry: `voice.local` → `172.24.35.45`
2. Update Caddyfile:
   ```caddyfile
   https://voice.local {
       reverse_proxy host.docker.internal:3000
   }

   https://voice.local:7880 {
       reverse_proxy livekit:7880
   }
   ```
3. Update .env.local: `LIVEKIT_URL=wss://voice.local:7880`
4. Restart services

### HTTP to HTTPS Redirect

To automatically redirect HTTP to HTTPS, uncomment in Caddyfile:

```caddyfile
http://172.24.35.45, http://localhost {
    redir https://{host}{uri} permanent
}
```

Then restart Caddy:
```bash
just caddy-restart
```

### Multiple Host Support

To support multiple IPs or domains:

```caddyfile
https://172.24.35.45, https://192.168.1.100, https://voice.local {
    reverse_proxy host.docker.internal:3000
}
```

## Related Documentation

- **[HTTPS Setup Guide](HTTPS_SETUP.md)** - Detailed setup and troubleshooting
- **[Quick Start](../HTTPS_QUICKSTART.md)** - 5-minute setup guide
- **[Testing Guide](TESTING_GUIDE.md)** - How to test the system
- **[Deployment Summary](../HTTPS_SETUP_SUMMARY.md)** - Technical changes overview

## FAQ

**Q: Why not use Let's Encrypt?**
A: Let's Encrypt requires a public domain name. For local network access with IP addresses, self-signed certificates are the standard approach.

**Q: Can I use mDNS (.local domains)?**
A: Yes! Configure your local DNS and update Caddyfile. Most home routers don't support this by default.

**Q: Does this work over VPN?**
A: Yes, if the VPN routes traffic to your local network. You may need to use the VPN-assigned IP.

**Q: Can I access from outside my network?**
A: Not with this setup. For WAN access, you need port forwarding, dynamic DNS, and proper security hardening. Consider using a cloud deployment instead.

**Q: Why port 7443 instead of 7880?**
A: Port 7443 is HTTPS (via Caddy), port 7880 is plain HTTP (direct to LiveKit). Browsers require WebSocket Secure (WSS) when the page is HTTPS.

**Q: Do I need to restart everything when IP changes?**
A: Only Caddy (`just caddy-restart`) and Next.js need restarts. Docker services continue running.

## Support

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review [HTTPS Setup Guide](HTTPS_SETUP.md) for detailed help
3. Check Caddy logs: `just caddy-logs`
4. Verify firewall: `sudo ufw status`
5. Test endpoints: `just test-https`
