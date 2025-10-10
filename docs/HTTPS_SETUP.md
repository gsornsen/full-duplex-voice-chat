# HTTPS Setup for Local Network Access

This guide explains how to set up HTTPS for accessing the web client from other machines on your local network. This is necessary because browsers require HTTPS to grant camera/microphone permissions (except for localhost).

## Overview

We use **Caddy** as a reverse proxy to provide:
- HTTPS for the Next.js web client (port 443)
- WSS (WebSocket Secure) for LiveKit (port 7443, proxies to 7880)
- Self-signed certificates for local development

## Architecture

```
[Remote Device on LAN]
         |
         | https://172.24.35.45 (port 443)
         v
    [Caddy Proxy] ----------> [Next.js Web Client] (host:3000)
         |
         | wss://172.24.35.45:7443
         v
    [Caddy Proxy] ----------> [LiveKit Server] (docker:7880)
```

## Quick Start

### 1. Find Your Host IP Address

**Linux/Mac:**
```bash
hostname -I | awk '{print $1}'
# Example output: 172.24.35.45
```

**Windows (PowerShell):**
```powershell
(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias "Ethernet*" | Select-Object -First 1).IPAddress
```

**Windows (Command Prompt):**
```cmd
ipconfig
# Look for "IPv4 Address" under your network adapter
```

### 2. Update Caddyfile

Edit `/home/gerald/git/full-duplex-voice-chat/Caddyfile` and replace `172.24.35.45` with your actual host IP:

```caddyfile
# Replace this IP address with your host machine's IP
https://172.24.35.45, https://localhost {
    # ... rest of config
}

https://172.24.35.45:7880, https://localhost:7880 {
    # ... rest of config
}
```

**Note:** You can specify multiple IP addresses separated by commas if your machine has multiple network interfaces.

### 3. Create Web Client Environment File

Create `/home/gerald/git/full-duplex-voice-chat/src/client/web/.env.local`:

```bash
# Replace with your host IP
LIVEKIT_URL=wss://172.24.35.45:7443
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret1234567890abcdefghijklmn
```

**Important:** Use `wss://` (WebSocket Secure) instead of `ws://`, and use port `7443` instead of `7880`.

### 4. Start the Services

**Option A: Start everything with Docker Compose**
```bash
# Start Docker services (includes Caddy)
docker compose up --build

# In another terminal, start the Next.js web client on the host
cd src/client/web
pnpm install
pnpm dev
```

**Option B: Start Caddy separately (if you already have services running)**
```bash
# Start just the Caddy service
docker compose up caddy

# Or use standalone Caddy (not recommended)
caddy run --config Caddyfile
```

### 5. Access from Another Device

1. **Open your browser** on another device connected to the same network
2. **Navigate to** `https://172.24.35.45` (use your actual IP)
3. **Accept the self-signed certificate** (see instructions below)
4. **Allow camera/microphone permissions** when prompted

## Accepting Self-Signed Certificates

Since we're using self-signed certificates for local development, you'll need to accept the certificate warning in your browser.

### Chrome/Edge

1. Navigate to `https://172.24.35.45`
2. You'll see "Your connection is not private"
3. Click **Advanced**
4. Click **Proceed to 172.24.35.45 (unsafe)**
5. Repeat for `https://172.24.35.45:7443` (LiveKit endpoint)

**Pro tip:** Type `thisisunsafe` while on the warning page (no input box needed) to bypass the warning.

### Firefox

1. Navigate to `https://172.24.35.45`
2. You'll see "Warning: Potential Security Risk Ahead"
3. Click **Advanced**
4. Click **Accept the Risk and Continue**
5. Repeat for `https://172.24.35.45:7443`

### Safari (iOS/macOS)

1. Navigate to `https://172.24.35.45`
2. Tap **Show Details**
3. Tap **visit this website**
4. Tap **Visit Website**
5. Repeat for `https://172.24.35.45:7443`

**Note:** On iOS, you may need to install the certificate profile:
1. Settings > General > About > Certificate Trust Settings
2. Enable full trust for the Caddy certificate

## Port Reference

| Port | Protocol | Service | Purpose |
|------|----------|---------|---------|
| 443  | HTTPS    | Web Client | Main entry point for browser clients |
| 7443 | WSS      | LiveKit | WebSocket Secure for LiveKit WebRTC |
| 7880 | WS       | LiveKit | Direct WebSocket (localhost only) |
| 3000 | HTTP     | Next.js | Web client dev server (host only) |
| 80   | HTTP     | Caddy | Optional HTTP redirect to HTTPS |

## Troubleshooting

### Can't connect from another device

1. **Check firewall:** Ensure ports 443 and 7443 are open
   ```bash
   # Linux: Check firewall status
   sudo ufw status

   # Linux: Allow ports
   sudo ufw allow 443/tcp
   sudo ufw allow 7443/tcp

   # Windows: Check Windows Firewall settings
   # Add inbound rules for ports 443 and 7443
   ```

2. **Verify IP address:** Make sure you're using the correct host IP
   ```bash
   hostname -I | awk '{print $1}'
   ```

3. **Check Docker network:** Ensure Caddy can reach LiveKit
   ```bash
   docker compose exec caddy wget -O- http://livekit:7880
   ```

4. **Check Caddy logs:**
   ```bash
   docker compose logs caddy
   # Or view detailed logs
   docker compose exec caddy tail -f /var/log/caddy/web.log
   docker compose exec caddy tail -f /var/log/caddy/livekit.log
   ```

### Certificate errors persist

1. **Clear browser cache:** Sometimes browsers cache certificate decisions
2. **Restart Caddy:** Regenerate certificates
   ```bash
   docker compose restart caddy
   ```

3. **Check Caddy certificate generation:**
   ```bash
   docker compose exec caddy caddy list-certificates
   ```

### Web client can't connect to LiveKit

1. **Verify .env.local:** Ensure you're using `wss://` and port `7443`
2. **Check browser console:** Look for WebSocket connection errors
3. **Test LiveKit directly:**
   ```bash
   # From your browser's console (on the HTTPS page)
   new WebSocket('wss://172.24.35.45:7443')
   ```

4. **Verify LiveKit is running:**
   ```bash
   docker compose ps livekit
   docker compose logs livekit
   ```

### Next.js dev server not reachable from Caddy

1. **Check Next.js is running on host:**
   ```bash
   curl http://localhost:3000
   ```

2. **Verify host.docker.internal mapping:**
   ```bash
   docker compose exec caddy ping host.docker.internal
   ```

3. **Try using host IP instead:** In Caddyfile, replace `host.docker.internal:3000` with your actual host IP

## Production Considerations

This setup is for **local development only**. For production deployment:

1. **Use real certificates:** Get certificates from Let's Encrypt
2. **Configure proper domain:** Use a real domain name instead of IP addresses
3. **Harden security:** Review Caddy security settings
4. **Use proper secrets:** Generate strong API keys for LiveKit
   ```bash
   docker run --rm livekit/livekit-server generate-keys
   ```

## Advanced Configuration

### Multiple Network Interfaces

If your machine has multiple IPs (e.g., Ethernet + WiFi), add all of them:

```caddyfile
https://172.24.35.45, https://192.168.1.100, https://localhost {
    # ... config
}
```

### Custom Domain with DNS

If you control your local DNS (e.g., Pi-hole, router):

1. Add DNS entry: `voice.local` → `172.24.35.45`
2. Update Caddyfile:
   ```caddyfile
   https://voice.local {
       # ... config
   }
   ```

### HTTP to HTTPS Redirect

Uncomment the redirect section in Caddyfile:

```caddyfile
http://172.24.35.45, http://localhost {
    redir https://{host}{uri} permanent
}
```

## References

- [Caddy Documentation](https://caddyserver.com/docs/)
- [LiveKit Server Configuration](https://docs.livekit.io/realtime/server/configuration/)
- [WebRTC and HTTPS Requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#security)
