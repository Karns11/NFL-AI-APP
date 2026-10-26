# LineupIQ — Pi Setup Guide

## Prerequisites
- Raspberry Pi 5 (8GB) with Docker installed
- DuckDNS account with a subdomain

---

## 1. DuckDNS (free dynamic DNS)

1. Go to duckdns.org, sign in with Google/GitHub
2. Create a subdomain (e.g. `lineupiq`) → your URL will be `lineupiq.duckdns.org`
3. Note your token
4. Set up IP auto-update on the Pi:
   ```bash
   crontab -e
   # Add (replace YOUR_SUBDOMAIN and YOUR_TOKEN):
   */5 * * * * curl -s "https://www.duckdns.org/update?domains=YOUR_SUBDOMAIN&token=YOUR_TOKEN&ip=" > /dev/null
   ```

---

## 2. Clone the repo and create .env

```bash
cd ~
git clone https://github.com/Karns11/NFL-AI-APP.git lineupiq
cd lineupiq
nano .env
```

`.env` contents:
```
ANTHROPIC_API_KEY=your_key_here
SECRET_KEY=your_secret_key_here
```

(`SECURE_COOKIES=true` is set automatically by docker-compose.yml)

---

## 3. Start the app

```bash
docker compose up -d --build
```

First build takes 10–15 min (downloads and compiles heavy ML packages).
Subsequent builds are fast (layer cache).

Check it's running:
```bash
docker logs lineupiq
```

---

## 4. Install and configure Caddy

```bash
sudo apt install -y caddy

# Copy config and edit with your subdomain
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
sudo nano /etc/caddy/Caddyfile   # replace YOUR_SUBDOMAIN

sudo systemctl enable caddy
sudo systemctl restart caddy
```

Caddy automatically provisions an HTTPS certificate via Let's Encrypt.

---

## 5. Port forward on your router

Log into your router admin page (usually 192.168.1.1 or 192.168.0.1) and add:
- Port **80** TCP → Pi's local IP
- Port **443** TCP → Pi's local IP

Find your Pi's local IP:
```bash
hostname -I
```

---

## 6. Verify

Visit `https://YOUR_SUBDOMAIN.duckdns.org` — you should see the login page.
Share this URL with anyone you want to access the app.

---

## Updating the app

```bash
cd ~/lineupiq
git pull
docker compose up -d --build
```

User accounts and data are stored in the `lineupiq_data` Docker volume and are unaffected by rebuilds.

---

## Adding a second app later

Add a new service to a shared `docker-compose.yml` and add a new block to the Caddyfile:
```
otherapp.duckdns.org {
    reverse_proxy localhost:9090
}
```
