# Pi Setup

## 1. DuckDNS (free dynamic DNS)
1. Go to duckdns.org, sign in, create a subdomain (e.g. `lineupiq`)
2. Note your token
3. Set up auto-update cron so your home IP stays current:
   ```
   crontab -e
   # Add this line:
   */5 * * * * curl -s "https://www.duckdns.org/update?domains=lineupiq&token=YOUR_TOKEN&ip=" > /dev/null
   ```

## 2. Install dependencies
```bash
sudo apt update && sudo apt install -y python3-pip caddy git
pip3 install -r requirements.txt
```

## 3. Clone the repo
```bash
cd ~
git clone https://github.com/Karns11/NFL-AI-APP.git NFL-AI-Agent
cd NFL-AI-Agent
cp .env.example .env   # then fill in your values
```

## 4. .env on the Pi
```
ANTHROPIC_API_KEY=your_key
SECRET_KEY=your_secret_key
SECURE_COOKIES=true
```

## 5. Install the systemd service
```bash
sudo cp deploy/lineupiq.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable lineupiq
sudo systemctl start lineupiq
```

## 6. Configure Caddy
```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile
# Edit it: replace YOUR_SUBDOMAIN with your actual DuckDNS subdomain
sudo nano /etc/caddy/Caddyfile
sudo systemctl restart caddy
```

## 7. Router port forwarding
Forward ports **80** and **443** to your Pi's local IP.
Find Pi's local IP: `hostname -I`

## 8. Verify
Visit `https://YOUR_SUBDOMAIN.duckdns.org` — should see the login page.

## Updating the app
```bash
cd ~/NFL-AI-Agent
git pull
sudo systemctl restart lineupiq
```
