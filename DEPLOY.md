# Deploying Spectra-Lab on a Raspberry Pi (venv + systemd + Cloudflare)

This runs the app natively in a Python virtual environment, supervised by
systemd, and exposes it on a subdomain through a Cloudflare tunnel — the same
pattern as your existing `ha.tarikhassio.online` setup.

---

## 1. Copy the app to the Pi

From your computer (adjust IP/user to your Pi):

```bash
scp -r soil-app pi@192.168.100.187:/home/pi/
```

Or `git clone` it if you put it in a repo. The folder should contain
`app.py`, `models.py`, `utils.py`, `requirements.txt`, `setup.sh`,
`spectra-lab.service`, and `.streamlit/config.toml`.

---

## 2. Install dependencies

SSH into the Pi and run the setup script:

```bash
ssh pi@192.168.100.187
cd /home/pi/soil-app
bash setup.sh
```

This creates `.venv/` and installs scikit-learn, scipy, streamlit, etc.
On a Pi 5 with prebuilt ARM wheels it takes a couple of minutes. If a wheel
has to compile, the apt packages installed by the script (OpenBLAS, gfortran)
let it succeed.

Quick manual test before setting up the service:

```bash
source .venv/bin/activate
streamlit run app.py
# visit http://192.168.100.187:8501 from your LAN, then Ctrl-C to stop
```

---

## 3. Run it as a service

Edit the three `CHANGE-ME` lines in `spectra-lab.service` (user and the two
paths) if your username or folder differ from `pi` / `/home/pi/soil-app`,
then install it:

```bash
sudo cp spectra-lab.service /etc/systemd/system/spectra-lab.service
sudo systemctl daemon-reload
sudo systemctl enable --now spectra-lab
systemctl status spectra-lab        # should show "active (running)"
```

It now starts on boot and restarts if it crashes. Follow logs with:

```bash
journalctl -u spectra-lab -f
```

The app is listening on `0.0.0.0:8501` (set in `.streamlit/config.toml`).

---

## 4. Expose it via Cloudflare

You already run a Cloudflare tunnel for Home Assistant, so add a hostname to
the **same** tunnel — no new tunnel or port-forward needed.

### Option A — Dashboard (Zero Trust)
1. Cloudflare Zero Trust → **Networks → Tunnels** → your existing tunnel → **Configure**.
2. **Public Hostname → Add a public hostname.**
   - **Subdomain:** `soil` (gives `soil.tarikhassio.online`)
   - **Domain:** `tarikhassio.online`
   - **Type:** `HTTP`
   - **URL:** `localhost:8501`
3. Save. DNS is created automatically; it's live in under a minute.

### Option B — config file (`cloudflared`)
If your tunnel is defined in `~/.cloudflared/config.yml`, add an ingress rule
**above** the catch-all:

```yaml
ingress:
  - hostname: soil.tarikhassio.online
    service: http://localhost:8501
  # ... your existing ha.tarikhassio.online rule ...
  - service: http_status:404
```

Then restart the tunnel:

```bash
sudo systemctl restart cloudflared
```

Visit **https://soil.tarikhassio.online** — Cloudflare handles TLS; Streamlit
runs plain HTTP locally, which is why `enableCORS=false` is set in the config.

---

## 5. (Recommended) Put it behind a login

It's a public URL, so protect it. Cloudflare **Zero Trust → Access →
Applications → Add an application** (self-hosted), point it at
`soil.tarikhassio.online`, and add a policy allowing only your email(s) or a
one-time PIN. This gates the app at the edge before traffic ever reaches the Pi.

---

## Updating the app later

```bash
# copy new files over, then:
sudo systemctl restart spectra-lab
```

If `requirements.txt` changed:

```bash
cd /home/pi/soil-app
source .venv/bin/activate
pip install -r requirements.txt
sudo systemctl restart spectra-lab
```

---

## Troubleshooting

- **Service won't start** → `journalctl -u spectra-lab -e`; usually a wrong path
  in the unit file or the venv not built.
- **502 from Cloudflare** → the service isn't running or the ingress URL/port
  doesn't match `8501`.
- **Slow first model fit** → expected on a Pi for Random Forest / SVR with many
  trees; lower RF trees or untick those models for big datasets.
- **Out-of-memory on large libraries** → Kennard-Stone builds a full N×N distance
  matrix; switch to the random split for several-thousand-sample sets.
