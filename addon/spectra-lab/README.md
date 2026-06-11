# Spectra-Lab — Home Assistant Add-on

Soil property prediction from spectra (pXRF / vis-NIR / MIR), running as a Home
Assistant add-on. Opens directly in the HA sidebar via ingress.

## Install (local add-on)

1. On the machine running Home Assistant, copy the **`spectra-lab`** folder into
   the `/addons` directory (the "local add-ons" folder). Easiest ways:
   - **Samba / SSH add-on:** drop the folder into `\\<HA-IP>\addons\` or
     `/addons/` over SSH.
   - **VS Code / File editor add-on:** create `/addons/spectra-lab/` and add the
     files.

   The final layout must be:
   ```
   /addons/spectra-lab/
     config.yaml
     Dockerfile
     run.sh
     app.py
     models.py
     utils.py
     requirements.txt
   ```

2. In Home Assistant: **Settings → Add-ons → Add-on Store**, click the **⋮**
   menu (top-right) → **Check for updates** (or **Reload**). The
   **Spectra-Lab** add-on appears under **Local add-ons**.

3. Click it → **Install**. The first build downloads Python wheels and takes a
   few minutes on a Pi 5. (It builds on a Debian base, so scikit-learn/scipy
   install as prebuilt aarch64 wheels — no compiling, which is why the earlier
   Alpine pip install failed.)

4. **Start** the add-on. Enable **"Show in sidebar"** to pin it.

5. Open it from the sidebar — ingress handles the URL and auth, so no extra
   port or login is needed.

## Direct LAN access (optional)

The add-on also exposes port **8501**, so you can reach it at
`http://<HA-IP>:8501` too. Remove the port mapping in the add-on's
**Configuration** tab if you only want sidebar/ingress access.

## Exposing it externally

Since it's in the HA sidebar behind ingress, your existing Cloudflare tunnel to
Home Assistant already covers it — open HA remotely and the Spectra-Lab panel is
right there, no separate hostname required.

## Persistent storage

The add-on maps `/share` (read/write). Exported `.joblib` model bundles you
download go through your browser as normal; if you want server-side storage,
point exports at `/share/spectra-lab`.

## Updating

Bump `version` in `config.yaml`, replace the changed files in
`/addons/spectra-lab/`, then **Reload** the store and use the add-on's
**Rebuild** option.
