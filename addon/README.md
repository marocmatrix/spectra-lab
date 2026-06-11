# Spectra-Lab — Home Assistant Add-on

This folder packages Spectra-Lab as a Home Assistant add-on. There are two ways
to install it.

## Method 1 — Add-on repository (install by URL, recommended)

1. Home Assistant → **Settings → Add-ons → Add-on Store**.
2. Top-right **⋮ → Repositories**.
3. Paste: `https://github.com/marocmatrix/spectra-lab`
4. **Add**, then close. **Spectra-Lab** appears in the store under this repo.
5. Click it → **Install** → **Start**, and enable **Show in sidebar**.

HA reads `addon/repository.yaml` and the `addon/spectra-lab/` add-on folder
automatically.

## Method 2 — Local add-on (copy files)

Copy the `spectra-lab/` subfolder into your HA `/addons/` directory, then reload
the Add-on Store (**⋮ → Check for updates**). It shows under **Local add-ons**.
See `spectra-lab/README.md` for details.

## Notes

- Built on a Debian Python base so scikit-learn/scipy install as prebuilt
  aarch64 wheels — no compiling on the Pi (which is why a bare Alpine `pip
  install` failed).
- Ingress is enabled: the app opens in the HA sidebar with no extra port or
  login, and your existing Cloudflare tunnel to HA covers remote access.
