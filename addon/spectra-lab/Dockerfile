# HA add-on Dockerfile — Debian-based Python so scikit-learn/scipy install as
# prebuilt aarch64 wheels (no compiling on the Pi). We deliberately do NOT use
# the Alpine HA base image here, to avoid the musl/wheel problem.
FROM python:3.12-slim

# curl is needed by run.sh to query the Supervisor ingress entry
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# App dependencies (prebuilt wheels on Debian/glibc)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# App code
WORKDIR /app
COPY app.py models.py utils.py /app/
COPY run.sh /run.sh
RUN chmod +x /run.sh

CMD ["/run.sh"]
