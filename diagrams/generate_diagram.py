"""
generate_diagram.py
-------------------
Generates the IoT system architecture diagram for the RT-IoT2 project.
Run once to produce system_diagram.png / .svg / .pdf; not part of the ML pipeline.

Dependencies (all pip-installable):
    pip install matplotlib numpy
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Editable color palette ────────────────────────────────────────────────────
COLORS = {
    "orange":  "#FF9500",   # Dashboard / Visualization layer
    "purple":  "#AF52DE",   # Cloud / Storage / Processing layer
    "green":   "#34C759",   # Network / Gateway layer
    "teal":    "#007AFF",   # Perception layer (sensors / actuators)
    "coral":   "#FF453A",   # Edge / Data gateway nodes
    "arrow":   "#000000",
    "bg":      "#FFFFFF",
    "panel":   "#F2F2F7",   # sidebar and legend background
    "label":   "#1C1C1E",
    "gray":    "#6C6C70",
    "divider": "#C6C6C8",
}

# ── Canvas ────────────────────────────────────────────────────────────────────
CANVAS_W, CANVAS_H = 24, 13   # inches — wide for 5 bottom columns
DPI = 300
# Data coords: x in [0,1], y in [0,1]
# We reserve x ∈ [0, 0.07] for the left sidebar layer labels.
# All boxes live in x ∈ [0.075, 1.0].

# ── Editable node definitions ─────────────────────────────────────────────────
# x, y = bottom-left corner in data coords
# w, h = width / height in data coords
NODES = {

    # ── Row 5 — Dashboard / Visualization  (y = 0.830) ───────────────────────
    "dash_thing": dict(
        x=0.080, y=0.830, w=0.415, h=0.130,
        color="orange",
        title="Thing Dashboard",
        sub="DNN Attack Classifier  ·  TensorFlow",
        lines=[
            "Real-time attack-type heatmap & confidence scores",
            "Per-device traffic timeline with alert markers",
            "WebSocket push refresh (250 ms latency)",
            "PDF / CSV export for compliance & audit",
        ],
    ),
    "dash_alert": dict(
        x=0.510, y=0.830, w=0.415, h=0.130,
        color="orange",
        title="Alert & Response System",
        sub="LSTM Flow Duration Predictor  ·  TensorFlow",
        lines=[
            "SNS push notifications + PagerDuty webhook",
            "SIEM integration — Splunk / Elastic SIEM",
            "Auto-block via firewall / WAF API rules",
            "Auto-ticket creation — Jira / ServiceNow",
        ],
    ),

    # ── Row 4 — Cloud / Storage / ML  (y = 0.630) ────────────────────────────
    "cloud_ingest": dict(
        x=0.080, y=0.630, w=0.260, h=0.155,
        color="purple",
        title="Cloud Ingest",
        sub="LoRaWAN-Core  ·  AWS IoT Core",
        lines=[
            "MQTT broker · TLS mutual auth (X.509 certs)",
            "Device shadow registry & IoT rule engine",
            "LoRaWAN NS → S3 / Kinesis fan-out",
            "Throughput: up to 50 K messages / sec",
        ],
    ),
    "data_lake": dict(
        x=0.370, y=0.630, w=0.260, h=0.155,
        color="purple",
        title="Data Lake",
        sub="Amazon S3  +  Athena",
        lines=[
            "Parquet columnar storage (Snappy compression)",
            "Partitioned by date, hour & device ID",
            "Athena SQL for ad-hoc exploration",
            "Lifecycle: hot → warm → Glacier cold",
        ],
    ),
    "ml_proc": dict(
        x=0.660, y=0.630, w=0.265, h=0.155,
        color="purple",
        title="ML Processing",
        sub="Kubeflow  ·  KServe  ·  LangChain",
        lines=[
            "Model 1 — DNN Attack Classifier (83 features)",
            "Model 2 — LSTM Flow Duration Predictor",
            "KServe auto-scaling inference endpoints",
            "LangChain RAG for alert explainability",
        ],
    ),

    # ── Row 3 — Parse / Queue tier  (y = 0.450) ──────────────────────────────
    "mqtt_parser": dict(
        x=0.080, y=0.450, w=0.260, h=0.135,
        color="purple",
        title="MQTT Parser",
        sub="Eclipse Mosquitto  ·  Paho Bridge",
        lines=[
            "Topic normalisation & Avro schema validation",
            "QoS 0 / 1 / 2 relay with session persistence",
            "Protocol support: MQTT 5.0 & MQTT-SN",
        ],
    ),
    "parse_queue": dict(
        x=0.370, y=0.450, w=0.260, h=0.135,
        color="purple",
        title="Parse Queue",
        sub="Apache Kafka  ·  Confluent Schema Registry",
        lines=[
            "Durable ordered log — 7-day retention",
            "Avro-serialised labeled flow records",
            "Consumer groups for ML & storage fanout",
        ],
    ),
    "flow_engine": dict(
        x=0.660, y=0.450, w=0.265, h=0.135,
        color="purple",
        title="Tube Flow Engine",
        sub="Apache Flink  ·  Kinesis Data Analytics",
        lines=[
            "Stateful CEP — detect multi-step attack chains",
            "Feature engineering: IAT, flag counts, byte stats",
            "Sliding-window aggregations: 30 s / 5 min",
        ],
    ),

    # ── Row 2 — Network Capture  (y = 0.305)  full-width ─────────────────────
    "net_capture": dict(
        x=0.080, y=0.305, w=0.845, h=0.105,
        color="green",
        title="Network Capture Layer  —  Zeek IDS + Flowmeter Plugin  ·  Passive network tap",
        sub=None,
        lines=[
            "Extracts 83-feature labeled flow records  ·  Protocols monitored: MQTT (1883), HTTPS, ZigBee, BLE, LoRaWAN",
            "Flowmeter plugin computes IAT, flag counts, byte stats  ·  Dataset output: ~123 K labeled flows (RT-IoT2022)",
        ],
    ),

    # ── Row 1 — Perception / Edge devices  (y = 0.060) ───────────────────────
    "thing_node": dict(
        x=0.080, y=0.060, w=0.148, h=0.200,
        color="teal",
        title="Thing Node",
        sub="ThingSpeak LED Sensor",
        lines=[
            "DHT11 temp / humidity",
            "WiFi 802.11b/g/n",
            "ESP8266 MCU",
            "Sample rate: 1 Hz",
            "Location: smart-home",
        ],
    ),
    "actuator_node": dict(
        x=0.243, y=0.060, w=0.148, h=0.200,
        color="teal",
        title="Actuator Node",
        sub="Wipro Smart Bulb",
        lines=[
            "Power & lux telemetry",
            "ZigBee / BLE 4.2",
            "ARM Cortex-M0+",
            "Sample rate: 0.5 Hz",
            "Location: office lighting",
        ],
    ),
    "mqtt_broker": dict(
        x=0.406, y=0.060, w=0.148, h=0.200,
        color="teal",
        title="MQTT Broker Node",
        sub="Mosquitto / Raspberry Pi Zero",
        lines=[
            "DS18B20 temperature",
            "Ethernet / WiFi",
            "Raspberry Pi Zero",
            "Sample rate: 5 Hz",
            "Location: industrial rack",
        ],
    ),
    "data_gw": dict(
        x=0.569, y=0.060, w=0.148, h=0.200,
        color="coral",
        title="Data Gateway",
        sub="Raspberry Pi 4 / Jetson Nano",
        lines=[
            "Local MQTT broker",
            "libpcap packet sniffing",
            "Threshold anomaly filter",
            "4-core ARM · 4 GB RAM",
            "TLS 1.3 · MQTT-S upload",
        ],
    ),
    "iot_sensors": dict(
        x=0.777, y=0.060, w=0.148, h=0.200,
        color="teal",
        title="IoT Sensors",
        sub="Amazon Alexa / Voice Hub",
        lines=[
            "Microphone array",
            "WiFi 802.11ac",
            "ARM Cortex-A9",
            "Event-driven sampling",
            "Location: living room",
        ],
    ),
}

# ── Arrow definitions ──────────────────────────────────────────────────────────
# (x1, y1, x2, y2, label, color_key)
# Coordinates target box edges directly.
ARROWS = [
    # Perception → Network Capture (bottom edge y=0.305)
    (0.154, 0.260, 0.154, 0.305, "WiFi",       "green"),
    (0.317, 0.260, 0.270, 0.305, "ZigBee/BLE", "green"),
    (0.480, 0.260, 0.480, 0.305, "MQTT/ETH",   "green"),
    (0.643, 0.260, 0.560, 0.305, "LAN/TLS",    "green"),
    (0.851, 0.260, 0.851, 0.305, "WiFi",       "green"),
    # Network Capture → Parse tier (top edge y=0.410)
    (0.210, 0.410, 0.210, 0.450, "raw flows",  "purple"),
    (0.500, 0.410, 0.500, 0.450, "raw flows",  "purple"),
    (0.792, 0.410, 0.792, 0.450, "raw flows",  "purple"),
    # Parse tier → Cloud/Storage/ML (top of parse = 0.585, bottom of cloud = 0.630)
    (0.210, 0.585, 0.210, 0.630, "MQTT-S/TLS", "purple"),
    (0.500, 0.585, 0.500, 0.630, "S3 PUT",     "purple"),
    (0.792, 0.585, 0.792, 0.630, "features",   "purple"),
    # Cloud Ingest → Data Lake (horizontal, y = 0.715 mid of row 4)
    (0.340, 0.715, 0.370, 0.715, "S3 PUT",     "purple"),
    # Data Lake → ML Processing (horizontal)
    (0.630, 0.715, 0.660, 0.715, "batch",      "purple"),
    # Cloud Ingest → Thing Dashboard
    (0.210, 0.785, 0.210, 0.830, "predictions","orange"),
    # ML Processing → Alert System
    (0.792, 0.785, 0.792, 0.830, "alerts",     "orange"),
    # Data Lake → Thing Dashboard (diagonal)
    (0.500, 0.785, 0.370, 0.830, "CSV export", "orange"),
]

# ── Sidebar layer labels ───────────────────────────────────────────────────────
# (y_center, label, color_key)
LAYER_LABELS = [
    (0.160, "Perception\nLayer",        "teal"),
    (0.358, "Network\nCapture",         "green"),
    (0.517, "Parse /\nQueue",           "purple"),
    (0.707, "Cloud /\nStorage /\nML",   "purple"),
    (0.895, "Dashboard\n& Alerts",      "orange"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_node(ax, node: dict):
    """Flat rounded box: solid fill, white border, bold title, divider, bullets."""
    x, y, w, h = node["x"], node["y"], node["w"], node["h"]
    color = COLORS[node["color"]]

    # Box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.010",
        facecolor=color,
        edgecolor="white",
        linewidth=2.2,
        zorder=3,
    )
    ax.add_patch(box)

    # ── Title area: top 28 % of box height ──────────────────────────────────
    title_h = h * 0.28
    title_y = y + h - title_h

    # White horizontal divider line below title
    ax.plot(
        [x + 0.008, x + w - 0.008],
        [title_y, title_y],
        color="white", lw=1.0, alpha=0.55, zorder=4,
    )

    # Title text (centered in title band)
    ax.text(
        x + w / 2, title_y + title_h / 2,
        node["title"],
        ha="center", va="center",
        fontsize=8.8, fontweight="bold",
        color="white", zorder=5,
    )

    # ── Sub-title (first line below divider) ─────────────────────────────────
    body_top = title_y - 0.008
    if node.get("sub"):
        ax.text(
            x + w / 2, body_top,
            node["sub"],
            ha="center", va="top",
            fontsize=6.6, style="italic",
            color="white", alpha=0.90, zorder=5,
        )
        body_top -= 0.026

    # ── Bullet lines ──────────────────────────────────────────────────────────
    n = len(node["lines"])
    available = body_top - (y + 0.010)
    line_gap = min(0.026, available / max(n, 1))
    for i, line in enumerate(node["lines"]):
        ax.text(
            x + 0.012, body_top - i * line_gap,
            f"• {line}",
            ha="left", va="top",
            fontsize=6.3, color="white", zorder=5,
        )


def draw_arrow(ax, x1, y1, x2, y2, label="", color_key="arrow"):
    """Thin directed arrow with a small italic label beside the midpoint."""
    color = "#000000"  # all arrows are black
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=1.5,
            mutation_scale=11,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=6,
    )
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        # Place label to the right for vertical arrows, above for horizontal
        if abs(x2 - x1) < 0.01:          # vertical
            ax.text(mx + 0.012, my, label, ha="left", va="center",
                    fontsize=5.8, color=color, style="italic", zorder=7)
        else:                              # horizontal / diagonal
            ax.text(mx, my + 0.012, label, ha="center", va="bottom",
                    fontsize=5.8, color=color, style="italic", zorder=7)


# ─────────────────────────────────────────────────────────────────────────────
# Build canvas
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("auto")
ax.axis("off")
fig.patch.set_facecolor(COLORS["bg"])
ax.set_facecolor(COLORS["bg"])

# ── Title block ───────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.980,
    "RT-IoT2  ·  IoT Intrusion Detection System Architecture",
    ha="center", va="top",
    fontsize=17, fontweight="bold", color=COLORS["label"],
)
fig.text(
    0.5, 0.960,
    "RT-IoT2022 Dataset  ·  DNN Attack Classifier  +  LSTM Flow Duration Predictor",
    ha="center", va="top",
    fontsize=9.5, color=COLORS["gray"], style="italic",
)

# ── Left sidebar: layer labels ────────────────────────────────────────────────
# Draw a light panel behind the sidebar
sidebar = FancyBboxPatch(
    (0.0, 0.04), 0.072, 0.940,
    boxstyle="round,pad=0.005",
    facecolor=COLORS["panel"], edgecolor=COLORS["divider"],
    linewidth=0.8, zorder=2,
)
ax.add_patch(sidebar)

for y_c, lbl, col_key in LAYER_LABELS:
    col = COLORS[col_key]
    # Colored accent bar
    bar = FancyBboxPatch(
        (0.004, y_c - 0.060), 0.008, 0.120,
        boxstyle="round,pad=0.002",
        facecolor=col, edgecolor="none", zorder=3,
    )
    ax.add_patch(bar)
    ax.text(
        0.020, y_c, lbl,
        ha="left", va="center",
        fontsize=7.2, color=COLORS["label"],
        fontweight="bold",
        zorder=4,
    )

# ── Draw all nodes ────────────────────────────────────────────────────────────
for node in NODES.values():
    draw_node(ax, node)

# ── Draw all arrows ───────────────────────────────────────────────────────────
for x1, y1, x2, y2, lbl, col_key in ARROWS:
    draw_arrow(ax, x1, y1, x2, y2, lbl, col_key)

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=COLORS["teal"],   label="Perception Layer — Sensors & Actuators"),
    mpatches.Patch(color=COLORS["coral"],  label="Edge / Data Gateway"),
    mpatches.Patch(color=COLORS["green"],  label="Network Capture Layer"),
    mpatches.Patch(color=COLORS["purple"], label="Cloud Storage & ML Processing"),
    mpatches.Patch(color=COLORS["orange"], label="Dashboard & Alert Layer"),
]
leg = ax.legend(
    handles=legend_items,
    loc="lower right",
    fontsize=8,
    framealpha=1.0,
    edgecolor=COLORS["divider"],
    facecolor="white",
    bbox_to_anchor=(0.998, -0.06),
    title="Layer Legend",
    title_fontsize=8.5,
)
leg.get_title().set_fontweight("bold")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
BASE = "system_diagram"  # saved into diagrams/ alongside this script
plt.tight_layout(rect=[0, 0.01, 1, 0.945])

plt.savefig(f"{BASE}.png", dpi=DPI, bbox_inches="tight", facecolor=COLORS["bg"])
print(f"Saved {BASE}.png  ({DPI} dpi)")

plt.savefig(f"{BASE}.svg", format="svg", bbox_inches="tight", facecolor=COLORS["bg"])
print(f"Saved {BASE}.svg")

plt.savefig(f"{BASE}.pdf", format="pdf", bbox_inches="tight", facecolor=COLORS["bg"])
print(f"Saved {BASE}.pdf")
