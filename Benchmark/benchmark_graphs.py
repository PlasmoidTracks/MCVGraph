# benchmark_plots_final.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

# Load results
df = pd.read_csv("bench_results.csv")

sns.set(style="whitegrid", context="talk")

# 1. FPS vs. number of points (scatter scaling scenarios)
fps_cols = [c for c in df.columns if c.startswith("fps_canvas")]
df_scaling = df[df["scenario"].str.contains("scatter_scaling")]

plt.figure(figsize=(10, 6))
for scenario, subdf in df_scaling.groupby("scenario"):
    subdf = subdf.groupby("points")[fps_cols].mean().reset_index()
    subdf["fps_mean"] = subdf[fps_cols].mean(axis=1)
    plt.plot(subdf["points"], subdf["fps_mean"], marker="o", label=scenario)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Number of points")
plt.ylabel("Mean FPS across canvases")
plt.title("FPS vs. dataset size (scaling scenarios)")
plt.legend()
# Denser ticks
plt.gca().yaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,4,5,6,7,8,9], numticks=20))
plt.gca().xaxis.set_major_locator(LogLocator(base=10, subs=[1,2,5], numticks=20))
plt.tight_layout()
plt.savefig("fig_fps_scaling.png", dpi=300)
plt.close()

# 2. Latency vs. number of points (one plot per scenario)
df_lat = df.dropna(subset=["select_latency_ms", "clear_latency_ms"])
for scenario, subdf in df_lat.groupby("scenario"):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=subdf, x="points", y="select_latency_ms",
                 marker="o", label="Select latency")
    sns.lineplot(data=subdf, x="points", y="clear_latency_ms",
                 marker="x", linestyle="--", label="Clear latency")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of points")
    plt.ylabel("Latency (ms)")
    plt.title(f"Latency vs. dataset size – {scenario}")
    plt.legend(loc="upper left")
    # Denser ticks
    plt.gca().yaxis.set_major_locator(LogLocator(base=10, subs=[1,2,3,4,5,6,7,8,9], numticks=20))
    plt.gca().xaxis.set_major_locator(LogLocator(base=10, subs=[1,2,5], numticks=20))
    plt.tight_layout()
    fname = f"fig_latency_{scenario}.png".replace(" ", "_")
    plt.savefig(fname, dpi=300)
    plt.close()

# 3. Streaming throughput (FPS, no variance)
df_stream = df[df["scenario"] == "scatter_streaming_updates"].copy()
df_stream["fps"] = df_stream["frames"] / df_stream["duration_s"]

plt.figure(figsize=(8, 6))
sns.barplot(data=df_stream, x="points", y="fps", errorbar=None)
plt.ylabel("Updates per second")
plt.xlabel("Number of points")
plt.title("Streaming throughput")
plt.tight_layout()
plt.savefig("fig_streaming.png", dpi=300)
plt.close()

print("Saved: fig_fps_scaling.png, per-scenario latency plots, fig_streaming.png")
