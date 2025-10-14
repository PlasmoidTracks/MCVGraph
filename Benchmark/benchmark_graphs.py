# benchmark_plots_final.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LogLocator

# Ensure output directory exists
output_dir = os.path.join(".", "Benchmark", "graphs")
os.makedirs(output_dir, exist_ok=True)

# Load results
df = pd.read_csv("bench_results.csv")

sns.set(style="whitegrid", context="talk")

# 1. FPS vs. number of points (scatter scaling scenarios)
fps_cols = [c for c in df.columns if c.startswith("fps_canvas")]
df_scaling = df[df["scenario"].str.contains("scatter_scaling")]

order = ["scatter_scaling_3_canvases", "scatter_scaling_6_canvases", "scatter_scaling_12_canvases"]
alphas = [0.2, 0.2, 0.125]

plt.figure(figsize=(10, 6))
for scenario, alpha in zip(order, alphas):
    if scenario not in df_scaling["scenario"].unique():
        continue
    subdf = df_scaling[df_scaling["scenario"] == scenario]
    grouped = subdf.groupby("points")[fps_cols]
    stats = pd.DataFrame({
        "points": grouped.mean().index,
        "fps_mean": grouped.mean().mean(axis=1),
        "fps_min": grouped.min().min(axis=1),
        "fps_max": grouped.max().max(axis=1),
    }).reset_index(drop=True)
    
    line, = plt.plot(stats["points"], stats["fps_mean"], marker="o", label=scenario)
    color = line.get_color()
    plt.fill_between(
        stats["points"],
        stats["fps_min"],
        stats["fps_max"],
        color=color,
        alpha=alpha,
    )

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
plt.savefig(os.path.join(output_dir, "fig_fps_scaling.png"), dpi=300)
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
    fname = os.path.join(output_dir, f"fig_latency_{scenario}.png".replace(" ", "_"))
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
plt.savefig(os.path.join(output_dir, "fig_streaming.png"), dpi=300)
plt.close()

print("Saved: fig_fps_scaling.png, per-scenario latency plots, fig_streaming.png")
