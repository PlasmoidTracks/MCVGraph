# main_sonified_scatter.py
#
# Standalone demo:
# - Generates 2D scatter data that *encodes* a 1D audio waveform (chirp + short harmonic burst) mixed with noise.
# - One ScatterPlot shows the 2D points.
# - Multiple LinePlots each apply a different transform from the same 2D data → 1D waveform.
#   * "Passthrough (random order)" sounds like noise (because points are shuffled).
#   * "Sorted by X (reveals signal)" restores the original time order → audible ascending tone + burst.
#   * "Sorted by X + smoothed" further denoises the signal for clarity.
#
# Click View → Layer Actions → "Play waveform" on a LinePlot to hear it.
# (Requires sounddevice installed for playback.)

import os
import sys
import numpy as np
from PyQt5 import QtWidgets

# Adjust import path relative to this file so "MCVGraph" package is found (same approach as your example).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MCVGraph import DataSource, ScatterPlot, LinePlot, HeatmapPlot, Canvas


# -----------------------------
# Data synthesis (2D scatter; polar encoding)
# -----------------------------
def synth_scatter_polar_fm(
    *,
    seed: int = 7,
    fs: int = 22050,           # audio sample rate for playback
    duration_s: float = 2.5,   # total duration encoded in the scatter
    f_carrier: float = 330.0,  # FM carrier frequency (Hz)
    f_mod: float = 3.7,        # FM modulator frequency (Hz)
    beta: float = 8.0,         # FM modulation index (radians)
    sweep_hz: float = 120.0,   # slow linear sweep added to carrier (Hz over whole duration)
    r_min: float = 0.05,       # inner radius (avoid origin singularity)
    r_max: float = 1.0,        # outer radius
    jitter_r: float = 0.002,   # radial jitter
    jitter_xy: float = 0.004,  # Cartesian jitter (applied after polar->Cartesian)
) -> tuple[np.ndarray, int]:
    """
    Generate an FM sound, normalize to [0,1], and encode as a 2D scatter where:
      - time is the distance from origin, r ∈ [r_min, r_max]
      - amplitude (normalized 0..1) is the angle θ ∈ [-π, π]
    Returns:
        data: np.ndarray of shape (N, 2) with columns (x, y), row order shuffled.
        fs:   sample rate for playback.
    """
    rng = np.random.default_rng(seed)

    N = int(fs * duration_s)
    t = np.arange(N, dtype=np.float64) / fs
    T = duration_s

    # FM audio: y(t) = sin(2π (f_c t + 0.5*sweep*t^2) + β sin(2π f_m t))
    # Add a gentle amplitude envelope to avoid clicks.
    sweep = sweep_hz / T
    phase_carrier = 2.0 * np.pi * (f_carrier * t + 0.5 * sweep * t * t)
    phase_mod = beta * np.sin(2.0 * np.pi * f_mod * t)
    y = np.sin(phase_carrier + phase_mod)

    # Optional gentle attack/release (Hann fade 5% at ends)
    fade = max(1, int(0.05 * N))
    if fade > 1:
        win = np.hanning(2 * fade)
        env = np.ones_like(y)
        env[:fade] = win[:fade]
        env[-fade:] = win[-fade:]
        y = y * env

    # Normalize audio to [0,1] for angle encoding
    a01 = 0.5 * (y / max(1e-12, np.max(np.abs(y))) + 1.0)
    a01 = np.clip(a01, 0.0, 1.0)

    # Map time -> radius (monotone)
    r = r_min + (r_max - r_min) * (t / T)
    r = r + rng.normal(0.0, jitter_r, size=N)

    # Map amplitude -> angle θ in [-π, π]; θ = 2π*a01 - π
    theta = 2.0 * np.pi * a01 - np.pi

    # Polar -> Cartesian, with small jitter
    x = r * np.cos(theta)
    y_scatter = r * np.sin(theta)
    x += rng.normal(0.0, jitter_xy, size=N)
    y_scatter += rng.normal(0.0, jitter_xy, size=N)

    # Shuffle rows to hide time in row order
    perm = rng.permutation(N)
    data = np.column_stack([x, y_scatter])[perm]

    return data, fs


# -----------------------------
# Transforms for LinePlots
# -----------------------------
def transform_passthrough_y(data: np.ndarray) -> np.ndarray:
    """
    Naive baseline: use the raw y coordinate. With polar encoding this is not the audio amplitude.
    """
    return data[:, 1]

def argsort_identity(data: np.ndarray) -> np.ndarray:
    return np.arange(len(data), dtype=int)

def argsort_by_x(data: np.ndarray) -> np.ndarray:
    """
    Sort by x coordinate. This is NOT the true time for polar encoding, but useful for comparison.
    """
    return np.argsort(data[:, 0], kind="mergesort")

def argsort_by_radius(data: np.ndarray) -> np.ndarray:
    """
    Recover time by sorting by radius r = sqrt(x^2 + y^2), which encodes time monotonically.
    """
    x = data[:, 0]
    y = data[:, 1]
    r = np.hypot(x, y)
    # mergesort is stable and matches prior code style
    return np.argsort(r, kind="mergesort")

def transform_angle_to_audio_amplitude(data: np.ndarray) -> np.ndarray:
    """
    Decode amplitude from angle:
        θ = atan2(y, x) in (-π, π], a01 = (θ + π) / (2π), audio = 2*a01 - 1
    Output is per-row (unsorted) so that argsort_by_radius can impose time order.
    """
    x = data[:, 0]
    y = data[:, 1]
    theta = np.arctan2(y, x)
    a01 = (theta + np.pi) / (2.0 * np.pi)      # in [0,1]
    audio = 2.0 * a01 - 1.0                    # back to [-1,1]
    return audio.astype(np.float64)


# -----------------------------
# Spectrogram utilities
# -----------------------------
def _spectrogram_matrix(wave: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256) -> np.ndarray:
    wave = np.asarray(wave, dtype=np.float64).reshape(-1)
    if wave.size == 0:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)

    # Zero-pad to at least one full frame
    if wave.size < n_fft:
        pad = n_fft - wave.size
        wave = np.pad(wave, (0, pad), mode="constant")

    win = np.hanning(n_fft).astype(np.float64)
    n_frames = 1 + (wave.size - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1

    mags = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = wave[start:start + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size), mode="constant")
        X = np.fft.rfft(frame * win, n=n_fft)
        mags[i] = np.abs(X)

    # Return as (freq_bins, time_frames) for HeatmapPlot (H x W)
    return mags.T.astype(np.float32)

def log_normalize_spec(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    arr = np.maximum(arr, 1e-12)
    log_arr = np.log1p(arr)
    vmin = np.nanmin(log_arr)
    vmax = np.nanmax(log_arr)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros_like(log_arr, dtype=np.float32)
    return ((log_arr - vmin) / (vmax - vmin)).astype(np.float32)

def make_spectrogram_transform(transform_fn, argsort_fn, fs: int, n_fft: int = 1024, hop: int = 256):
    """
    Build a HeatmapPlot.transform for a given LinePlot's (transform, argsort),
    so the spectrogram matches what the LinePlot actually plays/displays.
    """
    def _tx(data: np.ndarray) -> np.ndarray:
        y_view = np.asarray(transform_fn(data)).reshape(-1)
        order = np.asarray(argsort_fn(data), dtype=int)
        order = order[(order >= 0) & (order < y_view.size)]
        if order.size != y_view.size:
            order = np.arange(y_view.size, dtype=int)
        y_plot = y_view[order]
        spec = _spectrogram_matrix(y_plot, fs=fs, n_fft=n_fft, hop=hop)
        # spec is (freq_bins, time_frames).
        # Transpose → (time_frames, freq_bins), then flip freq so 0Hz at bottom.
        spec_tf = spec.T[:, ::1]
        return spec_tf
    return _tx


# -----------------------------
# Main App
# -----------------------------
def main():
    # 1) Make the polar-encoded data
    data, fs = synth_scatter_polar_fm(
        seed=7,
        fs=11025,
        duration_s=2,
        f_carrier=330.0,
        f_mod=500,
        beta=3.0,
        sweep_hz=220.0,
        r_min=0.00,
        r_max=1.0,
        jitter_r=3e-5*0,
        jitter_xy=3e-5*0,
    )
    ds = DataSource(data)

    # 2) Qt app
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    # 3) Scatter (2D)
    canvas_scatter = Canvas()
    canvas_scatter.set_graph_name("Scatter: Polar-encoded Sound (time=r, amp=angle)")
    scatter = ScatterPlot(ds, focus=True, color_non_selected="gray")
    scatter.set_transform(lambda d: d[:, 0:2])
    canvas_scatter.plot(scatter)
    canvas_scatter.set_axis_label("x", "x = r cos(θ)")
    canvas_scatter.set_axis_label("y", "y = r sin(θ)")
    canvas_scatter.show()

    # 4) LinePlot A (naive): raw y with current row order → meaningless / noise-like
    canvas_lp_naive = Canvas()
    canvas_lp_naive.set_graph_name("Line A: Naive y (row order)")
    lp_naive = LinePlot(
        data_source=ds,
        sample_rate=fs,
        transform=transform_passthrough_y,     # raw y
        argsort_func=argsort_by_x,         # current (shuffled) order
        name="NaiveY",
        selection_color="red",
        focus=True,
    )
    canvas_lp_naive.plot(lp_naive)
    canvas_lp_naive.set_axis_label("x", "Time (s) — current row order")
    canvas_lp_naive.set_axis_label("y", "Value (raw y, not amplitude)")
    # Spectrogram A
    n_fft = 1024
    hop = 256
    freq_bins = n_fft // 2 + 1
    spec_tx_naive = make_spectrogram_transform(transform_passthrough_y, argsort_identity, fs, n_fft=n_fft, hop=hop)
    heat_naive = HeatmapPlot(
        data_source=ds,
        transform=spec_tx_naive,
        normalizer=log_normalize_spec,
        scale_x=hop / fs,
        scale_y=2.0 / freq_bins,
        graph_name="Spectrogram (naive y)",
    )
    heat_naive.set_translation(0.0, -1.0)
    canvas_lp_naive.plot(heat_naive)
    canvas_lp_naive.show()

    # 5) LinePlot B (decoded): order by radius; amplitude from angle
    canvas_lp_dec = Canvas()
    canvas_lp_dec.set_graph_name("Line B: Decoded (time=r sort, amp=angle→[-1,1])")
    lp_dec = LinePlot(
        data_source=ds,
        sample_rate=fs,
        transform=transform_angle_to_audio_amplitude,  # decode θ→amplitude
        argsort_func=argsort_by_radius,                # impose time via r
        name="DecodedPolar",
        selection_color="lime",
        focus=True,
    )
    canvas_lp_dec.plot(lp_dec)
    canvas_lp_dec.set_axis_label("x", "Time (s) — recovered by sorting radius r")
    canvas_lp_dec.set_axis_label("y", "Amplitude (decoded from angle)")

    # Spectrogram B
    n_fft = 1024
    hop = 256
    freq_bins = n_fft // 2 + 1
    spec_tx_dec = make_spectrogram_transform(transform_angle_to_audio_amplitude, argsort_by_radius, fs, n_fft=n_fft, hop=hop)
    heat_dec = HeatmapPlot(
        data_source=ds,
        transform=spec_tx_dec,
        normalizer=log_normalize_spec,
        scale_x=hop / fs,
        scale_y=2.0 / freq_bins,
        graph_name="Spectrogram (decoded)",
    )
    heat_dec.set_translation(0.0, -1.0)
    canvas_lp_dec.plot(heat_dec)

    canvas_lp_dec.show()

    # Run
    app.exec()


if __name__ == "__main__":
    main()
