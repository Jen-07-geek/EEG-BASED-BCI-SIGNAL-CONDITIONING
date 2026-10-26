import io, os, tempfile, subprocess
from urllib.parse import urlparse, unquote
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.signal import resample_poly
import streamlit as st
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

try:
    import mne
except ImportError:
    mne = None

try:
    import lameenc
except ImportError:
    lameenc = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

st.set_page_config(page_title="EEG BCI Signal Conditioning", page_icon="〰️", layout="wide")
st.title("EEG-Based BCI Preprocessing Unit")
st.write("Upload an EEG file or fetch a direct downloadable file URL, process it, view graphs, and export results.")
st.info("Educational simulation only. Software gain numerically scales recorded values; it is not physical amplification or a medical device.")
st.caption("Supported: CSV, TSV, TXT, EDF. URL mode needs a direct file URL, not a dataset landing page.")

with st.sidebar:
    st.header("Processing settings")
    gain = st.number_input("Software gain (×)", min_value=0.1, max_value=1000.0, value=5.0, step=0.5)
    seconds = st.number_input("Seconds shown in graphs", min_value=1, max_value=120, value=10, step=1)
    mains = st.selectbox("Mains notch", [50.0, 60.0, 0.0], index=0,
                         format_func=lambda x: "Off" if x == 0 else f"{x:g} Hz")
    low = st.number_input("Overall high-pass cutoff (Hz)", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
    high = st.number_input("Overall low-pass cutoff (Hz)", min_value=10.0, max_value=100.0, value=40.0, step=1.0)

st.subheader("1. Choose dataset source")
source = st.radio("Dataset source", ["Upload a file", "Direct download URL"], horizontal=True)
uploaded = None
url = ""
if source == "Upload a file":
    uploaded = st.file_uploader("Upload EEG dataset", type=["csv", "tsv", "txt", "edf"])
    if uploaded is not None:
        st.session_state.eeg_bytes = uploaded.getvalue()
        st.session_state.eeg_name = uploaded.name
        st.session_state.eeg_source = "upload"
else:
    url = st.text_input("Direct EEG file URL", placeholder="https://example.org/recording.edf")
    st.caption("If the URL points to a webpage, login page, or ZIP archive, download the file manually instead.")
    if st.button("Fetch dataset from URL", type="primary"):
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("Enter a valid http:// or https:// URL.")
            max_bytes = 300 * 1024 * 1024
            chunks, total = [], 0
            progress = st.progress(0, text="Connecting to dataset server...")
            with st.spinner("Downloading EEG file..."):
                with requests.get(url.strip(), stream=True, timeout=(12, 20),
                                  headers={"User-Agent": "Mozilla/5.0 EEG-Student-Project/1.0"},
                                  allow_redirects=True) as resp:
                    resp.raise_for_status()
                    length = resp.headers.get("Content-Length")
                    expected = int(length) if length and length.isdigit() else None
                    if expected and expected > max_bytes:
                        raise ValueError("File exceeds the 300 MB download limit.")
                    content_type = resp.headers.get("Content-Type", "").lower()
                    disposition = resp.headers.get("Content-Disposition", "")
                    for chunk in resp.iter_content(256 * 1024):
                        if chunk:
                            total += len(chunk)
                            if total > max_bytes:
                                raise ValueError("Download stopped: file exceeds 300 MB.")
                            chunks.append(chunk)
                            if expected:
                                progress.progress(min(total / expected, 1.0), text=f"Downloaded {total/1024:.0f} KB of {expected/1024:.0f} KB")
                            else:
                                progress.progress(0, text=f"Downloaded {total/1024:.0f} KB... (server did not provide file size)")
                    data = b"".join(chunks)
            progress.progress(1.0, text="Download complete")

            # Prefer the URL filename, then Content-Disposition, then file signature/content type.
            name = unquote(os.path.basename(urlparse(resp.url).path)) or ""
            if not os.path.splitext(name)[1] and "filename=" in disposition.lower():
                import re
                match = re.search(r'filename\*?=(?:UTF-8\'\')?[\"\']?([^\"\';]+)', disposition, re.I)
                if match:
                    name = unquote(match.group(1).strip())
            ext = os.path.splitext(name)[1].lower()
            if data.startswith(b"0\x00") or data[:8].startswith(b"\x00"):
                pass  # Do not infer EDF from a generic binary signature alone.
            if ext not in (".csv", ".tsv", ".txt", ".edf"):
                if data[:8] == b"0       " or b"EDF" in data[:64]:
                    name = "downloaded_eeg.edf"
                elif "edf" in content_type:
                    name = "downloaded_eeg.edf"
                elif "csv" in content_type or "text/plain" in content_type or "text/csv" in content_type:
                    name = "downloaded_eeg.csv"
                else:
                    raise ValueError(f"Downloaded the response ({len(data)/1024:.1f} KB), but could not identify its format. Use a direct link ending in .edf, .csv, .tsv, or .txt. Server content type: {content_type or 'unknown'}.")
            if not data:
                raise ValueError("The URL returned an empty file.")
            # Catch common landing-page/login responses before trying to parse them as EEG data.
            if data[:500].lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("The URL returned a webpage, not an EEG file. Use the direct recording URL, not a dataset page.")
            st.session_state.eeg_bytes = data
            st.session_state.eeg_name = name
            st.session_state.eeg_source = "url"
            st.success(f"Downloaded {name} ({len(data)/1024:.1f} KB).")
        except Exception as e:
            st.error(f"Could not download URL: {e}")

if "eeg_bytes" not in st.session_state:
    st.session_state.eeg_bytes = None
    st.session_state.eeg_name = None
    st.session_state.eeg_source = None

data = st.session_state.eeg_bytes
filename = st.session_state.eeg_name
if source == "Upload a file" and uploaded is None:
    data, filename = None, None
if source == "Direct download URL" and st.session_state.eeg_source != "url":
    data, filename = None, None

def bandpass(x, lo, hi, fs):
    if not (0 < lo < hi < fs/2):
        raise ValueError(f"Filter {lo:g}–{hi:g} Hz invalid for sampling rate {fs:g} Hz; Nyquist is {fs/2:g} Hz.")
    return sosfiltfilt(butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), x)

def notch_filter(x, freq, fs):
    if freq == 0:
        return x.copy(), "Notch filter disabled."
    if freq >= fs/2:
        return x.copy(), f"Skipped {freq:g} Hz notch because it is at/above Nyquist ({fs/2:g} Hz)."
    b, a = iirnotch(freq, Q=30, fs=fs)
    return filtfilt(b, a, x), f"{freq:g} Hz notch applied."

def read_table(raw_bytes, ext):
    sep = "\t" if ext == ".tsv" else None
    df = pd.read_csv(io.BytesIO(raw_bytes), sep=sep, engine="python", comment="#")
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].notna().any()]
    if not numeric:
        raise ValueError("No numeric columns found. The file must contain numeric EEG samples.")
    return df, numeric


def make_mp3(signal_values, fs, duration_limit=120):
    """Create an audible educational sonification by modulating a 440 Hz carrier."""
    if lameenc is None:
        raise RuntimeError("MP3 export dependency is missing. Run: python -m pip install lameenc")
    x = np.asarray(signal_values, dtype=np.float64)
    if len(x) < 2:
        raise ValueError("Not enough samples to create audio.")
    # Keep MP3 files manageable; use at most 120 seconds of EEG.
    x = x[:min(len(x), int(fs * duration_limit))]
    target_fs = 22050
    # Resample the EEG envelope to standard audio sample rate while preserving duration.
    from fractions import Fraction
    ratio = Fraction(target_fs / fs).limit_denominator(10000)
    envelope = resample_poly(x, ratio.numerator, ratio.denominator)
    if len(envelope) < 2:
        raise ValueError("Not enough samples after audio resampling.")
    envelope = envelope - np.mean(envelope)
    peak = np.max(np.abs(envelope))
    if peak > 0:
        envelope = envelope / peak
    # Amplitude-modulate a tone so low-frequency EEG changes are audible.
    t_audio = np.arange(len(envelope), dtype=np.float64) / target_fs
    carrier = np.sin(2 * np.pi * 440.0 * t_audio)
    audio = (0.15 + 0.75 * (envelope + 1.0) / 2.0) * carrier
    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(target_fs)
    encoder.set_channels(1)
    encoder.set_quality(5)
    # lameenc may return bytearray; Streamlit download_button expects bytes-like data.
    encoded_audio = encoder.encode(pcm.tobytes())
    flushed_audio = encoder.flush()
    return bytes(encoded_audio) + bytes(flushed_audio)

def make_mp4(t_values, raw_values, processed_values, theta_values, alpha_values, beta_values,
             seconds_to_show, fs, gain_value, audio_bytes=None):
    """Render an animated EEG visualization and optionally mux in the EEG sonification soundtrack."""
    if imageio_ffmpeg is None:
        raise RuntimeError("MP4 export dependency is missing. Run: python -m pip install imageio-ffmpeg")
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path

    # Keep rendering reasonably quick for a student project demo.
    fps = 10
    n_frames = min(100, max(30, int(seconds_to_show * fps)))
    total_n = min(len(t_values), max(2, int(seconds_to_show * fs)))
    t = np.asarray(t_values[:total_n])
    signals = [
        (np.asarray(raw_values[:total_n]), "Raw EEG"),
        (np.asarray(processed_values[:total_n]), f"Processed EEG (gain {gain_value:g}×)"),
        (np.asarray(theta_values[:total_n]), "Theta (4–8 Hz)"),
        (np.asarray(alpha_values[:total_n]), "Alpha (8–13 Hz)"),
        (np.asarray(beta_values[:total_n]), "Beta (13–30 Hz)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(10, 8), sharex=True)
    lines = []
    for ax, (values, label) in zip(axes, signals):
        line, = ax.plot([], [], linewidth=0.8)
        lines.append(line)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.25)
        if len(values):
            vmin, vmax = np.nanpercentile(values, [1, 99])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = np.nanmin(values)-1, np.nanmax(values)+1
            pad = max((vmax-vmin)*0.1, 1e-9)
            ax.set_ylim(vmin-pad, vmax+pad)
    axes[-1].set_xlabel("Time (seconds)")
    axes[0].set_title("EEG Signal Conditioning — Animated Visualization")
    for ax in axes:
        ax.set_xlim(float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0])+1/fs)
    fig.tight_layout()

    frame_ends = np.linspace(2, total_n, n_frames, dtype=int)
    def update(frame_index):
        end = frame_ends[frame_index]
        for line, (values, _) in zip(lines, signals):
            line.set_data(t[:end], values[:end])
        return lines

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)
    silent_path = None
    final_path = None
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            silent_path = tmp.name
        anim.save(silent_path, writer=FFMpegWriter(fps=fps, bitrate=1200))
        if audio_bytes:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_audio:
                audio_path = tmp_audio.name
                tmp_audio.write(bytes(audio_bytes))
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_final:
                final_path = tmp_final.name
            # Mux the MP3 sonification into the animated MP4. -shortest keeps durations aligned.
            command = [
                ffmpeg_path, "-y", "-i", silent_path, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                "-shortest", "-movflags", "+faststart", final_path
            ]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            if result.returncode != 0:
                raise RuntimeError("Could not add audio to MP4: " + result.stderr[-700:])
            chosen_path = final_path
        else:
            chosen_path = silent_path
        with open(chosen_path, "rb") as f:
            video_bytes = f.read()
        return video_bytes
    finally:
        for path in (silent_path, final_path, audio_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        plt.close(fig)

if data is None:
    st.markdown("### How to use\n1. Upload a supported EEG file or fetch a direct download URL.\n2. Select the signal channel and specify sampling rate/units when prompted.\n3. Click **Process EEG**.\n4. View and download both graphs and processed CSV.")
    st.stop()

st.success(f"Dataset ready: **{filename}**")
try:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".edf":
        if mne is None:
            raise ValueError("EDF support requires MNE. Install it with: python -m pip install mne")
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose="ERROR")
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks) == 0:
                raise ValueError("No EEG channels found in EDF.")
            channels = [raw.ch_names[i] for i in picks]
            channel = st.selectbox("Select EEG channel", channels)
            fs = float(raw.info["sfreq"])
            signal = raw.get_data(picks=[channel])[0] * 1e6
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass
        time_axis = np.arange(len(signal))/fs
    else:
        table, numeric = read_table(data, ext)
        channel = st.selectbox("Select EEG signal column", numeric)
        fs = float(st.number_input("Sampling rate (Hz)", min_value=1.0, max_value=10000.0, value=160.0, step=1.0))
        units = st.selectbox("Selected column units", ["µV (microvolts)", "mV (millivolts)", "V (volts)", "Already scaled / unknown"])
        vals = pd.to_numeric(table[channel], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if units == "mV (millivolts)": vals = vals * 1000
        elif units == "V (volts)": vals = vals * 1e6
        signal = vals
        time_axis = np.arange(len(signal))/fs

    signal = np.nan_to_num(np.asarray(signal, dtype=float))
    if len(signal) < 100:
        raise ValueError("Fewer than 100 valid samples. Select a longer recording.")
    if not (0 < low < high < fs/2):
        raise ValueError(f"Overall passband is invalid for sampling rate {fs:g} Hz. Need 0 < low < high < Nyquist ({fs/2:g} Hz).")

    if st.button("Process EEG", type="primary", key="process"):
        notched, note = notch_filter(signal, mains, fs)
        filtered = bandpass(notched, low, high, fs)
        amplified = filtered * gain
        theta = bandpass(filtered, 4, 8, fs)
        alpha = bandpass(filtered, 8, 13, fs)
        beta = bandpass(filtered, 13, 30, fs)
        n = min(len(signal), int(seconds*fs))
        t = time_axis[:n]
        st.success("Signal processing completed.")
        st.write(f"**File:** {filename} · **Channel:** {channel} · **Sampling rate:** {fs:g} Hz · **Samples:** {len(signal):,}")
        st.caption(note + " Software gain is numerical scaling, not physical amplification.")

        fig1, ax = plt.subplots(figsize=(12, 5))
        ax.plot(t, signal[:n], label="Raw EEG", linewidth=.8)
        ax.plot(t, amplified[:n], label=f"Processed EEG (software gain {gain:g}×)", linewidth=.8)
        ax.set_title("Graph 1: Raw EEG vs Processed EEG")
        ax.set_xlabel("Time (seconds)"); ax.set_ylabel("Amplitude (input scale; processed is scaled)")
        ax.grid(True, alpha=.3); ax.legend(); fig1.tight_layout()
        st.subheader("Graph 1 — Raw vs Processed EEG")
        st.pyplot(fig1, use_container_width=True)
        b1 = io.BytesIO(); fig1.savefig(b1, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download Graph 1 (PNG)", b1.getvalue(), "Graph1_Raw_vs_Processed_EEG.png", "image/png")

        fig2, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        series = [(signal, "Raw EEG Input", None), (theta, "Theta (4–8 Hz)", "tab:orange"),
                  (alpha, "Alpha (8–13 Hz)", "tab:green"), (beta, "Beta (13–30 Hz)", "tab:red")]
        for ax, (values, title, color) in zip(axes, series):
            kw = {"linewidth": .8, "label": title}
            if color: kw["color"] = color
            ax.plot(t, values[:n], **kw); ax.set_title(title, fontsize=10)
            ax.set_ylabel("Amplitude"); ax.grid(True, alpha=.3); ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time (seconds)")
        fig2.suptitle("Graph 2: Raw EEG and Separate Frequency Bands", y=1.0); fig2.tight_layout()
        st.subheader("Graph 2 — Raw EEG, Theta, Alpha, and Beta")
        st.pyplot(fig2, use_container_width=True)
        b2 = io.BytesIO(); fig2.savefig(b2, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Download Graph 2 (PNG)", b2.getvalue(), "Graph2_Raw_Theta_Alpha_Beta_Separate.png", "image/png")

        out = pd.DataFrame({"Time_seconds": time_axis, "Raw_EEG_input_units": signal,
            "Notch_filtered": notched, "Conditioned_EEG": filtered, "Software_gain_output": amplified,
            "Theta_4_8_Hz": theta, "Alpha_8_13_Hz": alpha, "Beta_13_30_Hz": beta})
        st.download_button("Download processed data (CSV)", out.to_csv(index=False).encode(), "EEG_BCI_Processed_Dataset.csv", "text/csv")

        st.subheader("Additional exports — audio and video")
        st.caption("MP3 is an educational sonification: EEG amplitude modulates an audible 440 Hz tone. MP4 animates the raw, processed, theta, alpha and beta traces. These are visual/audio representations, not diagnostic outputs.")
        export_col1, export_col2 = st.columns(2)
        mp3_bytes = None
        with export_col1:
            try:
                mp3_bytes = make_mp3(filtered, fs)
                st.download_button("Download EEG audio (MP3)", mp3_bytes,
                                   "EEG_Signal_Sonification.mp3", "audio/mpeg")
            except Exception as audio_error:
                st.warning(f"MP3 export unavailable: {audio_error}")
        with export_col2:
            try:
                with st.spinner("Creating animated MP4 with EEG visuals and audio soundtrack… this may take a little while."):
                    mp4_bytes = make_mp4(
                        time_axis, signal, amplified, theta, alpha, beta,
                        seconds, fs, gain, audio_bytes=mp3_bytes
                    )
                st.download_button("Download EEG video with audio (MP4)", mp4_bytes,
                                   "EEG_Signal_Animation_with_Audio.mp4", "video/mp4")
                st.caption("The MP4 contains animated raw/processed/theta/alpha/beta plots plus the EEG sonification soundtrack.")
            except Exception as video_error:
                st.warning(f"MP4 export unavailable: {video_error}")
        with st.expander("Processing details"):
            st.write(f"Overall passband: {low:g}–{high:g} Hz; software gain: {gain:g}×. {note}")
            st.write("Correct sampling rate and units must come from the dataset documentation. Offline filtering is for recorded data, not real-time streaming.")
        plt.close(fig1); plt.close(fig2)
except Exception as e:
    st.error(f"Could not read or process this dataset: {e}")
    st.write("Check file format, channel, units, sampling rate, and filter cutoffs.")
