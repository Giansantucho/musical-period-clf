"""
Para cada una de las 3 obras (una por período), genera:

  1) <label>_tiempo_frecuencia.png
     - Señal de audio en el dominio del tiempo
     - Espectrograma (STFT, Hann) en el dominio tiempo-frecuencia
     - Transcripción automática: nota dominante detectada por frame
       (pico espectral más fuerte de cada ventana, mapeado a la nota
       más cercana)

  2) <label>_fft_global.png
     - FFT estándar de la señal completa (sin resolución temporal):
       qué notas aparecen en todo el fragmento, con picos detectados
     - Barras con el top 10 de notas por energía relativa

No depende de preprocess_maestro.py / train_mlp.py ni los modifica.

Uso:
    python scripts/analisis_tiempo_frecuencia_maestro.py
"""

from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- Paths -----------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "maestro-v3.0.0"
RESULTS_DIR = REPO_ROOT / "results"

SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512

NOTE_FREQ_MIN = 60.0     # ~B1, debajo del registro práctico de piano
NOTE_FREQ_MAX = 2100.0   # ~C7, cubre casi todo el teclado

# --- Obras, una por período --------------------------------------------------

OBRAS = [
    {
        "label": "Barroco",
        "slug": "barroco_bach_bwv846",
        "subtitle": "J. S. Bach — Preludio y Fuga en Do M., BWV 846",
        "path": RAW_DIR / "2014" / "MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.wav",
        "offset": 0.0,
        "duration": 15.0,
    },
    {
        "label": "Clasicismo",
        "slug": "clasicismo_mozart_k330",
        "subtitle": "W. A. Mozart — Sonata en Do M., K. 330",
        "path": RAW_DIR / "2015" / "MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--2.wav",
        "offset": 0.0,
        "duration": 15.0,
    },
    {
        "label": "Romanticismo",
        "slug": "romanticismo_chopin_sonata3",
        "subtitle": "F. Chopin — Sonata No. 3",
        "path": RAW_DIR / "2006" / "MIDI-Unprocessed_20_R1_2006_01-04_ORIG_MID--AUDIO_20_R1_2006_04_Track04_wav.wav",
        "offset": 60.0,
        "duration": 15.0,
    },
]


# --- Helpers -----------------------------------------------------------------

def load_fragment(obra):
    y, sr = librosa.load(str(obra["path"]), sr=SAMPLE_RATE, mono=True,
                          offset=obra["offset"], duration=obra["duration"])
    if y.size == 0:
        raise ValueError(f"Audio vacío para {obra['label']} ({obra['path']})")
    return y, sr


def detect_notes_per_frame(S_mag, freqs, silence_ratio=0.08):
    """Para cada frame (columna), devuelve la nota del pico espectral más
    fuerte dentro de [NOTE_FREQ_MIN, NOTE_FREQ_MAX], o None si el frame
    está por debajo del umbral de silencio relativo al máximo global."""
    band_mask = (freqs >= NOTE_FREQ_MIN) & (freqs <= NOTE_FREQ_MAX)
    band_freqs = freqs[band_mask]
    band_mag = S_mag[band_mask, :]

    global_max = band_mag.max()
    threshold = global_max * silence_ratio

    notes = []
    for frame in band_mag.T:
        peak_idx = np.argmax(frame)
        peak_mag = frame[peak_idx]
        if peak_mag < threshold:
            notes.append(None)
            continue
        freq = band_freqs[peak_idx]
        note = librosa.hz_to_note(freq)
        notes.append(note)
    return notes


def note_sort_key(note_name):
    """Clave de orden musical (grave -> agudo) para nombres tipo 'C4'."""
    midi = librosa.note_to_midi(note_name)
    return midi


def plot_tiempo_frecuencia(obra, y, sr, out_path):
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    S_mag = np.abs(D)
    S_db = librosa.amplitude_to_db(S_mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    times = librosa.frames_to_time(np.arange(S_mag.shape[1]), sr=sr, hop_length=HOP_LENGTH)

    notes_per_frame = detect_notes_per_frame(S_mag, freqs)
    present_notes = sorted({n for n in notes_per_frame if n is not None}, key=note_sort_key)
    note_to_y = {n: i for i, n in enumerate(present_notes)}

    fig, (ax_wave, ax_spec, ax_notes) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True,
        gridspec_kw={"height_ratios": [1, 1.6, 1.3]},
    )

    # 1) Señal en el tiempo
    t_wave = np.arange(len(y)) / sr
    ax_wave.plot(t_wave, y, linewidth=0.6, color="#6a5acd")
    ax_wave.set_ylabel("Amplitud")
    ax_wave.set_title(f"{obra['label']} — Análisis espectral y detección de notas\n"
                       f"{obra['subtitle']}  ·  STFT + ventana de Hann", fontsize=12)

    # 2) Espectrograma
    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="linear",
        cmap="magma", ax=ax_spec,
    )
    ax_spec.set_ylim(0, NOTE_FREQ_MAX)
    ax_spec.set_ylabel("Frecuencia (Hz)")
    fig.colorbar(img, ax=ax_spec, format="%+2.0f dB", pad=0.01, label="Magnitud (dB)")

    # 3) Transcripción: nota dominante por frame
    xs, ys = [], []
    for t, n in zip(times, notes_per_frame):
        if n is None:
            continue
        xs.append(t)
        ys.append(note_to_y[n])
    ax_notes.scatter(xs, ys, s=8, color="#e94f8a")
    ax_notes.set_yticks(range(len(present_notes)))
    # Si hay muchas notas distintas (rango amplio), mostrar solo 1 de cada
    # 2-3 etiquetas para que no se superpongan, sin tocar los puntos graficados.
    n_notes = len(present_notes)
    label_step = 1 if n_notes <= 15 else (2 if n_notes <= 30 else 3)
    thinned_labels = [
        note if i % label_step == 0 else ""
        for i, note in enumerate(present_notes)
    ]
    ax_notes.set_yticklabels(thinned_labels, fontsize=7)
    ax_notes.set_xlabel("Tiempo (s)")
    ax_notes.set_ylabel("Nota")
    ax_notes.set_title("Transcripción automática — nota dominante por frame", fontsize=10)
    ax_notes.grid(axis="x", alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado: {out_path}")


def plot_fft_global(obra, y, sr, out_path):
    # FFT estándar de la señal completa (sin ventaneo en el tiempo: una sola
    # FFT de todo el fragmento -> sin resolución temporal, solo "qué aparece")
    n = len(y)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    band_mask = (freqs >= NOTE_FREQ_MIN) & (freqs <= NOTE_FREQ_MAX)
    band_freqs = freqs[band_mask]
    band_spec = spectrum[band_mask]

    freq_resolution = sr / n
    min_separation_hz = 15.0
    distance = max(1, int(min_separation_hz / freq_resolution))

    peak_idx, _ = find_peaks(
        band_spec,
        height=band_spec.max() * 0.03,
        distance=distance,
        prominence=band_spec.max() * 0.01,
    )

    peak_freqs = band_freqs[peak_idx]
    peak_mags = band_spec[peak_idx]
    peak_notes = [librosa.hz_to_note(f) for f in peak_freqs]

    # Top 10 por magnitud
    order = np.argsort(peak_mags)[::-1][:10]
    top_freqs = peak_freqs[order]
    top_mags = peak_mags[order]
    top_notes = [peak_notes[i] for i in order]
    top_pct = 100 * top_mags / top_mags.max()

    fig, (ax_fft, ax_bar) = plt.subplots(2, 1, figsize=(11, 7.5))

    ax_fft.plot(band_freqs, band_spec, linewidth=0.8, color="#6a5acd")
    ax_fft.scatter(top_freqs, top_mags, color="#e94f8a", zorder=5, s=25)
    for f, m, note in zip(top_freqs, top_mags, top_notes):
        ax_fft.annotate(note, (f, m), textcoords="offset points", xytext=(0, 6),
                         fontsize=8, ha="center", color="#333333")
    ax_fft.set_xlim(NOTE_FREQ_MIN, NOTE_FREQ_MAX)
    ax_fft.set_xlabel("Frecuencia (Hz)")
    ax_fft.set_ylabel("Magnitud")
    ax_fft.set_title(
        f"{obra['label']} — FFT global (comparativa)\n{obra['subtitle']}\n"
        "FFT estándar del fragmento completo — sin resolución temporal",
        fontsize=11,
    )

    # Barras ordenadas descendente por magnitud
    bar_order = np.argsort(top_pct)[::-1]
    ax_bar.bar(
        [top_notes[i] for i in bar_order],
        [top_pct[i] for i in bar_order],
        color=plt.cm.magma(np.linspace(0.3, 0.85, len(bar_order))),
    )
    for i, idx in enumerate(bar_order):
        ax_bar.text(i, top_pct[idx] + 1.5, f"{top_pct[idx]:.0f}%",
                     ha="center", fontsize=8)
    ax_bar.set_ylabel("Magnitud relativa (%)")
    ax_bar.set_title("Top 10 notas por energía (FFT global)", fontsize=10)
    ax_bar.set_ylim(0, 110)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardado: {out_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for obra in OBRAS:
        if not obra["path"].exists():
            raise FileNotFoundError(
                f"No encontré el audio de '{obra['label']}' en {obra['path']}. "
                f"¿Corriste descargar_audios_maestro.py?"
            )
        print(f"Procesando {obra['label']}: {obra['path'].name}")
        y, sr = load_fragment(obra)

        plot_tiempo_frecuencia(
            obra, y, sr, RESULTS_DIR / f"{obra['slug']}_tiempo_frecuencia.png"
        )
        plot_fft_global(
            obra, y, sr, RESULTS_DIR / f"{obra['slug']}_fft_global.png"
        )

    print("\nListo. Imágenes guardadas en results/")


if __name__ == "__main__":
    main()