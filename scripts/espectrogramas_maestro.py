"""
Genera el espectrograma (STFT) de tres fragmentos representativos de
MAESTRO -uno por período- para usar como material visual del video.

No alimenta al clasificador: es un script totalmente aparte de
preprocess_maestro.py / train_mlp.py.

Uso:
    python scripts/espectrogramas_maestro.py
"""

from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# --- Paths ---------------------------------------------------------------
# Resueltas respecto a la ubicación del script, no al directorio desde donde
# se lo invoque: funciona igual corriéndolo desde la raíz del repo o desde
# adentro de scripts/.

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "maestro-v3.0.0"
RESULTS_DIR = REPO_ROOT / "results"

SAMPLE_RATE = 22050     # mismo valor que preprocess_maestro.py
N_FFT = 2048            # ventana de 2048 muestras (~93 ms a 22050 Hz)
HOP_LENGTH = 512        # salto entre ventanas (~23 ms), 75% de solapamiento

# --- Obras elegidas, una por período --------------------------------------
# offset / duration: segundo de inicio y duración del fragmento a graficar.
# Ajustá el offset si el fragmento cae en un silencio o no es representativo.

OBRAS = [
    {
        "label": "Barroco",
        "subtitle": "J. S. Bach — Preludio y Fuga en Do M., BWV 846",
        "path": RAW_DIR / "2014" / "MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.wav",
        "offset": 0.0,
        "duration": 15.0,
    },
    {
        "label": "Clasicismo",
        "subtitle": "W. A. Mozart — Sonata en Do M., K. 330",
        "path": RAW_DIR / "2015" / "MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--2.wav",
        "offset": 0.0,
        "duration": 15.0,
    },
    {
        "label": "Romanticismo",
        "subtitle": "F. Chopin — Sonata No. 3",
        "path": RAW_DIR / "2006" / "MIDI-Unprocessed_20_R1_2006_01-04_ORIG_MID--AUDIO_20_R1_2006_04_Track04_wav.wav",
        "offset": 60.0,   # salteamos el primer minuto; ajustar si hace falta
        "duration": 15.0,
    },
]


def compute_spectrogram(path: Path, offset: float, duration: float):
    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True,
                          offset=offset, duration=duration)
    if y.size == 0:
        raise ValueError(f"Audio vacío al cargar {path} (offset={offset})")

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann")
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db, sr


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(OBRAS), figsize=(16, 4.5), sharey=True)

    last_img = None
    for ax, obra in zip(axes, OBRAS):
        if not obra["path"].exists():
            raise FileNotFoundError(
                f"No encontré el audio de '{obra['label']}' en {obra['path']}. "
                f"¿Corriste descargar_audios_maestro.py?"
            )

        print(f"Procesando {obra['label']}: {obra['path'].name}")
        S_db, sr = compute_spectrogram(obra["path"], obra["offset"], obra["duration"])

        last_img = librosa.display.specshow(
            S_db, sr=sr, hop_length=HOP_LENGTH,
            x_axis="time", y_axis="log", cmap="magma", ax=ax,
        )
        ax.set_title(obra["label"], fontsize=15, fontweight="bold")
        ax.set_xlabel("Tiempo (s)")
        ax.text(
            0.02, 0.02, obra["subtitle"], transform=ax.transAxes,
            fontsize=9, color="white", va="bottom",
        )

    axes[0].set_ylabel("Frecuencia (Hz)")
    fig.colorbar(last_img, ax=axes, format="%+2.0f dB", fraction=0.025, pad=0.02)
    fig.suptitle(
        "Espectrograma (STFT, Hann, N=2048, hop=512) por período histórico",
        fontsize=13,
    )

    out_path = RESULTS_DIR / "espectrograma_comparacion.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\nGuardado: {out_path}")


if __name__ == "__main__":
    main()