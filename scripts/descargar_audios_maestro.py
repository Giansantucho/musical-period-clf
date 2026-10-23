"""
Descarga solo 3 archivos de audio puntuales del dataset MAESTRO v3.0.0
sin bajar el .zip completo (101GB).

Requisitos:
    pip install remotezip

Uso:
    python descargar_audios_maestro.py
"""

from pathlib import Path
from remotezip import RemoteZip

ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"

# Un fragmento identificador de cada archivo (no hace falta la ruta completa
# ni mayúsculas exactas: busca por coincidencia dentro del nombre interno del zip).
OBRAS = {
    "barroco_bach_bwv846": "MIDI-UNPROCESSED_16-18_R1_2014_MID--AUDIO_16_R1_2014_wav--1.wav",
    "clasico_mozart_k330": "MIDI-Unprocessed_R1_D2-13-20_mid--AUDIO-from_mp3_20_R1_2015_wav--2.wav",
    "romantico_chopin_sonata3": "MIDI-Unprocessed_20_R1_2006_01-04_ORIG_MID--AUDIO_20_R1_2006_04_Track04_wav.wav",
}

OUT_DIR = Path("data/raw/maestro-v3.0.0")


def main():
    print(f"Conectando al zip remoto (solo lee el índice, no baja todo)...")
    with RemoteZip(ZIP_URL) as zf:
        names = zf.namelist()
        print(f"Índice leído: {len(names)} archivos en total dentro del zip.\n")

        to_extract = {}
        for label, fragment in OBRAS.items():
            matches = [n for n in names if fragment.lower() in n.lower()]
            if not matches:
                print(f"[FALTA] No encontré ningún archivo que contenga: {fragment}")
                continue
            if len(matches) > 1:
                print(f"[AVISO] Varias coincidencias para {label}, uso la primera:")
                for m in matches:
                    print("   ", m)
            to_extract[label] = matches[0]
            print(f"[OK] {label} -> {matches[0]}")

        print("\nDescargando (por rango HTTP, solo estos archivos)...")
        for label, internal_path in to_extract.items():
            # Conserva la estructura <año>/<archivo>.wav dentro de data/raw/maestro-v3.0.0
            rel_path = Path(internal_path)
            # Si el zip trae un prefijo tipo "maestro-v3.0.0/", lo sacamos
            parts = rel_path.parts
            if parts[0].lower().startswith("maestro-v3"):
                rel_path = Path(*parts[1:])

            dest = OUT_DIR / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)

            with RemoteZip(ZIP_URL) as zf2:
                data = zf2.read(internal_path)
            dest.write_bytes(data)
            print(f"  guardado: {dest}  ({len(data) / 1e6:.1f} MB)")

    print("\nListo. Archivos guardados en data/raw/maestro-v3.0.0/")


if __name__ == "__main__":
    main()
