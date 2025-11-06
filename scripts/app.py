import os
import parselmouth
import pandas as pd

audio_folder = "audios/"
output_csv = "resultados_voz.csv"

audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]
results = []

def analizar_voz(file_path):
    snd = parselmouth.Sound(file_path)

    # Análisis de pitch (F0)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values > 0]  # Filtrar valores válidos
    f0_mean = float(f0_values.mean()) if len(f0_values) > 0 else 0.0

    # Análisis de jitter y shimmer
    point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 300)
    
    # Jitter local (más confiable)
    jitter_local = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    
    # Shimmer local (más confiable)
    shimmer_local = parselmouth.praat.call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # Harmonicity (HNR)
    harmonicity = parselmouth.praat.call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr_mean = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

    return f0_mean, jitter_local, shimmer_local, hnr_mean

for file in audio_files:
    wav_path = os.path.join(audio_folder, file)
    try:
        print(f"Analizando: {file}")
        f0, jitter, shimmer, hnr = analizar_voz(wav_path)
        results.append({
            "Archivo": file,
            "F0_Hz": round(f0, 2),
            "Jitter_porcentaje": round(jitter * 100, 4),  # Más decimales para precisión
            "Shimmer_porcentaje": round(shimmer * 100, 4),
            "HNR_dB": round(hnr, 2)
        })
        print(f"  ✓ Completado: F0={round(f0,2)}Hz, Jitter={round(jitter*100,4)}%, Shimmer={round(shimmer*100,4)}%, HNR={round(hnr,2)}dB")
    except Exception as e:
        print(f"  ✗ Error analizando {file}: {e}")
        # Agregar fila con valores nulos para mantener consistencia
        results.append({
            "Archivo": file,
            "F0_Hz": None,
            "Jitter_porcentaje": None,
            "Shimmer_porcentaje": None,
            "HNR_dB": None
        })

# Crear DataFrame y guardar
df = pd.DataFrame(results)

print("\n" + "="*50)
print("RESULTADOS COMPLETOS")
print("="*50)
print(df)

# Estadísticas descriptivas
print("\n" + "="*50)
print("ESTADÍSTICAS DESCRIPTIVAS")
print("="*50)
print(df.describe())

# Guardar CSV
df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"\n✅ Resultados guardados en: {output_csv}")
print(f"📊 Archivos procesados: {len(audio_files)}")
print(f"📁 Archivos exitosos: {len([r for r in results if r['F0_Hz'] is not None])}")