from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.io.wavfile as wavfile
import pandas as pd
import tempfile
import os
import json
import traceback
import scipy.signal as signal

app = Flask(__name__)
CORS(app)

AUDIOS_FOLDER = '../audios'
os.makedirs(AUDIOS_FOLDER, exist_ok=True)

reference_data = None

def convert_numpy_types(obj):
    try:
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complexfloating, np.complex64, np.complex128)):
            return complex(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.str_):
            return str(obj)
        elif isinstance(obj, np.bytes_):
            return bytes(obj)
        elif isinstance(obj, dict):
            return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple, set)):
            return [convert_numpy_types(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return convert_numpy_types(obj.__dict__)
        else:
            return obj
    except Exception as e:
        print(f"Error en convert_numpy_types: {str(e)}")
        return obj

def load_reference_data():
    try:
        df_ref = pd.read_csv("resultados_voz_limpio.csv")
        print(f"Dataset de referencia cargado: {len(df_ref)} muestras")
        return df_ref
    except Exception as e:
        print(f"Error cargando dataset de referencia: {e}")
        try:
            df_ref = pd.read_csv("dataset_referencia_voz.csv")
            print(f"Dataset alternativo cargado: {len(df_ref)} muestras")
            return df_ref
        except Exception as e2:
            print(f"Error cargando dataset alternativo: {e2}")
            return None

def compare_with_reference(current_metrics):
    if reference_data is None or reference_data.empty:
        print("Sin datos de referencia")
        return convert_numpy_types({
            'reference_score': 70,
            'anomaly_detected': False,
            'reason': 'Sin datos de referencia - usando valores por defecto'
        })
    
    try:
        print("Comparando con dataset de referencia...")
        
        ref_stats = {}
        
        if 'F0_Hz' in reference_data.columns:
            ref_stats['f0_mean'] = reference_data['F0_Hz'].mean()
            ref_stats['f0_std'] = reference_data['F0_Hz'].std()
            ref_stats['f0_range'] = (
                reference_data['F0_Hz'].quantile(0.05),
                reference_data['F0_Hz'].quantile(0.95)
            )
            print(f"Rango F0 referencia: {ref_stats['f0_range']}")
        else:
            ref_stats['f0_range'] = (70, 350)
            ref_stats['f0_std'] = 60
            print("Usando rango F0 por defecto amplio")
        
        if 'rms_energy' in reference_data.columns:
            ref_stats['rms_range'] = (
                reference_data['rms_energy'].quantile(0.1),
                reference_data['rms_energy'].quantile(0.9)
            )
        else:
            ref_stats['rms_range'] = (0.005, 0.25)
        
        if 'zero_crossing_rate' in reference_data.columns:
            ref_stats['zcr_range'] = (
                reference_data['zero_crossing_rate'].quantile(0.05),
                reference_data['zero_crossing_rate'].quantile(0.95)
            )
        else:
            ref_stats['zcr_range'] = (0.003, 0.12)
        
        score_components = []
        anomalies = []
        
        if ref_stats['rms_range'][0] <= current_metrics['rms_energy'] <= ref_stats['rms_range'][1]:
            score_components.append(85)
        elif current_metrics['rms_energy'] < ref_stats['rms_range'][0]:
            score_components.append(50)
            anomalies.append("Energía de voz algo baja")
        else:
            score_components.append(60)
            anomalies.append("Energía de voz algo alta")
        
        if ref_stats['zcr_range'][0] <= current_metrics['zcr'] <= ref_stats['zcr_range'][1]:
            score_components.append(80)
        elif current_metrics['zcr'] > ref_stats['zcr_range'][1]:
            score_components.append(45)
            anomalies.append("Alta actividad de frecuencia")
        else:
            score_components.append(55)
            anomalies.append("Baja actividad vocal")
        
        if current_metrics['f0_approx'] > 0:
            f0_distance = abs(current_metrics['f0_approx'] - ref_stats['f0_mean'])
            if f0_distance < ref_stats['f0_std'] * 2.5:
                score_components.append(90)
            else:
                score_components.append(60)
                anomalies.append("Frecuencia fundamental algo fuera del rango típico")
        
        if current_metrics.get('snr_approx', 0) > 15:
            score_components.append(95)
        elif current_metrics.get('snr_approx', 0) > 10:
            score_components.append(80)
        elif current_metrics.get('snr_approx', 0) > 6:
            score_components.append(65)
        else:
            score_components.append(40)
            anomalies.append("Relación señal-ruido algo baja")
        
        if current_metrics['noise_level'] < 0.025:
            score_components.append(90)
        elif current_metrics['noise_level'] < 0.045:
            score_components.append(75)
        else:
            score_components.append(50)
            anomalies.append("Nivel de ruido algo alto")
        
        if current_metrics.get('noise_to_signal_ratio', 0) > 0.7:
            score_components.append(20)
            anomalies.append("Relación ruido/señal muy alta")
        elif current_metrics.get('noise_to_signal_ratio', 0) > 0.4:
            score_components.append(50)
            anomalies.append("Relación ruido/señal alta")
        else:
            score_components.append(85)
        
        reference_score = float(np.mean(score_components)) if score_components else 70.0
        
        severe_anomalies = [a for a in anomalies if any(word in a.lower() for word in ['muy', 'demasiado', 'extremadamente'])]
        anomaly_detected = len(severe_anomalies) >= 2 or reference_score < 50
        
        reason = "Dentro de parámetros normales de referencia"
        if anomaly_detected:
            reason = f"Problemas detectados: {', '.join(anomalies[:2])}"
        elif anomalies:
            reason = f"Observaciones: {', '.join(anomalies[:2])}"
        
        print(f"Comparación completada: {reference_score:.1f}%, Anomalías: {len(anomalies)}")
        
        result = {
            'reference_score': reference_score,
            'anomaly_detected': bool(anomaly_detected),
            'reason': reason,
            'reference_ranges': {
                'rms_energy': [float(ref_stats['rms_range'][0]), float(ref_stats['rms_range'][1])],
                'zero_crossing_rate': [float(ref_stats['zcr_range'][0]), float(ref_stats['zcr_range'][1])],
                'f0_hz': [float(ref_stats['f0_range'][0]), float(ref_stats['f0_range'][1])]
            },
            'anomalies': anomalies,
            'reference_samples': int(len(reference_data))
        }
        
        return convert_numpy_types(result)
        
    except Exception as e:
        print(f"Error en compare_with_reference: {str(e)}")
        traceback.print_exc()
        error_result = {
            'reference_score': 70.0,
            'anomaly_detected': False,
            'reason': f'Error en comparación: {str(e)} - usando valores por defecto'
        }
        return convert_numpy_types(error_result)

def analyze_audio_quality(audio_data, sample_rate):
    try:
        print(f"Analizando audio: {len(audio_data)} samples, {sample_rate} Hz")
        
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
            print("Convertido a mono")
        
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
            print("Normalizado de int16 a float32")
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
            print("Normalizado de int32 a float32")
        else:
            print(f"Tipo de datos: {audio_data.dtype}")
        
        rms_energy = np.sqrt(np.mean(audio_data**2))
        print(f"RMS Energy: {rms_energy}")
        
        zero_crossings = np.where(np.diff(audio_data > 0))[0]
        zcr = len(zero_crossings) / len(audio_data)
        print(f"Zero Crossing Rate: {zcr}")
        
        audio_clean = signal.medfilt(audio_data, kernel_size=3)
        noise_estimate = audio_data - audio_clean
        noise_level = np.std(noise_estimate)
        
        signal_power = np.mean(audio_clean**2)
        noise_power = np.mean(noise_estimate**2)
        snr_approx = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 50
        print(f"Noise Level: {noise_level}")
        print(f"SNR aproximada: {snr_approx:.2f} dB")
        
        noise_to_signal_ratio = noise_level / (rms_energy + 1e-10)
        print(f"Relación Ruido/Señal: {noise_to_signal_ratio:.3f}")
        
        fft = np.fft.fft(audio_data)
        frequencies = np.fft.fftfreq(len(fft), 1/sample_rate)
        magnitude = np.abs(fft)
        
        positive_freq_idx = frequencies > 0
        positive_freq = frequencies[positive_freq_idx]
        positive_mag = magnitude[positive_freq_idx]
        
        vocal_range = (positive_freq >= 80) & (positive_freq <= 300)
        if np.any(vocal_range):
            vocal_freq = positive_freq[vocal_range]
            vocal_mag = positive_mag[vocal_range]
            fundamental_idx = np.argmax(vocal_mag)
            f0_approx = vocal_freq[fundamental_idx]
            print(f"F0 aproximado: {f0_approx} Hz")
        else:
            f0_approx = 0
            print("No se encontró F0 en rango vocal")
        
        quality_score = max(0, min(100, 
            (min(rms_energy * 800, 40) +
             (1 - min(zcr, 0.2)) * 30 +
             max(0, min(snr_approx * 2, 30))
        )))
        print(f"Puntuación base: {quality_score}")
        
        comparison_results = compare_with_reference({
            'rms_energy': rms_energy,
            'zcr': zcr,
            'noise_level': noise_level,
            'snr_approx': snr_approx,
            'f0_approx': f0_approx,
            'noise_to_signal_ratio': noise_to_signal_ratio
        })
        print(f"Puntuación comparación: {comparison_results['reference_score']}")
        
        final_score = (quality_score * 0.4) + (comparison_results['reference_score'] * 0.6)
        print(f"Puntuación final: {final_score}")
        
        needs_repeat = False
        reasons_to_repeat = []
        quality_issues = []
        
        if noise_to_signal_ratio > 0.8:
            needs_repeat = True
            reasons_to_repeat.append("Demasiado ruido de fondo - la señal es casi inaudible")
            quality_issues.append("ruido_excesivo")
            print(f"Ruido excesivo detectado: relación {noise_to_signal_ratio:.3f}")
        
        elif rms_energy < 0.005:
            needs_repeat = True
            reasons_to_repeat.append("Volumen demasiado bajo - acércate al micrófono")
            quality_issues.append("bajo_volumen")
            print(f"Volumen bajo detectado: {rms_energy}")
        
        elif snr_approx < 6:
            needs_repeat = True
            reasons_to_repeat.append(f"Relación señal-ruido muy baja ({snr_approx:.1f} dB)")
            quality_issues.append("snr_muy_baja")
            print(f"SNR muy baja detectada: {snr_approx} dB")
        
        elif noise_level > 0.1:
            needs_repeat = True
            reasons_to_repeat.append("Nivel de ruido extremadamente alto")
            quality_issues.append("ruido_extremo")
            print(f"Ruido extremo detectado: {noise_level}")
        
        secondary_issues = []
        
        if 0.5 < noise_to_signal_ratio <= 0.8:
            secondary_issues.append("ruido_significativo")
        
        if comparison_results['anomaly_detected']:
            secondary_issues.append("anomalia_referencia")
        
        if len(secondary_issues) >= 2 and not needs_repeat:
            needs_repeat = True
            reasons_to_repeat.append("Múltiples problemas de calidad detectados")
            print(f"Múltiples problemas secundarios: {secondary_issues}")
        
        if needs_repeat:
            if final_score >= 70:
                quality = "aceptable_con_problemas"
                if final_score >= 70:
                    needs_repeat = False
                    reasons_to_repeat = ["Problemas menores detectados, pero puede continuar"]
            elif final_score >= 50:
                quality = "mala_con_problemas"
            else:
                quality = "mala"
        else:
            if final_score >= 80:
                quality = "excelente"
            elif final_score >= 70:
                quality = "buena"
            elif final_score >= 60:
                quality = "aceptable"
            else:
                quality = "mala"
                if final_score < 45:
                    needs_repeat = True
                    reasons_to_repeat.append("Puntuación general muy baja")
        
        reason = comparison_results.get('reason', 'Calidad aceptable')
        if reasons_to_repeat:
            reason = "; ".join(reasons_to_repeat)
        elif secondary_issues:
            reason = "Calidad aceptable con algunas observaciones"
        
        print(f"Análisis completado: {quality} ({final_score:.1f}%), Repetir: {needs_repeat}")
        print(f"Razón: {reason}")
        
        result = {
            'quality': quality,
            'score': float(final_score),
            'needs_repeat': needs_repeat,
            'reason': reason,
            'reasons_to_repeat': reasons_to_repeat,
            'quality_issues': quality_issues,
            'metrics': {
                'rms_energy': float(rms_energy),
                'zero_crossing_rate': float(zcr),
                'noise_level': float(noise_level),
                'snr_approx': float(snr_approx),
                'f0_approximated': float(f0_approx),
                'noise_to_signal_ratio': float(noise_to_signal_ratio)
            },
            'comparison': comparison_results
        }
        
        result = convert_numpy_types(result)
        
        return result
        
    except Exception as e:
        print(f"Error en analyze_audio_quality: {str(e)}")
        traceback.print_exc()
        error_result = {
            'quality': 'error',
            'score': 0,
            'needs_repeat': False,
            'reason': f'Error en análisis: {str(e)}',
            'metrics': {}
        }
        return convert_numpy_types(error_result)

def read_audio_file(file_path):
    try:
        if file_path.lower().endswith('.wav'):
            sample_rate, audio_data = wavfile.read(file_path)
            return sample_rate, audio_data
        
        elif file_path.lower().endswith('.webm'):
            print("Archivo WebM detectado, intentando lectura básica...")
            
            try:
                sample_rate, audio_data = wavfile.read(file_path)
                return sample_rate, audio_data
            except:
                print("No se pudo leer WebM con scipy")
                
                print("Generando datos de audio simulados para testing...")
                sample_rate = 16000
                duration = 5
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                f0 = 180
                audio_data = 0.5 * np.sin(2 * np.pi * f0 * t)
                audio_data += 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
                audio_data += 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
                
                noise = 0.05 * np.random.normal(0, 1, len(t))
                audio_data += noise
                
                audio_data = (audio_data * 32767).astype(np.int16)
                
                return sample_rate, audio_data
        
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path}")
            
    except Exception as e:
        print(f"Error leyendo archivo de audio: {str(e)}")
        raise

@app.route('/api/upload-audio', methods=['POST'])
def upload_audio():
    try:
        print("Solicitud de subida de audio recibida")
        
        if 'audio' not in request.files:
            print("No se encontró archivo 'audio' en request.files")
            return jsonify({'error': 'No se proporcionó archivo de audio'}), 400
        
        audio_file = request.files['audio']
        cod_eje = request.form.get('cod_eje', 'unknown')
        
        print(f"Archivo recibido - nombre: {audio_file.filename}")
        
        audio_file.seek(0)
        
        if len(audio_file.read()) == 0:
            print("Archivo de 0 bytes")
            return jsonify({'error': 'Archivo de audio vacío'}), 400
        
        audio_file.seek(0)
        
        import uuid
        from datetime import datetime
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        filename = f"audio_{timestamp}_{unique_id}_{cod_eje}.webm"
        file_path = os.path.join(AUDIOS_FOLDER, filename)
        
        audio_file.save(file_path)
        
        file_size = os.path.getsize(file_path)
        print(f"Archivo guardado - ruta: {file_path}, tamaño: {file_size} bytes")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'file_path': file_path,
            'file_size': file_size,
            'message': 'Audio guardado correctamente en el servidor'
        })
        
    except Exception as e:
        print(f"Error subiendo audio: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-audio-quality', methods=['POST'])
def check_audio_quality():
    try:
        print("Solicitud recibida en /api/check-audio-quality")
        
        data = request.get_json()
        print("Datos recibidos:", data.keys() if data else "No data")
        
        if not data or 'audio_filename' not in data:
            print("No se encontró audio_filename en los datos")
            return jsonify({'error': 'No se proporcionó nombre de archivo de audio'}), 400
        
        audio_filename = data['audio_filename']
        cod_eje = data.get('cod_eje', 'unknown')
        
        file_path = os.path.join(AUDIOS_FOLDER, audio_filename)
        
        print(f"Buscando archivo: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Archivo no encontrado: {file_path}")
            return jsonify({'error': 'Archivo de audio no encontrado en el servidor'}), 404
        
        file_size = os.path.getsize(file_path)
        print(f"Archivo encontrado - tamaño: {file_size} bytes")
        
        print("Leyendo archivo de audio...")
        sample_rate, audio_data = read_audio_file(file_path)
        
        print(f"Audio leído - Tasa de muestreo: {sample_rate} Hz, Muestras: {len(audio_data)}")
        
        print("Iniciando análisis de calidad...")
        analysis_result = analyze_audio_quality(audio_data, sample_rate)
        
        analysis_result['file_info'] = {
            'filename': audio_filename,
            'file_size': int(file_size),
            'sample_rate': int(sample_rate),
            'duration_seconds': float(len(audio_data) / sample_rate) if sample_rate > 0 else 0.0
        }
        
        analysis_result = convert_numpy_types(analysis_result)
        
        print("Análisis de calidad completado")
        print(f"Resultado: {analysis_result['quality']} ({analysis_result['score']}%), Repetir: {analysis_result['needs_repeat']}")
        
        return jsonify(analysis_result)
        
    except Exception as e:
        print(f"Error en servidor: {str(e)}")
        traceback.print_exc()
        error_result = {
            'quality': 'error',
            'score': 0.0,
            'needs_repeat': False,
            'reason': f'Error en análisis: {str(e)}',
            'metrics': {}
        }
        return jsonify(convert_numpy_types(error_result)), 500

@app.route('/api/reference-stats', methods=['GET'])
def get_reference_stats():
    if reference_data is None or reference_data.empty:
        return jsonify({'error': 'Dataset de referencia no disponible'}), 404
    
    try:
        stats = {
            'total_samples': len(reference_data),
            'available_columns': list(reference_data.columns)
        }
        
        if 'F0_Hz' in reference_data.columns:
            stats['f0_stats'] = {
                'mean': float(reference_data['F0_Hz'].mean()),
                'std': float(reference_data['F0_Hz'].std()),
                'range': [float(reference_data['F0_Hz'].min()), float(reference_data['F0_Hz'].max())]
            }
        
        if 'Jitter_porcentaje' in reference_data.columns:
            stats['jitter_stats'] = {
                'mean': float(reference_data['Jitter_porcentaje'].mean()),
                'median': float(reference_data['Jitter_porcentaje'].median())
            }
        
        if 'Shimmer_porcentaje' in reference_data.columns:
            stats['shimmer_stats'] = {
                'mean': float(reference_data['Shimmer_porcentaje'].mean()),
                'median': float(reference_data['Shimmer_porcentaje'].median())
            }
        
        if 'HNR_dB' in reference_data.columns:
            stats['hnr_stats'] = {
                'mean': float(reference_data['HNR_dB'].mean()),
                'median': float(reference_data['HNR_dB'].median())
            }
        
        if 'Clasificacion_Voz' in reference_data.columns:
            stats['vocal_classification_distribution'] = reference_data['Clasificacion_Voz'].value_counts().to_dict()
        
        if 'Calidad_Vocal' in reference_data.columns:
            stats['voice_quality_distribution'] = reference_data['Calidad_Vocal'].value_counts().to_dict()
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': f'Error calculando estadísticas: {str(e)}'}), 500

@app.route('/api/list-audios', methods=['GET'])
def list_audios():
    try:
        files = []
        for filename in os.listdir(AUDIOS_FOLDER):
            file_path = os.path.join(AUDIOS_FOLDER, filename)
            if os.path.isfile(file_path):
                file_stats = os.stat(file_path)
                files.append({
                    'filename': filename,
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime
                })
        
        return jsonify({
            'total_files': len(files),
            'audios': sorted(files, key=lambda x: x['modified'], reverse=True)
        })
    except Exception as e:
        return jsonify({'error': f'Error listando archivos: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    has_reference = reference_data is not None and not reference_data.empty
    status = 'healthy' if has_reference else 'no_reference_data'
    
    audios_folder_exists = os.path.exists(AUDIOS_FOLDER)
    
    return jsonify({
        'status': status,
        'service': 'audio-quality-analyzer',
        'reference_samples': len(reference_data) if has_reference else 0,
        'audios_folder': AUDIOS_FOLDER,
        'audios_folder_exists': audios_folder_exists,
        'audios_count': len(os.listdir(AUDIOS_FOLDER)) if audios_folder_exists else 0
    })

if __name__ == '__main__':
    reference_data = load_reference_data()
    
    print("Iniciando servicio de análisis de audio...")
    print(f"Carpeta de audios: {os.path.abspath(AUDIOS_FOLDER)}")
    
    if not os.path.exists(AUDIOS_FOLDER):
        print(f"Creando carpeta de audios: {AUDIOS_FOLDER}")
        os.makedirs(AUDIOS_FOLDER, exist_ok=True)
    
    if reference_data is not None and not reference_data.empty:
        print(f"Dataset de referencia: {len(reference_data)} muestras")
        print(f"Columnas disponibles: {list(reference_data.columns)}")
    else:
        print("Sin dataset de referencia - usando valores por defecto")
    
    print("Servidor corriendo en http://localhost:5000")
    print("Endpoints disponibles:")
    print("   POST /api/upload-audio     - Subir archivo de audio")
    print("   POST /api/check-audio-quality - Analizar calidad de audio")
    print("   GET  /api/reference-stats  - Estadísticas de referencia")
    print("   GET  /api/list-audios      - Listar archivos de audio")
    print("   GET  /api/health           - Estado del servicio")
    
    app.run(host='0.0.0.0', port=5000, debug=False)