from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
import tempfile
import os
import time
import gc
from scipy.signal import find_peaks
import parselmouth
from parselmouth.praat import call
import warnings
import soundfile as sf
from datetime import datetime

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

class ClinicalVocalAnalyzer:
    def __init__(self):
        self.sample_rate = 44100
        self.clinical_norms = {
            'pitch': {
                'male': {'min': 85, 'max': 180, 'mean': 120},
                'female': {'min': 165, 'max': 265, 'mean': 215},
                'child': {'min': 250, 'max': 400, 'mean': 300}
            },
            'jitter': {'normal': 1.04, 'pathological': 3.8},
            'shimmer': {'normal': 3.81, 'pathological': 11.0},
            'hnr': {'good': 20, 'acceptable': 15, 'poor': 10}
        }

    def convert_to_serializable(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return [self.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj

    def safe_temp_file_operation(self, y, sr, operation_func):
        temp_file = None
        temp_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_path = temp_file.name
            temp_file.close()
            sf.write(temp_path, y, sr)
            time.sleep(0.1)
            result = operation_func(temp_path)
            return result
        except Exception:
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    gc.collect()
                    time.sleep(0.1)
                    for attempt in range(3):
                        try:
                            os.chmod(temp_path, 0o777)
                            os.unlink(temp_path)
                            break
                        except PermissionError:
                            if attempt < 2:
                                time.sleep(0.2)
                except:
                    pass

    def calculate_pitch_periods_accurate(self, y, sr, f0_min=50, f0_max=500):
        y_filtered = librosa.effects.preemphasis(y)
        
        try:
            f0 = librosa.yin(y_filtered, 
                             fmin=f0_min, 
                             fmax=f0_max, 
                             sr=sr,
                             threshold=0.1,
                             win_length=int(sr*0.04),
                             hop_length=int(sr*0.01))
        except TypeError:
            try:
                f0 = librosa.yin(y_filtered, 
                                 fmin=f0_min, 
                                 fmax=f0_max, 
                                 sr=sr,
                                 hop_length=int(sr*0.01))
            except:
                f0 = librosa.yin(y_filtered, fmin=f0_min, fmax=f0_max, sr=sr)
        
        try:
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=int(sr*0.01))
        except:
            times = librosa.frames_to_time(np.arange(len(f0)), sr=sr)
        
        voiced_mask = (f0 > 0) & (f0 >= f0_min) & (f0 <= f0_max)
        voiced_f0 = f0[voiced_mask]
        
        if len(times) > len(f0):
            times = times[:len(f0)]
        elif len(times) < len(f0):
            times = np.linspace(0, len(y)/sr, len(f0))
            
        voiced_times = times[voiced_mask]
        
        if len(voiced_f0) < 10:
            return None, None, None
        
        median_f0 = np.median(voiced_f0)
        f0_std = np.std(voiced_f0)
        outlier_mask = np.abs(voiced_f0 - median_f0) < 2 * f0_std
        
        clean_f0 = voiced_f0[outlier_mask]
        clean_times = voiced_times[outlier_mask]
        
        if len(clean_f0) < 5:
            return None, None, None
            
        return clean_f0, clean_times, voiced_mask.sum()

    def calculate_jitter_accurate(self, f0_values, method='local'):
        if len(f0_values) < 3:
            return 0.0
        
        periods = 1.0 / f0_values
        
        if method == 'local':
            period_diffs = np.abs(np.diff(periods))
            mean_period = np.mean(periods[:-1])
            jitter = np.mean(period_diffs) / mean_period * 100
        elif method == 'rap':
            if len(periods) < 3:
                return 0.0
            rap_diffs = []
            for i in range(1, len(periods) - 1):
                expected = (periods[i-1] + periods[i] + periods[i+1]) / 3
                rap_diffs.append(abs(periods[i] - expected))
            jitter = np.mean(rap_diffs) / np.mean(periods) * 100
        elif method == 'ppq5':
            if len(periods) < 5:
                return 0.0
            ppq_diffs = []
            for i in range(2, len(periods) - 2):
                expected = np.mean(periods[i-2:i+3])
                ppq_diffs.append(abs(periods[i] - expected))
            jitter = np.mean(ppq_diffs) / np.mean(periods) * 100
        else:
            return 0.0
        
        return float(jitter)

    def calculate_shimmer_accurate(self, y, sr, f0_values, times, method='local'):
        if len(f0_values) < 3:
            return 0.0
        
        amplitudes = []
        hop_length = int(sr * 0.01)
        
        for i, (f0, time) in enumerate(zip(f0_values, times)):
            if f0 <= 0:
                continue
                
            sample_idx = int(time * sr)
            period_samples = int(sr / f0)
            
            start_idx = max(0, sample_idx - period_samples // 2)
            end_idx = min(len(y), sample_idx + period_samples // 2)
            
            if end_idx - start_idx < period_samples // 2:
                continue
                
            period_signal = y[start_idx:end_idx]
            
            if len(period_signal) > 0:
                rms_amplitude = np.sqrt(np.mean(period_signal ** 2))
                amplitudes.append(rms_amplitude)
        
        if len(amplitudes) < 3:
            return 0.0
        
        amplitudes = np.array(amplitudes)
        
        if method == 'local':
            amp_diffs = np.abs(np.diff(amplitudes))
            mean_amplitude = np.mean(amplitudes[:-1])
            if mean_amplitude > 0:
                shimmer = np.mean(amp_diffs) / mean_amplitude * 100
            else:
                shimmer = 0.0
        elif method == 'local_db':
            if np.any(amplitudes <= 0):
                return 0.0
            amp_db = 20 * np.log10(amplitudes + 1e-12)
            shimmer = np.mean(np.abs(np.diff(amp_db)))
        elif method == 'apq3':
            if len(amplitudes) < 3:
                return 0.0
            apq_diffs = []
            for i in range(1, len(amplitudes) - 1):
                expected = (amplitudes[i-1] + amplitudes[i] + amplitudes[i+1]) / 3
                if expected > 0:
                    apq_diffs.append(abs(amplitudes[i] - expected))
            if len(apq_diffs) > 0 and np.mean(amplitudes) > 0:
                shimmer = np.mean(apq_diffs) / np.mean(amplitudes) * 100
            else:
                shimmer = 0.0
        elif method == 'apq5':
            if len(amplitudes) < 5:
                return 0.0
            apq_diffs = []
            for i in range(2, len(amplitudes) - 2):
                expected = np.mean(amplitudes[i-2:i+3])
                if expected > 0:
                    apq_diffs.append(abs(amplitudes[i] - expected))
            if len(apq_diffs) > 0 and np.mean(amplitudes) > 0:
                shimmer = np.mean(apq_diffs) / np.mean(amplitudes) * 100
            else:
                shimmer = 0.0
        else:
            return 0.0
        
        return float(shimmer)

    def calculate_perturbation_measures_corrected(self, y, sr):
        if len(y) < sr * 1.0:
            return {
                'jitter': {'local': 0.0, 'rap': 0.0, 'ppq5': 0.0, 'absolute': 0.0},
                'shimmer': {'local': 0.0, 'local_db': 0.0, 'apq3': 0.0, 'apq5': 0.0}
            }
        
        y_norm = librosa.util.normalize(y)
        rms_energy = np.sqrt(np.mean(y_norm ** 2))
        if rms_energy < 0.01:
            return {
                'jitter': {'local': 0.0, 'rap': 0.0, 'ppq5': 0.0, 'absolute': 0.0},
                'shimmer': {'local': 0.0, 'local_db': 0.0, 'apq3': 0.0, 'apq5': 0.0}
            }
        
        f0_values, times, voiced_count = self.calculate_pitch_periods_accurate(y_norm, sr)
        
        if f0_values is None or len(f0_values) < 10:
            return {
                'jitter': {'local': 0.0, 'rap': 0.0, 'ppq5': 0.0, 'absolute': 0.0},
                'shimmer': {'local': 0.0, 'local_db': 0.0, 'apq3': 0.0, 'apq5': 0.0}
            }
        
        jitter_local = self.calculate_jitter_accurate(f0_values, 'local')
        jitter_rap = self.calculate_jitter_accurate(f0_values, 'rap')
        jitter_ppq5 = self.calculate_jitter_accurate(f0_values, 'ppq5')
        
        periods = 1.0 / f0_values
        if len(periods) > 1:
            period_diffs = np.abs(np.diff(periods))
            jitter_absolute = np.mean(period_diffs) * 1000000
        else:
            jitter_absolute = 0.0
        
        shimmer_local = self.calculate_shimmer_accurate(y_norm, sr, f0_values, times, 'local')
        shimmer_local_db = self.calculate_shimmer_accurate(y_norm, sr, f0_values, times, 'local_db')
        shimmer_apq3 = self.calculate_shimmer_accurate(y_norm, sr, f0_values, times, 'apq3')
        shimmer_apq5 = self.calculate_shimmer_accurate(y_norm, sr, f0_values, times, 'apq5')
        
        return {
            'jitter': {
                'local': jitter_local,
                'rap': jitter_rap,
                'ppq5': jitter_ppq5,
                'absolute': jitter_absolute
            },
            'shimmer': {
                'local': shimmer_local,
                'local_db': shimmer_local_db,
                'apq3': shimmer_apq3,
                'apq5': shimmer_apq5
            }
        }

    def calculate_perturbation_praat_corrected(self, temp_path):
        try:
            sound = parselmouth.Sound(temp_path)
            duration = sound.get_total_duration()
            
            if duration < 1.0:
                return None
                
            pitch_rough = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
            f0_values = pitch_rough.selected_array['frequency']
            f0_values_voiced = f0_values[f0_values > 0]
            
            if len(f0_values_voiced) < 10:
                return None
                
            mean_f0 = np.mean(f0_values_voiced)
            min_period = 0.8 / min(mean_f0 * 1.5, 600)
            max_period = 1.25 / max(mean_f0 * 0.7, 50)
            
            pitch = sound.to_pitch(
                time_step=0.01,
                pitch_floor=max(50, mean_f0 * 0.6),
                pitch_ceiling=min(600, mean_f0 * 1.8),
                max_number_of_candidates=15,
                very_accurate=True,
                silence_threshold=0.03,
                voicing_threshold=0.45,
                octave_cost=0.01,
                octave_jump_cost=0.35,
                voiced_unvoiced_cost=0.14
            )
            
            jitter_results = {}
            shimmer_results = {}
            
            try:
                jitter_local = call(pitch, "Get jitter (local)", 0, 0, min_period, max_period, 1.3)
                jitter_results['local'] = float(jitter_local * 100) if not np.isnan(jitter_local) else 0.0
            except:
                jitter_results['local'] = 0.0
                
            try:
                jitter_absolute = call(pitch, "Get jitter (absolute)", 0, 0, min_period, max_period, 1.3)
                jitter_results['absolute'] = float(jitter_absolute * 1000000) if not np.isnan(jitter_absolute) else 0.0
            except:
                jitter_results['absolute'] = 0.0
                
            try:
                jitter_rap = call(pitch, "Get jitter (rap)", 0, 0, min_period, max_period, 1.3)
                jitter_results['rap'] = float(jitter_rap * 100) if not np.isnan(jitter_rap) else 0.0
            except:
                jitter_results['rap'] = 0.0
                
            try:
                jitter_ppq5 = call(pitch, "Get jitter (ppq5)", 0, 0, min_period, max_period, 1.3)
                jitter_results['ppq5'] = float(jitter_ppq5 * 100) if not np.isnan(jitter_ppq5) else 0.0
            except:
                jitter_results['ppq5'] = 0.0
            
            try:
                shimmer_local = call(sound, "Get shimmer (local)", 0, 0, min_period, max_period, 1.3, 1.6)
                shimmer_results['local'] = float(shimmer_local * 100) if not np.isnan(shimmer_local) else 0.0
            except:
                shimmer_results['local'] = 0.0
                
            try:
                shimmer_local_db = call(sound, "Get shimmer (local_dB)", 0, 0, min_period, max_period, 1.3, 1.6)
                shimmer_results['local_db'] = float(shimmer_local_db) if not np.isnan(shimmer_local_db) else 0.0
            except:
                shimmer_results['local_db'] = 0.0
                
            try:
                shimmer_apq3 = call(sound, "Get shimmer (apq3)", 0, 0, min_period, max_period, 1.3, 1.6)
                shimmer_results['apq3'] = float(shimmer_apq3 * 100) if not np.isnan(shimmer_apq3) else 0.0
            except:
                shimmer_results['apq3'] = 0.0
                
            try:
                shimmer_apq5 = call(sound, "Get shimmer (apq5)", 0, 0, min_period, max_period, 1.3, 1.6)
                shimmer_results['apq5'] = float(shimmer_apq5 * 100) if not np.isnan(shimmer_apq5) else 0.0
            except:
                shimmer_results['apq5'] = 0.0
            
            return {
                'jitter': jitter_results,
                'shimmer': shimmer_results
            }
            
        except Exception:
            return None

    def load_audio_from_blob(self, audio_data):
        try:
            if audio_data.startswith(b'RIFF'):
                extension = '.wav'
            elif b'webm' in audio_data[:100] or audio_data.startswith(b'\x1a\x45\xdf\xa3'):
                extension = '.webm'
            elif audio_data.startswith(b'ID3') or audio_data.startswith(b'\xff\xfb'):
                extension = '.mp3'
            elif b'mp4' in audio_data[:100] or audio_data.startswith(b'\x00\x00\x00'):
                extension = '.mp4'
            else:
                extension = '.webm'

            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name

            try:
                if extension == '.wav':
                    y, sr = librosa.load(tmp_path, sr=self.sample_rate)
                else:
                    try:
                        y, sr = librosa.load(tmp_path, sr=self.sample_rate)
                    except Exception:
                        try:
                            y, sr = sf.read(tmp_path)
                            if sr != self.sample_rate:
                                import scipy.signal
                                y = scipy.signal.resample(y, int(len(y) * self.sample_rate / sr))
                                sr = self.sample_rate
                        except Exception:
                            raise Exception(f"No se pudo cargar el audio")

            finally:
                try:
                    time.sleep(0.1)
                    os.unlink(tmp_path)
                except:
                    pass

            if y is None or len(y) == 0:
                raise Exception("El audio cargado está vacío")

            y = np.array(y, dtype=np.float32)
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            return y, sr

        except Exception:
            return None, None

    def calculate_fundamental_frequency_advanced(self, y, sr):
        def pitch_operation(temp_path):
            try:
                sound = parselmouth.Sound(temp_path)
                pitch = sound.to_pitch(pitch_floor=75.0, pitch_ceiling=600.0)
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values != 0]

                if len(pitch_values) > 0:
                    result = {
                        'mean': float(np.mean(pitch_values)),
                        'std': float(np.std(pitch_values)),
                        'min': float(np.min(pitch_values)),
                        'max': float(np.max(pitch_values)),
                        'median': float(np.median(pitch_values)),
                        'range': float(np.max(pitch_values) - np.min(pitch_values)),
                        'coefficient_variation': float(np.std(pitch_values) / np.mean(pitch_values) * 100) if np.mean(pitch_values) > 0 else 0.0
                    }
                    return result
                else:
                    return {key: 0.0 for key in ['mean', 'std', 'min', 'max', 'median', 'range', 'coefficient_variation']}

            except Exception:
                return {key: 0.0 for key in ['mean', 'std', 'min', 'max', 'median', 'range', 'coefficient_variation']}

        result = self.safe_temp_file_operation(y, sr, pitch_operation)
        return result if result is not None else {key: 0.0 for key in ['mean', 'std', 'min', 'max', 'median', 'range', 'coefficient_variation']}

    def calculate_formants_clinical(self, y, sr):
        def formants_operation(temp_path):
            try:
                sound = parselmouth.Sound(temp_path)
                formant = sound.to_formant_burg()
                duration = sound.get_total_duration()
                if duration < 0.1:
                    return {f'F{i}': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0} for i in range(1, 5)}

                time_points = np.linspace(0.1, duration - 0.1, min(10, int(duration * 5)))
                formant_data = {'F1': [], 'F2': [], 'F3': [], 'F4': []}

                for time_point in time_points:
                    for i, formant_num in enumerate(['F1', 'F2', 'F3', 'F4'], 1):
                        try:
                            f = call(formant, "Get value at time", i, time_point, "Hertz", "Linear")
                            if f and not np.isnan(f) and f > 0:
                                formant_data[formant_num].append(float(f))
                        except:
                            pass

                formant_stats = {}
                for formant_name, values in formant_data.items():
                    if values:
                        formant_stats[formant_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values))
                        }
                    else:
                        formant_stats[formant_name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

                return formant_stats

            except Exception:
                return {f'F{i}': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0} for i in range(1, 5)}

        result = self.safe_temp_file_operation(y, sr, formants_operation)
        return result if result is not None else {f'F{i}': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0} for i in range(1, 5)}

    def calculate_voice_quality_measures(self, y, sr):
        def quality_operation(temp_path):
            try:
                sound = parselmouth.Sound(temp_path)

                hnr_mean = hnr_std = 0.0
                try:
                    harmonicity = sound.to_harmonicity()
                    hnr_mean = call(harmonicity, "Get mean", 0, 0)
                    hnr_std = call(harmonicity, "Get standard deviation", 0, 0)
                    if np.isnan(hnr_mean):
                        hnr_mean = 0.0
                    if np.isnan(hnr_std):
                        hnr_std = 0.0
                except Exception:
                    hnr_mean = hnr_std = 0.0

                intensity_mean = intensity_std = 0.0
                try:
                    intensity = sound.to_intensity()
                    intensity_mean = call(intensity, "Get mean", 0, 0)
                    intensity_std = call(intensity, "Get standard deviation", 0, 0)
                    if np.isnan(intensity_mean):
                        intensity_mean = 0.0
                    if np.isnan(intensity_std):
                        intensity_std = 0.0
                except Exception:
                    intensity_mean = intensity_std = 0.0

                spectral_center_gravity = spectral_std = spectral_skewness = spectral_kurtosis = 0.0
                try:
                    spectrum = sound.to_spectrum()
                    spectral_center_gravity = call(spectrum, "Get centre of gravity", 1)
                    spectral_std = call(spectrum, "Get standard deviation", 1)
                    spectral_skewness = call(spectrum, "Get skewness", 1)
                    spectral_kurtosis = call(spectrum, "Get kurtosis", 1)

                    if np.isnan(spectral_center_gravity):
                        spectral_center_gravity = 0.0
                    if np.isnan(spectral_std):
                        spectral_std = 0.0
                    if np.isnan(spectral_skewness):
                        spectral_skewness = 0.0
                    if np.isnan(spectral_kurtosis):
                        spectral_kurtosis = 0.0

                except Exception:
                    spectral_center_gravity = spectral_std = spectral_skewness = spectral_kurtosis = 0.0

                return {
                    'hnr': {
                        'mean': float(hnr_mean),
                        'std': float(hnr_std)
                    },
                    'intensity': {
                        'mean': float(intensity_mean),
                        'std': float(intensity_std)
                    },
                    'spectral': {
                        'center_gravity': float(spectral_center_gravity),
                        'std': float(spectral_std),
                        'skewness': float(spectral_skewness),
                        'kurtosis': float(spectral_kurtosis)
                    }
                }

            except Exception:
                return {
                    'hnr': {'mean': 0.0, 'std': 0.0},
                    'intensity': {'mean': 0.0, 'std': 0.0},
                    'spectral': {'center_gravity': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
                }

        result = self.safe_temp_file_operation(y, sr, quality_operation)
        return result if result is not None else {
            'hnr': {'mean': 0.0, 'std': 0.0},
            'intensity': {'mean': 0.0, 'std': 0.0},
            'spectral': {'center_gravity': 0.0, 'std': 0.0, 'skewness': 0.0, 'kurtosis': 0.0}
        }

    def calculate_perturbation_measures(self, y, sr):
        def perturbation_operation(temp_path):
            return self.calculate_perturbation_praat_corrected(temp_path)
        
        result = self.safe_temp_file_operation(y, sr, perturbation_operation)
        
        if result is None:
            result = self.calculate_perturbation_measures_corrected(y, sr)
        
        return result

    def assess_disorder_risk(self, analysis_results, patient_info):
        disorder_risk = {}

        jitter = analysis_results.get('perturbation', {}).get('jitter', {}).get('local', 0)
        shimmer = analysis_results.get('perturbation', {}).get('shimmer', {}).get('local', 0)
        hnr = analysis_results.get('voice_quality', {}).get('hnr', {}).get('mean', 0)
        pitch_std = analysis_results.get('pitch', {}).get('std', 0)

        nodules_risk = 0
        if jitter > 1.04: 
            nodules_risk += min((jitter - 1.04) * 10, 25)
        if shimmer > 3.81: 
            nodules_risk += min((shimmer - 3.81) * 5, 25)
        if hnr < 20 and hnr > 0: 
            nodules_risk += min((20 - hnr) * 2, 20)
        if pitch_std > 20: 
            nodules_risk += min((pitch_std - 20) * 0.75, 15)

        nodules_level = 'low' if nodules_risk < 30 else 'medium' if nodules_risk < 60 else 'high'

        disorder_risk['vocal_nodules'] = {
            'name': 'Nódulos Vocales',
            'probability': min(int(nodules_risk), 85),
            'level': nodules_level,
            'score': min(int(nodules_risk), 85)
        }

        paralysis_risk = 0
        formants = analysis_results.get('formants', {})
        f1_std = formants.get('F1', {}).get('std', 0)
        f2_std = formants.get('F2', {}).get('std', 0)

        if hnr < 15 and hnr > 0: 
            paralysis_risk += min((15 - hnr) * 2, 30)
        if jitter > 3.0: 
            paralysis_risk += min((jitter - 3.0) * 8, 25)
        if f1_std > 100: 
            paralysis_risk += min((f1_std - 100) * 0.2, 20)
        if f2_std > 200: 
            paralysis_risk += min((f2_std - 200) * 0.075, 15)

        paralysis_level = 'low' if paralysis_risk < 35 else 'medium' if paralysis_risk < 65 else 'high'

        disorder_risk['vocal_paralysis'] = {
            'name': 'Parálisis Vocal',
            'probability': min(int(paralysis_risk), 90),
            'level': paralysis_level,
            'score': min(int(paralysis_risk), 90)
        }

        age = patient_info.get('age', '')
        if age and str(age).isdigit() and int(age) > 60:
            presbyphonia_risk = 0
            if jitter > 0.8: 
                presbyphonia_risk += min((jitter - 0.8) * 25, 20)
            if shimmer > 2.5: 
                presbyphonia_risk += min((shimmer - 2.5) * 15, 20)
            if hnr < 18 and hnr > 0: 
                presbyphonia_risk += min((18 - hnr) * 1.4, 25)

            presbyphonia_level = 'low' if presbyphonia_risk < 25 else 'medium' if presbyphonia_risk < 50 else 'high'

            disorder_risk['presbyphonia'] = {
                'name': 'Presbifonia',
                'probability': min(int(presbyphonia_risk), 75),
                'level': presbyphonia_level,
                'score': min(int(presbyphonia_risk), 75)
            }

        return disorder_risk

    def calculate_stability_score(self, analysis_results):
        score = 100.0
        jitter = analysis_results.get('perturbation', {}).get('jitter', {}).get('local', 0)
        shimmer = analysis_results.get('perturbation', {}).get('shimmer', {}).get('local', 0)
        hnr = analysis_results.get('voice_quality', {}).get('hnr', {}).get('mean', 0)
        pitch_cv = analysis_results.get('pitch', {}).get('coefficient_variation', 0)

        if jitter > 1.04:
            score -= min((jitter - 1.04) * 20, 30)
        if shimmer > 3.81:
            score -= min((shimmer - 3.81) * 10, 25)
        if hnr < 20 and hnr > 0:
            score -= min((20 - hnr) * 2, 25)
        if pitch_cv > 10:
            score -= min((pitch_cv - 10) * 2, 20)

        return max(float(score), 0.0)

    def clinical_analyze_audio(self, audio_data, metadata):
        y, sr = self.load_audio_from_blob(audio_data)

        if y is None:
            return {'error': 'No se pudo procesar el audio'}

        duration = len(y) / sr
        if duration < 1.0:
            return {'error': f'Audio demasiado corto: {duration:.2f}s < 1.0s'}

        max_amplitude = np.max(np.abs(y))
        if max_amplitude < 0.01:
            return {'error': 'Audio parece ser silencio o tiene amplitud muy baja'}

        y = librosa.util.normalize(y)
        intervals = librosa.effects.split(y, top_db=20)
        if len(intervals) > 0:
            y = np.concatenate([y[start:end] for start, end in intervals])

        if len(y) < sr * 1.0:
            return {'error': 'Audio demasiado corto para análisis clínico'}

        results = {}

        try:
            pitch_results = self.calculate_fundamental_frequency_advanced(y, sr)
            results['pitch'] = pitch_results

            formants_results = self.calculate_formants_clinical(y, sr)
            results['formants'] = formants_results

            perturbation_results = self.calculate_perturbation_measures(y, sr)
            results['perturbation'] = perturbation_results

            voice_quality_results = self.calculate_voice_quality_measures(y, sr)
            results['voice_quality'] = voice_quality_results

            results['stability_score'] = self.calculate_stability_score(results)

            patient_info = {
                'age': metadata.get('patient_age'),
                'gender': metadata.get('patient_gender'),
                'diagnosis': metadata.get('diagnosis')
            }
            results['disorder_risk'] = self.assess_disorder_risk(results, patient_info)

            results['pitch'] = results['pitch']['mean']
            results['jitter'] = results['perturbation']['jitter']['local']
            results['shimmer'] = results['perturbation']['shimmer']['local']
            results['hnr'] = results['voice_quality']['hnr']['mean']

            formants_array = []
            for i in range(1, 5):
                formant_key = f'F{i}'
                if formant_key in formants_results and 'mean' in formants_results[formant_key]:
                    formants_array.append(formants_results[formant_key]['mean'])
                else:
                    formants_array.append(0.0)

            results['formants'] = formants_array

            results['session_metadata'] = {
                'analysis_date': datetime.now().isoformat(),
                'mode': metadata.get('mode', 'rehabilitation'),
                'exercise': metadata.get('exercise', 'sustained_a'),
                'patient_info': patient_info,
                'audio_duration': float(len(y) / sr)
            }

            results = self.convert_to_serializable(results)
            return results

        except Exception as e:
            return {'error': f'Error en análisis clínico: {str(e)}'}

clinical_analyzer = ClinicalVocalAnalyzer()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Clínica de Análisis Vocal</title>
    </head>
    <body>
        <h1>API Clínica de Análisis Vocal</h1>
        <p>Sistema especializado para rehabilitación vocal, diagnóstico de trastornos y entrenamiento profesional.</p>
        <h3>Endpoints:</h3>
        <ul>
            <li><strong>POST /api/clinical_analyze</strong> - Análisis clínico completo</li>
            <li><strong>GET /api/health</strong> - Estado del sistema</li>
        </ul>
    </body>
    </html>
    """

@app.route('/api/clinical_analyze', methods=['POST'])
def clinical_analyze():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No se encontró archivo de audio'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'Archivo vacío'}), 400

        metadata = {
            'mode': request.form.get('mode', 'rehabilitation'),
            'exercise': request.form.get('exercise', 'sustained_a'),
            'patient_age': request.form.get('patient_age', ''),
            'patient_gender': request.form.get('patient_gender', ''),
            'diagnosis': request.form.get('diagnosis', '')
        }

        audio_data = audio_file.read()
        if len(audio_data) == 0:
            return jsonify({'error': 'Archivo de audio vacío'}), 400

        results = clinical_analyzer.clinical_analyze_audio(audio_data, metadata)

        if 'error' in results:
            return jsonify(results), 400

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': f'Error del servidor: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '3.0.0-optimized',
        'capabilities': [
            'corrected_perturbation_analysis',
            'clinical_pitch_analysis',
            'formant_detection',
            'voice_quality_assessment',
            'disorder_risk_evaluation'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)