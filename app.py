from flask import Flask, render_template, request, redirect, url_for, Response, session, send_file, jsonify, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import sqlite3
import os
import math
import uuid
from scipy.spatial.distance import cosine, euclidean
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from scipy.signal import savgol_filter, find_peaks
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
import pywt
from scipy.spatial.distance import cityblock
import json
from datetime import datetime
from functools import wraps

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB = "gait_forensics.db"
app.secret_key = "supersecretkey"

def init_db():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            gender TEXT,
            role TEXT,
            gait_vector TEXT,
            video_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS comparisons (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      profile1_id TEXT,
                      profile2_id TEXT,
                      similarity_score REAL,
                      confidence REAL,
                      component_scores TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (profile1_id) REFERENCES profiles (id),
                      FOREIGN KEY (profile2_id) REFERENCES profiles (id)
                      )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Mediapipe Setup ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,  # Increased complexity for better accuracy
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.8,  # Higher confidence threshold
    min_tracking_confidence=0.8
)

# ---------- Enhanced Feature Extraction ----------
def calculate_angle(a, b, c):
    """Calculate angle between three points with robust error handling"""
    try:
        ba = a - b
        bc = c - b
        
        ba_norm = np.linalg.norm(ba)
        bc_norm = np.linalg.norm(bc)
        
        if ba_norm < 1e-8 or bc_norm < 1e-8:
            return 0.0
            
        cosine_angle = np.dot(ba, bc) / (ba_norm * bc_norm)
        cosine_angle = np.clip(cosine_angle, -1, 1)
        return np.degrees(np.arccos(cosine_angle))
    except:
        return 0.0

def extract_comprehensive_frame_features(landmarks, frame_shape):
    """Extract enhanced frame-level features with comprehensive biomechanical analysis"""
    h, w = frame_shape[:2]
    
    def get_coords(idx):
        if idx < len(landmarks):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
        return np.array([0, 0])
    
    # Extract all major joints with enhanced set
    joints = {}
    key_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]  # Added nose and all limbs
    
    for idx in key_indices:
        joints[idx] = get_coords(idx)
    
    # Calculate comprehensive angles with error handling
    angles = {}
    
    # Leg angles - comprehensive set
    angles['left_hip_knee_ankle'] = calculate_angle(joints[23], joints[25], joints[27])
    angles['right_hip_knee_ankle'] = calculate_angle(joints[24], joints[26], joints[28])
    angles['left_knee_flexion'] = calculate_angle(joints[23], joints[25], joints[27])
    angles['right_knee_flexion'] = calculate_angle(joints[24], joints[26], joints[28])
    angles['left_hip_flexion'] = calculate_angle(joints[11], joints[23], joints[25])
    angles['right_hip_flexion'] = calculate_angle(joints[12], joints[24], joints[26])
    
    # Arm angles
    angles['left_shoulder_elbow_wrist'] = calculate_angle(joints[11], joints[13], joints[15])
    angles['right_shoulder_elbow_wrist'] = calculate_angle(joints[12], joints[14], joints[16])
    angles['left_elbow_flexion'] = calculate_angle(joints[11], joints[13], joints[15])
    angles['right_elbow_flexion'] = calculate_angle(joints[12], joints[14], joints[16])
    
    # Torso and posture angles
    shoulder_center = (joints[11] + joints[12]) / 2
    hip_center = (joints[23] + joints[24]) / 2
    angles['torso_vertical'] = calculate_angle(
        shoulder_center, 
        hip_center, 
        hip_center + np.array([0, -100])
    )
    
    # Spatial relationships with enhanced metrics
    spatial_features = {}
    spatial_features['step_width'] = np.linalg.norm(joints[27] - joints[28])
    spatial_features['shoulder_width'] = np.linalg.norm(joints[11] - joints[12])
    spatial_features['hip_width'] = np.linalg.norm(joints[23] - joints[24])
    spatial_features['stride_length'] = np.linalg.norm(joints[23] - joints[24])  # Hip separation as proxy
    
    # Height normalization with robust estimation
    height_estimate = np.linalg.norm(shoulder_center - (joints[27] + joints[28]) / 2)
    if height_estimate > 0:
        for key in ['step_width', 'shoulder_width', 'hip_width', 'stride_length']:
            spatial_features[key] /= height_estimate
    
    # Velocity and acceleration features
    velocity_features = []
    if hasattr(extract_comprehensive_frame_features, 'prev_joints'):
        prev_joints = extract_comprehensive_frame_features.prev_joints
        for idx in key_indices:
            if idx in prev_joints and idx in joints:
                velocity = np.linalg.norm(joints[idx] - prev_joints[idx])
                velocity_features.append(velocity)
    else:
        velocity_features = [0] * len(key_indices)
    
    extract_comprehensive_frame_features.prev_joints = joints
    
    # Enhanced symmetry metrics
    symmetry = {}
    symmetry['leg_angle_symmetry'] = abs(angles['left_hip_knee_ankle'] - angles['right_hip_knee_ankle'])
    symmetry['arm_angle_symmetry'] = abs(angles['left_shoulder_elbow_wrist'] - angles['right_shoulder_elbow_wrist'])
    symmetry['hip_flexion_symmetry'] = abs(angles['left_hip_flexion'] - angles['right_hip_flexion'])
    
    # Posture and balance features
    posture = {}
    nose_pos = joints[0]
    posture['body_lean'] = np.linalg.norm(nose_pos - hip_center)
    posture['balance_ratio'] = np.linalg.norm(joints[27] - joints[28]) / (height_estimate + 1e-8)
    
    # Gait phase estimation
    gait_phase = {}
    left_ankle_height = joints[27][1]
    right_ankle_height = joints[28][1]
    gait_phase['ankle_height_diff'] = abs(left_ankle_height - right_ankle_height)
    gait_phase['stance_ratio'] = min(left_ankle_height, right_ankle_height) / (max(left_ankle_height, right_ankle_height) + 1e-8)
    
    # Confidence based on landmark visibility
    confidence = np.mean([landmarks[i].visibility for i in key_indices if i < len(landmarks)])
    
    # Combine all features into comprehensive vector
    feature_vector = (
        list(angles.values()) +
        list(spatial_features.values()) +
        velocity_features +
        list(symmetry.values()) +
        list(posture.values()) +
        list(gait_phase.values()) +
        [confidence]
    )
    
    return feature_vector

def extract_temporal_dynamics(feature_sequence):
    """Extract comprehensive temporal dynamics and patterns"""
    if len(feature_sequence) < 10:
        return np.zeros(200)  # Return zeros if insufficient data
    
    features = np.array(feature_sequence)
    
    # Apply robust smoothing
    smoothed_features = []
    for i in range(features.shape[1]):
        try:
            window_size = min(15, len(features) // 2 * 2 + 1)
            if window_size >= 5 and window_size <= len(features):
                smoothed = savgol_filter(features[:, i], window_size, 3, mode='nearest')
                smoothed_features.append(smoothed)
            else:
                smoothed_features.append(features[:, i])
        except:
            smoothed_features.append(features[:, i])
    
    features = np.column_stack(smoothed_features)
    
    # Comprehensive statistical features
    mean_features = np.mean(features, axis=0)
    std_features = np.std(features, axis=0)
    median_features = np.median(features, axis=0)
    range_features = np.ptp(features, axis=0)
    skewness = np.mean((features - mean_features) ** 3, axis=0) / (std_features ** 3 + 1e-8)
    kurtosis = np.mean((features - mean_features) ** 4, axis=0) / (std_features ** 4 + 1e-8)
    
    # Enhanced velocity and acceleration analysis
    velocities = np.diff(features, axis=0)
    accelerations = np.diff(velocities, axis=0) if len(velocities) > 1 else np.zeros_like(velocities)
    
    vel_mean = np.mean(velocities, axis=0) if len(velocities) > 0 else np.zeros(features.shape[1])
    vel_std = np.std(velocities, axis=0) if len(velocities) > 0 else np.zeros(features.shape[1])
    vel_max = np.max(np.abs(velocities), axis=0) if len(velocities) > 0 else np.zeros(features.shape[1])
    
    acc_mean = np.mean(accelerations, axis=0) if len(accelerations) > 0 else np.zeros(features.shape[1])
    acc_std = np.std(accelerations, axis=0) if len(accelerations) > 0 else np.zeros(features.shape[1])
    
    # Advanced periodicity analysis
    autocorr_features = []
    for i in range(min(12, features.shape[1])):
        signal = features[:, i]
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) >= 8:
            # Get multiple lags for better periodicity analysis
            autocorr_norm = autocorr[:8] / (autocorr[0] + 1e-8)
            autocorr_features.extend(autocorr_norm)
        else:
            autocorr_features.extend([0] * 8)
    
    # Combine all temporal features
    temporal_dynamics = np.concatenate([
        mean_features, std_features, median_features, range_features,
        skewness[:20], kurtosis[:20],  # Limit higher order statistics
        vel_mean, vel_std, vel_max,
        acc_mean, acc_std,
        autocorr_features[:96]  # Limit autocorrelation features
    ])
    
    return temporal_dynamics

def extract_frequency_domain_features(feature_sequence):
    """Extract comprehensive frequency domain characteristics"""
    features = np.array(feature_sequence)
    if len(features) < 10:
        return np.zeros(60)
    
    freq_features = []
    
    for i in range(min(10, features.shape[1])):
        signal = features[:, i]
        
        # FFT analysis with windowing
        fft_result = fft(signal)
        fft_magnitude = np.abs(fft_result)
        
        # Get dominant frequencies and their magnitudes
        if len(fft_magnitude) > 3:
            # Exclude DC component and get top frequencies
            dominant_indices = np.argsort(fft_magnitude[1:min(10, len(fft_magnitude)//2)])[-5:]
            dominant_freqs = fft_magnitude[1:min(10, len(fft_magnitude)//2)][dominant_indices]
            freq_features.extend(dominant_freqs)
        else:
            freq_features.extend([0] * 5)
    
    # Spectral centroid and spread
    for i in range(min(2, features.shape[1])):
        signal = features[:, i]
        if len(signal) > 1:
            fft_result = fft(signal)
            magnitudes = np.abs(fft_result[:len(fft_result)//2])
            frequencies = np.arange(len(magnitudes))
            
            if np.sum(magnitudes) > 0:
                spectral_centroid = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
                spectral_spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitudes) / np.sum(magnitudes))
                freq_features.extend([spectral_centroid, spectral_spread])
            else:
                freq_features.extend([0, 0])
        else:
            freq_features.extend([0, 0])
    
    # Pad to consistent length
    while len(freq_features) < 60:
        freq_features.append(0)
    
    return np.array(freq_features[:60])

def extract_wavelet_features(feature_sequence):
    """Extract multi-resolution wavelet transform features"""
    features = np.array(feature_sequence)
    if len(features) < 8:
        return np.zeros(60)
    
    wavelet_features = []
    
    try:
        for i in range(min(6, features.shape[1])):
            signal = features[:, i]
            
            if len(signal) >= 8:
                # Multi-level wavelet decomposition
                coeffs = pywt.wavedec(signal, 'db4', level=min(4, int(np.log2(len(signal)))))
                
                for coeff in coeffs:
                    if len(coeff) > 0:
                        # Extract multiple statistical features from coefficients
                        wavelet_features.extend([
                            np.mean(coeff),
                            np.std(coeff),
                            np.max(np.abs(coeff)),
                            np.min(coeff),
                            np.median(coeff)
                        ])
                    else:
                        wavelet_features.extend([0, 0, 0, 0, 0])
            else:
                wavelet_features.extend([0] * 30)  # 5 features * 6 levels
    except Exception as e:
        wavelet_features = [0] * 60
    
    # Ensure consistent length
    return np.array(wavelet_features[:60])

def extract_gait_cycle_characteristics(feature_sequence):
    """Extract detailed gait cycle specific features"""
    features = np.array(feature_sequence)
    if len(features) < 15:
        return np.zeros(30)
    
    gait_cycle_features = []
    
    try:
        # Use multiple signals for gait cycle detection
        step_widths = [frame[12] for frame in features]  # Step width index
        hip_angles = [frame[0] for frame in features]   # Left hip angle index
        
        # Enhanced peak detection for step events
        peaks_step, _ = find_peaks(step_widths, height=np.mean(step_widths), distance=5, prominence=0.01)
        peaks_hip, _ = find_peaks(hip_angles, height=np.mean(hip_angles), distance=5, prominence=1.0)
        
        # Gait cycle analysis
        if len(peaks_step) >= 2:
            cycle_periods = np.diff(peaks_step)
            gait_cycle_features.extend([
                np.mean(cycle_periods),
                np.std(cycle_periods),
                len(peaks_step),  # Step count
                np.mean([step_widths[p] for p in peaks_step]),  # Average step width
                np.std([step_widths[p] for p in peaks_step]),   # Step width variability
                np.max(cycle_periods),
                np.min(cycle_periods)
            ])
        else:
            gait_cycle_features.extend([0] * 7)
        
        # Rhythm and regularity analysis
        if len(step_widths) > 1:
            step_changes = np.diff(step_widths)
            gait_cycle_features.extend([
                np.mean(step_changes),
                np.std(step_changes),
                np.mean(np.abs(step_changes)),
                np.correlate(step_widths, step_widths, mode='valid')[0] / len(step_widths)
            ])
        else:
            gait_cycle_features.extend([0] * 4)
        
        # Symmetry and coordination
        if len(peaks_step) >= 2 and len(peaks_hip) >= 2:
            step_hip_correlation = np.corrcoef(
                step_widths[:min(len(step_widths), len(hip_angles))],
                hip_angles[:min(len(step_widths), len(hip_angles))]
            )[0, 1] if not np.isnan(np.corrcoef(
                step_widths[:min(len(step_widths), len(hip_angles))],
                hip_angles[:min(len(step_widths), len(hip_angles))]
            )[0, 1]) else 0
            gait_cycle_features.append(step_hip_correlation)
        else:
            gait_cycle_features.append(0)
            
    except Exception as e:
        gait_cycle_features = [0] * 12
    
    # Pad to consistent length
    while len(gait_cycle_features) < 30:
        gait_cycle_features.append(0)
    
    return np.array(gait_cycle_features[:30])

def extract_advanced_gait_features(video_path):
    """Extract comprehensive gait features with robust error handling"""
    cap = cv2.VideoCapture(video_path)
    features_sequence = []
    frame_count = 0
    max_frames = 180  # 6 seconds at 30fps
    
    # Reset previous joints
    if hasattr(extract_comprehensive_frame_features, 'prev_joints'):
        delattr(extract_comprehensive_frame_features, 'prev_joints')
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if not results.pose_landmarks:
            continue
            
        try:
            # Extract comprehensive frame features
            frame_features = extract_comprehensive_frame_features(
                results.pose_landmarks.landmark, frame.shape
            )
            features_sequence.append(frame_features)
            frame_count += 1
        except Exception as e:
            continue
    
    cap.release()
    
    if len(features_sequence) < 30:
        return None
        
    try:
        # Extract multiple feature representations
        temporal_features = extract_temporal_dynamics(features_sequence)
        frequency_features = extract_frequency_domain_features(features_sequence)
        wavelet_features = extract_wavelet_features(features_sequence)
        gait_cycle_features = extract_gait_cycle_characteristics(features_sequence)
        
        # Combine all feature types
        combined_features = np.concatenate([
            temporal_features,
            frequency_features,
            wavelet_features,
            gait_cycle_features
        ])
        
        return combined_features.tolist()
    except Exception as e:
        return None

# ---------- Advanced Comparison Algorithm ----------
def calculate_feature_weights(vec1, vec2):
    """Calculate adaptive weights for different feature types"""
    n_features = min(len(vec1), len(vec2))
    weights = np.ones(n_features)
    
    # Feature type based weighting
    if n_features > 100:
        # Temporal features (first ~40%)
        temporal_end = int(n_features * 0.4)
        weights[:temporal_end] *= 1.6
        
        # Frequency features (next ~20%)
        freq_start = temporal_end
        freq_end = freq_start + int(n_features * 0.2)
        weights[freq_start:freq_end] *= 1.4
        
        # Wavelet features (next ~20%)
        wavelet_start = freq_end
        wavelet_end = wavelet_start + int(n_features * 0.2)
        weights[wavelet_start:wavelet_end] *= 1.3
        
        # Gait cycle features (last ~20%)
        cycle_start = n_features - int(n_features * 0.2)
        weights[cycle_start:] *= 1.8
    
    # Variance-based weighting
    feature_variance = np.var([vec1[:n_features], vec2[:n_features]], axis=0)
    high_var_mask = feature_variance > np.percentile(feature_variance, 50)
    weights[high_var_mask] *= 1.2
    
    return weights / (np.sum(weights) + 1e-8)

def apply_nonlinear_scaling(score):
    """Apply non-linear scaling to emphasize high similarity scores"""
    if score < 0.3:
        return score * 0.7  # Strong penalty for very low scores
    elif score < 0.6:
        return score * 0.9  # Moderate penalty for low scores
    elif score < 0.8:
        return score        # Linear for medium scores
    elif score < 0.9:
        return 0.8 + (score - 0.8) * 1.5  # Boost high scores
    else:
        return 0.95 + (score - 0.9) * 0.5  # Saturate very high scores

def advanced_gait_comparison(vec1, vec2):
    """Advanced multi-metric gait comparison with comprehensive analysis"""
    if vec1 is None or vec2 is None or len(vec1) == 0 or len(vec2) == 0:
        return {
            "final_score": 0.0,
            "confidence": 0.0,
            "component_scores": {},
            "interpretation": "Invalid input vectors"
        }
    
    vec1 = np.array(vec1, dtype=np.float64)
    vec2 = np.array(vec2, dtype=np.float64)
    
    # Ensure compatible dimensions
    min_len = min(len(vec1), len(vec2))
    if min_len < 50:  # Minimum feature length requirement
        return {
            "final_score": 0.0,
            "confidence": 0.0,
            "component_scores": {},
            "interpretation": "Insufficient feature data"
        }
    
    vec1 = vec1[:min_len]
    vec2 = vec2[:min_len]
    
    # Robust normalization
    try:
        vec1_norm = (vec1 - np.mean(vec1)) / (np.std(vec1) + 1e-8)
        vec2_norm = (vec2 - np.mean(vec2)) / (np.std(vec2) + 1e-8)
    except:
        vec1_norm = vec1
        vec2_norm = vec2
    
    similarity_metrics = {}
    
    # 1. Enhanced Cosine Similarity
    try:
        cos_sim = cosine_similarity([vec1_norm], [vec2_norm])[0][0]
        similarity_metrics['cosine'] = max(0.0, float(cos_sim))
    except:
        similarity_metrics['cosine'] = 0.0
    
    # 2. Dynamic Time Warping
    try:
        dtw_distance = dtw.distance_fast(vec1_norm, vec2_norm)
        max_possible_dtw = dtw.distance_fast(np.zeros_like(vec1_norm), np.ones_like(vec2_norm))
        dtw_similarity = 1.0 - (dtw_distance / (max_possible_dtw + 1e-8))
        similarity_metrics['dtw'] = max(0.0, dtw_similarity)
    except:
        similarity_metrics['dtw'] = 0.0
    
    # 3. Pearson Correlation
    try:
        correlation = np.corrcoef(vec1_norm, vec2_norm)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        similarity_metrics['correlation'] = max(0.0, (correlation + 1.0) / 2.0)
    except:
        similarity_metrics['correlation'] = 0.0
    
    # 4. Manhattan Similarity
    try:
        manhattan_dist = cityblock(vec1_norm, vec2_norm)
        max_manhattan = cityblock(np.zeros_like(vec1_norm), np.ones_like(vec1_norm))
        manhattan_sim = 1.0 - (manhattan_dist / (max_manhattan + 1e-8))
        similarity_metrics['manhattan'] = max(0.0, manhattan_sim)
    except:
        similarity_metrics['manhattan'] = 0.0
    
    # 5. Euclidean Similarity
    try:
        euclidean_dist = np.linalg.norm(vec1_norm - vec2_norm)
        max_euclidean = np.linalg.norm(np.ones_like(vec1_norm) - np.zeros_like(vec1_norm))
        euclidean_sim = 1.0 - (euclidean_dist / (max_euclidean + 1e-8))
        similarity_metrics['euclidean'] = max(0.0, euclidean_sim)
    except:
        similarity_metrics['euclidean'] = 0.0
    
    # 6. Feature-weighted similarity
    try:
        feature_weights = calculate_feature_weights(vec1, vec2)
        weighted_similarity = np.sum(feature_weights * np.minimum(np.abs(vec1_norm), np.abs(vec2_norm))) / np.sum(feature_weights)
        similarity_metrics['weighted'] = max(0.0, weighted_similarity)
    except:
        similarity_metrics['weighted'] = 0.0
    
    # Adaptive metric weighting based on reliability
    metric_weights = {
        'cosine': 0.22,
        'dtw': 0.20,
        'correlation': 0.18,
        'weighted': 0.15,
        'manhattan': 0.13,
        'euclidean': 0.12
    }
    
    # Calculate final weighted score
    final_score = 0.0
    for metric, weight in metric_weights.items():
        if metric in similarity_metrics:
            final_score += similarity_metrics[metric] * weight
    
    # Confidence calculation based on metric agreement and quality
    metric_scores = [similarity_metrics.get(metric, 0) for metric in metric_weights.keys()]
    score_std = np.std(metric_scores)
    score_range = max(metric_scores) - min(metric_scores) if metric_scores else 0
    
    confidence = max(0.3, 1.0 - score_std * 1.2 - score_range * 0.5)
    
    # Apply non-linear scaling
    final_score = apply_nonlinear_scaling(final_score)
    
    # Ensure valid ranges
    final_score = max(0.0, min(1.0, final_score))
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "final_score": float(final_score),
        "confidence": float(confidence),
        "component_scores": {k: float(v) for k, v in similarity_metrics.items()},
        "interpretation": interpret_advanced_similarity(final_score, confidence)
    }

def interpret_advanced_similarity(score, confidence):
    """Provide detailed interpretation of similarity results"""
    if confidence < 0.5:
        base_interpretation = "Low confidence - "
    elif confidence < 0.75:
        base_interpretation = "Medium confidence - "
    else:
        base_interpretation = "High confidence - "
    
    if score >= 0.90:
        return base_interpretation + "VERY STRONG MATCH: Almost certainly the same person"
    elif score >= 0.85:
        return base_interpretation + "STRONG MATCH: Highly likely same person"
    elif score >= 0.78:
        return base_interpretation + "GOOD MATCH: Probably same person"
    elif score >= 0.70:
        return base_interpretation + "MODERATE MATCH: Possibly same person"
    elif score >= 0.60:
        return base_interpretation + "WEAK MATCH: Unlikely same person"
    elif score >= 0.50:
        return base_interpretation + "VERY WEAK MATCH: Probably different persons"
    else:
        return base_interpretation + "NO MATCH: Different persons"

# ---------- Enhanced Real-Time Gait Recognizer ----------
class AdvancedRealTimeGaitRecognizer:
    def __init__(self):
        self.frame_buffer = []
        self.buffer_size = 90  # 3 seconds at 30fps
        self.current_match = "Unknown"
        self.current_confidence = 0.0
        self.match_history = []
        self.known_profiles = self.load_profiles()
        self.min_confidence_threshold = 0.78
        self.required_consensus = 0.7
        self.current_landmarks = None
        
    def load_profiles(self):
        """Load profiles with comprehensive validation"""
        conn = sqlite3.connect(DB)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, gait_vector FROM profiles")
        profiles = []
        for row in cursor.fetchall():
        
            try:
                vec = json.loads(row[2])
                # print(vec,"vector")
                if vec and len(vec) >= 150:  # Minimum feature length requirement
                    profiles.append({
                        'id': row[0],
                        'name': row[1],
                        'vector': np.array(vec)
                    })
            except Exception as e:
                continue
        conn.close()
        print(f"Loaded {len(profiles)} validated profiles for real-time recognition")
        return profiles
    
    def is_good_pose(self, pose_landmarks):
        """Enhanced pose quality assessment"""
        if not pose_landmarks:
            return False
            
        key_points = [11, 12, 23, 24, 25, 26, 27, 28]  # Major body joints
        visibilities = []
        
        for i in key_points:
            if i < len(pose_landmarks.landmark):
                visibilities.append(pose_landmarks.landmark[i].visibility)
        
        if len(visibilities) < 6:  # Need most key points
            return False
            
        avg_visibility = np.mean(visibilities)
        return avg_visibility > 0.75
    
    def assess_buffer_quality(self):
        """Comprehensive buffer quality assessment"""
        if len(self.frame_buffer) < 15:
            return 0.0
            
        try:
            features = np.array(self.frame_buffer)
            
            # Movement/variation assessment
            temporal_variance = np.mean(np.var(features, axis=0))
            movement_quality = min(1.0, temporal_variance / 0.02)
            
            # Consistency assessment
            feature_std = np.mean(np.std(features, axis=0))
            consistency_quality = max(0, 1 - feature_std * 8)
            
            # Completeness assessment (non-zero features)
            non_zero_ratio = np.mean(features != 0)
            completeness_quality = non_zero_ratio
            
            return (movement_quality + consistency_quality + completeness_quality) / 3
        except:
            return 0.0
    
    def process_frame(self, frame, pose_landmarks):
        """Enhanced frame processing with comprehensive analysis"""
        self.current_landmarks = pose_landmarks
        
        if pose_landmarks and self.is_good_pose(pose_landmarks):
            try:
                current_features = extract_comprehensive_frame_features(
                    pose_landmarks.landmark, frame.shape
                )
                self.frame_buffer.append(current_features)
                
                # Maintain buffer size
                if len(self.frame_buffer) > self.buffer_size:
                    self.frame_buffer.pop(0)
                
                # Perform recognition with quality threshold
                if len(self.frame_buffer) >= 45:  # 1.5 seconds minimum
                    buffer_quality = self.assess_buffer_quality()
                    if buffer_quality > 0.65:
                        self.analyze_gait_pattern()
            except Exception as e:
                print(f"Error processing frame: {e}")
        
        return self.annotate_frame(frame)
    
    def analyze_gait_pattern(self):
        """Enhanced gait pattern analysis with multiple validation layers"""
        try:
            temporal_features = extract_temporal_dynamics(self.frame_buffer)
            
            if temporal_features is None or np.all(temporal_features == 0) or len(self.known_profiles) == 0:
                self.update_match("Unknown", 0.0)
                return
            
            best_match = None
            best_score = 0.0
            candidate_scores = []
            
            # Compare with all known profiles
            for profile in self.known_profiles:
                comparison_result = advanced_gait_comparison(
                    temporal_features, profile['vector'].tolist()
                )
                score = comparison_result["final_score"]
                confidence = comparison_result["confidence"]
                
                # Combined score considering both similarity and confidence
                combined_score = score * confidence
                
                candidate_scores.append((profile['name'], combined_score, score, confidence))
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = profile['name']
            
            # Sort candidates by score
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Multi-layer validation
            if best_match and len(candidate_scores) > 0:
                best_candidate = candidate_scores[0]
                
                # Check confidence threshold
                if best_candidate[1] >= self.min_confidence_threshold:
                    # Check score margin over second best
                    if len(candidate_scores) > 1:
                        score_margin = best_candidate[1] - candidate_scores[1][1]
                        margin_adequate = score_margin > 0.12
                    else:
                        margin_adequate = True
                    
                    # Check absolute score quality
                    absolute_score_adequate = best_candidate[2] > 0.70
                    
                    # Check confidence quality
                    confidence_adequate = best_candidate[3] > 0.65
                    
                    if margin_adequate and absolute_score_adequate and confidence_adequate:
                        self.update_match(best_match, best_candidate[1])
                        return
            
            # If no confident match found
            self.update_match("Unknown", best_score if candidate_scores else 0.0)
            
        except Exception as e:
            print(f"Error in gait pattern analysis: {e}")
            self.update_match("Unknown", 0.0)
    
    def update_match(self, match, score):
        """Stable match updating with temporal consistency"""
        self.match_history.append((match, score))
        
        # Keep limited history
        if len(self.match_history) > 20:
            self.match_history.pop(0)
        
        # Apply temporal consistency filter with enhanced logic
        if len(self.match_history) >= 10:
            recent_matches = [m for m, s in self.match_history[-10:]]
            match_counts = {}
            
            for m in recent_matches:
                match_counts[m] = match_counts.get(m, 0) + 1
            
            # Find majority match with enhanced criteria
            if match_counts:
                majority_match, majority_count = max(match_counts.items(), key=lambda x: x[1])
                consensus_ratio = majority_count / len(recent_matches)
                
                if consensus_ratio >= self.required_consensus:
                    # Calculate weighted average score for the majority match
                    majority_scores = [s for m, s in self.match_history[-10:] if m == majority_match]
                    if majority_scores:
                        avg_score = sum(majority_scores) / len(majority_scores)
                        
                        # Only update if the majority match is significantly better
                        current_scores = [s for m, s in self.match_history[-5:] if m == self.current_match]
                        current_avg = sum(current_scores) / len(current_scores) if current_scores else 0
                        
                        if avg_score >= current_avg * 0.9:  # 90% of current score
                            self.current_match = majority_match
                            self.current_confidence = avg_score
                            return
        
        # Default to current decision with smoothing
        self.current_match = match
        self.current_confidence = score
    
    def annotate_frame(self, frame):
        """Enhanced frame annotation with comprehensive information"""
        # Draw pose landmarks if available
        if self.current_landmarks:
            mp_drawing.draw_landmarks(
                frame, self.current_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        # Display recognition results with enhanced visualization
        if self.current_match != "Unknown":
            print(self.current_match,"hhhh")
            # Color coding based on confidence
            if self.current_confidence > 0.85:
                color = (0, 255, 0)  # Green - high confidence
                status = "HIGH CONFIDENCE MATCH"
            elif self.current_confidence > 0.75:
                color = (0, 255, 255)  # Yellow - good confidence
                status = "GOOD MATCH"
            else:
                color = (0, 165, 255)  # Orange - medium confidence
                status = "MEDIUM CONFIDENCE"
            
            # Main identification display
            cv2.putText(frame, f"IDENTIFIED: {self.current_match}", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            cv2.putText(frame, f"Confidence: {self.current_confidence:.1%}", (30, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Status: {status}", (30, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            # Unknown person display
            cv2.putText(frame, "UNKNOWN PERSON", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if self.current_confidence > 0:
                cv2.putText(frame, f"Best Match Score: {self.current_confidence:.1%}", (30, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # System status information
        cv2.putText(frame, f"Analysis Frames: {len(self.frame_buffer)}/{self.buffer_size}", (30, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Profiles Loaded: {len(self.known_profiles)}", (30, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Buffer quality indicator
        buffer_quality = self.assess_buffer_quality()
        quality_color = (0, 255, 0) if buffer_quality > 0.7 else (0, 255, 255) if buffer_quality > 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Data Quality: {buffer_quality:.1%}", (30, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # Instructions
        cv2.putText(frame, "Walk naturally for 2-3 seconds for identification", 
                   (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

# Global recognizer instance
gait_recognizer = AdvancedRealTimeGaitRecognizer()
gait_recognizer.load_profiles()

# ---------- Admin Authentication ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == '1234':
            session['admin'] = True
            return redirect(url_for('index'))
        return render_template('admin_login.html', error="Invalid credentials")
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login'))

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('admin'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return wrapper

# ---------- Routes ----------
@app.route('/')
@admin_required
def index():
    return render_template("index.html")

@app.route('/admin/profiles')
@admin_required
def profiles():
    return render_template('profiles.html')

@app.route('/admin/compare')
@admin_required
def compare():
    return render_template('compare.html')

@app.route('/admin/realtime')
@admin_required
def admin_realtime():
    return render_template('realtime.html')

@app.route('/upload', methods=['POST'])
@admin_required
def upload_profile():
    name = request.form.get("name")
    age = request.form.get("age")
    gender = request.form.get("gender")
    role = request.form.get("role")
    file = request.files.get("video")

    if not all([name, age, gender, role, file]):
        return jsonify({"error": "All fields are required"}), 400

    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Use advanced feature extraction
    gait_signature = extract_advanced_gait_features(path)
    if gait_signature is None:
        os.remove(path)
        return jsonify({"error": "Could not extract sufficient gait features from video"}), 500

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    profile_id = uuid.uuid4().hex
    cursor.execute("""
        INSERT INTO profiles (id, name, age, gender, role, gait_vector, video_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (profile_id, name, age, gender, role, json.dumps(gait_signature), path))
    conn.commit()
    conn.close()

    return jsonify({"message": "Profile saved successfully", "id": profile_id})

@app.route('/profiles', methods=['GET'])
@admin_required
def get_profiles():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, age, gender, role, gait_vector, created_at
        FROM profiles ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    profiles = []
    for r in rows:
        gait_vector = json.loads(r[5]) if r[5] else []
        profiles.append({
            "id": r[0],
            "name": r[1],
            "age": r[2],
            "gender": r[3],
            "role": r[4],
            "feature_length": len(gait_vector),
            "created_at": r[6]
        })
    return jsonify(profiles)

@app.route('/delete_profile/<profile_id>', methods=['DELETE'])
@admin_required
def delete_profile(profile_id):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("SELECT video_path FROM profiles WHERE id = ?", (profile_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return jsonify({"error": "Profile not found"}), 404

    video_path = row[0]
    if os.path.exists(video_path):
        os.remove(video_path)

    cursor.execute("DELETE FROM profiles WHERE id = ?", (profile_id,))
    conn.commit()
    conn.close()
    return jsonify({"message": "Profile deleted successfully"})

@app.route('/profile/<profile_id>', methods=['GET'])
@admin_required
def view_profile(profile_id):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, age, gender, role, gait_vector, video_path, created_at
        FROM profiles WHERE id = ?
    """, 
    (profile_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return "Profile not found", 404

    gait_vector = json.loads(row[5]) if row[5] else []
    profile = {
        "id": row[0],
        "name": row[1],
        "age": row[2],
        "gender": row[3],
        "role": row[4],
        "feature_length": len(gait_vector),
        "video_path": row[6],
        "created_at": row[7]
    }

    return render_template("view_profile.html", profile=profile)

@app.route('/uploads/<path:filename>')
@admin_required
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/compare', methods=['POST'])
@admin_required
def compare_profiles():
    data = request.json
    id1, id2 = data.get("id1"), data.get("id2")

    if not id1 or not id2:
        return jsonify({"error": "Both profile IDs are required"}), 400

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name, gait_vector FROM profiles WHERE id=?", (id1,))
    v1 = cursor.fetchone()
    cursor.execute("SELECT name, gait_vector FROM profiles WHERE id=?", (id2,))
    v2 = cursor.fetchone()
    conn.close()

    if not v1 or not v2:
        return jsonify({"error": "One or both profiles not found"}), 404

    try:
        vec1 = json.loads(v1[1]) if v1[1] else []
        vec2 = json.loads(v2[1]) if v2[1] else []
    except:
        return jsonify({"error": "Invalid gait vector data"}), 500

    if not vec1 or not vec2:
        return jsonify({"error": "No gait features available for comparison"}), 400

    # Perform advanced comparison
    comparison_result = advanced_gait_comparison(vec1, vec2)
    
    # Store comparison in database
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""INSERT INTO comparisons 
                      (profile1_id, profile2_id, similarity_score, confidence, component_scores) 
                      VALUES (?, ?, ?, ?, ?)""",
                   (id1, id2, 
                    comparison_result["final_score"], 
                    comparison_result["confidence"],
                    json.dumps(comparison_result["component_scores"])))
    conn.commit()
    conn.close()

    return jsonify({
        "profile1": {"id": id1, "name": v1[0]},
        "profile2": {"id": id2, "name": v2[0]},
        "similarity_score": round(comparison_result["final_score"] * 100, 2),
        "confidence": round(comparison_result["confidence"] * 100, 2),
        "component_scores": comparison_result["component_scores"],
        "interpretation": comparison_result["interpretation"]
    })

@app.route('/comparison_history', methods=['GET'])
@admin_required
def get_comparison_history():
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT c.id, p1.name as name1, p2.name as name2, 
               c.similarity_score, c.confidence, c.created_at
        FROM comparisons c
        JOIN profiles p1 ON c.profile1_id = p1.id
        JOIN profiles p2 ON c.profile2_id = p2.id
        ORDER BY c.created_at DESC
        LIMIT 20
    """)
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        history.append({
            "id": r[0],
            "profile1": r[1],
            "profile2": r[2],
            "similarity_score": round(r[3] * 100, 2),
            "confidence": round(r[4] * 100, 2),
            "created_at": r[5]
        })
    
    return jsonify(history)

@app.route('/report/<id1>/<id2>')
@admin_required
def generate_report(id1, id2):
    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("SELECT name, gait_vector FROM profiles WHERE id=?", (id1,))
    p1 = cursor.fetchone()
    cursor.execute("SELECT name, gait_vector FROM profiles WHERE id=?", (id2,))
    p2 = cursor.fetchone()
    conn.close()

    if not p1 or not p2:
        return jsonify({"error": "Profiles not found"}), 404

    try:
        vec1 = json.loads(p1[1]) if p1[1] else []
        vec2 = json.loads(p2[1]) if p2[1] else []
    except:
        return jsonify({"error": "Invalid gait vector data"}), 500

    comparison_result = advanced_gait_comparison(vec1, vec2)

    # Generate PDF report
    report_path = f"report_{id1}_{id2}.pdf"
    doc = SimpleDocTemplate(report_path)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=16,
        spaceAfter=30
    )
    
    story = []
    
    # Title
    story.append(Paragraph("Forensic Gait Analysis Report", title_style))
    story.append(Spacer(1, 10))
    
    # Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Profile Information
    story.append(Paragraph("Profile Information", styles['Heading2']))
    profile_data = [
        ['Profile', 'Name', 'Feature Vector Length'],
        ['A', p1[0], str(len(vec1))],
        ['B', p2[0], str(len(vec2))]
    ]
    
    profile_table = Table(profile_data)
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 20))
    
    # Comparison Results
    story.append(Paragraph("Comparison Results", styles['Heading2']))
    
    results_data = [
        ['Metric', 'Score'],
        ['Overall Similarity', f"{round(comparison_result['final_score'] * 100, 2)}%"],
        ['Confidence Level', f"{round(comparison_result['confidence'] * 100, 2)}%"],
        ['Cosine Similarity', f"{round(comparison_result['component_scores'].get('cosine', 0) * 100, 2)}%"],
        ['DTW Similarity', f"{round(comparison_result['component_scores'].get('dtw', 0) * 100, 2)}%"],
        ['Correlation', f"{round(comparison_result['component_scores'].get('correlation', 0) * 100, 2)}%"]
    ]
    
    results_table = Table(results_data)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Interpretation
    story.append(Paragraph("Expert Interpretation", styles['Heading2']))
    interpretation = comparison_result["interpretation"]
    story.append(Paragraph(f"<b>Conclusion:</b> {interpretation}", styles['Normal']))
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Disclaimer:</b> This analysis should be used as supplementary evidence and verified by forensic experts.", styles['Italic']))
    
    doc.build(story)
    return send_file(report_path, as_attachment=True)

def generate_enhanced_realtime_frames():
    """Generate frames with enhanced real-time gait recognition"""
    cap = cv2.VideoCapture(1)  # Use default camera
    
    # Set camera resolution for better accuracy
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Enhanced real-time gait recognition started...")
    print("Make sure person walks naturally in front of camera for 2-3 seconds")
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        # Process frame for gait recognition
        annotated_frame = gait_recognizer.process_frame(frame, results.pose_landmarks)
        
        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/realtime')
@admin_required
def realtime():
    return Response(generate_enhanced_realtime_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime_status')
@admin_required
def realtime_status():
    """API endpoint to get current recognition status"""
    status = {
        "current_match": gait_recognizer.current_match,
        "confidence": round(gait_recognizer.current_confidence * 100, 2),
        "buffer_frames": len(gait_recognizer.frame_buffer),
        "profiles_loaded": len(gait_recognizer.known_profiles),
        "is_identified": gait_recognizer.current_match != "Unknown",
        "buffer_quality": round(gait_recognizer.assess_buffer_quality() * 100, 2)
    }
    return jsonify(status)

@app.route('/reset_realtime')
@admin_required
def reset_realtime():
    """Reset the real-time recognizer"""
    gait_recognizer.frame_buffer = []
    gait_recognizer.current_match = "Unknown"
    gait_recognizer.current_confidence = 0.0
    gait_recognizer.match_history = []
    # Reload profiles in case new ones were added
    gait_recognizer.known_profiles = gait_recognizer.load_profiles()
    return jsonify({"message": "Real-time recognizer reset successfully"})

@app.route('/identify', methods=['POST'])
@admin_required
def identify_person():
    file = request.files.get('video')
    if not file:
        return jsonify(error="Video file is required"), 400
    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    gaitsignature = extract_advanced_gait_features(path)
    os.remove(path)
    if gaitsignature is None:
        return jsonify(error="Could not extract sufficient gait features from video"), 500

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, age, gender, role, gait_vector FROM profiles")
    profiles = cursor.fetchall()
    conn.close()
    best_match = None
    best_confidence = -1  # Initialize to a very low value

    for profile in profiles:
        profile_id, name, age, gender, role, gaitvector_json = profile
        gaitvector = json.loads(gaitvector_json) if gaitvector_json else None
        if not gaitvector:
            continue
        
        comparison_result = advanced_gait_comparison(gaitsignature, gaitvector)
        print(comparison_result,"result")
        confidence = comparison_result.get('final_score', 0)

        
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = {
                "id": profile_id,
                "name": name,
                "age": age,
                "gender": gender,
                "role": role,
                "confidence": confidence,
                "similarity": comparison_result.get('final_score', 0),
                "interpretation": comparison_result.get('interpretation', '')
            }

    if best_match is None:
        return jsonify(message="No matching profile found"), 404

    return jsonify(best_match)


@app.route("/admin/identify")
def identify_page():
    return render_template("identify_admin.html")

# ---------- Run Application ----------
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=4500)