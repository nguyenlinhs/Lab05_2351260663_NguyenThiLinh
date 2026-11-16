"""
CSE457 - XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
Sinh viên: Nguyễn Thị Linh
MSSV: 2351260663
Lớp: 65TTNT - Trường Đại học Thủy Lợi
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from numpy.linalg import norm

# Thiết lập bảng mã UTF-8 cho console
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ==========================================
# CẤU HÌNH THÍ NGHIỆM CHUẨN (CONFIG)
# ==========================================
CONFIG = {
    "sr": 16000,
    "n_fft": 512,
    "win_length": 400,   # 25 ms @ 16 kHz
    "hop_length": 160,   # 10 ms @ 16 kHz
    "window": "hann",
    "n_mfcc": 20,
    "beta_default": 1.5,
    "floor_default": 0.02,
    "target_snr_list": [0.0, 5.0, 10.0],
    "mixing_matrix_easy": np.array([[1.0, 0.6], [0.4, 1.0]], dtype=np.float64),
    "mixing_matrix_hard": np.array([[1.0, 0.9], [0.95, 1.0]], dtype=np.float64),
    "random_state": 42
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "lab5_data")
AUDIO_OUT_DIR = os.path.join(BASE_DIR, "results", "audio")
FIG_OUT_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
os.makedirs(FIG_OUT_DIR, exist_ok=True)

# Cấu hình font và style vẽ đồ thị
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

# ==========================================
# CÁC HÀM TIỆN ÍCH CỐT LÕI (UTILITIES)
# ==========================================
def load16k(path):
    x, sr = librosa.load(path, sr=CONFIG["sr"], mono=True)
    return x.astype(np.float64), sr

def save16k(path, x):
    # Tránh clipping khi lưu PCM 16-bit
    peak = np.max(np.abs(x))
    if peak > 1.0:
        x = x / peak * 0.95
    sf.write(path, x.astype(np.float32), CONFIG["sr"], subtype='PCM_16')

def power(x):
    return np.mean(np.asarray(x, dtype=np.float64)**2) + 1e-12

def add_noise_at_snr(s, v, snr_db):
    if len(v) < len(s):
        v = np.tile(v, int(np.ceil(len(s)/len(v))))
    v = v[:len(s)]
    alpha = np.sqrt(power(s) / (power(v) * 10**(snr_db / 10.0)))
    y = s + alpha * v
    return y, alpha

def snr_db(ref, test):
    L = min(len(ref), len(test))
    ref, test = ref[:L], test[:L]
    err = test - ref
    return float(10.0 * np.log10(np.sum(ref**2) / (np.sum(err**2) + 1e-12)))

def enhance_stft(y, noise, sr=16000, beta=1.5, floor=0.02,
                 n_fft=512, win_length=400, hop_length=160):
    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann')
    
    # Ước lượng nhiễu từ noise audio
    N = librosa.stft(noise[:len(y)], n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window='hann')
    Pn = np.mean(np.abs(N)**2, axis=1, keepdims=True)
    Py = np.abs(Y)**2
    
    # 1. Spectral Subtraction
    Pss = np.maximum(Py - beta * Pn, floor * Pn)
    Sss = np.sqrt(Pss) * np.exp(1j * np.angle(Y))
    x_ss = librosa.istft(Sss, hop_length=hop_length,
                         win_length=win_length, length=len(y))
    
    # 2. Wiener Filter
    Ps = np.maximum(Py - Pn, 0.0)
    H = Ps / (Ps + Pn + 1e-12)
    Sw = H * Y
    x_w = librosa.istft(Sw, hop_length=hop_length,
                        win_length=win_length, length=len(y))
    
    return x_ss, x_w, Y, Sss, Sw

def speaker_embedding(x, sr=16000, n_mfcc=20, use_delta=True, use_cmn=False):
    M = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc,
                             n_fft=CONFIG["n_fft"], win_length=CONFIG["win_length"],
                             hop_length=CONFIG["hop_length"], window='hann')
    if use_cmn:
        M = M - np.mean(M, axis=1, keepdims=True)
        
    if use_delta:
        D = librosa.feature.delta(M)
        e = np.concatenate([M.mean(axis=1), M.std(axis=1),
                            D.mean(axis=1), D.std(axis=1)])
    else:
        e = np.concatenate([M.mean(axis=1), M.std(axis=1)])
        
    return e / (norm(e) + 1e-12)

def cosine(a, b):
    return float(np.dot(a, b) / (norm(a) * norm(b) + 1e-12))

def rms_normalize(x):
    return x / (np.sqrt(np.mean(x**2)) + 1e-12)

# ==========================================
# THỰC NGHIỆM CHÍNH VÀ CASE STUDY
# ==========================================
def main():
    print("=" * 70)
    print("CSE457 - LAB 5: SPEECH ENHANCEMENT, SPEAKER ID & ICA SEPARATION")
    print("Sinh viên: Nguyễn Thị Linh - MSSV: 2351260663")
    print("=" * 70)
    
    results = {}
    
    # ------------------------------------------------------------------
    # PHẦN 1: CHUẨN HÓA DỮ LIỆU & TẠO NOISY SPEECH (MỤC 4.1 & 5.1)
    # ------------------------------------------------------------------
    print("\n>>> [1/5] Khối 1: Chuẩn hóa dữ liệu & Tạo Noisy Speech...")
    clean_ref_path = os.path.join(DATA_DIR, "clean", "spk01_test_01.wav")
    fan_noise_path = os.path.join(DATA_DIR, "noise", "fan.wav")
    
    s_clean, _ = load16k(clean_ref_path)
    v_fan, _ = load16k(fan_noise_path)
    
    save16k(os.path.join(AUDIO_OUT_DIR, "clean_reference.wav"), s_clean)
    
    noisy_dict = {}
    snr_verify = {}
    for snr_target in CONFIG["target_snr_list"]:
        y_noisy, alpha = add_noise_at_snr(s_clean, v_fan, snr_target)
        snr_meas = snr_db(s_clean, y_noisy)
        noisy_dict[int(snr_target)] = y_noisy
        snr_verify[int(snr_target)] = {
            "target": snr_target,
            "measured": snr_meas,
            "diff": abs(snr_meas - snr_target),
            "alpha": alpha
        }
        save16k(os.path.join(AUDIO_OUT_DIR, f"noisy_{int(snr_target)}dB.wav"), y_noisy)
        print(f"  SNR mục tiêu: {snr_target:4.1f} dB | SNR đo được: {snr_meas:6.3f} dB | Sai lệch: {abs(snr_meas - snr_target):.4f} dB (<= 0.2 dB)")
    
    results["snr_verification"] = snr_verify
    
    # Vẽ Figure 1: Waveforms và Spectrograms của Clean và Noisy
    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    time_axis = np.linspace(0, len(s_clean)/CONFIG["sr"], len(s_clean))
    
    signals = [
        ("Clean Reference", s_clean),
        ("Noisy 10 dB", noisy_dict[10]),
        ("Noisy 5 dB", noisy_dict[5]),
        ("Noisy 0 dB", noisy_dict[0])
    ]
    
    for row, (title, sig) in enumerate(signals):
        axes[row, 0].plot(time_axis, sig, color='#1f77b4', lw=0.8)
        axes[row, 0].set_title(f"Waveform: {title}", fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel("Biên độ")
        axes[row, 0].set_ylim([-1.05, 1.05])
        if row == 3:
            axes[row, 0].set_xlabel("Thời gian (giây)")
            
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)), ref=np.max)
        img = librosa.display.specshow(D, sr=CONFIG["sr"], hop_length=160, x_axis='time', y_axis='linear',
                                      ax=axes[row, 1], cmap='magma')
        axes[row, 1].set_title(f"Spectrogram: {title}", fontsize=11, fontweight='bold')
        axes[row, 1].set_ylabel("Tần số (Hz)")
        if row == 3:
            axes[row, 1].set_xlabel("Thời gian (giây)")
            
    fig.colorbar(img, ax=axes[:, 1], format="%+2.0f dB", fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig1_clean_noisy_wave_spec.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 1: fig1_clean_noisy_wave_spec.png")
    
    # ------------------------------------------------------------------
    # PHẦN 2: SPECTRAL SUBTRACTION & WIENER FILTER (MỤC 4.2, 4.3 & 5.2)
    # ------------------------------------------------------------------
    print("\n>>> [2/5] Khối 2: Spectral Subtraction và Wiener Filtering...")
    noisy_5db = noisy_dict[5]
    
    # Chạy khảo sát tham số beta (1.0, 1.5, 2.0)
    beta_exp = {}
    ss_audio_dict = {}
    for beta in [1.0, 1.5, 2.0]:
        x_ss, _, _, Sss, _ = enhance_stft(noisy_5db, v_fan, beta=beta, floor=CONFIG["floor_default"])
        snr_out = snr_db(s_clean, x_ss)
        delta_snr = snr_out - 5.0
        beta_exp[str(beta)] = {
            "beta": beta,
            "snr_out": snr_out,
            "delta_snr": delta_snr
        }
        ss_audio_dict[beta] = x_ss
        print(f"  Spectral Subtraction (beta={beta:.1f}): SNR_out = {snr_out:5.2f} dB, Delta SNR = {delta_snr:+5.2f} dB")
        
    # Chạy Wiener filter
    x_ss_default, x_wiener_default, Y_noisy, S_ss_default, S_wiener = enhance_stft(
        noisy_5db, v_fan, beta=CONFIG["beta_default"], floor=CONFIG["floor_default"]
    )
    snr_wiener = snr_db(s_clean, x_wiener_default)
    delta_wiener = snr_wiener - 5.0
    print(f"  Wiener Filter (MMSE): SNR_out = {snr_wiener:5.2f} dB, Delta SNR = {delta_wiener:+5.2f} dB")
    
    save16k(os.path.join(AUDIO_OUT_DIR, "enhanced_ss.wav"), x_ss_default)
    save16k(os.path.join(AUDIO_OUT_DIR, "enhanced_wiener.wav"), x_wiener_default)
    
    results["enhancement_5db"] = {
        "spectral_subtraction_beta": beta_exp,
        "wiener": {"snr_out": snr_wiener, "delta_snr": delta_wiener}
    }
    
    # Khảo sát floor gamma: 0.0, 0.01, 0.05
    gamma_exp = {}
    for fl in [0.0, 0.01, 0.05]:
        x_ss_fl, _, _, _, _ = enhance_stft(noisy_5db, v_fan, beta=1.5, floor=fl)
        snr_out_fl = snr_db(s_clean, x_ss_fl)
        gamma_exp[str(fl)] = {
            "floor": fl,
            "snr_out": snr_out_fl,
            "delta_snr": snr_out_fl - 5.0
        }
    results["spectral_floor_gamma"] = gamma_exp
    
    # Khảo sát Input SNR (0, 5, 10 dB) với SS và Wiener (TN1)
    tn1_results = {}
    for s_in in [0, 5, 10]:
        y_cur = noisy_dict[s_in]
        x_ss_cur, x_w_cur, _, _, _ = enhance_stft(y_cur, v_fan, beta=1.5, floor=0.02)
        tn1_results[f"{s_in}dB"] = {
            "input_snr": s_in,
            "ss_snr_out": snr_db(s_clean, x_ss_cur),
            "ss_delta_snr": snr_db(s_clean, x_ss_cur) - s_in,
            "wiener_snr_out": snr_db(s_clean, x_w_cur),
            "wiener_delta_snr": snr_db(s_clean, x_w_cur) - s_in
        }
    results["tn1_input_snr_survey"] = tn1_results
    
    # Vẽ Figure 2: Phổ của Spectral Subtraction khi thay đổi beta
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for idx, beta in enumerate([1.0, 1.5, 2.0]):
        sig = ss_audio_dict[beta]
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(D, sr=CONFIG["sr"], hop_length=160, x_axis='time', y_axis='linear',
                                ax=axes[idx], cmap='magma')
        axes[idx].set_title(f"Spectral Subtraction (β = {beta:.1f}) - ΔSNR: {beta_exp[str(beta)]['delta_snr']:+.2f} dB",
                            fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("Tần số (Hz)")
    axes[2].set_xlabel("Thời gian (giây)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig2_spectral_subtraction_effect.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 2: fig2_spectral_subtraction_effect.png")
    
    # Vẽ Figure 3: So sánh phổ Noisy, SS và Wiener
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    specs = [
        ("Noisy Speech (SNR = 5 dB)", noisy_5db),
        ("Spectral Subtraction (β=1.5, floor=0.02)", x_ss_default),
        ("Wiener Filter (MMSE)", x_wiener_default)
    ]
    for idx, (title, sig) in enumerate(specs):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(D, sr=CONFIG["sr"], hop_length=160, x_axis='time', y_axis='linear',
                                ax=axes[idx], cmap='viridis')
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].set_ylabel("Tần số (Hz)")
    axes[2].set_xlabel("Thời gian (giây)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig3_wiener_filter_comparison.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 3: fig3_wiener_filter_comparison.png")
    
    # ------------------------------------------------------------------
    # PHẦN 3: SPEAKER ENROLLMENT & IDENTIFICATION (MỤC 4.4 & 5.3)
    # ------------------------------------------------------------------
    print("\n>>> [3/5] Khối 3: Speaker Enrollment, Identification & Verification...")
    speakers = ["spk01", "spk02", "spk03"]
    
    def build_speaker_models(n_tokens=4, use_delta=True, use_cmn=False):
        models = {}
        for spk in speakers:
            embs = []
            for t_idx in range(1, n_tokens + 1):
                fpath = os.path.join(DATA_DIR, "clean", f"{spk}_enroll_{t_idx:02d}.wav")
                audio, _ = load16k(fpath)
                emb = speaker_embedding(audio, use_delta=use_delta, use_cmn=use_cmn)
                embs.append(emb)
            mean_emb = np.mean(embs, axis=0)
            models[spk] = mean_emb / (norm(mean_emb) + 1e-12)
        return models

    # Baseline Model: 4 tokens, MFCC + Delta, No CMN
    models_baseline = build_speaker_models(n_tokens=4, use_delta=True, use_cmn=False)
    
    # Kiểm tra tập Test độc lập (spk01, spk02, spk03 x test_01, test_02 = 6 utterances)
    def evaluate_identification(models, use_delta=True, use_cmn=False, audio_mode="clean"):
        y_true = []
        y_pred = []
        score_matrix = []
        
        for spk_idx, spk in enumerate(speakers):
            for t_idx in [1, 2]:
                fpath = os.path.join(DATA_DIR, "clean", f"{spk}_test_{t_idx:02d}.wav")
                audio, _ = load16k(fpath)
                
                # Biến đổi nếu thử nghiệm trên noisy hoặc enhanced
                if audio_mode == "noisy_5db":
                    audio, _ = add_noise_at_snr(audio, v_fan, 5.0)
                elif audio_mode == "enhanced_ss":
                    y_n, _ = add_noise_at_snr(audio, v_fan, 5.0)
                    audio, _, _, _, _ = enhance_stft(y_n, v_fan, beta=1.5, floor=0.02)
                elif audio_mode == "enhanced_wiener":
                    y_n, _ = add_noise_at_snr(audio, v_fan, 5.0)
                    _, audio, _, _, _ = enhance_stft(y_n, v_fan, beta=1.5, floor=0.02)
                    
                emb = speaker_embedding(audio, use_delta=use_delta, use_cmn=use_cmn)
                scores = [cosine(emb, models[s]) for s in speakers]
                pred = speakers[int(np.argmax(scores))]
                
                y_true.append(spk)
                y_pred.append(pred)
                score_matrix.append({
                    "test_utterance": f"{spk}_test_{t_idx:02d}",
                    "true_speaker": spk,
                    "predicted": pred,
                    "scores": {s: round(float(scores[i]), 4) for i, s in enumerate(speakers)}
                })
                
        acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
        return acc, y_true, y_pred, score_matrix

    acc_base, y_true, y_pred, score_table = evaluate_identification(models_baseline)
    print(f"  Speaker ID Baseline Accuracy: {acc_base * 100:.1f}%")
    results["speaker_id_baseline"] = {
        "accuracy": acc_base,
        "score_table": score_table
    }
    
    # Ablations (TN4, TN5, TN6):
    # Ablation 1: MFCC-only (no delta)
    models_mfcc_only = build_speaker_models(n_tokens=4, use_delta=False, use_cmn=False)
    acc_mfcc_only, _, _, _ = evaluate_identification(models_mfcc_only, use_delta=False, use_cmn=False)
    
    # Ablation 2: With CMN
    models_cmn = build_speaker_models(n_tokens=4, use_delta=True, use_cmn=True)
    acc_cmn, _, _, _ = evaluate_identification(models_cmn, use_delta=True, use_cmn=True)
    
    # Ablation 3: 1 token and 2 tokens
    models_1token = build_speaker_models(n_tokens=1, use_delta=True, use_cmn=False)
    acc_1token, _, _, _ = evaluate_identification(models_1token, use_delta=True, use_cmn=False)
    
    models_2token = build_speaker_models(n_tokens=2, use_delta=True, use_cmn=False)
    acc_2token, _, _, _ = evaluate_identification(models_2token, use_delta=True, use_cmn=False)
    
    results["speaker_id_ablations"] = {
        "baseline_mfcc_delta": acc_base,
        "mfcc_only": acc_mfcc_only,
        "with_cmn": acc_cmn,
        "1_enrollment_token": acc_1token,
        "2_enrollment_tokens": acc_2token,
        "4_enrollment_tokens": acc_base
    }
    print(f"  Ablation MFCC-only: {acc_mfcc_only*100:.1f}% | With CMN: {acc_cmn*100:.1f}% | 1 Token: {acc_1token*100:.1f}%")
    
    # Thí nghiệm tích hợp 4.8: Speaker ID trên Noisy và Enhanced
    acc_noisy, _, _, score_table_noisy = evaluate_identification(models_baseline, audio_mode="noisy_5db")
    acc_ss, _, _, score_table_ss = evaluate_identification(models_baseline, audio_mode="enhanced_ss")
    acc_wiener, _, _, score_table_wiener = evaluate_identification(models_baseline, audio_mode="enhanced_wiener")
    
    results["speaker_id_noise_enhancement_comparison"] = {
        "clean": acc_base,
        "noisy_5db": acc_noisy,
        "enhanced_ss": acc_ss,
        "enhanced_wiener": acc_wiener
    }
    print(f"  Robustness check -> Clean: {acc_base*100:.1f}%, Noisy 5dB: {acc_noisy*100:.1f}%, SS: {acc_ss*100:.1f}%, Wiener: {acc_wiener*100:.1f}%")

    # Vẽ Figure 4: Confusion Matrix cho Baseline và Ablation
    from sklearn.metrics import confusion_matrix
    cm_base = confusion_matrix(y_true, y_pred, labels=speakers)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
                xticklabels=speakers, yticklabels=speakers, ax=ax, cbar=False)
    ax.set_title("Confusion Matrix: Speaker Identification (Test Set)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted Speaker")
    ax.set_ylabel("True Speaker")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig4_speaker_id_confusion_matrix.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 4: fig4_speaker_id_confusion_matrix.png")

    # ------------------------------------------------------------------
    # PHẦN 4: SPEAKER VERIFICATION, FAR & FRR (MỤC 4.5 & 5.4)
    # ------------------------------------------------------------------
    print("\n>>> [4/5] Khối 4: Speaker Verification, quét ngưỡng Dev & đánh giá Test...")
    # Tạo Dev trials: spk01..spk03 x dev_01..dev_02
    dev_scores = []
    for s_idx, spk in enumerate(speakers):
        for dev_idx in [1, 2]:
            fpath = os.path.join(DATA_DIR, "clean", f"{spk}_dev_{dev_idx:02d}.wav")
            audio, _ = load16k(fpath)
            emb = speaker_embedding(audio)
            for m_spk in speakers:
                score = cosine(emb, models_baseline[m_spk])
                label = 1 if (m_spk == spk) else 0  # 1: genuine, 0: impostor
                dev_scores.append((score, label))
                
    thresholds = np.linspace(-1.0, 1.0, 401)
    dev_rows = []
    for th in thresholds:
        genuine = [s for s, l in dev_scores if l == 1]
        impostor = [s for s, l in dev_scores if l == 0]
        frr = np.mean(np.array(genuine) < th)
        far = np.mean(np.array(impostor) >= th)
        dev_rows.append((float(th), float(far), float(frr)))
        
    best_dev = min(dev_rows, key=lambda z: abs(z[1] - z[2]))
    theta_star = best_dev[0]
    dev_eer = (best_dev[1] + best_dev[2]) / 2.0
    print(f"  Development Set -> Locked Theta*: {theta_star:.4f} | Dev FAR: {best_dev[1]:.4f} | Dev FRR: {best_dev[2]:.4f}")
    
    # Đánh giá trên Test trials ĐỘC LẬP bằng ngưỡng theta_star đã khóa
    test_scores = []
    for s_idx, spk in enumerate(speakers):
        for t_idx in [1, 2]:
            fpath = os.path.join(DATA_DIR, "clean", f"{spk}_test_{t_idx:02d}.wav")
            audio, _ = load16k(fpath)
            emb = speaker_embedding(audio)
            for m_spk in speakers:
                score = cosine(emb, models_baseline[m_spk])
                label = 1 if (m_spk == spk) else 0
                test_scores.append((score, label))
                
    test_genuine = [s for s, l in test_scores if l == 1]
    test_impostor = [s for s, l in test_scores if l == 0]
    test_frr = float(np.mean(np.array(test_genuine) < theta_star))
    test_far = float(np.mean(np.array(test_impostor) >= theta_star))
    print(f"  Test Set (Locked Theta* = {theta_star:.4f}) -> Test FAR: {test_far:.4f} | Test FRR: {test_frr:.4f}")
    
    results["speaker_verification"] = {
        "dev_locked_threshold": theta_star,
        "dev_far_at_theta": best_dev[1],
        "dev_frr_at_theta": best_dev[2],
        "dev_eer_approx": dev_eer,
        "test_far": test_far,
        "test_frr": test_frr
    }
    
    # Vẽ Figure 5: Đồ thị FAR - FRR theo ngưỡng theta
    ths = [r[0] for r in dev_rows]
    fars = [r[1] for r in dev_rows]
    frrs = [r[2] for r in dev_rows]
    
    plt.figure(figsize=(8, 5))
    plt.plot(ths, fars, label="FAR (False Acceptance Rate)", color="#d62728", lw=2)
    plt.plot(ths, frrs, label="FRR (False Rejection Rate)", color="#1f77b4", lw=2)
    plt.axvline(theta_star, color='black', linestyle='--', lw=1.5,
                label=f"Locked θ* = {theta_star:.3f} (Dev EER)")
    plt.scatter([theta_star], [test_far], color='#d62728', s=80, zorder=5, label=f"Test FAR: {test_far:.2f}")
    plt.scatter([theta_star], [test_frr], color='#1f77b4', s=80, zorder=5, label=f"Test FRR: {test_frr:.2f}")
    plt.title("Speaker Verification: FAR and FRR Trade-off vs Decision Threshold θ", fontsize=11, fontweight='bold')
    plt.xlabel("Verification Threshold θ")
    plt.ylabel("Tỷ lệ lỗi (Error Rate)")
    plt.xlim([-0.2, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig5_speaker_verification_far_frr.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 5: fig5_speaker_verification_far_frr.png")

    # ------------------------------------------------------------------
    # PHẦN 5: BSS VÀ FASTICA TÁCH HAI NGUỒN NÓI (MỤC 4.6, 4.7 & 5.5)
    # ------------------------------------------------------------------
    print("\n>>> [5/5] Khối 5: Blind Source Separation (FastICA) trên hỗn hợp 2 mic...")
    s1_raw, _ = load16k(os.path.join(DATA_DIR, "separation", "source_A.wav"))
    s2_raw, _ = load16k(os.path.join(DATA_DIR, "separation", "source_B.wav"))
    
    L = min(len(s1_raw), len(s2_raw))
    S = np.vstack([rms_normalize(s1_raw[:L]), rms_normalize(s2_raw[:L])])
    
    def run_ica_experiment(A, add_noise=False, noise_snr=20.0):
        X = A @ S
        if add_noise:
            # Thêm nhiễu độc lập vào từng mic
            np.random.seed(CONFIG["random_state"])
            n1 = np.random.randn(L)
            n2 = np.random.randn(L)
            a1 = np.sqrt(power(X[0]) / (power(n1) * 10**(noise_snr/10.0)))
            a2 = np.sqrt(power(X[1]) / (power(n2) * 10**(noise_snr/10.0)))
            X[0] += a1 * n1
            X[1] += a2 * n2
            
        det_A = float(np.linalg.det(A))
        
        ica = FastICA(n_components=2, whiten='unit-variance', random_state=CONFIG["random_state"])
        Shat_raw = ica.fit_transform(X.T).T  # 2 x L
        
        # Ma trận tương quan tuyệt đối trước khi ghép
        C_raw = np.zeros((2, 2))
        C_signed = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                c = np.corrcoef(S[i], Shat_raw[j])[0, 1]
                C_signed[i, j] = c
                C_raw[i, j] = abs(c)
                
        # Giải quyết Permutation và Sign Ambiguity
        # Xét 2 khả năng ghép cặp: (0->0, 1->1) hoặc (0->1, 1->0)
        score_perm_direct = C_raw[0, 0] + C_raw[1, 1]
        score_perm_swap = C_raw[0, 1] + C_raw[1, 0]
        
        if score_perm_direct >= score_perm_swap:
            mapping = [0, 1]
        else:
            mapping = [1, 0]
            
        Shat_aligned = np.zeros_like(Shat_raw)
        C_aligned = np.zeros((2, 2))
        
        for i in range(2):
            j = mapping[i]
            est = Shat_raw[j].copy()
            # Xử lý Sign Ambiguity: nếu tương quan âm, nhân -1
            if C_signed[i, j] < 0:
                est = -est
            # Chuẩn hóa biên độ peak về 0.90
            est = est / (np.max(np.abs(est)) + 1e-12) * 0.90
            Shat_aligned[i] = est
            
        for i in range(2):
            for j in range(2):
                C_aligned[i, j] = np.corrcoef(S[i], Shat_aligned[j])[0, 1]
                
        return det_A, X, Shat_raw, Shat_aligned, C_raw, C_aligned

    # Thí nghiệm ma trận dễ (Easy): A = [[1.0, 0.6], [0.4, 1.0]]
    A_easy = CONFIG["mixing_matrix_easy"]
    det_easy, X_easy, Shat_raw_easy, Shat_easy, C_raw_easy, C_aligned_easy = run_ica_experiment(A_easy)
    print(f"  A Easy -> det(A) = {det_easy:.4f}")
    print(f"  Tương quan trước ghép (|C|):\n{C_raw_easy}")
    print(f"  Tương quan sau ghép và sửa dấu (C_aligned):\n{C_aligned_easy}")
    
    # Lưu audio mic mixtures và separated audio
    save16k(os.path.join(AUDIO_OUT_DIR, "mix_mic1.wav"), X_easy[0])
    save16k(os.path.join(AUDIO_OUT_DIR, "mix_mic2.wav"), X_easy[1])
    save16k(os.path.join(AUDIO_OUT_DIR, "separated_1.wav"), Shat_easy[0])
    save16k(os.path.join(AUDIO_OUT_DIR, "separated_2.wav"), Shat_easy[1])
    
    # Thí nghiệm ma trận khó gần suy biến (Hard): A = [[1.0, 0.9], [0.95, 1.0]] (TN7)
    A_hard = CONFIG["mixing_matrix_hard"]
    det_hard, X_hard, _, Shat_hard, C_raw_hard, C_aligned_hard = run_ica_experiment(A_hard)
    print(f"  A Hard (gần suy biến) -> det(A) = {det_hard:.4f}")
    print(f"  Tương quan sau ghép (Hard):\n{C_aligned_hard}")
    
    # Thí nghiệm ICA có thêm nhiễu vào microphone (TN8)
    det_noisy, _, _, Shat_noisy, _, C_aligned_noisy = run_ica_experiment(A_easy, add_noise=True, noise_snr=15.0)
    print(f"  ICA trên Noisy Mixture (SNR = 15 dB) -> Tương quan sau ghép:\n{C_aligned_noisy}")
    
    results["bss_ica"] = {
        "A_easy": {
            "det": det_easy,
            "C_raw": C_raw_easy.tolist(),
            "C_aligned": C_aligned_easy.tolist()
        },
        "A_hard_near_singular": {
            "det": det_hard,
            "C_aligned": C_aligned_hard.tolist()
        },
        "A_noisy_mixture_15db": {
            "C_aligned": C_aligned_noisy.tolist()
        }
    }
    
    # Vẽ Figure 6: Waveforms và Spectrograms của Sources, Mixtures và Separated Signals
    fig, axes = plt.subplots(6, 2, figsize=(14, 14))
    t_sig = np.linspace(0, L/CONFIG["sr"], L)
    ica_display_items = [
        ("Source 1 (s1)", S[0]),
        ("Source 2 (s2)", S[1]),
        ("Mixture Mic 1 (x1)", X_easy[0]),
        ("Mixture Mic 2 (x2)", X_easy[1]),
        ("Separated Source 1 (s_hat1)", Shat_easy[0]),
        ("Separated Source 2 (s_hat2)", Shat_easy[1])
    ]
    for row, (lbl, sig) in enumerate(ica_display_items):
        axes[row, 0].plot(t_sig, sig, lw=0.7, color='#2ca02c' if 'Separated' in lbl else ('#d62728' if 'Mixture' in lbl else '#1f77b4'))
        axes[row, 0].set_title(f"Waveform: {lbl}", fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel("Biên độ")
        axes[row, 0].set_ylim([-1.1, 1.1])
        if row == 5:
            axes[row, 0].set_xlabel("Thời gian (giây)")
            
        D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=512, hop_length=160)), ref=np.max)
        librosa.display.specshow(D, sr=CONFIG["sr"], hop_length=160, x_axis='time', y_axis='linear',
                                ax=axes[row, 1], cmap='plasma')
        axes[row, 1].set_title(f"Spectrogram: {lbl}", fontsize=10, fontweight='bold')
        axes[row, 1].set_ylabel("Tần số (Hz)")
        if row == 5:
            axes[row, 1].set_xlabel("Thời gian (giây)")
            
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig6_ica_separation_results.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 6: fig6_ica_separation_results.png")
    
    # Vẽ Figure 7: Ma trận tương quan ICA trước và sau ghép (Easy vs Hard)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.heatmap(C_raw_easy, annot=True, fmt='.3f', cmap='Greens', ax=axes[0], cbar=False,
                xticklabels=['Estimate 1', 'Estimate 2'], yticklabels=['Source 1', 'Source 2'])
    axes[0].set_title("A Easy: |Correlation| Raw (Trước ghép)", fontsize=10, fontweight='bold')
    
    sns.heatmap(C_aligned_easy, annot=True, fmt='.3f', cmap='Blues', ax=axes[1], cbar=False,
                xticklabels=['Estimate 1', 'Estimate 2'], yticklabels=['Source 1', 'Source 2'])
    axes[1].set_title("A Easy: Correlation Aligned (Sau ghép)", fontsize=10, fontweight='bold')
    
    sns.heatmap(C_aligned_hard, annot=True, fmt='.3f', cmap='Oranges', ax=axes[2], cbar=False,
                xticklabels=['Estimate 1', 'Estimate 2'], yticklabels=['Source 1', 'Source 2'])
    axes[2].set_title("A Hard (gần suy biến): Correlation Aligned", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_OUT_DIR, "fig7_ica_correlation_matrices.png"), dpi=300)
    plt.close()
    print("  -> Đã lưu Figure 7: fig7_ica_correlation_matrices.png")

    # ------------------------------------------------------------------
    # PHẦN 6: TỔNG HỢP VÀ LƯU KẾT QUẢ JSON
    # ------------------------------------------------------------------
    # Bảng tổng hợp 8 Thí nghiệm bắt buộc (TN1 - TN8)
    summary_8_experiments = [
        {"TN": 1, "Biến thay đổi": "Input SNR", "Giá trị": "0 / 5 / 10 dB",
         "Kết quả Delta SNR": f"0dB: {tn1_results['0dB']['ss_delta_snr']:+.2f} dB, 5dB: {tn1_results['5dB']['ss_delta_snr']:+.2f} dB, 10dB: {tn1_results['10dB']['ss_delta_snr']:+.2f} dB"},
        {"TN": 2, "Biến thay đổi": "Spectral Subtraction beta", "Giá trị": "1.0 / 1.5 / 2.0",
         "Kết quả Delta SNR": f"b=1.0: {beta_exp['1.0']['delta_snr']:+.2f} dB, b=1.5: {beta_exp['1.5']['delta_snr']:+.2f} dB, b=2.0: {beta_exp['2.0']['delta_snr']:+.2f} dB"},
        {"TN": 3, "Biến thay đổi": "Spectral Floor gamma", "Giá trị": "0 / 0.01 / 0.05",
         "Kết quả Delta SNR": f"g=0: {gamma_exp['0.0']['delta_snr']:+.2f} dB, g=0.01: {gamma_exp['0.01']['delta_snr']:+.2f} dB, g=0.05: {gamma_exp['0.05']['delta_snr']:+.2f} dB"},
        {"TN": 4, "Biến thay đổi": "Feature Speaker", "Giá trị": "MFCC vs MFCC+Delta",
         "Kết quả Accuracy": f"MFCC: {acc_mfcc_only*100:.1f}%, MFCC+Delta: {acc_base*100:.1f}%"},
        {"TN": 5, "Biến thay đổi": "CMN (Cepstral Mean Norm)", "Giá trị": "off / on",
         "Kết quả Accuracy": f"Off: {acc_base*100:.1f}%, On: {acc_cmn*100:.1f}%"},
        {"TN": 6, "Biến thay đổi": "Enrollment Tokens", "Giá trị": "1 / 2 / 4 tokens",
         "Kết quả Accuracy": f"1 token: {acc_1token*100:.1f}%, 2 tokens: {acc_2token*100:.1f}%, 4 tokens: {acc_base*100:.1f}%"},
        {"TN": 7, "Biến thay đổi": "Mixing Matrix A", "Giá trị": "Easy vs Hard (gần suy biến)",
         "Kết quả Tương quan chéo": f"det(Easy)=0.76 -> rho={C_aligned_easy[0,0]:.3f}; det(Hard)=0.145 -> rho={C_aligned_hard[0,0]:.3f}"},
        {"TN": 8, "Biến thay đổi": "ICA + Noise", "Giá trị": "Clean Mix vs Noisy Mix (15dB)",
         "Kết quả Tương quan": f"Clean: rho={C_aligned_easy[0,0]:.3f}; Noisy: rho={C_aligned_noisy[0,0]:.3f}"}
    ]
    results["summary_8_experiments"] = summary_8_experiments
    
    with open(os.path.join(BASE_DIR, "results", "experiment_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print("\n>>> Toàn bộ thực nghiệm hoàn thành xuất sắc!")
    print(f">>> Đã xuất dữ liệu kết quả JSON: results/experiment_results.json")
    print(f">>> Đã sinh các file âm thanh: results/audio/")
    print(f">>> Đã sinh 7 biểu đồ khoa học: results/figures/")

if __name__ == "__main__":
    main()
