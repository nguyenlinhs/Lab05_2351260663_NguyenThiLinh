import os
import io
import numpy as np
import soundfile as sf
import librosa
from gtts import gTTS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lab5_data")
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
NOISE_DIR = os.path.join(DATA_DIR, "noise")
SEP_DIR = os.path.join(DATA_DIR, "separation")

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(NOISE_DIR, exist_ok=True)
os.makedirs(SEP_DIR, exist_ok=True)

TARGET_SR = 16000

# 1. Generate clean utterances for 3 speakers
SPEAKERS = {
    "spk01": {
        "pitch_shift": 0.0,
        "speed": 1.0,
        "texts": {
            "enroll_01": "Xin chào quý vị, đây là hệ thống trợ lý thông minh phòng họp.",
            "enroll_02": "Tôi tên là Nguyễn Thị Linh, sinh viên khoa công nghệ thông tin.",
            "enroll_03": "Học phần xử lý âm thanh và tiếng nói cung cấp nhiều kiến thức chuyên sâu.",
            "enroll_04": "Khử nhiễu và nhận dạng người nói là hai bài toán rất quan trọng.",
            "dev_01": "Phòng họp thông minh có khả năng ghi âm và nhận diện từng thành viên.",
            "dev_02": "Chúng tôi đang kiểm tra ngưỡng xác thực trên tập phát triển hệ thống.",
            "test_01": "Hôm nay chúng ta cùng đánh giá chất lượng lọc nhiễu và nhận dạng.",
            "test_02": "Thuật toán xử lý tín hiệu số giúp tăng tỷ số tín hiệu trên nhiễu."
        }
    },
    "spk02": {
        "pitch_shift": -5.0,  # Male voice pitch shift
        "speed": 0.95,
        "texts": {
            "enroll_01": "Chào mọi người, tôi là thành viên thứ hai tham gia buổi họp hôm nay.",
            "enroll_02": "Dự án phát triển trợ lý ảo đang hoàn thiện giai đoạn tích hợp mô hình.",
            "enroll_03": "Thuật toán tách nguồn mù giúp phân tách các giọng nói nói đè lên nhau.",
            "enroll_04": "Ma trận trộn kênh âm thanh mô phỏng hai micro đặt trong không gian phòng.",
            "dev_01": "Các báo cáo tiến độ dự án cần được nộp trước hạn chót vào cuối tuần.",
            "dev_02": "Hệ thống ghi âm hai kênh cho phép ước lượng các nguồn tín hiệu độc lập.",
            "test_01": "Xin mời các thành viên kiểm tra micro và chuẩn bị bắt đầu cuộc họp.",
            "test_02": "Tách nguồn bằng phân tích thành phần độc lập đạt kết quả rất khả quan."
        }
    },
    "spk03": {
        "pitch_shift": 3.5,  # Higher pitched female / distinct vocal tract
        "speed": 1.08,
        "texts": {
            "enroll_01": "Tôi là người tham gia thứ ba, chịu trách nhiệm thẩm định chất lượng âm thanh.",
            "enroll_02": "Bộ lọc phi tuyến và biến đổi chuỗi thời gian ngắn mang lại hiệu quả cao.",
            "enroll_03": "Đặc trưng mel frequency cepstral coefficients giúp phân biệt giọng nói các cá nhân.",
            "enroll_04": "Khoảng cách cosine được áp dụng để so khớp vector đặc trưng người nói.",
            "dev_01": "Tập dữ liệu kiểm thử độc lập đảm bảo đánh giá khách quan và chính xác.",
            "dev_02": "Chỉ số false accept rate và false reject rate đạt mức cân bằng tối ưu.",
            "test_01": "Chúng tôi đã hoàn thành toàn bộ các bài thí nghiệm kiểm chứng kỹ thuật.",
            "test_02": "Kết quả phân loại trên ma trận nhầm lẫn cho độ chính xác tuyệt đối."
        }
    }
}

def generate_speech(text, pitch_steps, speed_rate, sr=TARGET_SR):
    fp = io.BytesIO()
    tts = gTTS(text, lang='vi')
    tts.write_to_fp(fp)
    fp.seek(0)
    y, orig_sr = librosa.load(fp, sr=sr, mono=True)
    
    # Trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=25)
    
    # Pitch shift if needed
    if abs(pitch_steps) > 0.1:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_steps)
        
    # Time stretch / speed change if needed
    if abs(speed_rate - 1.0) > 0.02:
        y = librosa.effects.time_stretch(y, rate=speed_rate)
        
    # Target duration 2.5 - 4.0s
    dur = len(y) / sr
    if dur < 2.2:
        # pad slightly
        pad_len = int((2.5 - dur) * sr / 2)
        y = np.pad(y, (pad_len, pad_len), mode='constant')
    elif dur > 4.2:
        y = y[:int(4.0 * sr)]
        
    # Normalize peak to 0.90 to avoid clipping
    peak = np.max(np.abs(y)) + 1e-12
    y = y / peak * 0.90
    return y.astype(np.float32)

print("--- 1. Generating clean utterances for 3 speakers ---")
for spk_id, spk_info in SPEAKERS.items():
    print(f"Generating for {spk_id}...")
    for tag, text in spk_info["texts"].items():
        fname = f"{spk_id}_{tag}.wav"
        fpath = os.path.join(CLEAN_DIR, fname)
        if not os.path.exists(fpath):
            audio = generate_speech(text, spk_info["pitch_shift"], spk_info["speed"])
            sf.write(fpath, audio, TARGET_SR, subtype='PCM_16')
            print(f"  Saved {fname}: {len(audio)/TARGET_SR:.2f}s")
        else:
            print(f"  Exists: {fname}")

# 2. Generate Noises: fan.wav and cafeteria.wav (duration 10.0s at 16kHz)
print("\n--- 2. Generating realistic noise files (fan.wav, cafeteria.wav) ---")
noise_dur = 10.0
n_samples = int(noise_dur * TARGET_SR)
t = np.linspace(0, noise_dur, n_samples, endpoint=False)

# Fan noise: stationary process with 50Hz hum + harmonics + aerodynamic pink/brown noise
fan_path = os.path.join(NOISE_DIR, "fan.wav")
if not os.path.exists(fan_path):
    np.random.seed(42)
    # 50 Hz motor hum and harmonics
    hum = (0.25 * np.sin(2 * np.pi * 50 * t) +
           0.15 * np.sin(2 * np.pi * 100 * t) +
           0.08 * np.sin(2 * np.pi * 150 * t) +
           0.05 * np.sin(2 * np.pi * 200 * t))
    
    # Aerodynamic turbulence: filtered Gaussian white noise
    white = np.random.randn(n_samples)
    # Bandpass filter around 100-900 Hz with 1/f falloff
    b_lp = np.exp(-np.linspace(0, 5, 256))
    b_lp /= np.sum(b_lp)
    turb = np.convolve(white, b_lp, mode='same')
    turb = turb / (np.std(turb) + 1e-12) * 0.20
    
    fan_signal = hum + turb
    fan_signal = fan_signal / (np.max(np.abs(fan_signal)) + 1e-12) * 0.85
    sf.write(fan_path, fan_signal.astype(np.float32), TARGET_SR, subtype='PCM_16')
    print(f"  Saved fan.wav: {len(fan_signal)/TARGET_SR:.2f}s")
else:
    print("  fan.wav already exists.")

# Cafeteria noise: non-stationary babble + ambient clinks
cafeteria_path = os.path.join(NOISE_DIR, "cafeteria.wav")
if not os.path.exists(cafeteria_path):
    np.random.seed(101)
    # Modulated speech-shaped noise (babble)
    mod1 = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    mod2 = 0.5 + 0.5 * np.sin(2 * np.pi * 4.2 * t + 1.2)
    mod3 = 0.5 + 0.5 * np.sin(2 * np.pi * 2.8 * t + 2.5)
    
    # Filtered colored noise for vowel formant energy (formants at 500Hz, 1500Hz, 2500Hz)
    babble_raw = np.random.randn(n_samples)
    from scipy.signal import butter, sosfilt
    sos_speech = butter(4, [300, 3400], btype='bandpass', fs=TARGET_SR, output='sos')
    babble = sosfilt(sos_speech, babble_raw)
    babble = babble / (np.std(babble) + 1e-12) * (0.15 * mod1 + 0.15 * mod2 + 0.10 * mod3)
    
    # Occasional cutlery clinks (impulsive damped sine waves)
    clinks = np.zeros(n_samples)
    clink_times = [1.2, 2.7, 4.5, 6.1, 7.8, 8.9]
    for ct in clink_times:
        idx = int(ct * TARGET_SR)
        L_clink = int(0.08 * TARGET_SR)
        if idx + L_clink < n_samples:
            t_c = np.linspace(0, 0.08, L_clink, endpoint=False)
            clink_wave = np.exp(-t_c * 60) * np.sin(2 * np.pi * 3200 * t_c) * 0.35
            clinks[idx:idx+L_clink] += clink_wave
            
    caf_signal = babble + clinks
    caf_signal = caf_signal / (np.max(np.abs(caf_signal)) + 1e-12) * 0.85
    sf.write(cafeteria_path, caf_signal.astype(np.float32), TARGET_SR, subtype='PCM_16')
    print(f"  Saved cafeteria.wav: {len(caf_signal)/TARGET_SR:.2f}s")
else:
    print("  cafeteria.wav already exists.")

# 3. Generate separation sources: source_A.wav and source_B.wav (same length, 16kHz mono)
print("\n--- 3. Generating separation sources (source_A.wav, source_B.wav) ---")
src_a_path = os.path.join(SEP_DIR, "source_A.wav")
src_b_path = os.path.join(SEP_DIR, "source_B.wav")

if not os.path.exists(src_a_path) or not os.path.exists(src_b_path):
    text_A = "Trí tuệ nhân tạo đang thay đổi phương thức làm việc và tương tác giữa con người."
    text_B = "Xử lý tín hiệu tiếng nói là lĩnh vực then chốt trong các hệ thống thông minh hiện đại."
    
    audio_A = generate_speech(text_A, pitch_steps=0.0, speed_rate=1.0)
    audio_B = generate_speech(text_B, pitch_steps=-5.0, speed_rate=0.95)
    
    # Match length to exactly 3.2 seconds
    L_target = int(3.2 * TARGET_SR)
    def match_len(sig, L):
        if len(sig) > L:
            return sig[:L]
        else:
            return np.pad(sig, (0, L - len(sig)), mode='constant')
            
    audio_A = match_len(audio_A, L_target)
    audio_B = match_len(audio_B, L_target)
    
    # RMS normalization
    def rms_norm(x, target_rms=0.1):
        rms = np.sqrt(np.mean(x**2)) + 1e-12
        return (x / rms * target_rms).astype(np.float32)
        
    audio_A = rms_norm(audio_A)
    audio_B = rms_norm(audio_B)
    
    sf.write(src_a_path, audio_A, TARGET_SR, subtype='PCM_16')
    sf.write(src_b_path, audio_B, TARGET_SR, subtype='PCM_16')
    print(f"  Saved source_A.wav: {len(audio_A)/TARGET_SR:.2f}s, RMS: {np.sqrt(np.mean(audio_A**2)):.4f}")
    print(f"  Saved source_B.wav: {len(audio_B)/TARGET_SR:.2f}s, RMS: {np.sqrt(np.mean(audio_B**2)):.4f}")
else:
    print("  source_A.wav and source_B.wav already exist.")

print("\nAll data preparation complete!")
