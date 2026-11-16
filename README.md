# CSE457 • XỬ LÝ ÂM THANH VÀ TIẾNG NÓI | LAB 5
## SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
### Từ khử nhiễu trong miền STFT đến xác định người nói và tách hai nguồn nói bằng ICA

* **Sinh viên thực hiện:** Nguyễn Thị Linh
* **Mã số sinh viên (MSSV):** 2351260663
* **Lớp:** 65TTNT
* **Khoa:** Công nghệ Thông tin - **Bộ môn:** Trí tuệ Nhân tạo
* **Trường:** Đại học Thủy Lợi

---

## 1. Cấu trúc thư mục dự án

```text
Lab05_2351260663_NguyenThiLinh/
├── Lab 5.pdf                     # Đề bài gốc bài thực hành số 5
├── lab5_data/                    # Dữ liệu âm thanh chuẩn hóa (16 kHz, mono, PCM 16-bit)
│   ├── clean/                    # 3 người nói x 8 utterances (4 enroll, 2 dev, 2 test)
│   │   ├── spk01_enroll_01.wav ... spk01_enroll_04.wav
│   │   ├── spk01_dev_01.wav, spk01_dev_02.wav
│   │   ├── spk01_test_01.wav, spk01_test_02.wav
│   │   ├── spk02_...
│   │   └── spk03_...
│   ├── noise/                    # Âm thanh nhiễu thực tế (độ dài 10.0s)
│   │   ├── fan.wav               # Nhiễu quạt dừng (Stationary noise)
│   │   └── cafeteria.wav         # Nhiễu quán ăn phi dừng (Non-stationary babble noise)
│   └── separation/               # 2 nguồn nói phục vụ phân tách BSS (cùng độ dài 3.2s)
│       ├── source_A.wav          # Người nói A (Nữ)
│       └── source_B.wav          # Người nói B (Nam)
├── results/                      # Kết quả sinh ra từ thực nghiệm
│   ├── audio/                    # Toàn bộ file âm thanh kết quả (.wav)
│   │   ├── clean_reference.wav   # Tín hiệu sạch đối chứng
│   │   ├── noisy_0dB.wav         # Tín hiệu nhiễm nhiễu quạt ở SNR = 0 dB
│   │   ├── noisy_5dB.wav         # Tín hiệu nhiễm nhiễu quạt ở SNR = 5 dB
│   │   ├── noisy_10dB.wav        # Tín hiệu nhiễm nhiễu quạt ở SNR = 10 dB
│   │   ├── enhanced_ss.wav       # Âm thanh làm sạch bằng Spectral Subtraction (β=1.5)
│   │   ├── enhanced_wiener.wav   # Âm thanh làm sạch bằng Wiener Filter (MMSE)
│   │   ├── mix_mic1.wav          # Tín hiệu hỗn hợp quan sát tại Micro 1 (x1)
│   │   ├── mix_mic2.wav          # Tín hiệu hỗn hợp quan sát tại Micro 2 (x2)
│   │   ├── separated_1.wav       # Nguồn 1 tách bằng FastICA (đã giải bất định)
│   │   └── separated_2.wav       # Nguồn 2 tách bằng FastICA (đã giải bất định)
│   ├── figures/                  # 7 biểu đồ khoa học độ phân giải cao (300 DPI)
│   │   ├── fig1_clean_noisy_wave_spec.png       # Dạng sóng & phổ Clean vs Noisy (0, 5, 10 dB)
│   │   ├── fig2_spectral_subtraction_effect.png # Khảo sát phổ SS theo hệ số β (1.0, 1.5, 2.0)
│   │   ├── fig3_wiener_filter_comparison.png    # So sánh phổ Noisy, SS và Wiener
│   │   ├── fig4_speaker_id_confusion_matrix.png # Ma trận nhầm lẫn nhận dạng người nói
│   │   ├── fig5_speaker_verification_far_frr.png# Đồ thị đánh đổi FAR - FRR và ngưỡng khóa θ*
│   │   ├── fig6_ica_separation_results.png      # Dạng sóng & phổ Nguồn gốc, Hỗn hợp, Tách
│   │   └── fig7_ica_correlation_matrices.png    # Heatmap tương quan ICA (Easy vs Hard)
│   └── experiment_results.json   # Lưu trữ toàn bộ chỉ số định lượng chi tiết
├── Lab05.py                      # Mã nguồn Python độc lập thực thi toàn bộ pipeline
├── Lab05.ipynb                   # Jupyter Notebook hoàn chỉnh có giải thích & output trực quan
├── prepare_data.py               # Script chuẩn bị và tổng hợp tập dữ liệu lab5_data/
├── generate_notebook.py          # Script sinh tự động cấu trúc notebook Lab05.ipynb
├── report_Lab05.md               # Báo cáo học thuật chi tiết 4-6 trang đầy đủ 12 câu hỏi
├── report.docx                   # File báo cáo Microsoft Word
├── report.pdf                    # File báo cáo PDF xuất chuẩn
└── README.md                     # Hướng dẫn chi tiết môi trường và cách thực thi
```

---

## 2. Thiết lập Môi trường thực thi

Khuyến nghị sử dụng Python 3.10 trở lên. Cài đặt các gói thư viện cần thiết thông qua lệnh:

```bash
pip install numpy scipy matplotlib seaborn librosa soundfile scikit-learn gtts python-docx
```

---

## 3. Hướng dẫn Chạy chương trình

### Cách 1: Chạy toàn bộ pipeline qua Python Script (Khuyến nghị)
```bash
python Lab05.py
```
* Script sẽ tự động đọc dữ liệu từ `lab5_data/`, chạy tuần tự 5 khối chức năng, in ra toàn bộ số liệu đo đạc (SNR, $\Delta\mathrm{SNR}$, độ chính xác phân loại, ngưỡng EER, ma trận tương quan FastICA), đồng thời xuất tất cả file âm thanh vào `results/audio/` và lưu các biểu đồ vào `results/figures/`.

### Cách 2: Chạy tương tác qua Jupyter Notebook
```bash
jupyter notebook Lab05.ipynb
```
hoặc mở trực tiếp file `Lab05.ipynb` trong Visual Studio Code để xem từng cell code kèm hình vẽ và bảng kết quả tương ứng.

---

## 4. Tóm tắt Kết quả Thực nghiệm Chính

1. **Khử nhiễu tiếng nói (Speech Enhancement):**
   * Sai số SNR đo được so với SNR mục tiêu: $\approx 0.0000$ dB (thỏa mãn yêu cầu khắt khe $\le 0.2$ dB).
   * Tại SNR đầu vào $5.0$ dB, Spectral Subtraction ($\beta=1.5$) nâng SNR lên $12.11$ dB ($\Delta\mathrm{SNR} = +7.11$ dB).
   * Bộ lọc Wiener tối ưu MMSE nâng SNR lên $13.93$ dB ($\Delta\mathrm{SNR} = +8.93$ dB), cho âm sắc mượt mà hơn và hạn chế tối đa hiện tượng *musical noise*.

2. **Nhận dạng và Xác thực người nói (Speaker Recognition):**
   * Mô hình Baseline (MFCC + $\Delta$, 4 tokens enrollment) đạt độ chính xác **100%** trên tập Test độc lập.
   * Quét ngưỡng xác thực trên Development Set xác định ngưỡng tối ưu $\theta^* = 0.9900$ (xấp xỉ điểm EER). Khi khóa $\theta^*$ và áp dụng lên Test Set độc lập, hệ thống đạt $\mathrm{FAR} = 0.0000$ và $\mathrm{FRR} = 0.0000$.
   * Thí nghiệm độ bền vững (Robustness): Trong môi trường nhiễu 5 dB, độ chính xác nhận dạng giảm mạnh xuống $33.3\%$. Spectral Subtraction không phục hồi được nhận dạng ($33.3\%$) do méo phi tuyến, trong khi Wiener Filter phục hồi độ chính xác lên **$66.7\%$**.

3. **Phân tách nguồn mù (Blind Source Separation - FastICA):**
   * Ma trận trộn dễ $A_{\text{easy}}$ ($\det = 0.76$): FastICA tách hoàn hảo 2 nguồn nói chồng lấn. Sau khi giải quyết bất định vị trí (*Permutation*) và dấu (*Sign*), hệ số tương quan chéo đạt tuyệt đối $|\rho| \ge 0.9998$.
   * Ma trận gần suy biến $A_{\text{hard}}$ ($\det = 0.145$): Quá trình tách nhạy cảm hơn với điều kiện số.
   * Thêm nhiễu micro (15 dB): Tương quan chéo giảm xuống $\approx 0.956$ do vi phạm giả thiết nguồn thuần túy.
