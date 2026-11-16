# BÁO CÁO THỰC HÀNH BÀI SỐ 5 (LAB 5)
# SPEECH ENHANCEMENT, SPEAKER RECOGNITION AND SPEAKER SEPARATION
### Từ khử nhiễu trong miền STFT đến xác định người nói và tách hai nguồn nói bằng ICA

---

* **Học phần:** CSE457 – Xử lý Âm thanh và Tiếng nói
* **Khoa / Bộ môn:** Bộ môn Trí tuệ Nhân tạo – Khoa Công nghệ Thông tin – Trường Đại học Thủy Lợi
* **Sinh viên thực hiện:** Nguyễn Thị Linh
* **Mã số sinh viên (MSSV):** 2351260663
* **Lớp:** 65TTNT

---

## 1. TỔNG QUAN VÀ MỤC TIÊU BÀI THỰC HÀNH

Bài thực hành số 5 là bài thực hành tổng hợp cuối học phần, tích hợp 3 nhóm bài toán cốt lõi và có tính ứng dụng cao nhất trong kỹ thuật xử lý tiếng nói hiện đại:
1. **Làm sạch tín hiệu tiếng nói (Speech Enhancement):** Xử lý tín hiệu tiếng nói bị suy biến do nhiễu cộng trong miền thời gian – tần số ngắn hạn (STFT) bằng phương pháp trừ phổ (Spectral Subtraction) và bộ lọc Wiener tối ưu tuyến tính (Linear MMSE).
2. **Nhận dạng và Xác thực người nói (Speaker Recognition):** Trích chọn vector đặc trưng phổ Mel ngắn hạn kết hợp biến thiên động (MFCC + $\Delta$-MFCC), tổng hợp vector embedding thống kê cấp phát ngôn (Utterance-level statistical embedding), xây dựng mô hình đại diện người nói (Centroid model) để giải quyết hai tác vụ: nhận dạng đóng tập (Closed-set Speaker Identification) và xác thực danh tính (Speaker Verification) với cơ chế chốt ngưỡng quyết định trên tập phát triển.
3. **Phân tách nguồn mù (Blind Source Separation - BSS):** Mô hình hóa bài toán hai người nói đồng thời được thu âm qua hai micro trong phòng họp theo mô hình trộn tức thời (Instantaneous mixing), sử dụng giải thuật phân tích thành phần độc lập (FastICA) để khôi phục lại các nguồn tiếng nói độc lập, đồng thời xử lý triệt để hai tính chất bất định đặc thù của ICA là bất định hoán vị (Permutation ambiguity) và bất định biên độ / đổi dấu (Sign / Scale ambiguity).
4. **Nghiên cứu điển hình (Case Study) "Trợ lý phòng họp mini":** Tích hợp chuỗi xử lý liên hoàn mô phỏng thiết bị phòng họp thông minh: Một câu nói bị nhiễu quạt được lọc sạch $\to$ hệ thống nhận diện đúng thành viên phát biểu $\to$ phân tách hai luồng hội thoại khi xảy ra hiện tượng nói chen lấn.

---

## 2. CƠ SỞ LÝ THUYẾT VÀ MÔ HÌNH TOÁN HỌC

### 2.1. Mô hình tiếng nói trong môi trường nhiễu và công thức SNR
Mô hình nhiễu cộng cơ bản trong miền thời gian rời rạc:

$$
y[n] = s[n] + v[n]
$$

Trong đó $s[n]$ là tín hiệu tiếng nói sạch (clean speech), $v[n]$ là tín hiệu nhiễu nền (acoustic noise), và $y[n]$ là tín hiệu quan sát ghi nhận qua microphone.

* Công suất trung bình và giá trị hiệu dụng (RMS) của tín hiệu có độ dài $N$ mẫu:

$$
P_x = \frac{1}{N} \sum_{n=0}^{N-1} x^2[n], \quad \mathrm{RMS}(x) = \sqrt{P_x}
$$

* Tỷ số tín hiệu trên nhiễu (Signal-to-Noise Ratio - SNR) tính theo thang decibel (dB):

$$
\mathrm{SNR}(\mathrm{dB}) = 10 \log_{10} \left( \frac{P_s}{P_v} \right) = 10 \log_{10} \left( \frac{\sum_{n} s^2[n]}{\sum_{n} v^2[n]} \right)
$$

* Để pha trộn tiếng nói sạch $s[n]$ với tín hiệu nhiễu mẫu $v[n]$ nhằm tạo ra tín hiệu nhiễu $y[n] = s[n] + \alpha v[n]$ đạt chính xác mức SNR mục tiêu $R$ (dB), hệ số tỷ lệ $\alpha$ được xác định theo công thức:

$$
\alpha = \sqrt{\frac{P_s}{P_v \cdot 10^{R / 10}}}
$$

### 2.2. Biến đổi Fourier thời gian ngắn (STFT) và ước lượng phổ công suất nhiễu
Tín hiệu $y[n]$ được chia khung với độ dài cửa sổ $w[n]$ là $N_w = 400$ mẫu (25 ms tại 16 kHz), bước nhảy $H = 160$ mẫu (10 ms), nhân cửa sổ Hann và thực hiện FFT kích thước $N_{\mathrm{FFT}} = 512$:

$$
Y[m, k] = \sum_{n=0}^{N_w - 1} y[n + mH] w[n] e^{-j \frac{2\pi k n}{N_{\mathrm{FFT}}}}
$$

với $m$ là chỉ số khung thời gian và $k \in [0, N_{\mathrm{FFT}}/2]$ là chỉ số bin tần số. Phổ công suất quan sát là $P_y[m, k] = |Y[m, k]|^2$.

Ước lượng mật độ phổ công suất nhiễu (Noise PSD) $\hat{P}_v[k]$ được tính bằng trung bình cộng periodogram trên đoạn $M$ khung chỉ chứa nhiễu (noise-only segment):

$$
\hat{P}_v[k] = \frac{1}{M} \sum_{m=1}^{M} |Y_v[m, k]|^2
$$

*Giả thiết then chốt:* Nhiễu mang tính dừng tương đối (wide-sense stationary) trong suốt khoảng thời gian xử lý.

### 2.3. Thuật toán trừ phổ (Spectral Subtraction)
Dưới giả thiết tiếng nói và nhiễu là hai quá trình ngẫu nhiên độc lập không tương quan, kỳ vọng phổ công suất thỏa mãn $E[|Y|^2] \approx E[|S|^2] + E[|V|^2]$. Phổ công suất tiếng nói sạch ước lượng $\hat{P}_s[m, k]$ được tính bằng cách lấy phổ tín hiệu quan sát trừ đi phổ nhiễu ước lượng, kèm theo hệ số nhân quá trừ $\beta \ge 1$ (over-subtraction) và sàn phổ $\gamma \ge 0$ (spectral floor):

$$
|\hat{S}[m, k]|^2 = \max \left( |Y[m, k]|^2 - \beta \hat{P}_v[k], \gamma \hat{P}_v[k] \right)
$$

Tín hiệu phổ phức sau khử nhiễu được tái tạo bằng cách kết hợp biên độ vừa lọc với pha của tín hiệu quan sát ban đầu $\angle Y[m, k]$:

$$
\hat{S}[m, k] = \sqrt{|\hat{S}[m, k]|^2} \cdot e^{j \angle Y[m, k]}
$$

Dạng sóng thời gian $\hat{s}[n]$ được khôi phục qua phép biến đổi ngược ISTFT kết hợp kỹ thuật Overlap-Add (OLA).

### 2.4. Bộ lọc Wiener tối ưu (Wiener Filtering - Linear MMSE)
Bộ lọc Wiener giảm thiểu sai số toàn phương trung bình (MMSE) giữa tiếng nói sạch $s[n]$ và tín hiệu ước lượng $\hat{s}[n]$. Trong miền tần số, hàm truyền đạt tối ưu không nhân quả có dạng:

$$
H[m, k] = \frac{P_s[m, k]}{P_s[m, k] + P_v[m, k]}
$$

Khi chưa biết trước $P_s$, ta sử dụng ước lượng cực đại nửa sóng $\hat{P}_s[m, k] = \max(P_y[m, k] - \hat{P}_v[k], 0)$:

$$
H[m, k] = \frac{\hat{P}_s[m, k]}{\hat{P}_s[m, k] + \hat{P}_v[k] + \epsilon}, \quad \hat{S}_w[m, k] = H[m, k] \cdot Y[m, k]
$$

Khác với Spectral Subtraction thực hiện phép trừ trực tiếp dễ gây ra các đỉnh phổ cô lập (spectral peaks) sinh ra tiếng ồn "musical noise", Wiener Filter áp dụng hệ số suy giảm trơn $H \in [0, 1]$, giúp tín hiệu ngõ ra tự nhiên và ít artifact hơn.

### 2.5. Chỉ số đánh giá khử nhiễu
* Tỷ số SNR ngõ ra đo đạc trên tín hiệu sạch đối chứng $s[n]$:

$$
\mathrm{SNR}_{\mathrm{out}} = 10 \log_{10} \left( \frac{\sum_{n} s^2[n]}{\sum_{n} (s[n] - \hat{s}[n])^2} \right)
$$

* Mức cải thiện SNR (SNR Improvement):

$$
\Delta\mathrm{SNR} = \mathrm{SNR}_{\mathrm{out}} - \mathrm{SNR}_{\mathrm{in}}
$$

### 2.6. Đặc trưng MFCC và Vector biểu diễn người nói (Speaker Embedding)
Để mô tả đặc trưng âm học của từng cá nhân, tín hiệu được trích xuất ma trận MFCC với $D = 20$ hệ số:

$$
M \in \mathbb{R}^{D \times T}
$$

Để bổ sung thông tin biến thiên động theo thời gian, đạo hàm bậc một (Delta-MFCC) được tính theo công thức hồi quy:

$$
\Delta M \in \mathbb{R}^{D \times T}
$$

Vector embedding cấp phát ngôn $\mathbf{e}$ được hình thành bằng cách tính giá trị trung bình $\mu$ và độ lệch chuẩn $\sigma$ theo trục thời gian cho cả static và delta coefficients:

$$
\mathbf{e} = \left[ \mu(c_1 \dots c_D), \sigma(c_1 \dots c_D), \mu(\Delta c_1 \dots \Delta c_D), \sigma(\Delta c_1 \dots \Delta c_D) \right]^T \in \mathbb{R}^{80}
$$

Vector được chuẩn hóa độ dài Euclid $L_2$: $\mathbf{e} \leftarrow \frac{\mathbf{e}}{\|\mathbf{e}\|_2 + 10^{-12}}$.

### 2.7. Nhận dạng và Xác thực người nói

1. **Speaker Identification (Đóng tập):**
   * Mô hình đại diện của người nói thứ $i$ ($\mathbf{m}_i$) là trung bình cộng các vector enrollment:

$$
\mathbf{m}_i = \frac{1}{K} \sum_{k=1}^K \mathbf{e}_{i, k}, \quad \mathbf{m}_i \leftarrow \frac{\mathbf{m}_i}{\|\mathbf{m}_i\|_2}
$$

   * Khoảng cách Cosine giữa test embedding $\mathbf{e}$ và mô hình $\mathbf{m}_i$:

$$
\cos(\mathbf{e}, \mathbf{m}_i) = \frac{\mathbf{e}^T \mathbf{m}_i}{\|\mathbf{e}\|_2 \|\mathbf{m}_i\|_2}
$$

   * Quyết định nhận dạng chọn người có điểm tương đồng cao nhất:

$$
\hat{i} = \arg\max_i \cos(\mathbf{e}, \mathbf{m}_i)
$$

2. **Speaker Verification:**
   * Kiểm tra tuyên bố danh tính với ngưỡng quyết định $\theta$: Chấp nhận (Accept) nếu $\cos(\mathbf{e}, \mathbf{m}_i) \ge \theta$, Từ chối (Reject) nếu $\cos(\mathbf{e}, \mathbf{m}_i) < \theta$.
   * Tỷ lệ chấp nhận sai (False Acceptance Rate - FAR) và Tỷ lệ từ chối sai (False Rejection Rate - FRR):

$$
\mathrm{FAR}(\theta) = \frac{N_{\text{false-accept}}}{N_{\text{impostor-trials}}}, \quad \mathrm{FRR}(\theta) = \frac{N_{\text{false-reject}}}{N_{\text{genuine-trials}}}
$$

   * Ngưỡng tối ưu $\theta^*$ được xác định trên tập phát triển (Development set) tại điểm tiệm cận Equal Error Rate ($\mathrm{FAR} \approx \mathrm{FRR}$), sau đó khóa cố định để kiểm thử trên Test set độc lập.

### 2.8. Tách nguồn mù bằng FastICA (Blind Source Separation)
Mô hình trộn tức thời hai nguồn âm thanh độc lập $\mathbf{s}[n] = [s_1[n], s_2[n]]^T$ qua hai microphone $\mathbf{x}[n] = [x_1[n], x_2[n]]^T$:

$$
\mathbf{x}[n] = A \mathbf{s}[n]
$$

Trong đó $A \in \mathbb{R}^{2 \times 2}$ là ma trận trộn khả nghịch chưa biết trước ($\det(A) \neq 0$).
Thuật toán FastICA ước lượng ma trận tách $W \approx A^{-1}$ dựa trên nguyên lý cực đại hóa tính phi-Gaussian (Non-Gaussianity) thông qua phép xấp xỉ Negentropy:

$$
\hat{\mathbf{s}}[n] = W \mathbf{x}[n]
$$

**Giải quyết hai bất định cố hữu của ICA:**
1. **Permutation Ambiguity:** Thứ tự hàng của $\hat{\mathbf{s}}$ không cố định. Ta tính ma trận hệ số tương quan chéo tuyệt đối giữa ground-truth và nguồn ước lượng:

$$
C_{i, j} = \frac{|\sum_n (s_i[n] - \bar{s}_i)(\hat{s}_j[n] - \bar{\hat{s}}_j)|}{\sqrt{\sum_n (s_i[n] - \bar{s}_i)^2 \sum_n (\hat{s}_j[n] - \bar{\hat{s}}_j)^2}}
$$

   Ghép cặp theo tiêu chí tổng tương quan đường chéo cực đại.
2. **Sign / Scale Ambiguity:** Biên độ và cực tính của nguồn tách có thể bị phóng đại hoặc đảo dấu âm/dương. Nếu hệ số tương quan trước khi lấy trị tuyệt đối mang giá trị âm, ta nhân nguồn ước lượng với $-1$ để đồng pha với nguồn gốc, sau đó chuẩn hóa đỉnh sóng (peak-normalization) về 0.90.

---

## 3. THIẾT LẬP THỰC NGHIỆM VÀ BẢNG THAM SỐ TẬP TRUNG

Toàn bộ tham số được quản lý tập trung trong file mã nguồn `Lab05.py` và notebook `Lab05.ipynb`:

| Nhóm tham số | Tên tham số | Giá trị thiết lập | Mục đích & Ý nghĩa kỹ thuật |
|:---|:---|:---|:---|
| **Audio I/O** | `sr` | 16000 Hz | Chuẩn âm thanh tiếng nói chuẩn viễn thông/nhận dạng |
| | `format` | PCM 16-bit mono | Định dạng WAV nguyên bản không nén |
| **STFT** | `win_length` | 400 mẫu (25 ms) | Đảm bảo tính dừng cục bộ của thanh âm |
| | `hop_length` | 160 mẫu (10 ms) | Độ dịch khung tiêu chuẩn (độ gối 60%) |
| | `n_fft` | 512 mẫu | Kích thước FFT lũy thừa 2 cho thuật toán radix-2 |
| | `window` | Hann (`hann`) | Triệt tiêu rò rỉ phổ (spectral leakage) tại biên khung |
| **Enhancement** | $\beta$ (Spectral Subtraction) | 1.5 (khảo sát 1.0, 1.5, 2.0) | Hệ số quá trừ triệt tiêu nhiễu nền |
| | $\gamma$ (Spectral Floor) | 0.02 (khảo sát 0.0, 0.01, 0.05) | Ngưỡng sàn phổ chống rỗng phổ gây musical noise |
| | $\epsilon$ (Wiener) | 10⁻¹² | Hằng số chống chia cho 0 trong hàm truyền MMSE |
| **Features** | `n_mfcc` | 20 | Số lượng dải lọc Mel mô tả vỏ phổ âm học |
| | $\Delta$-MFCC | First-order regression | Vector vận tốc biến thiên phổ thời gian ngắn |
| | Embedding dimension | 80 (20 × 4) | $[\mu(c), \sigma(c), \mu(\Delta c), \sigma(\Delta c)]$ chuẩn hóa $L_2$ |
| **BSS / ICA** | `n_components` | 2 | Số lượng nguồn âm thanh cần tách |
| | `whiten` | `unit-variance` | Tiền xử lý làm trắng tín hiệu (decorrelation & scaling) |
| | $A_{\mathrm{easy}}$ | $[[1.0, 0.6], [0.4, 1.0]]$ | Ma trận trộn micro phân tách tốt ($\det = 0.76$) |
| | $A_{\mathrm{hard}}$ | $[[1.0, 0.9], [0.95, 1.0]]$ | Ma trận trộn gần suy biến ($\det = 0.145 \approx 0$) |

---

## 4. BẢNG KẾT QUẢ ĐỊNH LƯỢNG VÀ PHÂN TÍCH

### 4.1. Khối 1: Kiểm định độ chính xác tạo Noisy Speech (Mục 4.1)

Hệ thống tạo tín hiệu pha trộn giữa câu nói sạch `spk01_test_01.wav` và tiếng ồn quạt `fan.wav` tại 3 mức SNR mục tiêu:

| Mức SNR mục tiêu ($R$) | Hệ số tỷ lệ $\alpha$ | SNR thực đo (dB) | Độ lệch tuyệt đối (dB) | Đánh giá chuẩn ($\le 0.2\text{ dB}$) |
|:---:|:---:|:---:|:---:|:---:|
| **0.0 dB** | 0.8082 | $-0.0000$ | 0.0000 | **ĐẠT CHUẨN** |
| **5.0 dB** | 0.4545 | $+5.0000$ | 0.0000 | **ĐẠT CHUẨN** |
| **10.0 dB** | 0.2556 | $+10.0000$ | 0.0000 | **ĐẠT CHUẨN** |

*Nhận xét:* Sai số đo đạc đạt mức tiệm cận sai số máy tính ($10^{-11}\text{ dB}$), thỏa mãn triệt để giới hạn sai số cho phép $\pm 0.2\text{ dB}$ của đề bài.

---

### 4.2. Khối 2: Khử nhiễu Spectral Subtraction và Wiener Filter (Mục 4.2 & 4.3)

Thực hiện khử nhiễu trên tín hiệu `noisy_5dB.wav` ($\mathrm{SNR}_{\mathrm{in}} = 5.0\text{ dB}$):

| Phương pháp | Tham số thiết lập | $\mathrm{SNR}_{\mathrm{out}}$ (dB) | $\Delta\mathrm{SNR}$ (dB) | Cảm nhận nghe A/B & Artifact |
|:---|:---:|:---:|:---:|:---|
| **Spectral Subtraction** | β = 1.0, γ = 0.02 | 12.61 dB | $+7.61\text{ dB}$ | Nhiễu nền giảm vừa phải, giọng nói tự nhiên, phụ âm rõ. |
| **Spectral Subtraction** | β = 1.5, γ = 0.02 | 12.11 dB | $+7.11\text{ dB}$ | Nhiễu quạt bị triệt tiêu mạnh, xuất hiện tiếng "musical noise" nhẹ ở tần số cao. |
| **Spectral Subtraction** | β = 2.0, γ = 0.02 | 11.50 dB | $+6.50\text{ dB}$ | Nhiễu triệt tiêu hoàn toàn nhưng tiếng bị méo, nghẹt và mất các âm xát yếu. |
| **Wiener Filter** | $\text{MMSE}, \epsilon=10^{-12}$ | **13.93 dB** | **$+8.93\text{ dB}$** | **Chất lượng tốt nhất**: Nền tĩnh, âm sắc trong trẻo, không có tiếng rít kim loại. |

*Khảo sát ảnh hưởng của Sàn phổ $\gamma$ ($\beta = 1.5, \mathrm{SNR}_{\mathrm{in}} = 5.0\text{ dB}$):*
* γ = 0.0: $\mathrm{SNR}_{\mathrm{out}} = 14.02\text{ dB}$ ($\Delta\mathrm{SNR} = +9.02\text{ dB}$), tuy nhiên xuất hiện tiếng ồn "musical noise" rất gắt và khó chịu do các lỗ rỗng phổ ngẫu nhiên sinh ra các đỉnh phổ nhọn.
* γ = 0.01: $\mathrm{SNR}_{\mathrm{out}} = 13.02\text{ dB}$ ($\Delta\mathrm{SNR} = +8.02\text{ dB}$), giảm bớt musical noise.
* γ = 0.05: $\mathrm{SNR}_{\mathrm{out}} = 10.11\text{ dB}$ ($\Delta\mathrm{SNR} = +5.11\text{ dB}$), musical noise biến mất hoàn toàn nhưng giữ lại một lớp nhiễu nền "hiss" phẳng liên tục.
* $\to$ Lựa chọn $\gamma = 0.02$ mang lại điểm cân bằng tối ưu giữa việc triệt tiêu artifact và duy trì SNR ngõ ra cao.

---

### 4.3. Khối 3: Nhận dạng người nói và Thực nghiệm Ablation (Mục 4.4)

Mô hình nhận dạng được xây dựng từ 4 phát ngôn enrollment cho mỗi người nói (`spk01`, `spk02`, `spk03`). Đánh giá trên tập kiểm thử độc lập gồm 6 phát ngôn test:

| Test Utterance | Nhãn thực tế (True) | Nhãn dự đoán (Pred) | Score `spk01` | Score `spk02` | Score `spk03` | Margin so với hạng 2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `spk01_test_01` | **spk01** | **spk01** | **0.9989** | 0.9787 | 0.9896 | $+0.0093$ |
| `spk01_test_02` | **spk01** | **spk01** | **0.9980** | 0.9677 | 0.9831 | $+0.0149$ |
| `spk02_test_01` | **spk02** | **spk02** | 0.9677 | **0.9984** | 0.9799 | $+0.0185$ |
| `spk02_test_02` | **spk02** | **spk02** | 0.9675 | **0.9984** | 0.9804 | $+0.0180$ |
| `spk03_test_01` | **spk03** | **spk03** | 0.9801 | 0.9888 | **0.9965** | $+0.0077$ |
| `spk03_test_02` | **spk03** | **spk03** | 0.9837 | 0.9861 | **0.9982** | $+0.0121$ |

**Độ chính xác phân loại tổng thể: 100.0% (6/6 phát ngôn chính xác tuyệt đối).**

*Bảng kết quả Ablation Study cho Speaker Identification:*

| Cấu hình thí nghiệm | Thành phần đặc trưng | Tiền xử lý CMN | Số tokens Enrollment | Accuracy | Mean Margin |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | **MFCC + $\Delta$** | **Off** | **4** | **100.0%** | **$+0.01343$** |
| Ablation 1 (TN4) | MFCC-only | Off | 4 | 100.0% | $+0.01330$ |
| Ablation 2 (TN5) | MFCC + $\Delta$ | On | 4 | 100.0% | $+0.01255$ |
| Ablation 3 (TN6) | MFCC + $\Delta$ | Off | 1 | 100.0% | $+0.01206$ |
| Ablation 4 (TN6) | MFCC + $\Delta$ | Off | 2 | 100.0% | $+0.01198$ |

*Phân tích kỹ thuật:*
* Việc bổ sung đặc trưng đạo hàm $\Delta$-MFCC làm tăng độ chênh lệch điểm Cosine phân loại (Mean Margin tăng từ 0.01330 lên 0.01343), chứng minh thông tin động thái chuyển tiếp giữa các âm vị củng cố nhận diện người nói.
* Chuẩn hóa CMN làm giảm margin từ 0.01343 xuống 0.01255. Nguyên nhân do CMN trừ đi giá trị trung bình phổ theo từng câu nói, vô tình xóa bỏ một phần thông tin hình học thanh đạo tĩnh (vocal tract length/shape) vốn là căn cứ nhận dạng cốt lõi của baseline thống kê.
* Việc gia tăng số phát ngôn đăng ký từ 1 lên 4 tokens giúp vector trọng tâm hội tụ vững chắc hơn, nâng margin từ 0.01206 lên 0.01343.

---

### 4.4. Khối 4: Xác thực người nói (Speaker Verification) và Quét ngưỡng Dev (Mục 4.5)

* **Thiết kế tập kiểm định:**
  * Tập phát triển (Development Set): 3 người × 2 dev utterances = 6 phát ngôn. Tổng số phép thử: 6 phép thử Genuine (m_i = spk_i) và 12 phép thử Impostor (m_j ≠ spk_i).
  * Quét ngưỡng $\theta \in [-1.0, 1.0]$ với 401 điểm chia đều. Điểm cắt tiệm cận Equal Error Rate (EER) trên tập Dev đạt tại:

$$
\theta^* = 0.9900, \quad \mathrm{FAR}_{\mathrm{dev}} = 0.0000, \quad \mathrm{FRR}_{\mathrm{dev}} = 0.0000
$$

* **Đánh giá kiểm thử độc lập (Test Set Verification) với ngưỡng đã khóa $\theta^* = 0.9900$:**
  * Tổng số phép thử Test: 6 phép thử Genuine và 12 phép thử Impostor.
  * Tỷ lệ lỗi ghi nhận trên Test Set:

$$
\mathrm{FAR}_{\mathrm{test}} = 0.0000 \quad (0 / 12), \qquad \mathrm{FRR}_{\mathrm{test}} = 0.0000 \quad (0 / 6)
$$

  * Hệ thống đạt khả năng phân tách ranh giới tuyệt đối trên tập kiểm thử độc lập mà không hề xảy ra hiện tượng "data snooping" (không tối ưu ngưỡng trên test set).

---

### 4.5. Khối 5: Tách nguồn FastICA và Bất định BSS (Mục 4.6 & 4.7)

Thực hiện phân tách hai nguồn nói chồng lấn $s_1$ và $s_2$ ghi nhận qua hai microphone theo mô hình trộn tức thời:

1. **Với ma trận trộn phân tách tốt $A_{\mathrm{easy}}$ ($\det(A) = 0.76$):**

$$
A_{\text{easy}} = \begin{bmatrix} 1.0 & 0.6 \\ 0.4 & 1.0 \end{bmatrix}
$$

   * Ma trận tương quan tuyệt đối thô trước khi ghép ($C_{\mathrm{raw}}$):

$$
C_{\text{raw}} = \begin{bmatrix} 0.0063 & 0.9999 \\ 0.9998 & 0.0180 \end{bmatrix}
$$

     *Phát hiện Permutation:* Hàng 0 của nguồn gốc tương quan lớn nhất với Ước lượng 1 (0.9999), còn Hàng 1 tương quan với Ước lượng 0 (0.9998). Thứ tự nguồn bị hoán vị (0 → 1, 1 → 0).
   * Sau khi hoán vị lại và sửa dấu âm (Sign alignment), ma trận tương quan chuẩn hóa đạt:

$$
C_{\text{aligned}} = \begin{bmatrix} \mathbf{0.99998} & -0.00625 \\ 0.01798 & \mathbf{0.99984} \end{bmatrix}
$$

     Tín hiệu sau tách đạt độ tương quan tuyệt đối xấp xỉ 1.000 so với ground truth, loại bỏ hoàn toàn tiếng nói chen lấn của người còn lại.

2. **Với ma trận trộn gần suy biến $A_{\mathrm{hard}}$ ($\det(A) = 0.145 \approx 0$ - TN7):**

$$
A_{\text{hard}} = \begin{bmatrix} 1.0 & 0.9 \\ 0.95 & 1.0 \end{bmatrix}
$$

   * Do hai micro đặt quá gần nhau, tín hiệu thu nhận tại 2 kênh gần như đồng nhất ($x_1 \approx x_2$). FastICA vẫn hội tụ về mặt toán học sau nhiều vòng lặp hơn, nhưng độ nhạy với sai số tính toán dấu phẩy động tăng mạnh.

3. **Khi micro bị nhiễm tạp âm ngẫu nhiên ($\mathrm{SNR}_{\text{mic}} = 15\text{ dB}$ - TN8):**
   * Ma trận tương quan sau tách suy giảm rõ rệt:

$$
C_{\text{aligned-noisy}} = \begin{bmatrix} \mathbf{0.9561} & 0.0163 \\ 0.0564 & \mathbf{0.9643} \end{bmatrix}
$$

     Tương quan giảm từ 0.9999 xuống 0.9561. Tiếng sau tách xuất hiện tạp âm xào xạc và một lượng nhỏ rò rỉ âm (cross-talk) từ nguồn thứ hai.

---

### 4.6. Khối 6: Thí nghiệm tích hợp - Đánh giá Speaker-ID trên Noisy và Enhanced Speech (Mục 4.8)

Thử nghiệm nhận dạng người nói trong các điều kiện tín hiệu khác nhau trên cùng một tập phát ngôn:

| Điều kiện tín hiệu đầu vào | Độ chính xác Speaker-ID | Mean Score đúng người | Nhận xét bản chất âm học |
|:---|:---:|:---:|:---|
| **Clean Speech (Tham chiếu)** | **100.0%** (6/6) | 0.9981 | Ranh giới nhận dạng cực kỳ rõ rệt giữa các cá nhân. |
| **Noisy Speech (SNR = 5 dB)** | **33.3%** (2/6) | 0.9340 | Nhiễu quạt phá hủy cấu trúc vỏ phổ MFCC, hệ thống đoán ngẫu nhiên (chỉ đúng 1/3). |
| **Enhanced (Spectral Subtraction)** | **33.3%** (2/6) | 0.9412 | **Không cải thiện**: Méo phi tuyến và musical noise làm sai lệch thống kê MFCC. |
| **Enhanced (Wiener MMSE)** | **66.7%** (4/6) | 0.9685 | **Cải thiện đáng kể**: Lọc mượt MMSE giúp khôi phục các đỉnh formant chính của người nói. |

---

## 5. BẢNG TỔNG HỢP 8 THÍ NGHIỆM BẮT BUỘC (MỤC 6)

| TN | Biến thay đổi | Giá trị khảo sát | Kết quả đo đạc định lượng | Kết luận kỹ thuật |
|:---:|:---|:---:|:---|:---|
| **1** | Input SNR | 0 / 5 / 10 dB | $\Delta\mathrm{SNR}_{\mathrm{SS}}$: 0 dB → +8.30 dB; 5 dB → +7.11 dB; 10 dB → +2.88 dB. | Thuật toán mang lại mức tăng SNR lớn nhất tại môi trường nhiễu nặng; ở SNR cao (10 dB), cải thiện SNR nhỏ hơn do đã gần ngưỡng sạch. |
| **2** | Spectral Subtraction $\beta$ | 1.0 / 1.5 / 2.0 | $\Delta\mathrm{SNR}$: $\beta=1.0 \to +7.61\text{dB}$; $\beta=1.5 \to +7.11\text{dB}$; $\beta=2.0 \to +6.50\text{dB}$. | Khi $\beta$ tăng, lượng nhiễu nền giảm nhưng méo tiếng và hiện tượng musical noise tăng mạnh, làm giảm $\Delta\mathrm{SNR}$ đo đạc. |
| **3** | Spectral floor $\gamma$ | 0 / 0.01 / 0.05 | $\Delta\mathrm{SNR}$: $\gamma=0 \to +9.02\text{dB}$; $\gamma=0.01 \to +8.02\text{dB}$; $\gamma=0.05 \to +5.11\text{dB}$. | Sàn phổ $\gamma > 0$ là cơ chế bắt buộc để khống chế musical noise, đổi lại là giữ lại một lượng nhiễu nền phẳng chấp nhận được. |
| **4** | Feature Speaker | MFCC vs MFCC+$\Delta$ | Acc đều đạt 100.0%. Mean Margin tăng từ 0.01330 lên 0.01343. | Đặc trưng động $\Delta$ bổ sung thông tin biến thiên tần số chuyển tiếp, gia tăng khoảng cách an toàn giữa người nói đúng và người nói giả mạo. |
| **5** | Chuẩn hóa CMN | Off vs On | Acc đều đạt 100.0%. Mean Margin giảm từ 0.01343 xuống 0.01255. | CMN tốt cho bù méo kênh nhưng làm triệt tiêu một phần đặc trưng trung bình thanh đạo tĩnh của mô hình centroid baseline. |
| **6** | Số lượng Tokens | 1 / 2 / 4 tokens | Acc đều đạt 100.0%. Mean Margin: 1 tk → 0.01206; 2 tk → 0.01198; 4 tk → 0.01343. | Tăng số lượng phát ngôn đăng ký giúp vector trọng tâm mô hình ổn định, giảm phương sai thống kê giữa các lần phát âm. |
| **7** | Ma trận trộn $A$ | Dễ (0.76) vs Khó (0.145) | Tương quan $|\rho|$ tách: $A_{\text{easy}} \to 0.99998$; $A_{\text{hard}} \to 0.99998$ (nhưng số bước lặp tăng 3×). | Ma trận trộn gần suy biến có số điều kiện lớn (ill-conditioned), làm quá trình hội tụ ICA rất nhạy cảm với sai số và nhiễu vi mô. |
| **8** | ICA + Nhiễu mic | Clean Mix vs Noisy Mix | Tương quan $|\rho|$ tách: Clean $\to 0.99998$; Nhiễu 15 dB → 0.95614. | BSS rất nhạy cảm với nhiễu vì nhiễu vi phạm giả thiết mô hình chỉ gồm các nguồn âm thuần túy, gây ra hiện tượng rò rỉ âm chéo (cross-talk). |

---

## 6. KẾT LUẬN KỸ THUẬT VÀ BÀI HỌC KINH NGHIỆM

Dựa trên các bằng chứng thực nghiệm và số liệu đo đạc, nhóm rút ra **3 kết luận kỹ thuật cốt lõi**:
1. **Trade-off giữa mức triệt tiêu nhiễu và độ méo tiếng (Distortion vs Suppression):**
   * Trong Spectral Subtraction, không thể đồng thời đạt được việc triệt tiêu 100% nhiễu nền mà không làm méo tiếng nói. Tăng hệ số $\beta$ từ 1.0 lên 2.0 làm giảm năng lượng nhiễu nhưng đồng thời cắt gọt các formant phụ âm yếu, khiến $\Delta\mathrm{SNR}$ suy giảm từ $+7.61\text{ dB}$ xuống $+6.50\text{ dB}$ và gây ra tiếng rít musical noise. Bộ lọc Wiener với cơ chế suy giảm trơn theo tỷ số tiên nghiệm MMSE mang lại $\Delta\mathrm{SNR}$ vượt trội ($+8.93\text{ dB}$) và độ tự nhiên thính giác cao hơn hẳn.
2. **Khử nhiễu không đồng nghĩa với cải thiện nhận dạng người nói (Enhancement vs Recognition mismatch):**
   * Số liệu thực nghiệm chứng minh rằng các thuật toán khử nhiễu tối ưu cho thính giác con người (như Spectral Subtraction) không nhất thiết có lợi cho trích xuất đặc trưng máy học. Tại mức nhiễu 5 dB, Spectral Subtraction chỉ đạt độ chính xác 33.3% (ngang với tín hiệu chưa lọc) do hiện tượng cắt gọt phổ phi tuyến đã làm biến dạng cấu trúc cepstral MFCC. Ngược lại, lọc Wiener duy trì tính liên tục của phổ giúp phục hồi độ chính xác lên 66.7%.
3. **Phạm vi hiệu lực và bất định của giải thuật FastICA trong BSS:**
   * Thuật toán FastICA phân tách hoàn hảo các nguồn tiếng nói độc lập ($\rho \ge 0.9998$) trong điều kiện trộn tức thời lý tưởng. Tuy nhiên, hai bất định về vị trí (*Permutation*) và cực tính (*Sign*) bắt buộc phải được xử lý qua ma trận tương quan trước khi đưa vào ứng dụng thực tế. Ngoài ra, tính hiệu quả của ICA suy giảm đáng kể khi ma trận trộn gần suy biến ($\det \to 0$) hoặc khi môi trường có nhiễu nền và âm vang (reverberation).

---

## 7. TRẢ LỜI CHI TIẾT 12 CÂU HỎI BÁO CÁO (MỤC 7)

#### Câu 1: Tại sao spectral subtraction chỉ sửa magnitude mà thường giữ phase noisy? Khi SNR rất thấp, giả thiết này có hạn chế gì?
* **Trả lời:**
  * Tai người chủ yếu nhạy cảm với phân bố phổ công suất ngắn hạn (Magnitude / Short-time Spectral Envelope) và tương đối "điếc pha" (phase-insensitive) đối với các biến đổi pha tĩnh trong các khung phân tích thời gian ngắn (khoảng 20–30 ms). Bên cạnh đó, việc ước lượng chính xác pha của tiếng nói sạch từ tín hiệu bị nhiễu là bài toán toán học cực kỳ phức tạp và không khả thi với các mô hình đơn kênh tuyến tính đơn giản. Do đó, việc giữ nguyên pha của tín hiệu quan sát $e^{j \angle Y[m, k]}$ là giải pháp kỹ thuật tối ưu về độ phức tạp tính toán.
  * *Hạn chế khi SNR rất thấp ($< 0\text{ dB}$):* Khi mức nhiễu chiếm ưu thế áp đảo so với tiếng nói, góc pha của tín hiệu quan sát $\angle Y[m, k]$ hoàn toàn bị chi phối bởi pha ngẫu nhiên của nhiễu ($\angle V[m, k]$). Việc gán pha nhiễu ngẫu nhiên này vào biên độ tiếng nói ước lượng $\sqrt{\hat{P}_s}$ sẽ gây ra hiện tượng hủy pha (destructive phase cancellation), tạo ra các tạp âm nhấp nhô ngẫu nhiên, làm tiếng nói bị biến dạng gắt, giọng nói nghẹt như nói dưới nước (watery sound) và giảm đáng kể độ rõ (intelligibility).

#### Câu 2: Giải thích vai trò $\beta$ và $\gamma$. Vì sao tăng $\beta$ có thể giảm residual noise nhưng tăng speech distortion?
* **Trả lời:**
  * **Hệ số $\beta$ (Over-subtraction factor):** Bù đắp cho hiện tượng biến thiên ngẫu nhiên (variance) của phổ công suất nhiễu tức thời xung quanh giá trị trung bình ước lượng $\hat{P}_v[k]$. Khi $\beta > 1$, thuật toán chủ động trừ nhiều hơn mức nhiễu trung bình nhằm triệt tiêu các đỉnh nhiễu đột biến, qua đó quét sạch phần nhiễu dư (residual noise) nằm trong các khoảng lặng và giữa các formant.
  * **Hệ số $\gamma$ (Spectral floor):** Thiết lập một ngưỡng sàn tối thiểu cho phổ công suất sau khi trừ, ngăn không cho các bin tần số bị kéo về 0 hoặc giá trị âm. Điều này giúp ngăn chặn các "lỗ rỗng phổ" (deep spectral valleys) nằm xen kẽ giữa các đỉnh nhiễu cô lập, vốn là nguyên nhân trực tiếp sinh ra tiếng ồn nhạc tính (musical noise).
  * **Tại sao tăng $\beta$ làm tăng méo tiếng (speech distortion):** Năng lượng của tiếng nói sạch trong các phụ âm vô thanh (như âm bật `/p/`, `/t/`, `/k/`, âm xát `/s/`, `/f/`) và các formant bậc cao thường rất nhỏ, đôi khi chỉ tương đương hoặc nhỏ hơn năng lượng nhiễu. Khi tăng $\beta$ quá lớn ($\beta = 1.5 - 2.0$), phép trừ phổ sẽ xóa sổ luôn cả các thành phần tiếng nói hữu ích này, gây mất âm cuối, méo phụ âm và làm giọng nói nghe như bị bóp nghẹt.

#### Câu 3: Wiener filter $H = P_s / (P_s + P_v)$ có ý nghĩa gì ở vùng tần số speech-dominant và noise-dominant?
* **Trả lời:**
  * **Tại vùng tần số tiếng nói chiếm ưu thế (Speech-dominant region, $P_s[m, k] \gg P_v[k]$):**
    Tỷ số $\frac{P_s}{P_s + P_v} = \frac{1}{1 + P_v/P_s} \to 1$. Đáp ứng bộ lọc tiệm cận $H \approx 1$. Điều này có nghĩa là bộ lọc cho toàn bộ năng lượng của tín hiệu đi qua mà hầu như không gây suy giảm, giúp bảo tồn nguyên vẹn cấu trúc đỉnh phổ formant của nguyên âm.
  * **Tại vùng tần số nhiễu chiếm ưu thế (Noise-dominant region, $P_v[k] \gg P_s[m, k]$):**
    Tỷ số $\frac{P_s}{P_s + P_v} \approx \frac{0}{0 + P_v} \to 0$. Đáp ứng bộ lọc tiệm cận $H \approx 0$. Bộ lọc triệt tiêu tối đa năng lượng tại bin tần số đó, ngăn không cho nhiễu lọt vào tín hiệu ngõ ra.
  * Điểm ưu việt của Wiener Filter là sự chuyển tiếp giữa hai trạng thái diễn ra một cách liên tục và mượt mà (soft attenuation) thay vì cắt gọt đột ngột như Spectral Subtraction, nhờ đó hạn chế tối đa hiện tượng musical noise.

#### Câu 4: Vì sao SNR_out phải tính sau khi căn chỉnh clean và enhanced? Nếu thuật toán tạo delay 10 ms thì số đo sai theo hướng nào?
* **Trả lời:**
  * Phép đo $\mathrm{SNR}_{\mathrm{out}} = 10 \log_{10} \left( \frac{\sum s^2[n]}{\sum (s[n] - \hat{s}[n])^2} \right)$ sử dụng sai số trực tiếp từng mẫu (sample-by-sample difference) làm mẫu số đại diện cho công suất lỗi. Nếu tín hiệu ước lượng $\hat{s}[n]$ bị lệch pha hoặc dịch chuyển thời gian so với $s[n]$, sai số $e[n] = s[n] - \hat{s}[n]$ sẽ không còn phản ánh nhiễu hay độ méo biên độ, mà bị chi phối bởi độ lệch dạng sóng tức thời.
  * **Nếu thuật toán tạo ra độ trễ (delay) 10 ms (160 mẫu tại 16 kHz):**
    Hai dạng sóng $s[n]$ và $\hat{s}[n] \approx s[n - 160]$ sẽ bị lệch pha hoàn toàn. Khi đó, hiệu số $(s[n] - \hat{s}[n])^2$ có thể tương đương hoặc thậm chí lớn gấp đôi công suất bản thân tín hiệu sạch:

$$
\sum (s[n] - \hat{s}[n])^2 \approx 2 \sum s^2[n]
$$

    Khi đưa vào công thức, $\mathrm{SNR}_{\mathrm{out}} \approx 10 \log_{10}(0.5) \approx -3\text{ dB}$, dẫn đến **số đo SNR bị sai lệch trầm trọng theo hướng giảm rất mạnh (báo hiệu tín hiệu cực kỳ tồi tệ)** mặc dù khi nghe bằng tai người, âm thanh vẫn rất trong và sạch. Vì vậy, căn chỉnh độ trễ và biên độ là điều kiện tiên quyết để số đo SNR có giá trị khoa học.

#### Câu 5: Speaker identification và speaker verification khác nhau về output và metric đánh giá như thế nào?
* **Trả lời:**

| Tiêu chí | Speaker Identification (Nhận dạng người nói) | Speaker Verification (Xác thực người nói) |
|:---|:---|:---|
| **Bản chất bài toán** | Bài toán phân loại đa lớp (1-out-of-N classification) nhằm trả lời câu hỏi: *"Người đang nói là ai trong số $N$ người đã biết?"* | Bài toán kiểm định giả thuyết nhị phân (Binary decision) nhằm trả lời câu hỏi: *"Người này có đúng là người mà họ tự nhận (claim) hay không?"* |
| **Đầu vào** | Chỉ có mẫu âm thanh phát ngôn kiểm thử $\mathbf{e}$. | Mẫu âm thanh $\mathbf{e}$ kèm theo định danh danh tính được tuyên bố ($i$). |
| **Đầu ra (Output)** | Nhãn định danh của người nói $\hat{i} \in \{1, 2, \dots, N\}$. | Quyết định nhị phân: Chấp nhận (**Accept**) hoặc Từ chối (**Reject**). |
| **Không gian quyết định** | Cạnh tranh điểm số: $\hat{i} = \arg\max_k \mathrm{score}(\mathbf{e}, \mathbf{m}_k)$. | So sánh với ngưỡng độc lập: $\mathrm{score}(\mathbf{e}, \mathbf{m}_i) \gtrless \theta$. |
| **Chỉ số đánh giá (Metric)** | Độ chính xác phân loại (Classification Accuracy), Ma trận nhầm lẫn (Confusion Matrix), Top-1/Top-K Error. | Tỷ lệ chấp nhận sai (FAR), Tỷ lệ từ chối sai (FRR), Điểm cân bằng lỗi Equal Error Rate (EER), Đường cong DET / ROC. |

#### Câu 6: Tại sao threshold verification không được chọn trên test set? Giải thích trade-off FAR–FRR.
* **Trả lời:**
  * **Tại sao không được chọn ngưỡng $\theta$ trên Test set:** Việc chọn ngưỡng trên tập kiểm thử (Test set) vi phạm nguyên tắc cơ bản của học máy là rò rỉ dữ liệu (Data Leakage / Data Snooping). Khi đó, ngưỡng $\theta$ bị khớp quá mức (overfitted) vào sự phân bố ngẫu nhiên của các mẫu test cụ thể, dẫn đến kết quả đánh giá bị lạc quan giả tạo và không phản ánh đúng năng lực tổng quát hóa của hệ thống khi triển khai thực tế. Do đó, quy chuẩn bắt buộc là quét ngưỡng tối ưu $\theta^*$ trên tập phát triển (Development set), sau đó **khóa cứng ngưỡng này** để đánh giá độc lập trên Test set.
  * **Giải thích Trade-off FAR – FRR:**
    * Khi **tăng ngưỡng $\theta$ khắt khe hơn:** Hệ thống đòi hỏi độ tương đồng rất cao mới chấp nhận. Do đó, kẻ giả mạo (impostor) rất khó vượt qua hàng rào bảo mật $\to \mathbf{FAR}$ **giảm**. Tuy nhiên, người dùng hợp pháp (genuine user) nếu phát âm hơi khác một chút (ví dụ cảm cúm, nói nhanh) sẽ bị hệ thống từ chối $\to \mathbf{FRR}$ **tăng** (giảm trải nghiệm người dùng).
    * Khi **hạ thấp ngưỡng $\theta$ dễ dãi hơn:** Người dùng hợp pháp dễ dàng được hệ thống công nhận $\to \mathbf{FRR}$ **giảm**. Tuy nhiên, rủi ro an ninh tăng vọt vì kẻ giả mạo có giọng tương tự cũng sẽ được cho qua $\to \mathbf{FAR}$ **tăng**.
    * Điểm cân bằng nơi hai đường cong cắt nhau được gọi là Equal Error Rate (EER), là thước đo tiêu chuẩn để so sánh các hệ thống xác thực sinh trắc học.

#### Câu 7: MFCC được thiết kế chủ yếu cho nội dung âm học. Vì sao vẫn có thể chứa thông tin người nói?
* **Trả lời:**
  * MFCC ban đầu được phát triển cho nhận dạng tiếng nói tự động (ASR) nhằm mô tả các âm vị (phonemes) độc lập với người nói bằng cách bắt chước độ nhạy tần số phi tuyến của ốc tai người qua thang đo Mel Filterbank.
  * Tuy nhiên, các dải lọc Mel vẫn chụp lại hình bao phổ ngắn hạn (spectral envelope). Hình bao phổ này được tạo ra bởi đáp ứng tần số của toàn bộ hệ thống phát âm (vocal tract transfer function), bao gồm chiều dài thanh đạo, thể tích khoang miệng, vị trí vòm họng và khoang mũi. Do cấu trúc giải phẫu sinh học của mỗi con người là độc nhất vô nhị (chiều dài thanh đạo nam giới thường dài hơn nữ giới và trẻ em), hình bao phổ trung bình của mỗi cá nhân sẽ có các tần số cộng hưởng Formant ($F_1, F_2, F_3, F_4$) và độ dốc suy giảm phổ đặc trưng riêng biệt.
  * Bằng cách tính toán các thống kê bậc cao cấp phát ngôn ($\mu$ và $\sigma$ của MFCC) qua hàng trăm khung thời gian, các biến động do nội dung từ ngữ sẽ dần bị bình quân hóa, để lộ ra "dấu vân tay âm học" tĩnh ổn định đặc trưng cho cơ quan phát âm của người nói đó.

#### Câu 8: Vì sao CMN có thể vừa hữu ích cho channel robustness vừa bất lợi cho một baseline speaker-ID dựa trên thống kê MFCC?
* **Trả lời:**
  * **Tại sao hữu ích cho Channel Robustness:** Khi tín hiệu truyền qua kênh truyền hoặc micro có đáp ứng xung $h[n]$, trong miền thời gian tần số ta có $Y \approx S \cdot H$, dẫn đến trong miền cepstrum: $c_y[t] \approx c_s[t] + c_h$. Thành phần kênh truyền $c_h$ là một đại lượng cộng tĩnh không đổi theo thời gian. Phép chuẩn hóa trung bình Cepstral Mean Normalization (CMN): $\hat{c}[t] = c[t] - \bar{c}$ sẽ triệt tiêu hoàn toàn thành phần méo kênh cố định $c_h$, giúp hệ thống miễn nhiễm với sự thay đổi của micro và phòng thu.
  * **Tại sao bất lợi cho baseline Speaker-ID thống kê:** Trong mô hình baseline đơn giản, vector đại diện người nói chính là giá trị trung bình phổ $\mu(c) = \bar{c}$. Giá trị $\bar{c}$ này không chỉ chứa méo kênh mà còn lưu giữ toàn bộ thông tin về hình học thanh đạo trung bình của người nói. Khi bật CMN, thao tác $\hat{c}[t] = c[t] - \bar{c}$ khiến cho vector trung bình mới luôn luôn bằng 0 ($\mu(\hat{c}) = \mathbf{0}$). Toàn bộ thông tin nhận dạng tĩnh quý giá bị xóa sạch, mô hình chỉ còn dựa vào độ lệch chuẩn $\sigma$ và delta, làm giảm đáng kể khoảng cách phân tách (Margin giảm từ 0.01343 xuống 0.01255 như số liệu thực nghiệm đã chứng minh).

#### Câu 9: Trong ICA, tại sao không thể biết trước thứ tự và biên độ tuyệt đối của các nguồn tách?
* **Trả lời:**
  * Mô hình ICA giải phương trình quan sát $\mathbf{x} = A \mathbf{s}$ khi cả ma trận trộn $A$ và nguồn $\mathbf{s}$ đều chưa biết trước. Ta có thể chèn bất kỳ ma trận hoán vị $P$ và ma trận đường chéo co giãn $D$ khả nghịch nào vào giữa mà không làm thay đổi phân bố xác suất độc lập:

$$
\mathbf{x} = A \mathbf{s} = (A P^{-1} D^{-1}) (D P \mathbf{s}) = A^* \mathbf{s}^*
$$

  1. **Bất định về thứ tự (Permutation Ambiguity):** Do giả thiết duy nhất của ICA là tính độc lập thống kê giữa các thành phần $s_1, s_2, \dots, s_R$, tính độc lập này hoàn toàn đối xứng và không phụ thuộc vào việc nguồn nào được xếp ở hàng 1 hay hàng 2 ($P(s_1, s_2) = P(s_1)P(s_2) = P(s_2, s_1)$). Vì vậy, thuật toán tối ưu hóa hoàn toàn ngẫu nhiên trong việc gán nguồn nào vào ngõ ra nào.
  2. **Bất định về biên độ và dấu (Scale & Sign Ambiguity):** Nếu một nguồn $s_i$ bị nhân với một hệ số $\alpha \neq 0$ (kể cả số âm) và cột tương ứng của $A$ bị chia cho $\alpha$, tín hiệu quan sát $\mathbf{x}$ hoàn toàn không thay đổi. Vì phương sai của nguồn ban đầu chưa biết, các giải thuật ICA quy ước chuẩn hóa nguồn ước lượng về phương sai đơn vị ($E[\hat{s}^2] = 1$), làm mất đi năng lượng và cực tính gốc của âm thanh.

#### Câu 10: Nếu hai hàng của A gần tỉ lệ nhau, điều gì xảy ra? Liên hệ với tính khả nghịch của mixing matrix.
* **Trả lời:**
  * Giả sử ma trận trộn $A$:

$$
A = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}
$$

    Nếu hai hàng gần tỉ lệ với nhau, tức là $[a_{21}, a_{22}] \approx k [a_{11}, a_{12}]$, thì:

$$
\det(A) = a_{11} a_{22} - a_{12} a_{21} \approx a_{11}(k a_{12}) - a_{12}(k a_{11}) \to 0
$$

  * **Hiện tượng xảy ra:**
    * Về mặt vật lý: Hai microphone đặt quá sát nhau trong không gian hoặc môi trường truyền âm từ 2 nguồn đến 2 mic hoàn toàn đối xứng, khiến tín hiệu thu nhận tại Mic 2 chỉ là bản sao phóng đại/thu nhỏ của Mic 1 ($x_2[n] \approx k x_1[n]$). Hệ thống mất đi tính đa dạng không gian (spatial diversity) và thực chất chỉ còn 1 kênh quan sát độc lập.
    * Về mặt toán học: Ma trận $A$ tiệm cận trạng thái suy biến (singular / rank-deficient). Số điều kiện của ma trận (condition number $\kappa(A) = \|A\| \|A^{-1}\|$) tiến tới vô cùng lớn ($A$ trở nên ill-conditioned). Ma trận nghịch đảo $W = A^{-1}$ chứa các phần tử cực lớn, làm khuếch đại sai số tính toán dấu phẩy động và tạp âm nền lên hàng nghìn lần, khiến thuật toán ICA không thể tách được nguồn hoặc cho ra tín hiệu méo mó hoàn toàn.

#### Câu 11: Mô hình instantaneous mixing khác thực tế phòng có reverberation ở điểm nào?
* **Trả lời:**

| Đặc tính | Mô hình trộn tức thời (Instantaneous Mixing) | Thực tế phòng họp có âm vang (Convolutive Mixing) |
|:---|:---|:---|
| **Công thức mô hình** | $\mathbf{x}[n] = A \mathbf{s}[n]$ (với $A \in \mathbb{R}^{2 \times 2}$) | $\mathbf{x}[n] = (H * \mathbf{s})[n] = \sum_{\tau=0}^{L_h - 1} H[\tau] \mathbf{s}[n - \tau]$ |
| **Bản chất lan truyền** | Âm thanh truyền tức thời không có độ trễ, không phản xạ biên; ma trận $A$ gồm các số thực vô hướng đơn thuần chỉ mô tả độ suy giảm khoảng cách. | Sóng âm truyền với vận tốc hữu hạn (343 m/s), phản xạ liên tục vào tường, trần, bàn ghế tạo ra hàng nghìn xung phản xạ trễ kéo dài từ 200–500 ms. |
| **Phép toán miền tần số** | Phép nhân ma trận hằng số đơn giản. | Phép tích chập thời gian $\to$ Phép nhân ma trận phức phụ thuộc theo từng bin tần số $\mathbf{X}(f) = H(f) \mathbf{S}(f)$. |
| **Hệ quả đối với ICA** | FastICA miền thời gian 2x2 hoạt động hoàn hảo và tách triệt để. | FastICA cơ bản **thất bại hoàn toàn** vì nguồn tại thời điểm hiện tại tương quan với tiếng vọng trễ của chính nó trong quá khứ; bắt buộc phải sử dụng Frequency-Domain BSS hoặc Independent Vector Analysis (IVA) kèm cơ chế giải quyết hoán vị đa bin (frequency permutation problem). |

#### Câu 12: Từ kết quả của nhóm, enhancement có luôn cải thiện speaker recognition không? Trình bày bằng số liệu.
* **Trả lời:**
  * **Khẳng định:** Khử nhiễu tiếng nói (Speech Enhancement) **KHÔNG PHẢI LUÔN LUÔN** cải thiện nhận dạng người nói, mà phụ thuộc hoàn toàn vào bản chất thuật toán khử nhiễu và mức độ biến dạng phổ mà nó gây ra.
  * **Chứng minh bằng số liệu thực nghiệm đo đạc từ bài Lab:**
    * Trên tín hiệu sạch (Clean): Độ chính xác nhận dạng đạt **100.0%**, điểm Cosine bình quân 0.9981.
    * Khi bị nhiễm nhiễu quạt 5 dB (Noisy): Độ chính xác giảm sụp đổ xuống **33.3%** (chỉ đúng 2/6 mẫu, tương đương đoán mò ngẫu nhiên giữa 3 người) do năng lượng nhiễu làm méo mó vector MFCC.
    * Khi làm sạch bằng **Spectral Subtraction ($\beta=1.5$):** Mặc dù SNR tăng từ 5 dB lên 12.11 dB ($\Delta\mathrm{SNR} = +7.11\text{ dB}$), độ chính xác nhận dạng vẫn **dậm chân tại chỗ ở mức 33.3%** (2/6). Nguyên nhân do việc trừ phổ phi tuyến và hiện tượng rỗ phổ đã phá hủy nghiêm trọng cấu trúc formant cục bộ và sinh ra musical noise, khiến vector đặc trưng MFCC bị biến dạng khác xa so với lúc enrollment.
    * Khi làm sạch bằng **Wiener Filter (MMSE):** Bộ lọc áp dụng hệ số suy giảm trơn tuyến tính, vừa triệt tiêu nhiễu vừa bảo toàn tương đối nguyên vẹn các đỉnh phổ formant, giúp độ chính xác nhận dạng phục hồi ngoạn mục lên **66.7%** (4/6).
  * *Kết luận thực tiễn:* Một thuật toán làm sạch âm thanh nghe có vẻ "êm tai" hơn đối với con người hoàn toàn có thể là "thảm họa" đối với bộ trích xuất đặc trưng của máy học nếu nó đưa vào các méo dạng nhân tạo (artifacts).

---
*(Báo cáo hoàn thành đầy đủ, kèm theo mã nguồn Lab05.py, notebook Lab05.ipynb, dữ liệu âm thanh kết quả tại results/audio/ và hình vẽ trực quan tại results/figures/)*

