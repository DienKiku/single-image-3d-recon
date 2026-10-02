# Single-Image to 3D — Flask Lite REST API & Web Viewer (`2dto3d-flask-lite`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask 3.0+](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r160%20%7C%20CDN-black.svg)](https://threejs.org/)
[![License: All Rights Reserved](https://img.shields.io/badge/License-Copyright%20Author-orange.svg)](#license)

> **Nhánh độc lập (`2dto3d-flask-lite`) của dự án [`single-image-3d-recon`](https://github.com/DienKiku/single-image-3d-recon)**: Cung cấp backend REST API siêu nhẹ trên nền **Flask** cùng giao diện Three.js Web Viewer tĩnh (Single-Page App). Nhánh này tối ưu cho việc nhúng vào các ứng dụng bên thứ ba (Web, Mobile, Desktop Electron, Microservices) mà không cần cài đặt Streamlit server.

---

## 🚀 Điểm Khác Biệt Giữa 2 Nhánh

| Tính năng | Nhánh `main` (Streamlit CAD Studio) | Nhánh `2dto3d-flask-lite` (Flask API) |
| :--- | :--- | :--- |
| **Giao diện chính** | Streamlit Interactive Studio hoàn chỉnh | Single-Page Web tĩnh (`static/viewer.html`) |
| **REST API** | Không có REST API riêng biệt | **Có sẵn REST API chuẩn (`/api/convert`)** |
| **Thư viện Web** | `streamlit>=1.40` | `flask>=3.0` |
| **Three.js** | Tích hợp sâu vào Streamlit Component | Nạp trực tiếp từ CDN (`unpkg.com/three@0.160.0`) |
| **Windows Marching Cubes** | Marching Cubes đa cơ chế | **Patch `skimage` siêu mượt, không cần C++ Build Tools** |
| **Mục đích sử dụng** | Nghiên cứu, thiết kế CAD, bóc tách mô hình & nướng PBR | **Triển khai dịch vụ API, nhúng ứng dụng, chạy nhẹ máy** |

---

## 🛠️ Cấu Trúc Thư Mục

```text
2dto3d-flask-lite/
├── app.py                # Flask server chính & định tuyến API
├── convert.py            # CLI dựng phù điêu 2.5D (Relief Mesh)
├── convert3d.py          # CLI dựng mô hình 3D 360° (TripoSR)
├── depth.py              # Bộ ước lượng độ sâu (Depth-Anything-V2 / MiDaS)
├── mc_fallback.py        # Patch Marching Cubes bằng scikit-image trên Windows
├── mesh.py               # Thuật toán tạo lưới 2.5D (Solid & Open)
├── postprocess.py        # Xoay trục Y-up, hàn lỗ, mượt Taubin, giảm mặt
├── requirements.txt      # Danh sách thư viện Python
├── tripo_engine.py       # Engine nạp TripoSR NeRF với cơ chế chunking 8192
├── static/
│   └── viewer.html       # Web client Three.js độc lập (Single-Page App)
├── uploads/              # Thư mục chứa ảnh tải lên tạm thời
└── outputs/              # Thư mục lưu trữ mô hình xuất ra (.glb, .obj, .stl)
```

---

## ⚡ Cài Đặt & Khởi Chạy

### 1. Kích hoạt môi trường Python
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

> [!NOTE]
> Trên hệ điều hành Windows, tệp [`mc_fallback.py`](mc_fallback.py) đã tự động thay thế `torchmcubes` bằng `scikit-image` (`skimage.measure.marching_cubes`). Do đó bạn **không cần cài Visual Studio C++ Build Tools** vẫn có thể chạy bình thường!

### 3. Chạy Flask Server
```bash
python app.py
```
Mở trình duyệt tại: **`http://127.0.0.1:5000`** để sử dụng giao diện web Three.js.

---

## 📡 Tài Liệu REST API

### `POST /api/convert`
Nhận ảnh 2D và tham số xử lý, trả về đường dẫn tệp 3D kết quả.

**Content-Type:** `multipart/form-data`

#### Tham số Form:
- `image` *(bắt buộc)*: Tệp ảnh RGB (`.png`, `.jpg`, `.jpeg`, `.webp`).
- `mode`: Chế độ dựng (`full3d` hoặc `relief`). Mặc định: `full3d`.
- `format`: Định dạng xuất (`glb`, `obj`, `stl`, `ply`). Mặc định: `glb`.
- `device`: Thiết bị tính toán (`auto`, `cuda`, `cpu`). Mặc định: `auto`.

**Tham số riêng cho chế độ `full3d`:**
- `mc_resolution`: Độ phân giải Marching Cubes (`128`, `256`, `384`). Mặc định: `256`.
- `foreground_ratio`: Tỷ lệ căn giữa chủ thể (`0.5` - `1.0`). Mặc định: `0.85`.
- `remove_bg`: Tự động tách nền (`true` hoặc `false`). Mặc định: `true`.
- `faces`: Giới hạn số mặt tam giác (`0` = giữ nguyên). Mặc định: `0`.
- `bake_uv`: Bake texture UV từ vertex colors (`true` hoặc `false`). Mặc định: `false`.

**Tham số riêng cho chế độ `relief` (Phù điêu 2.5D):**
- `model`: Model độ sâu (`MiDaS_small`, `DPT_Hybrid`, `DPT_Large`). Mặc định: `MiDaS_small`.
- `resolution`: Độ phân giải lưới (`128` - `1024`). Mặc định: `512`.
- `depth_scale`: Độ nổi khối (`0.05` - `1.0`). Mặc định: `0.35`.
- `smooth`: Mức làm mượt Gaussian (`0` - `15`). Mặc định: `3`.
- `solid`: Tạo khối đáy đặc cho in 3D (`true` hoặc `false`). Mặc định: `false`.

#### Phản hồi mẫu (JSON):
```json
{
  "url": "/outputs/0c74011110.glb",
  "filename": "0c74011110.glb",
  "size": "3.42 MB"
}
```

---

## 🤖 Cơ Chế Nạp Trọng Số Mô Hình (Weights)

Hệ thống được thiết kế để tự động hoạt động linh hoạt:
1. **Tự động tải (Mặc định):** Nếu thư mục `models/` chưa có sẵn, hệ thống sẽ tự động tải weights qua HuggingFace Hub:
   - TripoSR: `stabilityai/TripoSR`
   - Depth Anything: `depth-anything/Depth-Anything-V2-Small-hf`
   - U2-Net: Tự động tải qua thư viện `rembg`.
2. **Nạp Offline hoàn toàn (Tùy chọn):** Bạn có thể tải sẵn weights và đặt vào thư mục `models/`:
   - `models/TripoSR/model.ckpt` và `models/TripoSR/config.yaml`
   - `models/u2net/u2net.onnx`
   - `models/hf/` (HuggingFace cache).

---

## 💻 Sử Dụng Qua Dòng Lệnh (CLI)

### Chế độ Full 3D 360°:
```bash
python convert3d.py input.png -o outputs/model.glb -r 256 -fg 0.85
```

### Chế độ Phù điêu 2.5D Relief:
```bash
python convert.py input.png -o outputs/relief.glb --solid -d 0.35
```

---

## 📜 Bản Quyền & Giấy Phép
Dự án thuộc bản quyền tác giả **DienKiku**. Mọi quyền được bảo lưu. Xem nhánh chính tại [`DienKiku/single-image-3d-recon`](https://github.com/DienKiku/single-image-3d-recon).
