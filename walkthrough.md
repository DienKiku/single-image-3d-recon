# Báo Cáo Triển Khai: Chuyển Sang Thuần Stable Fast 3D & Khắc Phục Lỗi Zoom 1%

Tài liệu này tổng kết toàn bộ các thay đổi kiến trúc hệ thống: loại bỏ hoàn toàn mô hình *Depth to 3D*, chuyển toàn bộ pipeline sang chuẩn **Stable Fast 3D (SF3D)** của Stability AI, hoàn thiện script CLI độc lập chuẩn production (`scripts/run_sf3d.py`), và đại tu toàn diện bộ điều khiển Zoom của Three.js với độ chính xác từng 1%.

---

## 1. Tổng Kết Các Hạng Mục Đã Hoàn Thành

| Hạng Mục | Trạng Thái Trước | Giải Pháp Mới Triển Khai |
| :--- | :--- | :--- |
| **Lựa chọn Engine 3D** | Tồn tại đồng thời Depth-to-3D và SF3D gây phân tán | **Xóa bỏ hoàn toàn Depth-to-3D**; toàn bộ hệ thống (`app.py`, `components.py`, `ai_processor.py`) chỉ sử dụng duy nhất engine **Stable Fast 3D (SF3D)**. |
| **Script CLI Production** | Chưa có script CLI độc lập cho SF3D | Xây dựng file [scripts/run_sf3d.py](file:///d:/2d-to-3d/scripts/run_sf3d.py) với `argparse` chuẩn, hỗ trợ đầy đủ các cờ lệnh từ Stability AI. |
| **Xác thực Hugging Face Token** | Chưa hỗ trợ đăng nhập token trực tiếp | Tích hợp hàm `authenticate_huggingface` hỗ trợ cờ `--hf-token` và biến môi trường `HF_TOKEN` để tải trọng số gated `stabilityai/stable-fast-3d`. |
| **Tùy chọn Remeshing** | Chỉ xuất lưới mặc định | Hỗ trợ 3 quy trình: `none` (nguyên bản), `triangle` (Botsch & Kobbelt isotropic), `quad` (Field-aligned quads), kèm tham số `target_vertex_count`. |
| **Tối ưu VRAM ($\le 6\text{ GB}$)** | Có nguy cơ tràn bộ nhớ GPU | Bật chế độ `fp16` mixed precision, `torch.inference_mode()`, dọn dẹp `torch.cuda.empty_cache()` và hỗ trợ cờ `--device cpu` / `SF3D_USE_CPU=1`. |
| **Lỗi Thu Phóng (Zoom) Three.js** | Lăn chuột bị giật vọt lên 1000x hoặc thu nhỏ biến mất về 0.001x | **Đại tu cơ chế Zoom từng 1%**: Chặn sự kiện cuộn tự do của OrbitControls, thêm thanh trượt Slider % ($25\%\text{--}500\%$), 2 nút `[-1%]` và `[+1%]`, hiển thị nhãn `%` trực quan và khóa biên toán học. |

---

## 2. Kiến Trúc Hệ Thống (System Architecture) & Quy Trình Khép Kín

![End-to-End System Architecture](docs/images/fig0_system_architecture.png)

### A. Triplane NeRF & Isosurface Extraction
SF3D kế thừa tốc độ của TripoSR nhưng nâng cấp với mạng Transformer ước lượng biểu diễn 3D liên tục (SDF/NeRF) chỉ trong $< 0.5\text{s}$ trên GPU. Lưới bề mặt sau đó được trích xuất bằng thuật toán Marching Cubes/Tetrahedra với mật độ đỉnh tối ưu.

### B. Mở Trải UV Nhanh (Fast UV Parameterization)
Khác với phương pháp chiếu ảnh thông thường (planar/camera projection) vốn làm giãn mép hoặc chồng lấn vân ở các mặt khuất, SF3D sử dụng thuật toán trải UV chuyên dụng (dựa trên xatlas hoặc phân tích biểu đồ conformal giữ biên):
- Tự động phát hiện các đường cắt seam tối ưu dọc các cạnh gập sắc nét.
- Mở phẳng các mảng 3D thành các đảo UV (UV islands) 2D không chồng lấn.
- Tối ưu hóa hệ số lấp đầy (packing efficiency) để tận dụng tối đa diện tích bản đồ texture $1024 \times 1024$ hoặc $2048 \times 2048$.

### C. Tách Rời Ánh Sáng (Delighting) & Nướng Texture PBR
- **Loại bỏ bóng đổ tĩnh (Delighting):** Ảnh 2D chụp thực tế luôn chứa bóng đổ và vùng phản chiếu của môi trường. SF3D dùng mạng nơ-ron tách rời lớp chiếu sáng để thu về màu phản xạ thuần túy (**Base Color / Diffuse Albedo**).
- **Dự đoán tham số vật liệu PBR:**
  - **Roughness & Metallic:** Xác định độ nhám và tính kim loại của từng vùng vật liệu (nhựa mờ của thân máy in, kim loại của ốc vít hay bề mặt kính bóng loáng của nắp scan).
  - **Normal Map:** Lưu trữ chi tiết vi mô bề mặt giúp dựng hình sắc nét trong game engines.
- **Baking:** Rasterize các thuộc tính PBR trực tiếp lên tọa độ UV đã mở phẳng bằng GPU, đóng gói thành file chuẩn `.glb` tương thích hoàn toàn với Blender, Three.js, Unity và Unreal Engine.

---

## 3. Khắc Phục Triệt Để Lỗi Zoom Three.js (Từng 1%)

### A. Nguyên Nhân Gốc Rễ Của Lỗi Zoom 1000x / 0.001x
1. **Sự kiện cuộn chuột không được kiểm soát (Unbounded Wheel Delta):** Trên hệ điều hành Windows, chuột chơi game hoặc trackpad độ nhạy cao gửi hàng chục sự kiện `wheel` trong một phần giây với `deltaY` lớn.
2. **Hàm Dolly nhân lũy thừa của Three.js:** `OrbitControls` tính khoảng cách camera theo hàm nhân $0.95^{\text{delta}}$. Khi các sự kiện dồn ứ, khoảng cách camera bị chia nhỏ liên tục khiến camera đâm xuyên qua tâm vật thể (phóng to tới $1000\times$) hoặc bị đẩy lùi xa hàng chục nghìn đơn vị (teo nhỏ về $0.001\times$).

### B. Giải Pháp Đã Triển Khai
1. **Vô hiệu hóa zoom mặc định:** Đặt `controls.enableZoom = false` để OrbitControls không can thiệp vào khoảng cách camera.
2. **Khóa tỷ lệ Zoom theo công thức phân số chuẩn:**
   $$\text{Camera Distance} = \frac{\text{Base Distance}}{\text{ZoomPercent} / 100.0}$$
3. **Chặn biên cứng an toàn:** Giới hạn $25\% \le \text{ZoomPercent} \le 500\%$, bước nhảy đúng $1\%$.
4. **Bổ sung UI HUD tương tác trực tiếp:**
   - **Thanh trượt Slider:** Kéo trực tiếp từ $25\%$ đến $500\%$.
   - **Nút bấm vi chỉnh:** `[-1%]` và `[+1%]`.
   - **Huy hiệu %:** Hiển thị thời gian thực mức phóng to hiện tại.
   - **Bắt sự kiện bánh xe chuột:** Mỗi nấc lăn chuột điều chỉnh đúng $\pm 1\%$, không bao giờ bị trượt văng.
   - **Nút Reset View:** Khôi phục ngay về $100\%$.

---

## 4. Kết Quả Kiểm Nghiệm Thực Tế

### A. Kiểm thử CLI Standalone (`scripts/run_sf3d.py`)
```powershell
python scripts/run_sf3d.py -i assets/samples/sample_aruco_objects.png -o output/cli_test --texture-resolution 1024 --remesh triangle --target-vertex-count 10000 --target-dimensions-mm 380 500 360
```
- **Kết quả xuất file:**
  - `sample_aruco_objects.stl` ($1,767.9\text{ KB}$): Khối đặc kín nước 100% (`is_watertight: True`).
  - `sample_aruco_objects.glb` ($920.4\text{ KB}$): Chuẩn GLB kèm vector pháp tuyến đỉnh.
  - `sample_aruco_objects.obj` ($3,282.6\text{ KB}$): Kèm file `.mtl` và texture map.
- **Kích thước thực tế:** `380.1 x 496.8 x 378.1 mm`.

### B. Kiểm thử toàn bộ Test Suite (`pytest tests/ -v`)
```text
============================= 26 passed in 21.53s =============================
```
Mọi unit test đều vượt qua 100%.
