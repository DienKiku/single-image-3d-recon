import os
import uuid
import traceback
from flask import Flask, request, jsonify, send_from_directory

from convert import convert as convert_relief
from convert3d import convert3d

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
OUTPUT_DIR = os.path.join(BASE, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory("static", "viewer.html")


@app.route("/api/convert", methods=["POST"])
def api_convert():
    if "image" not in request.files:
        return jsonify({"error": "Thiếu file ảnh"}), 400

    f = request.files["image"]
    uid = uuid.uuid4().hex[:10]
    ext = os.path.splitext(f.filename)[1].lower() or ".png"
    src = os.path.join(UPLOAD_DIR, uid + ext)
    f.save(src)

    fmt = request.form.get("format", "glb")
    mode = request.form.get("mode", "full3d")      # full3d | relief
    out = os.path.join(OUTPUT_DIR, f"{uid}.{fmt}")

    try:
        if mode == "full3d":
            convert3d(
                src, out,
                mc_resolution=int(request.form.get("mc_resolution", 256)),
                foreground_ratio=float(request.form.get("foreground_ratio", 0.85)),
                remove_bg=str(request.form.get("remove_bg", "true")).lower() == "true",
                target_faces=int(request.form.get("faces", 0)),
                smooth_iter=int(request.form.get("smooth", 0)),
                bake_uv=str(request.form.get("bake_uv", "false")).lower() == "true",
                device=request.form.get("device", "auto"),
            )
        else:
            convert_relief(
                src, out,
                resolution=int(request.form.get("resolution", 512)),
                depth_scale=float(request.form.get("depth_scale", 0.35)),
                smooth=int(request.form.get("smooth", 3)),
                solid=str(request.form.get("solid", "false")).lower() == "true",
                model_name=request.form.get("model", "MiDaS_small"),
                device=request.form.get("device", "auto"),
            )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    size_mb = os.path.getsize(out) / 1048576
    return jsonify({
        "url": f"/outputs/{os.path.basename(out)}",
        "filename": os.path.basename(out),
        "size": f"{size_mb:.2f} MB",
    })


@app.route("/outputs/<path:name>")
def outputs(name):
    return send_from_directory(OUTPUT_DIR, name)


if __name__ == "__main__":
    print("Mở trình duyệt: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, threaded=True)
