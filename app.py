import os
import uuid
import cv2

from flask import Flask, request, jsonify, send_from_directory, abort, render_template
from werkzeug.utils import secure_filename

from pipeline_core import process_image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def build_file_links(request_id: str) -> dict:
    output_dir = os.path.join(app.config["OUTPUT_FOLDER"], request_id)

    candidate_files = [
        "mask_full.png",
        "overlay.png",
        "roi_mask_final.jpg",
        "roi_mask_skin.jpg",
        "roi.jpg",
        "final_overlay_full.png",
        "final_mask_full.png",
        "roi_final_mask_roi.png",
        "2_yolo_ppd_bbox_debug.jpg",
        "1_a4_bbox_debug.jpg",
        "1b_a4_quad_debug.jpg",
        "1c_a4_rectified.jpg",
        "roi_0.jpg",
        "roi_1_mask_skin.jpg",
        "roi_hsv_hist_with_selected_interval.png",
        "roi_hsv_2_mask_base.jpg",
        "roi_hsv_3_edges.jpg",
        "roi_hsv_4_candidates_debug.jpg",
        "roi_hsv_4b_region_from_edges.jpg",
        "roi_hsv_4c_region_inner.jpg",
        "roi_hsv_4d_region_largest_component.jpg",
        "roi_hsv_4e_mask_base_inner.jpg",
        "roi_hsv_4f_mask_largest_component.jpg",
        "roi_hsv_5_mask_final.jpg",
        "roi_rel_2_gray_smooth.jpg",
        "roi_rel_3_blackhat.jpg",
        "roi_rel_4_laplacian.jpg",
        "roi_rel_5_combined.jpg",
        "roi_rel_6_binary_inverted_refined.jpg",
        "roi_rel_7_mask_final.jpg",
        "0_original.jpg",
    ]

    links = {}
    for fname in candidate_files:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            links[fname] = f"/outputs/{request_id}/{fname}"

    return links


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})


@app.route("/process-image", methods=["POST"])
def process_image_endpoint():
    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Nenhum arquivo enviado no campo 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Nome de arquivo vazio."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "status": "error",
            "message": f"Extensão não permitida. Use: {sorted(ALLOWED_EXTENSIONS)}"
        }), 400

    original_name = secure_filename(file.filename)
    request_id = str(uuid.uuid4())
    ext = os.path.splitext(original_name)[1].lower()

    upload_filename = f"{request_id}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)

    try:
        file.save(upload_path)

        image = cv2.imread(upload_path)
        if image is None:
            return jsonify({
                "status": "error",
                "message": "Erro ao ler a imagem enviada."
            }), 400

        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], request_id)
        os.makedirs(output_dir, exist_ok=True)

        result = process_image(image, output_dir)

        if result is None:
            return jsonify({
                "status": "no_detection",
                "message": "Nenhuma ROI/lesão detectada.",
                "request_id": request_id,
                "files": build_file_links(request_id)
            }), 200

        response = {
            "status": "ok",
            "request_id": request_id,
            "area_px": result.get("area_px"),
            "area_mm2": result.get("area_mm2"),
            "radius_mm": result.get("radius_mm"),
            "files": build_file_links(request_id)
        }

        for extra_key in [
            "source",
            "accepted_hsv",
            "best_hsv_score",
            "best_rel_score",
            "arm_area_roi",
            "frac_final_roi"
        ]:
            if extra_key in result:
                response[extra_key] = result[extra_key]

        return jsonify(response), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "Erro interno ao processar a imagem.",
            "details": str(e)
        }), 500


@app.route("/outputs/<request_id>/<filename>", methods=["GET"])
def get_output_file(request_id, filename):
    directory = os.path.join(app.config["OUTPUT_FOLDER"], request_id)

    if not os.path.isdir(directory):
        abort(404, description="Diretório de saída não encontrado.")

    file_path = os.path.join(directory, filename)
    if not os.path.isfile(file_path):
        abort(404, description="Arquivo não encontrado.")

    return send_from_directory(directory, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)