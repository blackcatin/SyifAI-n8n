import os
import json
import shutil
from datetime import datetime

# ==========================================
# PATH CONFIG
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR    = os.path.join(CURRENT_DIR, "trained_models")
ACTIVE_DIR   = os.path.join(MODEL_DIR, "active")
REJECTED_DIR = os.path.join(MODEL_DIR, "rejected")

CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")
REGISTRY_JSON_FILE = os.path.join(CURRENT_DIR, "model_history.json")

DEFAULT_MODEL = "Fix_best_model_v1.pth"

# ==========================================
# THRESHOLD CONFIG
# ==========================================
MIN_IMPROVEMENT_MARGIN = 1.00  # poin persentase

for d in [MODEL_DIR, ACTIVE_DIR, REJECTED_DIR]:
    os.makedirs(d, exist_ok=True)


# ==========================================
# UTIL
# ==========================================
def load_history():
    if not os.path.exists(REGISTRY_JSON_FILE):
        return []
    try:
        with open(REGISTRY_JSON_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []


def save_history(history):
    with open(REGISTRY_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)


def update_pointer(model_filename):
    with open(CURRENT_MODEL_FILE, "w", encoding="utf-8") as f:
        f.write(model_filename)
    print(f"📌 current_model.txt diupdate → {model_filename}")


def get_current_active_filename() -> str:
    if not os.path.exists(CURRENT_MODEL_FILE):
        return DEFAULT_MODEL
    try:
        with open(CURRENT_MODEL_FILE, "r") as f:
            name = f.read().strip()
        if not name.endswith(".pth"):
            name += ".pth"
        return name
    except:
        return DEFAULT_MODEL


# ==========================================
# CORE REGISTRY
# ==========================================
def update_model_registry(
    new_model_path,
    version,
    accuracy,
    final_loss,
    total_data_retrain,
    execution_time,
    upload_success=False
):
    history = load_history()
    filename = os.path.basename(new_model_path)

    # 1. Pastikan Champion yang ditunjuk sistem benar-benar tersedia.
    active_filename = get_current_active_filename()
    active_path = os.path.join(ACTIVE_DIR, active_filename)

    if not os.path.isfile(active_path):
        raise FileNotFoundError(
            f"Champion aktif tidak ditemukan: {active_path}"
        )

    # Jangan sampai kandidat menimpa file Champion.
    if filename == active_filename:
        raise ValueError(
            "Nama file Challenger tidak boleh sama dengan Champion aktif."
        )

    if not os.path.isfile(new_model_path):
        raise FileNotFoundError(
            f"File Challenger tidak ditemukan: {new_model_path}"
        )

    # 2. Ambil catatan ACTIVE terbaru sebagai baseline Champion.

    champion_record = next(
        (
            item for item in reversed(history)
            if item.get("status") == "ACTIVE"
            and os.path.basename(item.get("model_path", ""))
            == active_filename
        ),
        None
    )

    if champion_record is None:
        raise ValueError(
            f"Baseline untuk Champion {active_filename} tidak ditemukan "
            "di riwayat. Registrasi dibatalkan sampai baseline terverifikasi."
        )

    try:
        baseline_accuracy = float(
            str(champion_record["accuracy"]).replace("%", "").strip()
        )
        challenger_accuracy = float(
            str(accuracy).replace("%", "").strip()
        )
    except (KeyError, TypeError, ValueError):
        raise ValueError(
            "Akurasi Champion atau Challenger tidak valid."
        )

    if not (
        0 <= baseline_accuracy <= 100
        and 0 <= challenger_accuracy <= 100
    ):
        raise ValueError("Nilai akurasi harus berada pada rentang 0–100.")

    improvement = challenger_accuracy - baseline_accuracy
    is_best = improvement >= MIN_IMPROVEMENT_MARGIN

    if is_best:
        rejection_reason = None
        status = "ACTIVE"
    else:
        rejection_reason = (
            f"Challenger ditolak: akurasi {challenger_accuracy:.2f}% "
            f"dibandingkan baseline Champion {baseline_accuracy:.2f}%, "
            f"peningkatan hanya {improvement:.2f} poin persentase. "
            f"Minimum improvement margin adalah "
            f"{MIN_IMPROVEMENT_MARGIN:.2f} poin persentase."
        )
        status = "REJECTED"

    # 3. Pindahkan Challenger sesuai hasil validation gate.
    if is_best:
        # Simpan kandidat dengan nama unik ke folder active/.
        destination_path = os.path.join(ACTIVE_DIR, filename)

        if os.path.exists(destination_path):
            raise FileExistsError(
                f"File tujuan sudah ada: {destination_path}"
            )

        shutil.copy2(new_model_path, destination_path)
        os.remove(new_model_path)

        # Tandai Champion sebelumnya sebagai riwayat model tergantikan.
        for item in history:
            if item.get("status") == "ACTIVE":
                item["status"] = "SUPERSEDED"

        # Ganti pointer hanya setelah kandidat tersimpan.
        update_pointer(filename)

        print(f"Model Challenger diterima: {filename}")
    else:
        rejected_path = os.path.join(REJECTED_DIR, filename)

        if os.path.exists(rejected_path):
            raise FileExistsError(
                f"File tujuan rejected sudah ada: {rejected_path}"
            )

        shutil.move(new_model_path, rejected_path)
        print(f"Model Challenger ditolak: {filename}")
        print(f"Alasan: {rejection_reason}")

    # 4. Catat keputusan dan baseline untuk audit.
    entry = {
        "version": version,
        "model_path": filename,
        "accuracy": f"{challenger_accuracy:.2f}%",
        "baseline_accuracy": f"{baseline_accuracy:.2f}%",
        "baseline_model_path": champion_record.get("model_path"),
        "minimum_improvement_margin": (
            f"{MIN_IMPROVEMENT_MARGIN:.2f} poin persentase"
        ),
        "improvement": f"{improvement:.2f} poin persentase",
        "final_loss": final_loss,
        "total_data_retrain": total_data_retrain,
        "execution_time": f"{execution_time}s",
        "status": status,
        "is_best_model": is_best,
        "rejection_reason": rejection_reason,
        "previous_model_filename": active_filename,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    history.append(entry)
    save_history(history)

    return entry

# ==========================================
# GET CURRENT MODEL PATH
# ==========================================
def get_current_model_path():
    filename = get_current_active_filename()

    active_path = os.path.join(ACTIVE_DIR, filename)
    if os.path.exists(active_path):
        return active_path

    if os.path.exists(ACTIVE_DIR):
        pth_files = sorted(
            [f for f in os.listdir(ACTIVE_DIR) if f.endswith(".pth")],
            reverse=True
        )
        if pth_files:
            update_pointer(pth_files[0])
            return os.path.join(ACTIVE_DIR, pth_files[0])

    direct_path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(direct_path):
        return direct_path

    return active_path


def get_current_model_version_name():
    try:
        return get_current_active_filename().replace(".pth", "")
    except:
        return "model_v1"


# ==========================================
# COMPAT LAYER
# ==========================================
def update_model_path(
    new_model_path, version, accuracy, final_loss,
    total_data_retrain, execution_time, upload_success=False
):
    return update_model_registry(
        new_model_path=new_model_path,
        version=version,
        accuracy=accuracy,
        final_loss=final_loss,
        total_data_retrain=total_data_retrain,
        execution_time=execution_time,
        upload_success=upload_success
    )