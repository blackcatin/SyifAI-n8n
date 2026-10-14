import io
import cv2
import traceback
import numpy as np
import uuid
import os
import time
import shutil
import mlflow
import asyncio
import copy
import mlflow.pytorch
import requests

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import func, text
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from huggingface_hub import HfApi
from PIL import Image
from huggingface_hub import HfApi, create_repo
from retrain import FOLDER_TO_LABEL
from pydantic import BaseModel
from typing import List, Optional

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

from image_quality import validate_image, analyze_image_quality
from preprocessing import preprocess_image
from model_registry import (
    get_current_model_path,
    get_current_model_version_name,
    get_current_active_filename,
    load_history,
    update_model_registry
)
from retrain import RETRAIN_DIR, run_pytorch_training
from database import engine, SessionLocal
from models import Base, Prediction

# ===============================
# INIT
# ===============================
Base.metadata.create_all(bind=engine)

CURRENT_DIR       = os.path.dirname(os.path.abspath(__file__))
RAW_FOLDER        = os.path.join(CURRENT_DIR, "uploads", "raw")
PROCESSED_FOLDER  = os.path.join(CURRENT_DIR, "uploads", "processed")
DATASET_TRAIN_DIR = os.path.join(CURRENT_DIR, "dataset", "train")

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "blackcatin/skin-disease-dataset")
LABEL_TO_FOLDER = {v: k for k, v in FOLDER_TO_LABEL.items()}

os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("SyifAI_Skin_Disease_Retraining")

# ===============================
# BUILD MODEL
# ===============================
def build_model(num_classes: int):
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in m.parameters():
        p.requires_grad = False
    for p in m.layer3.parameters():
        p.requires_grad = True
    for p in m.layer4.parameters():
        p.requires_grad = True
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256),
        nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.35),
        nn.Linear(128, num_classes)
    )
    return m

# ===============================
# LOAD MODEL
# ===============================
model         = None
MODEL_VERSION = get_current_model_version_name()
model_path    = get_current_model_path()

print(f"📦 [Registry] Versi aktif: {MODEL_VERSION}")

try:
    if os.path.exists(model_path):
        print(f"📂 Loading: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model      = build_model(num_classes=23)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        print("✅ Model berhasil dimuat")
    else:
        print("⚠️ File model tidak ditemukan, pakai arsitektur kosong")
        model = build_model(num_classes=23)
except Exception as e:
    print("❌ Gagal memuat model:", e)
    model = build_model(num_classes=23)

model.eval()

_model_lock      = asyncio.Lock()
_training_active = False
_predict_queue: asyncio.Queue = asyncio.Queue(maxsize=50)

# ===============================
# PYDANTIC SCHEMA
# ===============================
class TrainRecord(BaseModel):
    image_path: str
    predicted_class: str
    source: Optional[str] = "new"

class TrainRequest(BaseModel):
    records: Optional[List[TrainRecord]] = None

# ===============================
# FASTAPI
# ===============================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_labels = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis", "Bullous Disease",
    "Cellulitis, Impetigo & other Bacterial Infections", "Eczema",
    "Exanthems & Drug Eruptions", "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes, HPV & other STDs", "Light Diseases & Disorders of Pigmentation",
    "Lupus & other Connective Tissue diseases", "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus & other Nail Disease", "Poison Ivy & other Contact Dermatitis",
    "Psoriasis, Lichen Planus & related diseases",
    "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses & other Benign Tumors", "Systemic Disease",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria / Hives", "Vascular Tumors", "Vasculitis",
    "Warts, Molluscum & other Viral Infections"
]

_predict_worker_task = None

@app.on_event("startup")
async def startup_event():
    global _predict_worker_task
    _predict_worker_task = asyncio.create_task(predict_worker())
    print("✅ Predict queue worker started")

@app.on_event("shutdown")
async def shutdown_event():
    if _predict_worker_task:
        _predict_worker_task.cancel()
        print("🛑 Predict queue worker stopped")

# ===============================
# ROOT
# ===============================
@app.get("/")
def root():
    return {"message": "Skin Detector API aktif 🚀", "active_model": MODEL_VERSION}

# ===============================
# PREDICT WORKER
# ===============================
async def predict_worker():
    while True:
        try:
            image_tensor, future = await _predict_queue.get()
            try:
                async with _model_lock:
                    with torch.no_grad():
                        outputs               = model(image_tensor)
                        probs                 = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probs, 1)
                        class_idx             = predicted.item()
                        confidence_val        = confidence.item() * 100
                if not future.done():
                    future.set_result((class_idx, confidence_val))
            except Exception as e:
                if not future.done():
                    future.set_exception(e)
            finally:
                _predict_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Predict worker error: {e}")
            await asyncio.sleep(0.1)

# ===============================
# PREDICT
# ===============================
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global _training_active

    if _training_active:
        return {
            "status":  "unavailable",
            "message": "Model sedang diperbarui ke versi terbaru. Coba lagi dalam beberapa detik."
        }

    if _predict_queue.full():
        return {
            "status":  "overloaded",
            "message": f"Server sedang sibuk ({_predict_queue.qsize()} request dalam antrian). Coba lagi sebentar."
        }

    try:
        content      = await file.read()
        filename     = f"{uuid.uuid4()}.jpg"
        raw_filepath = os.path.join(RAW_FOLDER, filename)

        with open(raw_filepath, "wb") as f:
            f.write(content)

        image_cv         = cv2.imread(raw_filepath)
        quality_data     = analyze_image_quality(image_cv)
        blur_score       = float(quality_data["blur_score"])
        brightness_score = float(quality_data["brightness_score"])
        quality_score    = float(quality_data["quality_score"])
        is_valid         = bool(validate_image(image_cv, blur_score, brightness_score))

        processed_filepath = preprocess_image(raw_filepath, PROCESSED_FOLDER)
        image        = Image.open(processed_filepath).convert("RGB")
        image_tensor = transform(image).unsqueeze(0)

        loop   = asyncio.get_event_loop()
        future = loop.create_future()

        await _predict_queue.put((image_tensor, future))

        try:
            class_idx, confidence_percentage = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            return {
                "status":  "timeout",
                "message": "Request timeout setelah 30 detik. Server mungkin sedang sibuk."
            }

        result = class_labels[class_idx]

        db = SessionLocal()
        db.add(Prediction(
            image_path           = raw_filepath,
            processed_image_path = processed_filepath,
            predicted_class      = result,
            confidence           = round(confidence_percentage, 2),
            blur_score           = blur_score,
            brightness_score     = brightness_score,
            quality_score        = quality_score,
            is_valid             = is_valid,
            model_version        = MODEL_VERSION,
            used_for_retraining  = False,
            confidence_score     = confidence_percentage / 100.0,
            is_hard_example      = (confidence_percentage / 100.0) < 0.6
        ))
        db.commit()
        db.close()

        return {
            "filename":        filename,
            "predicted_class": result,
            "confidence":      round(confidence_percentage, 2),
            "raw_image":       raw_filepath,
            "processed_image": processed_filepath,
            "queue_size":      _predict_queue.qsize(),
            "quality": {
                "blur_score":            round(blur_score, 2),
                "brightness_score":      round(brightness_score, 2),
                "quality_score":         round(quality_score, 2),
                "is_valid_for_training": is_valid
            }
        }

    except Exception as e:
        print("❌ Predict error:", traceback.format_exc())
        return {"status": "failed", "message": "Predict gagal", "error": str(e)}

# ===============================
# QUEUE STATUS
# ===============================
@app.get("/queue-status")
def queue_status():
    return {
        "queue_size":      _predict_queue.qsize(),
        "queue_maxsize":   _predict_queue.maxsize,
        "is_full":         _predict_queue.full(),
        "training_active": _training_active,
    }

# ===============================
# TRAIN
# ===============================
@app.post("/train")
async def trigger_retraining(req: TrainRequest = None):
    global model, MODEL_VERSION, _training_active

    if _training_active:
        return {"status": "failed", "message": "Training sudah berjalan.", "is_best_model": False}

    db = SessionLocal()
    # [BUG FIX 1] — new_samples didefinisikan di luar if/else
    # supaya bisa dipakai di return statement (total_new_data)
    new_samples = []

    try:
        if req and req.records and len(req.records) >= 5:
            # Path A: data dikirim dari n8n via body (normal flow)
            subset_records = [
                {
                    "image_path":      r.image_path,
                    "predicted_class": r.predicted_class,
                    "source":          r.source
                }
                for r in req.records
            ]
            new_samples = req.records  # untuk total_new_data di return
            print(f"📦 Data dari n8n request body: {len(subset_records)} records")

        else:
            # Path B: fallback manual trigger (tanpa body / body kosong)
            new_samples = db.query(Prediction).filter(
                Prediction.is_valid == True,
                Prediction.used_for_retraining == False
            ).all()

            if len(new_samples) < 5:
                return {
                    "status":        "failed",
                    "message":       f"Data baru hanya {len(new_samples)}, minimum 5.",
                    "is_best_model": False
                }

            # [BUG FIX 2] — hard_samples dan stratified_rows
            # hanya ada di path fallback, bukan di path n8n
            hard_samples = db.query(Prediction).filter(
                Prediction.is_valid == True,
                Prediction.used_for_retraining == True,
                Prediction.is_hard_example == True
            ).order_by(Prediction.confidence_score.asc()).limit(150).all()

            stratified_rows = db.execute(text("""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY predicted_class ORDER BY RANDOM()
                    ) AS rn
                    FROM predictions
                    WHERE used_for_retraining = TRUE AND is_valid = TRUE
                ) t WHERE rn <= 10
            """)).fetchall()

            subset_records = []
            for s in new_samples:
                ip = s.processed_image_path or s.image_path
                if ip:
                    subset_records.append({
                        "image_path":      ip,
                        "predicted_class": s.predicted_class,
                        "source":          "new"
                    })
            for s in hard_samples:
                ip = s.processed_image_path or s.image_path
                if ip:
                    subset_records.append({
                        "image_path":      ip,
                        "predicted_class": s.predicted_class,
                        "source":          "hard"
                    })
            for row in stratified_rows:
                m  = row._mapping
                ip = m.get("processed_image_path") or m.get("image_path")
                pc = m.get("predicted_class")
                if ip and pc:
                    subset_records.append({
                        "image_path":      ip,
                        "predicted_class": pc,
                        "source":          "stratified"
                    })

            print(f"📚 Total subset fallback: {len(subset_records)} data")

        # Setup path model baru
        TARGET_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
        os.makedirs(TARGET_MODEL_DIR, exist_ok=True)

        current_version_name = get_current_model_version_name()
        try:
            next_version = int(
                current_version_name.replace("Fix_best_model_v", "").replace("model_v", "")
            ) + 1
        except:
            next_version = 2

        NEW_MODEL_VERSION = f"model_v{next_version}"
        new_model_path    = os.path.join(TARGET_MODEL_DIR, f"Fix_best_{NEW_MODEL_VERSION}.pth")

        print("🔧 Membuat deep copy model untuk training...")
        training_model = copy.deepcopy(model)
        training_model.train()
        model.eval()

        start_time      = time.time()
        training_result = run_pytorch_training(training_model, new_model_path, subset_records)
        del training_model
        exec_time       = round(time.time() - start_time, 2)

        if not os.path.exists(new_model_path) or os.path.getsize(new_model_path) < 1000:
            raise Exception("Training gagal menghasilkan model valid.")

        raw_accuracy = training_result.get("accuracy", 0.0)
        final_loss   = training_result.get("final_loss", 0.0)

        registry_result = update_model_registry(
            new_model_path     = new_model_path,
            version            = NEW_MODEL_VERSION,
            accuracy           = f"{float(raw_accuracy) * 100:.1f}%",
            final_loss         = final_loss,
            total_data_retrain = len(subset_records),
            execution_time     = exec_time,
            upload_success     = False
        )

        is_best = registry_result["is_best_model"]
        print("✅ Training selesai.")

        if is_best:
            print("🔄 Model BEST — reload model global...")
            _training_active = True
            try:
                async with _model_lock:
                    new_active_path = get_current_model_path()
                    checkpoint      = torch.load(new_active_path, map_location="cpu", weights_only=False)
                    model           = build_model(num_classes=23)
                    if isinstance(checkpoint, dict):
                        model.load_state_dict(checkpoint)
                    else:
                        model = checkpoint
                    model.eval()
                    MODEL_VERSION = get_current_model_version_name()
                    print(f"✅ Model global diperbarui → {MODEL_VERSION}")
            finally:
                _training_active = False
        else:
            print("❌ Model DITOLAK — model global tidak berubah.")

        # [BUG FIX 3] — total_new_data pakai len() yang aman untuk kedua path
        return {
            "status":                  "success",
            "is_best_model":           is_best,
            "model_version":           NEW_MODEL_VERSION,
            "model_path":              os.path.basename(new_model_path),
            "previous_model_filename": registry_result.get("previous_model_filename"),
            "val_accuracy":            f"{float(training_result.get('accuracy', 0)) * 100:.1f}%",
            "train_accuracy":          f"{float(training_result.get('train_accuracy', 0)) * 100:.1f}%",
            "accuracy":                f"{float(training_result.get('accuracy', 0)) * 100:.1f}%",
            "baseline_accuracy":       registry_result.get("baseline_accuracy", "N/A"),
            "threshold":               registry_result.get("threshold", "N/A"),
            "threshold_locked":        registry_result.get("threshold_locked", False),
            "rejection_reason":        registry_result.get("rejection_reason"),
            "final_loss":              final_loss,
            "total_data_retrain":      len(subset_records),
            "total_new_data":          len(new_samples),
            "execution_time":          exec_time,
            "retrained_at":            time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except Exception as e:
        _training_active = False
        print(f"❌ Error fatal /train: {e}")
        print(traceback.format_exc())
        return {"status": "failed", "is_best_model": False, "error": str(e)}
    finally:
        db.close()

# ===============================
# UPLOAD MODEL KE HUGGING FACE
# ===============================
@app.post("/upload-model")
async def upload_model(request: Request):
    temp_path = None
    try:
        content = await request.body()
        if len(content) < 1000:
            raise HTTPException(400, f"File terlalu kecil ({len(content)} byte)")

        token   = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_REPO_ID", "blackcatin/resnet18-syifai")
        if not token:
            raise HTTPException(500, "HF_TOKEN tidak ditemukan di .env")

        filename  = "Fix_best_model_latest.pth"
        cd_header = request.headers.get("content-disposition")
        if cd_header and "filename=" in cd_header:
            raw_name = cd_header.split("filename=")[1].strip('"').strip("'").strip()
            if raw_name and raw_name not in ("None", "null", ""):
                filename = raw_name

        q_filename = request.query_params.get("filename", "").strip()
        if q_filename and q_filename not in ("None", "null", ""):
            filename = q_filename

        temp_path = os.path.join(CURRENT_DIR, f"temp_{filename}")
        with open(temp_path, "wb") as f:
            f.write(content)

        HfApi(token=token).upload_file(
            path_or_fileobj=temp_path, path_in_repo=filename,
            repo_id=repo_id, repo_type="model",
            commit_message=f"Auto-archive model {filename} via n8n"
        )
        print(f"✅ Upload HuggingFace sukses: {filename}")
        return {
            "status":        "success",
            "uploaded_file": filename,
            "url":           f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        }

    except Exception as e:
        print("❌ ERROR UPLOAD:", repr(e))
        if "RepositoryNotFoundError" in str(e):
            raise HTTPException(404, "Repo tidak ditemukan. Cek HF_REPO_ID di .env!")
        raise HTTPException(500, str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ===============================
# ARCHIVE RETRAINING DATA
# ===============================
@app.post("/archive-retraining-data")
async def archive_retraining_data():
    db = SessionLocal()
    try:
        samples = db.query(Prediction).filter(
            Prediction.is_valid            == True,
            Prediction.used_for_retraining == False
        ).all()

        archived = []
        for sample in samples:
            if not os.path.exists(sample.image_path):
               continue

            old_raw_path       = sample.image_path
            old_processed_path = sample.processed_image_path

            dest = old_processed_path  # fallback
            if old_processed_path and os.path.exists(old_processed_path):
                class_dir = os.path.join(DATASET_TRAIN_DIR, sample.predicted_class)
                os.makedirs(class_dir, exist_ok=True)
                dest = os.path.join(class_dir, os.path.basename(old_processed_path))
                if not os.path.exists(dest):
                    shutil.copy2(old_processed_path, dest)

            archived.append({
                "id":                 sample.id,
                "predicted_class":    sample.predicted_class,
                "image_path":         dest,           # path permanen untuk /train
                "raw_image":          old_raw_path,
                "processed_image":    old_processed_path,
                "raw_filename":       os.path.basename(old_raw_path),
                "processed_filename": os.path.basename(old_processed_path) if old_processed_path else None,
            })

            sample.image_path           = dest
            sample.processed_image_path = dest
            sample.used_for_retraining  = True

        db.commit()
        return {"status": "success", "total_data": len(archived), "data": archived}

    except Exception as e:
        print("❌ Archive error:", traceback.format_exc())
        return {"status": "failed", "error": str(e)}
    finally:
        db.close()

# ===============================
# SYNC DATASET KE HF
# ===============================
@app.post("/sync-dataset-to-hf")
async def sync_dataset_to_hf(request: Request):
    token = os.getenv("HF_TOKEN")
    if not token:
        raise HTTPException(500, "HF_TOKEN tidak ditemukan di .env")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Body request harus JSON.")

    archived = body.get("data", [])
    if not archived:
        return {"status": "skipped", "message": "Tidak ada data baru untuk disinkronkan."}

    create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        private=True,
        exist_ok=True,
    )

    api = HfApi(token=token)
    uploaded, failed = [], []

    for item in archived:
        processed_path  = item.get("processed_image")
        predicted_class = item.get("predicted_class")

        if not processed_path or not os.path.exists(processed_path) or not predicted_class:
            failed.append({"item": item, "error": "Path tidak valid / file tidak ditemukan"})
            continue

        folder_name  = LABEL_TO_FOLDER.get(predicted_class, predicted_class.replace("/", "-"))
        filename     = os.path.basename(processed_path)
        path_in_repo = f"{folder_name}/{filename}"

        try:
            api.upload_file(
                path_or_fileobj=processed_path,
                path_in_repo=path_in_repo,
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                commit_message=f"Sync data retraining baru: {filename}",
            )
            uploaded.append(path_in_repo)
        except Exception as e:
            failed.append({"item": item, "error": str(e)})

    print(f"✅ Sync ke HF Dataset selesai: {len(uploaded)} berhasil, {len(failed)} gagal")

    return {
        "status":         "success",
        "total_uploaded": len(uploaded),
        "total_failed":   len(failed),
        "uploaded":       uploaded,
        "failed":         failed,
    }

# ===============================
# READ IMAGE
# ===============================
@app.get("/read-raw-image/{filename}")
def read_raw_image(filename: str):
    path = os.path.join(RAW_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File tidak ditemukan: {filename}")
    return FileResponse(path, media_type="image/jpeg", filename=filename)

@app.get("/read-processed-image/{filename}")
def read_processed_image(filename: str):
    path = os.path.join(PROCESSED_FOLDER, filename)
    if not os.path.exists(path):
        raise HTTPException(404, f"File tidak ditemukan: {filename}")
    return FileResponse(path, media_type="image/jpeg", filename=filename)

# ===============================
# DELETE LOCAL FILES
# ===============================
@app.delete("/delete-raw-local")
def delete_raw_local(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return {"status": "not_found", "path": image_path}
    os.remove(image_path)
    return {"status": "deleted", "path": image_path}

@app.delete("/delete-processed-local")
def delete_processed_local(image_path: str):
    if not image_path or not os.path.exists(image_path):
        return {"status": "not_found", "path": image_path}
    os.remove(image_path)
    return {"status": "deleted", "path": image_path}

# ===============================
# DELETE OLD MODEL
# ===============================
@app.delete("/delete-old-model")
def delete_old_model(filename: str):
    if not filename or filename in ("None", "null", ""):
        return {"status": "skipped", "message": "Tidak ada filename"}

    current_active = get_current_active_filename()
    if filename == current_active:
        return {"status": "skipped", "message": f"{filename} adalah model aktif, tidak dihapus"}

    for d in [os.path.join(CURRENT_DIR, "trained_models", "active"),
              os.path.join(CURRENT_DIR, "trained_models")]:
        target = os.path.join(d, filename)
        if os.path.exists(target):
            os.remove(target)
            print(f"🗑️ Model lama dihapus: {target}")
            return {"status": "deleted", "filename": filename}

    return {"status": "not_found", "message": f"{filename} tidak ditemukan"}

# ===============================
# MARK USED
# ===============================
@app.post("/mark-used")
async def mark_used():
    db = SessionLocal()
    try:
        rows = db.query(Prediction).filter(
            Prediction.is_valid == True,
            Prediction.used_for_retraining == False
        ).all()

        archived = []
        for row in rows:
            old_raw       = row.image_path
            old_processed = row.processed_image_path

            if row.processed_image_path and os.path.exists(row.processed_image_path):
                class_dir = os.path.join(DATASET_TRAIN_DIR, row.predicted_class)
                os.makedirs(class_dir, exist_ok=True)
                dest = os.path.join(class_dir, os.path.basename(row.processed_image_path))
                if not os.path.exists(dest):
                    shutil.copy2(row.processed_image_path, dest)
                row.processed_image_path = dest
                row.image_path           = dest

            row.used_for_retraining = True
            archived.append({
                "raw_image":       old_raw,
                "processed_image": old_processed,
            })

        db.commit()
        return {"status": "success", "marked": len(archived), "data": archived}
    finally:
        db.close()

# ===============================
# REPAIR PATHS
# ===============================
@app.post("/repair-paths")
async def repair_paths():
    db = SessionLocal()
    fixed  = 0
    broken = 0
    try:
        rows = db.query(Prediction).filter(
            Prediction.used_for_retraining == True
        ).all()
        for row in rows:
            current_path = row.processed_image_path or row.image_path
            if current_path and not os.path.exists(current_path):
                filename  = os.path.basename(current_path)
                class_dir = os.path.join(DATASET_TRAIN_DIR, row.predicted_class)
                new_path  = os.path.join(class_dir, filename)
                if os.path.exists(new_path):
                    row.processed_image_path = new_path
                    row.image_path           = new_path
                    fixed += 1
                else:
                    broken += 1
        db.commit()
        return {"fixed": fixed, "still_broken": broken}
    finally:
        db.close()

# ===============================
# ACTIVE MODEL FILENAME
# Dipanggil n8n sebelum training untuk tahu nama model aktif
# ===============================
@app.get("/active-model-filename")
def active_model_filename():
    return {
        "filename": get_current_active_filename(),
        "version":  get_current_model_version_name()
    }

# ===============================
# REGISTER TRAINED MODEL
# Dipanggil n8n setelah HF Space selesai training
# HF Space sudah upload model ke HF Model repo
# Endpoint ini: download model dari HF → update registry → reload model lokal
# ===============================
class RegisterModelRequest(BaseModel):
    version:            str
    model_filename:     str
    accuracy:           str     # ← ubah dari float ke str, terima "91.49%"
    final_loss:         float
    total_data_retrain: int
    execution_time:     float

@app.post("/register-trained-model")
async def register_trained_model(req: RegisterModelRequest):
    global model, MODEL_VERSION, _training_active

    print(f"\n{'='*60}")
    print(f"📥 Register model dari HF Space")
    print(f"   Version    : {req.version}")
    print(f"   Filename   : {req.model_filename}")
    print(f"   Val Acc    : {req.accuracy}")
    print(f"   Final Loss : {req.final_loss:.4f}")
    print(f"{'='*60}\n")

    try:
        token   = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_REPO_ID", "blackcatin/resnet18-syifai")

        if not token:
            raise HTTPException(500, "HF_TOKEN tidak ditemukan di .env")

        # 1. Download model dari HF Model repo ke trained_models/
        TARGET_MODEL_DIR = os.path.join(CURRENT_DIR, "trained_models")
        os.makedirs(TARGET_MODEL_DIR, exist_ok=True)
        save_path = os.path.join(TARGET_MODEL_DIR, req.model_filename)

        print(f"⬇️  Download model dari HF: {req.model_filename}")
        model_url = f"https://huggingface.co/{repo_id}/resolve/main/{req.model_filename}"
        response  = requests.get(
            model_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=300
        )

        if response.status_code != 200:
            raise HTTPException(
                500,
                f"Gagal download model dari HF: HTTP {response.status_code}"
            )

        with open(save_path, "wb") as f:
            f.write(response.content)

        print(f"✅ Model berhasil didownload: {save_path}")

        if os.path.getsize(save_path) < 1000:
            raise HTTPException(500, "File model yang didownload terlalu kecil, kemungkinan corrupt.")

        # 2. Update registry — bandingkan akurasi dengan model sebelumnya
        registry_result = update_model_registry(
            new_model_path     = save_path,
            version            = req.version,
            accuracy           = req.accuracy,    # ← kirim langsung string "91.49%"
            final_loss         = req.final_loss,
            total_data_retrain = req.total_data_retrain,
            execution_time     = req.execution_time,
            upload_success     = True
        )

        is_best = registry_result["is_best_model"]

        # 3. Kalau model BEST → reload model global di lokal
        if is_best:
            print("🔄 Model BEST — reload model global dari file .pth...")
            _training_active = True
            try:
                async with _model_lock:
                    new_active_path = get_current_model_path()
                    checkpoint      = torch.load(
                        new_active_path, map_location="cpu", weights_only=False
                    )
                    model = build_model(num_classes=23)
                    if isinstance(checkpoint, dict):
                        model.load_state_dict(checkpoint)
                    else:
                        model = checkpoint
                    model.eval()
                    MODEL_VERSION = get_current_model_version_name()
                    print(f"✅ Model global diperbarui → {MODEL_VERSION}")
            finally:
                _training_active = False
        else:
            # Model DITOLAK — hapus file yang baru didownload
            # supaya tidak memenuhi disk
            if os.path.exists(save_path):
                os.remove(save_path)
            print("❌ Model DITOLAK — model global tidak berubah, file dihapus.")

        try:
            acc_float = float(str(req.accuracy).replace("%", "")) / 100.0
        except:
            acc_float = 0.0

        return {
            "status":                  "success",
            "is_best_model":           is_best,
            "model_version":           req.version,
            "model_path":              req.model_filename,
            "previous_model_filename": registry_result.get("previous_model_filename"),
            "val_accuracy":            req.accuracy,                    # "91.49%"
            "accuracy":                req.accuracy,                    # "91.49%"
            "val_accuracy_float":      round(acc_float, 4),
            "baseline_accuracy":       registry_result.get("baseline_accuracy", "N/A"),
            "threshold":               registry_result.get("threshold", "N/A"),
            "threshold_locked":        registry_result.get("threshold_locked", False),
            "rejection_reason":        registry_result.get("rejection_reason"),
            "final_loss":              req.final_loss,
            "total_data_retrain":      req.total_data_retrain,
            "execution_time":          req.execution_time,
            "retrained_at":            time.strftime("%Y-%m-%d %H:%M:%S")
        }

    except HTTPException:
        raise
    except Exception as e:
        _training_active = False
        print(f"❌ Error register-trained-model: {e}")
        print(traceback.format_exc())
        raise HTTPException(500, str(e))
    
@app.delete("/delete-rejected-model")
def delete_rejected_model(filename: str):
    if not filename or filename in ("None", "null", ""):
        return {"status": "skipped", "message": "Tidak ada filename"}

    rejected_path = os.path.join(CURRENT_DIR, "trained_models", "rejected", filename)
    if os.path.exists(rejected_path):
        os.remove(rejected_path)
        print(f"🗑️ Model rejected dihapus: {rejected_path}")
        return {"status": "deleted", "filename": filename}

    return {"status": "not_found", "message": f"{filename} tidak ditemukan di rejected/"}