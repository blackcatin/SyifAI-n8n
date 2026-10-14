import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import mlflow
import mlflow.pytorch
 
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split
from PIL import Image

# ==========================================
# CONFIG PATH
# ==========================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(BASE_DIR, "trained_models")
RETRAIN_DIR = os.path.join(BASE_DIR, "uploads", "retraining")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "train")
 
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 0.0001
VAL_SPLIT    = 0.2       # 20% data untuk validasi
VAL_ACCURACY_THRESHOLD = 0.90  # threshold 90% dicek dari val acc, bukan train acc
 
device = torch.device("cpu")

# ==========================================
# MLFLOW CONFIG
# ==========================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT   = "SyifAI_Skin_Disease_Retraining"

# ==========================================
# TRANSFORM
# ==========================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Mapping nama folder disk → nama kelas resmi
FOLDER_TO_LABEL = {
    "Acne and Rosacea Photos":                                        "Acne and Rosacea",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions": "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis Photos":                                       "Atopic Dermatitis",
    "Bullous Disease Photos":                                         "Bullous Disease",
    "Cellulitis Impetigo and other Bacterial Infections":             "Cellulitis, Impetigo & other Bacterial Infections",
    "Eczema Photos":                                                  "Eczema",
    "Exanthems and Drug Eruptions":                                   "Exanthems & Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases":              "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes HPV and other STDs Photos":                               "Herpes, HPV & other STDs",
    "Light Diseases and Disorders of Pigmentation":                   "Light Diseases & Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases":                     "Lupus & other Connective Tissue diseases",
    "Melanoma Skin Cancer Nevi and Moles":                            "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus and other Nail Disease":                             "Nail Fungus & other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis":                 "Poison Ivy & other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases":          "Psoriasis, Lichen Planus & related diseases",
    "Scabies Lyme Disease and other Infestations and Bites":          "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses and other Benign Tumors":                   "Seborrheic Keratoses & other Benign Tumors",
    "Systemic Disease":                                               "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections":         "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria Hives":                                                "Urticaria / Hives",
    "Vascular Tumors":                                                "Vascular Tumors",
    "Vasculitis Photos":                                              "Vasculitis",
    "Warts Molluscum and other Viral Infections":                     "Warts, Molluscum & other Viral Infections",
}

CLASS_LABELS = [
    "Acne and Rosacea",
    "Actinic Keratosis, Basal Cell Carcinoma & other Malignant Lesions",
    "Atopic Dermatitis",
    "Bullous Disease",
    "Cellulitis, Impetigo & other Bacterial Infections",
    "Eczema",
    "Exanthems & Drug Eruptions",
    "Hair Loss, Alopecia & other Hair Diseases",
    "Herpes, HPV & other STDs",
    "Light Diseases & Disorders of Pigmentation",
    "Lupus & other Connective Tissue diseases",
    "Melanoma, Skin Cancer, Nevi & Moles",
    "Nail Fungus & other Nail Disease",
    "Poison Ivy & other Contact Dermatitis",
    "Psoriasis, Lichen Planus & related diseases",
    "Scabies, Lyme Disease & other Infestations & Bites",
    "Seborrheic Keratoses & other Benign Tumors",
    "Systemic Disease",
    "Tinea, Ringworm, Candidiasis & other Fungal Infections",
    "Urticaria / Hives",
    "Vascular Tumors",
    "Vasculitis",
    "Warts, Molluscum & other Viral Infections"
]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(CLASS_LABELS)}


def sample_from_local_dataset(samples_per_class: int = 10) -> list:
    records = []

    if not os.path.exists(DATASET_DIR):
        print(f"⚠️ DATASET_DIR tidak ditemukan: {DATASET_DIR}")
        return records

    for folder_name in os.listdir(DATASET_DIR):
        class_dir = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(class_dir):
            continue

        class_name = FOLDER_TO_LABEL.get(folder_name)

        if not class_name and folder_name in LABEL_TO_IDX:
            class_name = folder_name

        if not class_name:
            folder_lower = folder_name.lower()
            for label in CLASS_LABELS:
                if label.lower() == folder_lower:
                    class_name = label
                    break

        if not class_name:
            print(f"⚠️ Kelas tidak dikenal, dilewati: {folder_name}")
            continue

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not image_files:
            continue

        sampled = random.sample(image_files, min(samples_per_class, len(image_files)))
        for fname in sampled:
            records.append({
                "image_path":      os.path.join(class_dir, fname),
                "predicted_class": class_name,
                "source":          "stratified"
            })

    print(f"📂 Stratified dari lokal: {len(records)} gambar ({samples_per_class} per kelas)")
    return records


class SubsetDataset(Dataset):
    def __init__(self, records: list):
        valid = []
        for r in records:
            if os.path.exists(r["image_path"]):
                valid.append(r)
        skipped = len(records) - len(valid)
        if skipped > 0:
            print(f"⚠️ {skipped} file tidak ditemukan, dilewati.")
        self.records = valid

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            img = transform(img)
        except Exception as e:
            print(f"⚠️ Gagal baca gambar: {rec['image_path']} — {e}")
            img = torch.zeros(3, 224, 224)

        label  = LABEL_TO_IDX.get(rec["predicted_class"], 0)
        source = rec.get("source", "stratified")
        return img, label, source


def build_train_val_loaders(records: list):
    dataset   = SubsetDataset(records)
    total     = len(dataset)
 
    if total == 0:
        raise Exception("Semua file gambar tidak ditemukan. Training dibatalkan.")
 
    val_size   = max(1, int(total * VAL_SPLIT))
    train_size = total - val_size
 
    # random_split membagi secara acak tapi reproducible dengan generator
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
 
    print(f"📊 Split dataset: {train_size} train | {val_size} val")
 
    # Train loader: pakai WeightedSampler seperti sebelumnya
    # Tapi weight harus di-index ulang sesuai train_ds.indices
    train_weights = [
        3.0 if dataset.records[i].get("source") in ("new", "hard") else 1.0
        for i in train_ds.indices
    ]
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_weights),
        replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=train_sampler)
 
    # Val loader: tidak pakai sampler — shuffle=False supaya konsisten
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
 
    return train_loader, val_loader

def evaluate(model, val_loader, criterion):
    model.eval()
    val_loss    = 0.0
    val_correct = 0
    val_total   = 0
 
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            loss           = criterion(outputs, labels)
            _, preds       = torch.max(outputs, 1)
            val_correct   += (preds == labels).sum().item()
            val_total     += labels.size(0)
            val_loss      += loss.item()
 
    avg_val_loss = val_loss / max(1, len(val_loader))
    avg_val_acc  = val_correct / max(1, val_total)
 
    model.train()  # kembalikan ke train mode setelah evaluasi
    return avg_val_acc, avg_val_loss


# ==========================================
# TRAINING ENGINE
# ==========================================
def run_pytorch_training(model, model_save_path, subset_records: list):
 
    local_samples = sample_from_local_dataset(samples_per_class=10)
    all_records   = subset_records + local_samples
 
    total_new        = sum(1 for r in all_records if r.get('source') == 'new')
    total_hard       = sum(1 for r in all_records if r.get('source') == 'hard')
    total_stratified = sum(1 for r in all_records if r.get('source') == 'stratified')
 
    print(f"\n🏋️ [PyTorch Engine] Memulai fine-tuning ResNet18...")
    print(f"📊 Total dataset: {len(all_records)} data")
    print(f"   ↳ Baru (DB)    : {total_new}")
    print(f"   ↳ Hard (DB)    : {total_hard}")
    print(f"   ↳ Stratified   : {total_stratified}")
 
    # [P2-FIX] Pakai train/val loader, bukan satu loader
    train_loader, val_loader = build_train_val_loaders(all_records)
 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
 
    model.train()
 
    final_train_loss = 0.0
    final_train_acc  = 0.0
    final_val_acc    = 0.0   # [P2-FIX] ini yang dikirim ke registry & n8n
    final_val_loss   = 0.0
 
    model_name = os.path.basename(model_save_path).replace(".pth", "")
 
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_enabled = True
    except Exception as e:
        print(f"⚠️ MLFlow tidak tersedia, logging dilewati: {e}")
        mlflow_enabled = False
 
    run_context = mlflow.start_run(run_name=model_name) if mlflow_enabled else None
 
    try:
        if mlflow_enabled:
            mlflow.log_params({
                "epochs":           EPOCHS,
                "batch_size":       BATCH_SIZE,
                "learning_rate":    LR,
                "val_split":        VAL_SPLIT,
                "total_data":       len(all_records),
                "total_new":        total_new,
                "total_hard":       total_hard,
                "total_stratified": total_stratified,
                "model_name":       model_name,
            })
 
        for epoch in range(EPOCHS):
            # ── TRAIN ──
            running_loss = 0.0
            correct      = 0
            total        = 0
 
            model.train()
            for images, labels, _ in train_loader:
                images, labels = images.to(device), labels.to(device)
 
                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
 
                _, preds  = torch.max(outputs, 1)
                correct      += (preds == labels).sum().item()
                total        += labels.size(0)
                running_loss += loss.item()
 
            avg_train_loss = running_loss / max(1, len(train_loader))
            avg_train_acc  = correct / max(1, total)
 
            # [P2-FIX] ── VALIDASI setelah setiap epoch ──
            val_acc, val_loss = evaluate(model, val_loader, criterion)
 
            final_train_loss = avg_train_loss
            final_train_acc  = avg_train_acc
            final_val_acc    = val_acc
            final_val_loss   = val_loss
 
            print(
                f"📈 Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
 
            if mlflow_enabled:
                mlflow.log_metrics({
                    "train_accuracy": round(avg_train_acc, 4),
                    "train_loss":     round(avg_train_loss, 4),
                    "val_accuracy":   round(val_acc, 4),   # [P2-FIX] log val acc
                    "val_loss":       round(val_loss, 4),
                    "learning_rate":  round(scheduler.get_last_lr()[0], 8),
                }, step=epoch + 1)
 
            scheduler.step()
 
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"💾 Model saved: {model_save_path}")
 
        if mlflow_enabled:
            mlflow.log_metrics({
                "final_train_accuracy": round(final_train_acc, 4),
                "final_train_loss":     round(final_train_loss, 4),
                "final_val_accuracy":   round(final_val_acc, 4),
                "final_val_loss":       round(final_val_loss, 4),
            })
            mlflow.set_tags({
                "model_file": os.path.basename(model_save_path),
                "status":     "completed",
            })
            print(f"📊 MLFlow run logged: {model_name}")
 
    finally:
        if run_context:
            mlflow.end_run()
 
    model.eval()
    print(f"🔄 Model dikembalikan ke mode eval.")
 
    # [P2-FIX] Return val_accuracy sebagai "accuracy" utama
    # Ini yang akan dibandingkan dengan threshold 90% di n8n
    return {
        "accuracy":           round(final_val_acc, 4),    # ← val acc, bukan train acc
        "train_accuracy":     round(final_train_acc, 4),  # ← train acc tetap ada untuk referensi
        "final_loss":         round(final_val_loss, 4),
        "train_loss":         round(final_train_loss, 4),
        "total_data_retrain": len(all_records),
        "epochs":             EPOCHS
    }