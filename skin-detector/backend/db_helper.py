# Helper DB untuk pengujian manual CT/CD. Jalankan dari folder backend (butuh database.py & models.py).
# count   : jumlah data valid & belum dipakai retraining
# prepare N : sisihkan (hold) semua data belum-terpakai, lalu lepas N data terbaru yang file gambarnya ada -> tepat N data baru
# restore : kembalikan semua data yang masih disisihkan ke status belum dipakai
import sys, os, json
sys.path.insert(0, os.getcwd())
from database import SessionLocal
from models import Prediction

HELD = "held_ids.json"
db = SessionLocal()

def unused_q():
    return db.query(Prediction).filter(Prediction.is_valid == True, Prediction.used_for_retraining == False)

def release(ids):
    if ids:
        db.query(Prediction).filter(Prediction.id.in_(ids)).update({Prediction.used_for_retraining: False}, synchronize_session=False)
        db.commit()

cmd = sys.argv[1] if len(sys.argv) > 1 else "count"
if cmd == "count":
    print(unused_q().count())
elif cmd == "prepare":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    held = json.load(open(HELD)) if os.path.exists(HELD) else []
    # 1) sisihkan semua data yang sekarang 'belum dipakai' (backlog + hasil seed_data baru)
    rows = unused_q().all()
    for r in rows:
        r.used_for_retraining = True
        if r.id not in held:
            held.append(r.id)
    db.commit()
    # 2) lepas N data TERBARU yang file gambarnya masih ada di disk (archive melewati data tanpa file)
    cands = db.query(Prediction).filter(Prediction.id.in_(held), Prediction.is_valid == True).order_by(Prediction.id.desc()).all()
    take = [r.id for r in cands if r.image_path and os.path.exists(r.image_path)][:n]
    if len(take) < n:
        print(f"PERINGATAN: hanya {len(take)} data dengan file gambar ada. Jalankan seed_data dulu.")
    release(take)
    held = [i for i in held if i not in take]
    json.dump(held, open(HELD, "w"))
    print(unused_q().count())
elif cmd == "restore":
    held = json.load(open(HELD)) if os.path.exists(HELD) else []
    release(held)
    if os.path.exists(HELD):
        os.remove(HELD)
    print(unused_q().count())
db.close()