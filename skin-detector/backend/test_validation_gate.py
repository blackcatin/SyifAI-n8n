import os
import json
import tempfile
from pathlib import Path

import model_registry as r


def test_case(candidate_accuracy, expected_status):
    with tempfile.TemporaryDirectory(prefix="syifai_gate_test_") as temp:
        root = Path(temp)
        model_dir = root / "trained_models"
        active_dir = model_dir / "active"
        rejected_dir = model_dir / "rejected"

        active_dir.mkdir(parents=True)
        rejected_dir.mkdir(parents=True)

        # Arahkan semua operasi registry ke folder sementara.
        r.MODEL_DIR = str(model_dir)
        r.ACTIVE_DIR = str(active_dir)
        r.REJECTED_DIR = str(rejected_dir)
        r.CURRENT_MODEL_FILE = str(model_dir / "current_model.txt")
        r.REGISTRY_JSON_FILE = str(root / "model_history.json")

        champion = active_dir / "champion.pth"
        champion.write_bytes(b"dummy champion model")

        Path(r.CURRENT_MODEL_FILE).write_text(
            "champion.pth", encoding="utf-8"
        )

        history = [{
            "version": "champion",
            "model_path": "champion.pth",
            "accuracy": "95.74%",
            "status": "ACTIVE"
        }]

        Path(r.REGISTRY_JSON_FILE).write_text(
            json.dumps(history), encoding="utf-8"
        )

        candidate = root / "challenger_test.pth"
        candidate.write_bytes(b"dummy challenger model")

        result = r.update_model_registry(
            new_model_path=str(candidate),
            version="test_candidate",
            accuracy=candidate_accuracy,
            final_loss=0.1,
            total_data_retrain=5,
            execution_time=1
        )

        pointer = Path(r.CURRENT_MODEL_FILE).read_text(
            encoding="utf-8"
        ).strip()

        assert result["status"] == expected_status, (
            f"Expected {expected_status}, got {result['status']}"
        )

        if expected_status == "REJECTED":
            assert pointer == "champion.pth", (
                "Pointer Champion berubah saat kandidat ditolak"
            )
            assert (rejected_dir / candidate.name).exists()
            assert champion.exists()
        else:
            assert pointer == candidate.name
            assert (active_dir / candidate.name).exists()
            assert champion.exists()

        print(
            f"PASS | Challenger={candidate_accuracy}% | "
            f"Status={result['status']} | Pointer={pointer}"
        )


test_case(95.73, "REJECTED")  # Akurasi turun
test_case(96.73, "REJECTED")  # Naik 0,99 poin persentase
test_case(96.74, "ACTIVE")    # Naik tepat 1,00 poin persentase
test_case(97.00, "ACTIVE")    # Naik 1,26 poin persentase

print("\nSemua pengujian validation gate berhasil.")
