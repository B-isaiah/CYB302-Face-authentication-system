"""
Bulk enroll subjects from the ORL (Olivetti) face dataset.

Usage:
    python scripts/bulk_enroll_orl.py --enroll-count 5

First N images per subject become enrollment templates.
Remaining images can be used as test probes for evaluation.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from app.database import SessionLocal, engine, Base
from app.models import User
from app.services.feature_extraction import extract_embedding
from app.services.encryption import encrypt_embedding
from app.services.preprocessing import preprocess_image
from app.services.multimodal import extract_hog_features
from app.datasets.orl import download_orl, get_subjects


def bulk_enroll(enroll_count=5, force_download=False):
    download_orl(force=force_download)

    subjects = get_subjects()
    if not subjects:
        print("No ORL subjects found. Run the script again or check datasets/orl/")
        return

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        total = 0
        skipped = 0
        for subj_name, images in subjects:
            batch = images[:enroll_count]
            for img_path in batch:
                img = cv2.imread(str(img_path))
                if img is None:
                    skipped += 1
                    continue

                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                embedding = extract_embedding(img)
                if embedding is None:
                    skipped += 1
                    continue

                encrypted_face = encrypt_embedding(embedding)
                hog_feat = extract_hog_features(img)
                encrypted_hog = encrypt_embedding(hog_feat) if hog_feat is not None else None

                user = User(
                    name=subj_name,
                    image_paths=str(img_path),
                    face_embedding=encrypted_face,
                    hog_feature=encrypted_hog,
                )
                db.add(user)
                total += 1
            print(f"  {subj_name}: enrolled {len(batch)} images")

        db.commit()
        test_count = sum(len(imgs[enroll_count:]) for _, imgs in subjects)
        print(f"\nDone! Enrolled {total} records, skipped {skipped}")
        print(f"Remaining for testing: {test_count} images across {len(subjects)} subjects")
        print(f"\nNow run evaluation:")
        print(f"  POST /metrics/evaluate-dataset?enroll_count={enroll_count}")
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk enroll ORL face dataset")
    parser.add_argument(
        "--enroll-count", type=int, default=5,
        help="Images per subject to enroll (default: 5, max: 10)",
    )
    parser.add_argument(
        "--force-download", action="store_true",
        help="Re-download dataset even if already present",
    )
    args = parser.parse_args()
    bulk_enroll(args.enroll_count, args.force_download)
