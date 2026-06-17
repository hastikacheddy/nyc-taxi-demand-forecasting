"""
Model artifact integrity verification.
After training, a SHA-256 hash of the model artifact is written to mlflow as a tag.
Before inference, the hash is re-verified to detect tampering.

Pickle/joblib files can execute arbitrary code on load — this guard ensures only
the exact artifact that passed CI training is ever loaded into production.
"""
import hashlib
import logging
import os

logger = logging.getLogger(__name__)


def hash_file(path: str) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_model_artifact(artifact_path: str, expected_hash: str) -> None:
    """
    Raises RuntimeError if the artifact's SHA-256 does not match expected_hash.
    Call this before loading any joblib/pickle model artifact.
    """
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    actual_hash = hash_file(artifact_path)
    if not actual_hash == expected_hash:
        logger.critical(
            "MODEL INTEGRITY FAILURE: artifact=%s expected=%s actual=%s",
            artifact_path, expected_hash, actual_hash,
        )
        raise RuntimeError(
            f"Model artifact hash mismatch — possible tampering detected.\n"
            f"  Expected: {expected_hash}\n"
            f"  Actual:   {actual_hash}"
        )
    logger.info("Model integrity verified: %s [%s]", artifact_path, actual_hash[:12])
