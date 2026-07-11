from cryptography.fernet import Fernet
from app.config import SECRET_KEY
import base64
import hashlib
import numpy as np


def _get_fernet() -> Fernet:
    """Create a Fernet cipher from the configured secret key."""
    key = hashlib.sha256(SECRET_KEY.encode()).digest()
    fernet_key = base64.urlsafe_b64encode(key)
    return Fernet(fernet_key)


def encrypt_embedding(embedding: np.ndarray) -> bytes:
    """Encrypt a face embedding (128D numpy array) using Fernet symmetric encryption."""
    fernet = _get_fernet()
    data = embedding.tobytes()
    return fernet.encrypt(data)


def decrypt_embedding(encrypted: bytes) -> np.ndarray:
    """Decrypt a previously encrypted face embedding back to a numpy array."""
    fernet = _get_fernet()
    data = fernet.decrypt(encrypted)
    return np.frombuffer(data, dtype=np.float64)
