"""Normalisasi gambar resi -> matriks biner (bitstring 2D).

Tahapan (sesuai Bab 2.5 makalah):
RGB -> grayscale -> skala ke lebar tetap (rasio dipertahankan) -> potong/lengkapi
tinggi -> blur ringan -> thresholding (Otsu) -> normalisasi tema (latar = 1).
"""
import numpy as np
from PIL import Image, ImageFilter, ImageOps

TARGET_W, TARGET_H = 240, 496  # n = 119.040 bit per gambar


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b, sum_b = 0.0, 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


def load_image(file_or_path) -> Image.Image:
    img = Image.open(file_or_path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def _fit_canvas(gray: Image.Image) -> Image.Image:
    """Skala ke lebar TARGET_W (rasio tetap), lalu potong/lengkapi tinggi dari atas.

    Tata letak resi mengikuti lebar layar, jadi HP dengan rasio layar berbeda
    tetap sejajar; kelebihan/kekurangan ruang kosong ada di bagian bawah.
    """
    w, h = gray.size
    new_h = max(1, round(h * TARGET_W / w))
    gray = gray.resize((TARGET_W, new_h), Image.LANCZOS)
    if new_h >= TARGET_H:
        return gray.crop((0, 0, TARGET_W, TARGET_H))
    bg = gray.getpixel((2, new_h - 2))
    canvas = Image.new("L", (TARGET_W, TARGET_H), bg)
    canvas.paste(gray, (0, 0))
    return canvas


def binarize(img: Image.Image) -> np.ndarray:
    """Kembalikan array uint8 (TARGET_H, TARGET_W) berisi 0/1."""
    gray = _fit_canvas(img.convert("L"))
    gray = gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(gray, dtype=np.uint8)
    t = _otsu_threshold(arr)
    bits = (arr >= t).astype(np.uint8)
    # Normalisasi tema gelap/terang: piksel mayoritas (latar) dibuat bernilai 1
    if bits.mean() < 0.5:
        bits = 1 - bits
    return bits


def read_software_tag(file) -> str | None:
    """Baca tag EXIF 'Software' (misal Photoshop). Screenshot asli biasanya kosong."""
    try:
        file.seek(0)
        exif = Image.open(file).getexif()
        value = exif.get(305)
        file.seek(0)
        return str(value) if value else None
    except Exception:
        return None
