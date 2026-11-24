"""Database resi asli per provider + pembentukan model acuan.

Model per provider:
- consensus : vektor acuan x (voting mayoritas dari semua resi asli)
- mask      : posisi bit "tetap" yang dibandingkan. Dibentuk dari (a) posisi yang
              konsisten di semua resi asli, dikurangi (b) kotak kolom dinamis
              (nominal, nama, rekening, nomor transaksi, waktu) yang diisi manual
              di PROVIDER_CONFIGS.
- threshold : batas rasio perubahan bit, dihitung dengan leave-one-out
"""
import os
import re
import uuid
import numpy as np

from .preprocess import load_image, binarize, TARGET_W, TARGET_H

BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "references")

MIN_REFS = 3
RECOMMENDED_REFS = 10
STABLE_AGREEMENT = 0.9   # posisi dianggap tetap jika >= 90% referensi sepakat
MIN_THRESHOLD = 0.005    # batas bawah ambang (0,5 %)

# Kotak kolom dinamis dalam koordinat piksel resi ukuran asli 775 x 1600:
# (x0, y0, x1, y1). Isi kotak ini berbeda di tiap transaksi sehingga diabaikan.
SEABANK_DYNAMIC_BOXES = [
    (150, 215, 625, 315),    # nominal besar di header
    (310, 370, 725, 480),    # Dari: nama + rekening pengirim
    (290, 535, 725, 645),    # Ke: nama + bank/rekening penerima
    (520, 700, 725, 760),    # Jumlah Transfer
    (480, 805, 725, 865),    # Biaya Transfer (bisa gratis / berbayar)
    (520, 925, 725, 985),    # Jumlah Total
    (320, 1045, 725, 1125),  # No. Transaksi
    (540, 1190, 725, 1240),  # Metode Transaksi
    (540, 1300, 725, 1350),  # Waktu Pemrosesan
    (420, 1410, 725, 1465),  # Waktu Transaksi
]

# Kolom isi yang diperiksa gaya tampilannya (koordinat resi referensi 775 x 1600).
#  align : "right" = nilai rata kanan, "center" = di tengah kartu
#  group : "value" = ikut perbandingan warna tinta antar-kolom
#  glyph_lib   : karakter kolom ini masuk pustaka font asli
#  glyph_check : bentuk karakter kolom ini diuji terhadap pustaka (nominal)
SEABANK_FIELDS = {
    "nominal_header":   {"label": "Nominal (header)",  "box": (150, 215, 625, 315),  "align": "center"},
    "jumlah_transfer":  {"label": "Jumlah Transfer",   "box": (470, 700, 730, 765),  "align": "right",
                         "group": "value", "glyph_lib": True, "glyph_check": True},
    "jumlah_total":     {"label": "Jumlah Total",      "box": (470, 925, 730, 990),  "align": "right",
                         "group": "value", "glyph_lib": True, "glyph_check": True},
    "nama_penerima":    {"label": "Nama penerima",     "box": (290, 535, 730, 602),  "align": "right",
                         "group": "value", "glyph_lib": True},
    "waktu_transaksi":  {"label": "Waktu Transaksi",   "box": (400, 1405, 730, 1470), "align": "right",
                         "group": "value", "glyph_lib": True},
    "no_transaksi":     {"label": "No. Transaksi",     "box": (320, 1045, 730, 1125), "align": "right",
                         "group": "value", "glyph_lib": True},
    "nama_pengirim":    {"label": "Nama pengirim",     "box": (310, 370, 730, 437),  "align": "right",
                         "group": "value", "glyph_lib": True},
    "metode":           {"label": "Metode Transaksi",  "box": (540, 1190, 730, 1240), "align": "right",
                         "group": "value", "glyph_lib": True},
    "waktu_pemrosesan": {"label": "Waktu Pemrosesan",  "box": (540, 1300, 730, 1350), "align": "right",
                         "group": "value", "glyph_lib": True},
}

# Untuk menambah provider lain: salin pola SeaBank, isi ref_width dan dynamic_boxes.
PROVIDER_CONFIGS = {
    "SeaBank": {
        "type": "Bank",
        "ref_width": 775,
        "dynamic_boxes": SEABANK_DYNAMIC_BOXES,
        "ref_card": (26, 343, 748, 1492),   # kartu putih pada resi referensi 775 x 1600
        "fields": SEABANK_FIELDS,
    },
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def provider_dir(provider: str) -> str:
    return os.path.join(BASE_DIR, _slug(provider))


def list_references(provider: str) -> list[str]:
    d = provider_dir(provider)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.lower().endswith(".png"))


def save_reference(provider: str, uploaded_file) -> str:
    uploaded_file.seek(0)
    img = load_image(uploaded_file)
    img.thumbnail((720, 1600))  # hemat ruang penyimpanan (rasio tetap)
    os.makedirs(provider_dir(provider), exist_ok=True)
    path = os.path.join(provider_dir(provider), f"{uuid.uuid4().hex[:10]}.png")
    img.save(path, "PNG")
    return path


def delete_reference(path: str) -> None:
    if os.path.isfile(path):
        os.remove(path)


def dynamic_mask(provider: str) -> np.ndarray:
    """True = piksel termasuk kolom dinamis (diabaikan)."""
    cfg = PROVIDER_CONFIGS.get(provider, {})
    s = TARGET_W / cfg.get("ref_width", 775)
    m = np.zeros((TARGET_H, TARGET_W), dtype=bool)
    for x0, y0, x1, y1 in cfg.get("dynamic_boxes", []):
        m[int(y0 * s):int(np.ceil(y1 * s)), int(x0 * s):int(np.ceil(x1 * s))] = True
    return m


def _consensus_and_mask(stack: np.ndarray, dyn: np.ndarray):
    """stack: (n, H, W) berisi 0/1."""
    p = stack.mean(axis=0)
    lo = 1.0 - STABLE_AGREEMENT
    mask = ((p <= lo) | (p >= STABLE_AGREEMENT)) & ~dyn
    consensus = (p >= 0.5).astype(np.uint8)
    return consensus, mask


def build_model(provider: str):
    paths = list_references(provider)
    if len(paths) < MIN_REFS:
        return None

    stack = np.stack([binarize(load_image(p)) for p in paths])
    dyn = dynamic_mask(provider)
    consensus, mask = _consensus_and_mask(stack, dyn)

    # Kalibrasi ambang batas: tiap referensi asli diuji terhadap model dari
    # referensi lainnya, sehingga variasi alami resi asli ikut terhitung.
    ratios = []
    for i in range(len(stack)):
        others = np.delete(stack, i, axis=0)
        cons_i, mask_i = _consensus_and_mask(others, dyn)
        n_i = int(mask_i.sum())
        d_i = int(np.count_nonzero(stack[i][mask_i] != cons_i[mask_i]))
        ratios.append(d_i / n_i if n_i else 1.0)

    ratios = np.array(ratios)
    threshold = max(float(ratios.mean() + 3 * ratios.std()), MIN_THRESHOLD)

    return {
        "n_refs": len(paths),
        "consensus": consensus,
        "mask": mask,
        "threshold": threshold,
        "loo_ratios": ratios,
        "stable_fraction": float(mask.mean()),
    }
