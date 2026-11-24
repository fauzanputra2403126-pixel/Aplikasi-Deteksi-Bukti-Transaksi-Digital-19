"""Perhitungan Hamming Distance, Hamming Weight, dan vonis forensik."""
import numpy as np


def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """d(x,y) = jumlah posisi i dengan x_i != y_i (dihitung dengan perbandingan langsung)."""
    return int(np.count_nonzero(x != y))


def hamming_weight(v: np.ndarray) -> int:
    """wt(v) = jumlah elemen bukan nol."""
    return int(np.count_nonzero(v))


def forensic_compare(x: np.ndarray, y: np.ndarray, mask: np.ndarray, threshold: float) -> dict:
    """Bandingkan vektor acuan x dengan vektor uji y pada posisi tetap (mask)."""
    xm = x[mask].astype(np.uint8)
    ym = y[mask].astype(np.uint8)
    n = int(xm.size)

    e = np.bitwise_xor(xm, ym)          # vektor perubahan e = x XOR y
    d = hamming_distance(xm, ym)        # d(x,y)
    wt_e = hamming_weight(e)            # wt(x XOR y)
    wt_x, wt_y = hamming_weight(xm), hamming_weight(ym)

    ratio = d / n if n else 1.0
    bound_ok = abs(wt_x - wt_y) <= d    # |wt(x)-wt(y)| <= d(x,y)

    e_full = np.zeros_like(x, dtype=np.uint8)
    e_full[mask] = e

    if ratio <= threshold:
        verdict = "VALID_ASLI"
        title = "✅ LAYOUT SESUAI RESI ASLI"
        desc = ("Bagian tetap resi (logo, tata letak, label) cocok dengan koleksi resi asli "
                "provider ini. Tidak ditemukan penyimpangan bit yang signifikan.")
    elif ratio <= 2 * threshold:
        verdict = "MENCURIGAKAN"
        title = "⚠️ MENCURIGAKAN"
        desc = ("Terdapat penyimpangan bit di atas batas wajar resi asli, tetapi belum "
                "sejauh resi palsu. Bisa karena kompresi/perangkat berbeda atau suntingan "
                "kecil. Periksa peta error dan verifikasi manual.")
    else:
        verdict = "PALSU"
        title = "🚨 DICURIGAI PALSU / DIMANIPULASI"
        desc = ("Penyimpangan bit pada bagian tetap resi jauh melampaui variasi resi asli. "
                "Kemungkinan gambar diedit, dibuat ulang, atau bukan resi dari provider ini.")

    return {
        "verdict": verdict,
        "title_output": title,
        "description": desc,
        "total_bits": n,
        "hamming_distance": d,
        "hamming_weight": wt_e,
        "wt_x": wt_x,
        "wt_y": wt_y,
        "ratio": ratio,
        "bit_change_ratio": ratio * 100,
        "threshold": threshold,
        "is_verified": (d == wt_e) and bound_ok,
        "e_full": e_full,
    }
