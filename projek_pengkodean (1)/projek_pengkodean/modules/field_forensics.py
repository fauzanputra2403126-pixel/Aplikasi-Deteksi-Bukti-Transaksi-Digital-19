"""Pemeriksaan isi kolom resi (nominal, nama, tanggal, dst.) terhadap bekas penyuntingan.

Kolom isi berbeda di tiap transaksi, jadi isinya tidak bisa dibandingkan bit demi bit
dengan resi asli. Yang bisa dibandingkan adalah GAYA tampilannya, yang dibuat aplikasi
bank secara konsisten:

1. Posisi      : nilai rata kanan di garis yang sama; nominal header di tengah kartu.
2. Tinta       : warna/ketebalan teks sama di semua kolom nilai dalam satu gambar
                 (dibandingkan antar-kolom, jadi tahan terhadap resize/kompresi).
3. Bentuk huruf: tiap karakter diubah ke bitmap 12x20, lalu dicari karakter resi asli
                 terdekat dengan Hamming Distance (indikasi lemah, tidak pernah
                 memberi vonis sendirian).

Posisi kolom ditentukan relatif terhadap kartu putih, sehingga gambar yang diperkecil
atau digeser tetap dapat dipetakan.
"""
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

GLYPH_W, GLYPH_H = 12, 20
INK_LEVEL = 128          # piksel < 128 dianggap tinta
INK_TOLERANCE = 15       # selisih tinta antar-kolom yang masih wajar (skala 0-255)
MIN_ALIGN_TOL = 0.008    # toleransi posisi minimum (fraksi lebar kartu, ~6 px)
GLYPH_FLOOR = 8.0        # batas bawah ambang jarak karakter (% bit)


# ------------------------------------------------------------------ kartu & pemetaan
def _longest_run(flags):
    best_len, best_start, start = 0, 0, None
    for i, v in enumerate(list(flags) + [False]):
        if v and start is None:
            start = i
        if (not v) and start is not None:
            if i - start > best_len:
                best_len, best_start = i - start, start
            start = None
    return best_start, best_start + best_len - 1


def detect_card(gray: np.ndarray):
    """Kotak kartu putih (x0, y0, x1, y1) atau None bila tidak ditemukan (mis. mode gelap)."""
    h, w = gray.shape
    bg = np.median(gray[:20, :20])
    bright = np.percentile(gray, 99.5)
    if bright - bg < 3:
        return None
    m = gray >= (bg + bright) / 2
    band = m[int(h * .40):int(h * .80)]
    x0, x1 = _longest_run(band.mean(axis=0) > .5)
    if x1 - x0 < 0.5 * w:
        return None
    pad = int(.08 * (x1 - x0))
    inner = m[:, x0 + pad:x1 - pad]
    ys = np.where(inner.mean(axis=1) > .5)[0]
    if len(ys) == 0 or ys.max() - ys.min() < 0.3 * h:
        return None
    return int(x0), int(ys.min()), int(x1), int(ys.max())


def _map_box(box, ref_card, tc):
    rx0, ry0, rx1, ry1 = ref_card
    tx0, ty0, tx1, ty1 = tc
    sx = (tx1 - tx0) / (rx1 - rx0)
    sy = (ty1 - ty0) / (ry1 - ry0)
    return [int(round(v)) for v in (tx0 + (box[0] - rx0) * sx, ty0 + (box[1] - ry0) * sy,
                                    tx0 + (box[2] - rx0) * sx, ty0 + (box[3] - ry0) * sy)]


# ------------------------------------------------------------------ karakter (glyph)
def _empty_glyphs():
    return np.zeros((0, GLYPH_H, GLYPH_W), dtype=bool)


def _glyphs(ink: np.ndarray) -> np.ndarray:
    lab, n = ndi.label(ink)
    if n == 0:
        return _empty_glyphs()
    objs = ndi.find_objects(lab)
    hmax = max(o[0].stop - o[0].start for o in objs)
    items = []
    for i, o in enumerate(objs):
        if (o[0].stop - o[0].start) < 0.5 * hmax:   # abaikan titik/koma
            continue
        g = ((lab[o] == i + 1) * 255).astype(np.uint8)
        g = np.array(Image.fromarray(g).resize((GLYPH_W, GLYPH_H), Image.BILINEAR)) > 127
        items.append((o[1].start, g))
    items.sort(key=lambda t: t[0])
    return np.array([g for _, g in items], dtype=bool) if items else _empty_glyphs()


def _glyph_distance(glyphs: np.ndarray, library: np.ndarray):
    """Rata-rata jarak Hamming (% bit) tiap karakter ke karakter asli terdekat."""
    if len(glyphs) == 0 or len(library) == 0:
        return None
    lib = library.reshape(len(library), -1)
    out = []
    for g in glyphs:
        d = int((lib != g.reshape(1, -1)).sum(axis=1).min())
        out.append(d / g.size * 100)
    return float(np.mean(out))


# ------------------------------------------------------------------ pengukuran
def measure(img: Image.Image, cfg: dict):
    gray = np.array(img.convert("L"), dtype=np.float32)
    tc = detect_card(gray)
    if tc is None:
        return None
    cw = tc[2] - tc[0]
    H, W = gray.shape
    out = {}
    for key, f in cfg["fields"].items():
        x0, y0, x1, y1 = _map_box(f["box"], cfg["ref_card"], tc)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        crop = gray[y0:y1, x0:x1]
        ink = crop < INK_LEVEL if crop.size else np.zeros((0, 0), bool)
        if ink.sum() < 10:
            out[key] = None
            continue
        cols = np.where(ink.any(axis=0))[0]
        left, right = x0 + cols.min(), x0 + cols.max()
        out[key] = {
            "right": (right - tc[0]) / cw,
            "center": ((left + right) / 2 - (tc[0] + tc[2]) / 2) / cw,
            "ink": float(np.percentile(crop[ink], 10)),
            "glyphs": _glyphs(ink) if f.get("glyph_lib") else _empty_glyphs(),
        }
    return {"card": tc, "fields": out}


def _library(measurements, cfg):
    arrs = [m["fields"][k]["glyphs"] for m in measurements
            for k, f in cfg["fields"].items()
            if f.get("glyph_lib") and m["fields"].get(k) is not None]
    arrs = [a for a in arrs if len(a)]
    return np.concatenate(arrs) if arrs else _empty_glyphs()


def build_field_model(ref_paths, cfg):
    """Pelajari pola posisi & karakter dari resi asli. Butuh >= 2 gambar yang terbaca."""
    from .preprocess import load_image
    ms = [measure(load_image(p), cfg) for p in ref_paths]
    ms = [m for m in ms if m is not None]
    if len(ms) < 2:
        return None

    expected = {}
    for key, f in cfg["fields"].items():
        vals = [m["fields"][key][f["align"]] for m in ms if m["fields"].get(key) is not None]
        if vals:
            expected[key] = (float(np.median(vals)),
                             max(MIN_ALIGN_TOL, 3 * (max(vals) - min(vals))))

    # Kalibrasi ambang karakter: tiap resi asli diuji terhadap pustaka resi lainnya
    loo = []
    for i, m in enumerate(ms):
        lib_i = _library([x for j, x in enumerate(ms) if j != i], cfg)
        ds = [_glyph_distance(m["fields"][k]["glyphs"], lib_i)
              for k, f in cfg["fields"].items()
              if f.get("glyph_check") and m["fields"].get(k) is not None]
        ds = [d for d in ds if d is not None]
        if ds:
            loo.append(float(np.mean(ds)))
    glyph_thr = max(GLYPH_FLOOR, 1.5 * max(loo)) if loo else GLYPH_FLOOR

    return {"n": len(ms), "expected": expected, "library": _library(ms, cfg),
            "glyph_threshold": glyph_thr}


# ------------------------------------------------------------------ pemeriksaan
def check_fields(img: Image.Image, cfg: dict, model):
    if model is None:
        return {"available": False, "reason": "Model isi kolom belum bisa dibuat (butuh minimal 2 resi asli yang terbaca)."}
    m = measure(img, cfg)
    if m is None:
        return {"available": False, "reason": "Kartu putih resi tidak terdeteksi (mode gelap atau bukan resi SeaBank), "
                                              "pemeriksaan isi kolom dilewati."}
    fields = m["fields"]
    group = [k for k, f in cfg["fields"].items() if f.get("group") == "value" and fields.get(k) is not None]
    ref_w = cfg["ref_card"][2] - cfg["ref_card"][0]

    rows = []
    for key, f in cfg["fields"].items():
        meas = fields.get(key)
        row = {"key": key, "Kolom": f["label"], "Status": "OK", "Keterangan": "Sesuai pola resi asli"}
        if meas is None:
            row.update(Status="ANOMALI", Keterangan="Kolom kosong atau tidak terbaca")
            rows.append(row)
            continue

        issues, warns = [], []
        exp, tol = model["expected"].get(key, (None, None))
        if exp is not None:
            dev = meas[f["align"]] - exp
            if abs(dev) > tol:
                what = "rata kanan" if f["align"] == "right" else "posisi tengah"
                issues.append(f"{what} bergeser {dev * ref_w:+.0f} px dari pola asli")

        if f.get("group") == "value":
            others = [fields[k]["ink"] for k in group if k != key]
            if len(others) >= 3:
                ref_ink = float(np.median(others))
                if abs(meas["ink"] - ref_ink) > INK_TOLERANCE:
                    issues.append(f"tinta {'lebih gelap' if meas['ink'] < ref_ink else 'lebih terang'} "
                                  f"dari kolom lain ({meas['ink']:.0f} vs {ref_ink:.0f})")

        if f.get("glyph_check"):
            gd = _glyph_distance(meas["glyphs"], model["library"])
            if gd is not None and gd > model["glyph_threshold"]:
                warns.append(f"bentuk karakter menyimpang dari font asli "
                             f"({gd:.1f}% > {model['glyph_threshold']:.1f}%)")

        if issues:
            row.update(Status="ANOMALI", Keterangan="; ".join(issues + warns))
        elif warns:
            row.update(Status="CURIGA", Keterangan="; ".join(warns))
        rows.append(row)

    statuses = [r["Status"] for r in rows]
    verdict = "ANOMALI" if "ANOMALI" in statuses else "CURIGA" if "CURIGA" in statuses else "OK"
    return {"available": True, "rows": rows, "verdict": verdict,
            "n_anomali": statuses.count("ANOMALI"), "n_curiga": statuses.count("CURIGA")}


# ------------------------------------------------------------------ vonis gabungan
_LAYOUT_RANK = {"VALID_ASLI": 0, "MENCURIGAKAN": 1, "PALSU": 2}
_FIELD_RANK = {"OK": 0, "CURIGA": 1, "ANOMALI": 2}
_RANK_NAME = ["VALID_ASLI", "MENCURIGAKAN", "PALSU"]

FINAL_TITLES = {
    "VALID_ASLI": "✅ SESUAI POLA RESI ASLI",
    "MENCURIGAKAN": "⚠️ MENCURIGAKAN",
    "PALSU": "🚨 DICURIGAI PALSU / DIMANIPULASI",
}


def final_verdict(layout_verdict: str, field_verdict):
    """Vonis akhir = yang terburuk dari pemeriksaan tata letak dan isi kolom."""
    rank = max(_LAYOUT_RANK[layout_verdict], _FIELD_RANK.get(field_verdict, 0))
    return _RANK_NAME[rank]
