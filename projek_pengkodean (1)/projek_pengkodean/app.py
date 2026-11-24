import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from modules.preprocess import load_image, binarize, read_software_tag
from modules.reference_store import (
    PROVIDER_CONFIGS, MIN_REFS, RECOMMENDED_REFS,
    list_references, save_reference, delete_reference, build_model,
)
from modules.hamming import forensic_compare
from modules.field_forensics import build_field_model, check_fields, final_verdict, FINAL_TITLES

# ---------------------------------------------------------------- Konfigurasi
st.set_page_config(
    page_title="VerifyID - Multi-Bank Transaction Forensics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 100%); color: #f3f4f6; }
    .badge { color: white; padding: 18px 24px; border-radius: 14px; font-weight: 700;
             font-size: 26px; text-align: center; margin-bottom: 24px; }
    .badge-asli  { background: linear-gradient(135deg,#059669,#10b981); box-shadow: 0 4px 25px rgba(16,185,129,.45); }
    .badge-warn  { background: linear-gradient(135deg,#d97706,#f59e0b); box-shadow: 0 4px 25px rgba(245,158,11,.45); }
    .badge-palsu { background: linear-gradient(135deg,#dc2626,#ef4444); box-shadow: 0 4px 25px rgba(239,68,68,.45); }
</style>
""", unsafe_allow_html=True)

BADGE_CLASS = {"VALID_ASLI": "badge-asli", "MENCURIGAKAN": "badge-warn", "PALSU": "badge-palsu"}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


def admin_ok() -> bool:
    """Proteksi opsional: set ADMIN_PASSWORD di .streamlit/secrets.toml."""
    try:
        pwd = st.secrets.get("ADMIN_PASSWORD", "")
    except Exception:
        pwd = ""
    if not pwd:
        return True
    return st.text_input("Password admin", type="password") == pwd


def make_error_map(y: np.ndarray, mask: np.ndarray, e_full: np.ndarray) -> Image.Image:
    """Merah = bit berubah, biru = area dinamis yang diabaikan."""
    rgb = np.stack([y * 255] * 3, axis=-1).astype(np.uint8)
    ignored = ~mask
    rgb[ignored] = (rgb[ignored] * 0.5 + np.array([60, 60, 200]) * 0.5).astype(np.uint8)
    rgb[e_full == 1] = [255, 0, 0]
    return Image.fromarray(rgb)


# ---------------------------------------------------------------- Header
st.title("🛡️ VerifyID: Sistem Deteksi Transaksi Digital")
st.caption("Verifikasi keaslian resi m-Banking & e-Wallet berbasis Hamming Distance & Hamming Weight")
st.markdown("---")

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.header("⚙️ Parameter")
    provider = st.selectbox("Pilih m-Banking / e-Wallet:", list(PROVIDER_CONFIGS.keys()))
    tolerance = st.slider(
        "Toleransi ambang batas", 0.5, 5.0, 1.0, 0.1,
        help="Naikkan jika resi asli sering terdeteksi mencurigakan (positif palsu). "
             "Turunkan agar lebih ketat.",
    )
    n_refs = len(list_references(provider))
    st.markdown("---")
    st.info(f"**Provider:** {provider}\n\n**Referensi asli tersimpan:** {n_refs}\n\n"
            f"**Metode:** Coding Theory (XOR bit-wise)")

tab_verify, tab_db = st.tabs(["🔍 Verifikasi Resi", "📚 Database Referensi Asli"])

# ---------------------------------------------------------------- Tab 1: Verifikasi
with tab_verify:
    model = build_model(provider)
    if model is None:
        st.warning(f"Referensi asli untuk **{provider}** baru {n_refs}. Minimal {MIN_REFS} "
                   f"(disarankan {RECOMMENDED_REFS}+). Tambahkan di tab *Database Referensi Asli*.")
    elif model["n_refs"] < RECOMMENDED_REFS:
        st.info(f"Model memakai {model['n_refs']} referensi. Akurasi lebih baik jika "
                f"≥ {RECOMMENDED_REFS} referensi dari perangkat berbeda.")

    col_left, col_right = st.columns(2)
    y = None
    img = None

    with col_left:
        st.subheader("📤 Unggah Bukti Transaksi")
        uploaded = st.file_uploader(f"Pilih resi ({provider}) - JPG/PNG",
                                    type=["png", "jpg", "jpeg"], key="verify_upload")
        if uploaded:
            st.image(uploaded, caption=f"Gambar uji ({provider})")

    with col_right:
        st.subheader("🔬 Pratinjau Biner")
        if uploaded:
            software = read_software_tag(uploaded)
            uploaded.seek(0)
            img = load_image(uploaded)
            y = binarize(img)
            st.image((y * 255).astype(np.uint8), caption="Representasi biner gambar uji (y)")

    analyze = st.button("🚀 Verifikasi Keaslian Resi", type="primary",
                        disabled=(model is None or uploaded is None))

    if analyze and model is not None and y is not None:
        st.markdown("---")
        st.header("📋 Hasil Evaluasi Integritas Data")

        cfg = PROVIDER_CONFIGS[provider]
        thr = model["threshold"] * tolerance
        res = forensic_compare(model["consensus"], y, model["mask"], thr)

        fmodel = build_field_model(list_references(provider), cfg)
        fres = check_fields(img, cfg, fmodel)
        final = final_verdict(res["verdict"], fres["verdict"] if fres["available"] else None)

        st.markdown(f'<div class="badge {BADGE_CLASS[final]}">{FINAL_TITLES[final]}</div>',
                    unsafe_allow_html=True)
        st.write(f"**Pemeriksaan tata letak (Hamming):** {res['description']}")
        if fres["available"]:
            if fres["verdict"] == "OK":
                st.write("**Pemeriksaan isi kolom:** Posisi, tinta, dan bentuk huruf pada kolom nominal, "
                         "nama, dan waktu sesuai pola resi asli.")
            else:
                st.write(f"**Pemeriksaan isi kolom:** ditemukan {fres['n_anomali']} kolom anomali dan "
                         f"{fres['n_curiga']} kolom mencurigakan (rincian di bawah).")
        else:
            st.warning(fres["reason"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Bit Dibandingkan (n)", f"{res['total_bits']:,}")
        m2.metric("Hamming Distance d(x,y)", f"{res['hamming_distance']:,}")
        m3.metric("Hamming Weight wt(e)", f"{res['hamming_weight']:,}")
        m4.metric("Rasio Perubahan Bit", f"{res['bit_change_ratio']:.4f} %")

        st.caption(f"Ambang batas asli: {thr * 100:.4f} %  |  "
                   f"wt(x) = {res['wt_x']:,}, wt(y) = {res['wt_y']:,}  |  "
                   f"Area tetap yang dibandingkan: {model['stable_fraction'] * 100:.1f} % dari gambar")

        st.markdown("### 🗺️ Peta Perubahan Bit")
        c1, c2 = st.columns(2)
        with c1:
            st.image(make_error_map(y, model["mask"], res["e_full"]),
                     caption="Merah = bit berubah | Biru = area dinamis (diabaikan)")
        with c2:
            st.image((model["consensus"] * 255).astype(np.uint8),
                     caption="Vektor acuan x (konsensus resi asli)")

        st.markdown("### 📐 Verifikasi Formulasi Teori Pengkodean")
        if res["is_verified"]:
            st.success(f"✅ **Terbukti ekuivalen:** d(x,y) = wt(x ⊕ y) = **{res['hamming_distance']:,} bit**, "
                       f"dan |wt(x) − wt(y)| ≤ d(x,y) terpenuhi.")
        else:
            st.error("⚠️ Ketidaksesuaian perhitungan biner!")

        if fres["available"]:
            st.markdown("### 🔎 Pemeriksaan Isi Kolom (nominal, nama, tanggal)")
            st.caption("Kolom isi berbeda di tiap transaksi, jadi yang dibandingkan adalah gaya tampilannya: "
                       "posisi rata kanan/tengah, ketebalan tinta antar-kolom, dan bentuk karakter "
                       "(Hamming Distance pada bitmap karakter).")
            table = pd.DataFrame([{"Kolom": r["Kolom"], "Status": r["Status"], "Keterangan": r["Keterangan"]}
                                  for r in fres["rows"]])
            st.dataframe(table)

        if software:
            st.warning(f"Metadata gambar memuat tag Software: **{software}**. "
                       f"Screenshot asli biasanya tidak memiliki tag ini; periksa kemungkinan penyuntingan.")

        st.info("ℹ️ Suntingan yang meniru posisi, warna tinta, dan font aslinya dengan sempurna masih bisa lolos. "
                "Untuk transaksi penting, cocokkan juga dengan mutasi rekening.")
        st.caption("Hasil ini berupa indikasi kesesuaian dengan pola resi asli, bukan bukti hukum.")
    elif uploaded is None:
        st.info("👆 Pilih provider di sidebar lalu unggah gambar resi untuk diverifikasi.")

# ---------------------------------------------------------------- Tab 2: Database
with tab_db:
    st.subheader(f"📚 Referensi Resi Asli: {provider}")
    if not admin_ok():
        st.warning("Masukkan password admin untuk mengelola database referensi.")
    else:
        st.write("Unggah **hanya resi yang pasti asli**. Resi palsu yang ikut tersimpan akan merusak model. "
                 f"Gunakan variasi transaksi dan perangkat (minimal {MIN_REFS}, disarankan {RECOMMENDED_REFS}+).")

        files = st.file_uploader("Unggah resi asli (boleh banyak)", type=["png", "jpg", "jpeg"],
                                 accept_multiple_files=True,
                                 key=f"ref_upload_{st.session_state.uploader_key}")
        if st.button("💾 Simpan sebagai referensi asli", disabled=not files):
            for f in files:
                save_reference(provider, f)
            st.session_state.uploader_key += 1
            st.rerun()

        refs = list_references(provider)
        if refs:
            st.markdown(f"**{len(refs)} referensi tersimpan**")
            cols = st.columns(5)
            for i, path in enumerate(refs):
                with cols[i % 5]:
                    st.image(path)
                    if st.button("🗑️ Hapus", key=f"del_{path}"):
                        delete_reference(path)
                        st.rerun()

            mdl = build_model(provider)
            if mdl:
                st.markdown("---")
                a, b = st.columns(2)
                a.metric("Area tetap (dibandingkan)", f"{mdl['stable_fraction'] * 100:.1f} %")
                b.metric("Ambang batas hasil kalibrasi", f"{mdl['threshold'] * 100:.4f} %")
        else:
            st.info("Belum ada referensi untuk provider ini.")

        st.caption("Catatan: di Streamlit Cloud penyimpanan bersifat sementara. "
                   "Commit folder data/references ke GitHub agar referensi tidak hilang saat restart.")
