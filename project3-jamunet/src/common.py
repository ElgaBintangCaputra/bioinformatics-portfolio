"""
common.py — Fungsi & konstanta bersama untuk semua script JamuNet.

Isi file ini:
1. Lokasi folder (data mentah, cache API, hasil tabel, hasil gambar).
2. Fungsi `cached_request()` : memanggil API web SEKALI saja, lalu menyimpan
   jawabannya ke folder data/cache. Kalau script dijalankan ulang, jawaban
   dibaca dari file cache (tidak perlu internet lagi) -> hasil REPRODUSIBEL.

Kenapa perlu cache?
- Database online (PubChem, ChEMBL, dll.) bisa berubah dari waktu ke waktu.
- Internet bisa putus, server bisa sibuk (rate limit).
- Dengan cache, siapa pun bisa menjalankan ulang analisis dan mendapat angka
  yang sama persis dengan yang ada di README.
"""

import hashlib
import json
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# 1. Lokasi folder proyek
# ---------------------------------------------------------------------------
# __file__ = lokasi file ini (src/common.py). parent = src/, parent.parent = root proyek
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"                 # data input kurasi manual
CACHE_DIR = DATA_DIR / "cache"           # jawaban API yang disimpan
RESULTS_DIR = ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"       # tabel hasil (CSV)
FIG_DIR = RESULTS_DIR / "figures"        # gambar hasil (PNG/HTML)

# Buat folder kalau belum ada (exist_ok=True -> tidak error kalau sudah ada)
for folder in (DATA_DIR, CACHE_DIR, TABLE_DIR, FIG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# Identitas penyakit: Type 2 diabetes mellitus di ontologi MONDO
# (dipakai Open Targets Platform). Bisa dicek di https://platform.opentargets.org
T2D_ID = "MONDO_0005148"

# Warna konsisten untuk tiap tanaman (dipakai di beberapa grafik)
PLANT_COLORS = {
    "Sambiloto": "#2ca02c",
    "Brotowali": "#8c564b",
    "Kayu manis": "#d62728",
    "Kunyit": "#ff9f1c",
    "Mahkota dewa": "#9467bd",
    "Salam": "#17becf",
}


# ---------------------------------------------------------------------------
# 2. Pemanggil API dengan cache
# ---------------------------------------------------------------------------
def _cache_path(namespace: str, key: str) -> Path:
    """Buat nama file cache unik dari isi permintaan (pakai hash MD5)."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
    folder = CACHE_DIR / namespace
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{digest}.json"


def cached_request(namespace, url, params=None, json_body=None, method="GET",
                   data=None, retries=4, pause=0.25, timeout=60):
    """
    Kirim permintaan HTTP ke API dan simpan jawabannya (JSON) di cache.

    Parameter
    ---------
    namespace : nama subfolder cache, mis. "pubchem", "chembl"
    url       : alamat API
    params    : parameter query string (?a=1&b=2) dalam bentuk dict
    json_body : isi body JSON (untuk POST, mis. GraphQL / g:Profiler)
    data      : isi body form (untuk POST biasa, mis. STRING)
    method    : "GET" atau "POST"

    Hasil
    -----
    dict/list hasil decode JSON, atau None kalau server menjawab 404
    (artinya: data memang tidak ada, mis. senyawa tidak ditemukan).
    """
    key = json.dumps([method, url, params, json_body, data], sort_keys=True)
    path = _cache_path(namespace, key)

    # (a) Sudah pernah diminta? baca dari file saja.
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"]

    # (b) Belum ada di cache -> minta ke server (perlu internet).
    last_error = None
    for attempt in range(retries):
        try:
            if method == "GET":
                r = requests.get(url, params=params, timeout=timeout)
            else:
                r = requests.post(url, params=params, json=json_body,
                                  data=data, timeout=timeout)
            if r.status_code == 404:          # data tidak ada -> simpan None
                result = None
            elif r.status_code in (429, 500, 502, 503, 504):
                # server sibuk -> tunggu lalu coba lagi (makin lama makin panjang)
                last_error = f"HTTP {r.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            else:
                r.raise_for_status()
                result = r.json()
            # simpan ke cache bersama info permintaan (supaya bisa dilacak)
            path.write_text(json.dumps({"request": json.loads(key),
                                        "response": result}),
                            encoding="utf-8")
            time.sleep(pause)                 # sopan: jangan membanjiri server
            return result
        except requests.RequestException as e:
            last_error = str(e)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Gagal memanggil {url}: {last_error}")
