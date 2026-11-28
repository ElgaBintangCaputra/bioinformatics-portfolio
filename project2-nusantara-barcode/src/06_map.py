"""
06_map.py
=========
Langkah 6: Peta Sebaran Geografis Sampel DNA Barcode (Plotly Interaktif + Grafik Negara).

Skrip ini memvisualisasikan dari negara mana saja sampel COI diperoleh di GenBank:
- Jika catatan GenBank mencantumkan koordinat tepat (`lat_lon`), dipakai koordinat asli (`exact_lat_lon`).
- Jika hanya mencantumkan negara (`country`), kita petakan ke titik tengah negara (`country_centroid`)
  khusus untuk visualisasi agregat sambil tetap menandai perbedaannya secara jujur.
"""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from common import get_paths, load_config, setup_logger

COUNTRY_CENTROIDS = {
    "Indonesia": (-2.5, 118.0),
    "Malaysia": (4.2, 102.0),
    "Philippines": (12.8, 121.8),
    "Australia": (-25.3, 133.8),
    "India": (20.6, 78.9),
    "China": (35.9, 104.2),
    "Taiwan": (23.7, 121.0),
    "Japan": (36.2, 138.3),
    "Viet Nam": (14.1, 108.3),
    "Vietnam": (14.1, 108.3),
    "Thailand": (15.8, 100.9),
    "Singapore": (1.35, 103.82),
    "Papua New Guinea": (-6.3, 143.9),
    "Saudi Arabia": (23.9, 45.1),
    "South Africa": (-30.6, 22.9),
    "USA": (37.1, -95.7),
    "Sri Lanka": (7.87, 80.77),
    "Bangladesh": (23.7, 90.4),
    "Pakistan": (30.4, 69.3),
    "Iran": (32.4, 53.7),
    "Oman": (21.5, 55.9),
    "Brazil": (-14.2, -51.9),
    "Mexico": (23.6, -102.5),
    "New Caledonia": (-21.3, 165.6),
    "French Polynesia": (-17.7, -149.4),
    "Fiji": (-17.7, 178.1),
    "South Korea": (35.9, 127.8),
    "Myanmar": (21.9, 95.9),
}


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    logger = setup_logger(paths["log_file"])
    logger.info("== Langkah 6: Peta Sebaran Geografis Sampel ==")

    meta = pd.read_csv(paths["tables_dir"] / "02_aligned_metadata.csv")
    lats, lons, src_types = [], [], []
    for _, r in meta.iterrows():
        if pd.notna(r["lat"]) and pd.notna(r["lon"]):
            lats.append(float(r["lat"]))
            lons.append(float(r["lon"]))
            src_types.append("Koordinat asli GenBank")
        elif r["country"] in COUNTRY_CENTROIDS:
            la, lo = COUNTRY_CENTROIDS[r["country"]]
            lats.append(la)
            lons.append(lo)
            src_types.append("Pusat negara (tanpa lat_lon)")
        else:
            lats.append(None)
            lons.append(None)
            src_types.append("Tidak terpetakan")

    meta["plot_lat"] = lats
    meta["plot_lon"] = lons
    meta["coord_source"] = src_types
    meta.to_csv(paths["tables_dir"] / "06_metadata_with_coords.csv", index=False)

    mapped = meta.dropna(subset=["plot_lat", "plot_lon"]).copy()
    agg = (
        mapped.groupby(["country", "family", "plot_lat", "plot_lon", "coord_source"])
        .agg(jumlah_sekuens=("accession", "count"), jumlah_spesies=("species", "nunique"))
        .reset_index()
    )

    fig = px.scatter_geo(
        agg,
        lat="plot_lat",
        lon="plot_lon",
        color="family",
        size="jumlah_sekuens",
        symbol="coord_source",
        hover_name="country",
        hover_data=["family", "jumlah_sekuens", "jumlah_spesies", "coord_source"],
        color_discrete_map={"Lutjanidae": "#d62728", "Epinephelidae": "#2ca02c", "Scombridae": "#1f77b4"},
        title="Sebaran Geografis Sampel COI NusantaraBarcode (Fokus Indo-Pasifik)",
        projection="natural earth",
    )
    fig.update_geos(center={"lat": 0, "lon": 118}, projection_scale=2.2)
    fig.write_html(str(paths["figures_dir"] / "06_sampling_map.html"))

    # Grafik batang negara asal sampel (statis untuk README)
    country_counts = meta["country"].value_counts().head(12).iloc[::-1]
    sns.set_theme(style="whitegrid")
    fig_bar, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#d62728" if c == "Indonesia" else "#1f77b4" for c in country_counts.index]
    country_counts.plot(kind="barh", color=colors, ax=ax, alpha=0.9)
    ax.set_title("Asal Negara Sampel COI dalam Dataset (Merah = Indonesia)", fontweight="bold")
    ax.set_xlabel("Jumlah Sekuens")
    plt.tight_layout()
    plt.savefig(paths["figures_dir"] / "06_country_distribution.png", dpi=200)
    plt.close()
    logger.info(f"Selesai Langkah 6 ({len(mapped)}/{len(meta)} sekuens terpetakan).")


if __name__ == "__main__":
    main()
