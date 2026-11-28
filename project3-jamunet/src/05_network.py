"""
05_network.py
=============
Langkah 5: Membangun Jaringan Farmakologi (Tanaman–Senyawa–Target–Diabetes) & PPI STRING.

Dua jaringan yang dibangun:
1. Jaringan Bipartit/Tripartit: Tanaman Jamu -> Senyawa Aktif -> Protein Target (irisan T2D).
2. Jaringan Interaksi Protein-Protein (PPI) dari API STRING-DB (Homo sapiens,
   combined_score >= 0.7 = kepercayaan tinggi) untuk semua target yang beririsan dengan T2D.

Analisis Teori Graf (Network Biology):
- Degree centrality: berapa banyak mitra interaksi yang dimiliki satu protein.
- Betweenness centrality: seberapa sering protein menjadi "jembatan" terpendek antar jalur.
- Closeness centrality: seberapa cepat sinyal dari protein ini menjangkau seluruh jaringan.
- Deteksi komunitas (Greedy Modularity) untuk melihat modul biologis yang terbentuk.
- Output: gambar statis PNG + peta jaringan interaktif HTML (pyvis) untuk dashboard!
"""

import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import community
import pandas as pd
from pyvis.network import Network

from common import FIG_DIR, PLANT_COLORS, TABLE_DIR, cached_request

STRING_API = "https://string-db.org/api/json/network"


def fetch_string_ppi(genes: list[str], min_score: int = 700) -> pd.DataFrame:
    """Ambil jaringan PPI manusia dari STRING API dengan cache lokal."""
    if not genes:
        return pd.DataFrame(columns=["gene_a", "gene_b", "score"])
    resp = cached_request(
        "string",
        STRING_API,
        method="POST",
        data={
            "identifiers": "\r".join(sorted(genes)),
            "species": 9606,
            "required_score": min_score,
            "caller_identity": "jamunet_smk_portfolio",
        },
    )
    if not resp:
        return pd.DataFrame(columns=["gene_a", "gene_b", "score"])
    rows = []
    seen = set()
    for edge in resp:
        a = edge.get("preferredName_A")
        b = edge.get("preferredName_B")
        if not a or not b or a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in seen:
            continue
        seen.add(pair)
        rows.append({"gene_a": pair[0], "gene_b": pair[1], "score": float(edge["score"])})
    return pd.DataFrame(rows)


def buat_html_pyvis(G_tri: nx.Graph, out_html: str) -> None:
    """Bangun visualisasi HTML interaktif jaringan Tanaman-Senyawa-Target-T2D."""
    net = Network(height="620px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-3500, central_gravity=0.35, spring_length=130)

    node_colors = {
        "plant": "#2ca02c",
        "compound": "#ff9f1c",
        "target": "#1f77b4",
        "disease": "#d62728",
    }
    node_shapes = {
        "plant": "box",
        "compound": "dot",
        "target": "diamond",
        "disease": "star",
    }
    for n, attrs in G_tri.nodes(data=True):
        ntype = attrs.get("ntype", "target")
        deg = G_tri.degree(n)
        net.add_node(
            n,
            label=n,
            color=node_colors.get(ntype, "#888888"),
            shape=node_shapes.get(ntype, "dot"),
            size=14 + min(deg * 2.2, 26),
            title=f"{n} ({ntype}, derajat={deg})",
        )
    for u, v in G_tri.edges():
        net.add_edge(u, v, color="#bbbbbb")
    net.save_graph(out_html)


def main():
    print("== Langkah 5: Network Pharmacology & STRING PPI ==")
    overlap_df = pd.read_csv(TABLE_DIR / "04_overlapping_targets.csv")
    ct_df = pd.read_csv(TABLE_DIR / "03_compound_targets.csv")
    plant_df = pd.read_csv(TABLE_DIR / "01_plant_compound_table.csv")

    t2d_targets = sorted(overlap_df["gene_symbol"].unique())
    ct_t2d = ct_df[ct_df["gene_symbol"].isin(t2d_targets)].copy()

    # --- 1. Jaringan Tanaman -> Senyawa -> Target -> T2D ---
    G_tri = nx.Graph()
    G_tri.add_node("Diabetes Tipe 2", ntype="disease")

    active_compounds = set(ct_t2d["compound"].unique())
    for _, r in plant_df[plant_df["compound"].isin(active_compounds)].iterrows():
        G_tri.add_node(r["plant_id"], ntype="plant")
        G_tri.add_node(r["compound"], ntype="compound")
        G_tri.add_edge(r["plant_id"], r["compound"])

    for _, r in ct_t2d.iterrows():
        G_tri.add_node(r["gene_symbol"], ntype="target")
        G_tri.add_edge(r["compound"], r["gene_symbol"])
        G_tri.add_edge(r["gene_symbol"], "Diabetes Tipe 2")

    buat_html_pyvis(G_tri, str(FIG_DIR / "05_jamunet_interactive.html"))

    # --- 2. Jaringan PPI STRING antar target irisan ---
    ppi_df = fetch_string_ppi(t2d_targets, min_score=400)
    ppi_df.to_csv(TABLE_DIR / "05_string_ppi_edges.csv", index=False)

    G_ppi = nx.Graph()
    for g in t2d_targets:
        G_ppi.add_node(g)
    for _, r in ppi_df.iterrows():
        G_ppi.add_edge(r["gene_a"], r["gene_b"], weight=r["score"])

    deg_c = nx.degree_centrality(G_ppi)
    bet_c = nx.betweenness_centrality(G_ppi)
    clo_c = nx.closeness_centrality(G_ppi)
    deg_raw = dict(G_ppi.degree())

    # Deteksi komunitas
    comms = list(community.greedy_modularity_communities(G_ppi))
    comm_map = {}
    for cid, cset in enumerate(comms, start=1):
        for node in cset:
            comm_map[node] = cid

    hub_df = pd.DataFrame([
        {
            "gene_symbol": g,
            "degree": deg_raw[g],
            "degree_centrality": round(deg_c[g], 4),
            "betweenness_centrality": round(bet_c[g], 4),
            "closeness_centrality": round(clo_c[g], 4),
            "community": comm_map.get(g, 0),
        }
        for g in t2d_targets
    ]).merge(overlap_df[["gene_symbol", "n_compounds", "ot_score"]], on="gene_symbol", how="left")
    hub_df = hub_df.sort_values(["degree", "betweenness_centrality"], ascending=[False, False])
    hub_df.to_csv(TABLE_DIR / "05_hub_genes.csv", index=False)

    print(f"  Jaringan Tripartit: {G_tri.number_of_nodes()} node, {G_tri.number_of_edges()} edge")
    print(f"  Jaringan STRING PPI (score >= 0.4): {G_ppi.number_of_nodes()} node, {G_ppi.number_of_edges()} edge, {len(comms)} komunitas")
    print("  Top 10 Hub Genes:")
    print(hub_df[["gene_symbol", "degree", "betweenness_centrality", "n_compounds", "ot_score"]].head(10).to_string(index=False))

    # --- Plot Statis: (Kiri) Jaringan Tripartit, (Kanan) PPI Hub Genes ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    # Kiri: Layout berlapis Tanaman | Senyawa | Target | Penyakit
    pos_tri = {}
    plants = sorted([n for n, d in G_tri.nodes(data=True) if d["ntype"] == "plant"])
    comps = sorted([n for n, d in G_tri.nodes(data=True) if d["ntype"] == "compound"])
    # Ambil top 25 target berdasarkan jumlah senyawa agar label terbaca rapi
    top_targets = overlap_df.head(25)["gene_symbol"].tolist()
    sub_nodes = set(plants) | set(comps) | set(top_targets) | {"Diabetes Tipe 2"}
    G_sub = G_tri.subgraph(sub_nodes)
    comps_sub = [c for c in comps if G_sub.degree(c) > 1]

    for i, p in enumerate(plants):
        pos_tri[p] = (0.0, 1.0 - i / max(1, len(plants) - 1))
    for i, c in enumerate(comps_sub):
        pos_tri[c] = (1.0, 1.0 - i / max(1, len(comps_sub) - 1))
    for i, t in enumerate(top_targets):
        pos_tri[t] = (2.0, 1.0 - i / max(1, len(top_targets) - 1))
    pos_tri["Diabetes Tipe 2"] = (2.8, 0.5)

    G_draw = G_sub.subgraph(pos_tri.keys())
    col_map = {"plant": "#2ca02c", "compound": "#ff9f1c", "target": "#1f77b4", "disease": "#d62728"}
    node_cols = [col_map[G_draw.nodes[n]["ntype"]] for n in G_draw.nodes()]
    nx.draw_networkx_edges(G_draw, pos_tri, ax=axes[0], alpha=0.28, edge_color="#888888")
    nx.draw_networkx_nodes(G_draw, pos_tri, ax=axes[0], node_color=node_cols, node_size=140, alpha=0.92)
    nx.draw_networkx_labels(G_draw, pos_tri, ax=axes[0], font_size=6.8, font_weight="bold")
    axes[0].set_title("Jaringan Tanaman Jamu → Senyawa → Top 25 Target → T2D", fontweight="bold")
    axes[0].axis("off")

    # Kanan: Subgraf PPI terhubung (buang node terisolasi agar visual jelas)
    connected = [n for n in G_ppi.nodes() if G_ppi.degree(n) > 0]
    G_conn = G_ppi.subgraph(connected)
    pos_ppi = nx.spring_layout(G_conn, seed=42, k=0.55)
    sizes = [120 + deg_raw[n] * 45 for n in G_conn.nodes()]
    comms_col = [comm_map[n] for n in G_conn.nodes()]
    nx.draw_networkx_edges(G_conn, pos_ppi, ax=axes[1], alpha=0.35, edge_color="#777777")
    nx.draw_networkx_nodes(G_conn, pos_ppi, ax=axes[1], node_size=sizes, node_color=comms_col, cmap=plt.cm.Set2, alpha=0.92)
    # Beri label hanya pada top 20 hub agar tidak menumpuk
    top20 = set(hub_df.head(20)["gene_symbol"])
    labels_ppi = {n: n for n in G_conn.nodes() if n in top20}
    nx.draw_networkx_labels(G_conn, pos_ppi, labels=labels_ppi, ax=axes[1], font_size=8, font_weight="bold")
    axes[1].set_title("Jaringan PPI STRING Target Irisan (Ukuran = Derajat Hub)", fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_network_overview.png", dpi=220)
    plt.close()
    print("Selesai Langkah 5.")


if __name__ == "__main__":
    main()
