# 🧬 Panduan Opsional: Validasi Lanjutan dengan Molecular Docking (AutoDock Vina di Google Colab)

> [!NOTE]
> Bagian ini **opsional** sebagai nilai tambah wawancara PKL. Project utama **JamuNet** (Langkah 1–7) menggunakan *network pharmacology* dan kemiripan kimia untuk mempersempit kandidat dari puluhan senyawa menjadi Top 3–5 kandidat. Setelah punya kandidat teratas, langkah selanjutnya di laboratorium komputasi adalah **structure-based validation** dengan **Molecular Docking**!

---

## Kenapa perlu Molecular Docking setelah Network Pharmacology?

1. **Network pharmacology** menjawab: *"Senyawa mana yang punya catatan bioaktivitas terhadap protein terkait Diabetes Tipe 2?"*
2. **Molecular docking** menjawab: *"Bagaimana bentuk pasnya (pose ikatan 3D) dan perkiraan energi ikatan (kcal/mol) senyawa tersebut ketika masuk ke kantong aktif protein target?"*

---

## Alur Kerja Standar (Yang Wajib Diketahui Saat Wawancara)

1. **Ambil struktur 3D protein target** dari **RCSB PDB** (https://www.rcsb.org/), misalnya protein *hub* hasil analisis JamuNet (contoh: `PPARG`, `PTGS2`, `AKT1`, `EGFR`, atau `DPP4`). Pilih struktur beresolusi tinggi (< 2.5 Å) yang **sudah berisi ligan ko-kristal** (obat/inhibitor asli di dalam kantong ikatan).
2. **Siapkan protein & ligan** (bersihkan molekul air yang tidak berperan, tambahkan atom hidrogen polar, atur protonasi pH 7.4, ubah format ke `.pdbqt` menggunakan `meeko` / OpenBabel).
3. **WAJIB: Validasi *Redocking* (Kontrol Positif Struktur)!**
   - Keluarkan ligan ko-kristal asli dari file PDB, lalu lakukan docking ulang (*redocking*) ke kantong ikatan yang sama.
   - Hitung **RMSD** (*Root Mean Square Deviation*) antara posisi hasil docking dan posisi kristal aslinya.
   - **Standar diterima:** **RMSD < 2.0 Å**. Jika RMSD > 2.0 Å, parameter kotak (*grid box*) atau metode preparasi belum valid!
4. **Docking kandidat senyawa jamu + obat kontrol** (mis. pioglitazone untuk PPARG atau sitagliptin untuk DPP4) pada kotak (*grid box*) yang sudah tervalidasi.
5. **Analisis interaksi 2D/3D** (ikatan hidrogen, interaksi hidrofobik) dan pertimbangkan **Molecular Dynamics (OpenMM)** 10–50 ns untuk menguji kestabilan kompleks.

---

## Contoh Skrip Singkat untuk Google Colab (AutoDock Vina + Meeko)

```python
# Di Google Colab:
# !pip install -q vina meeko rdkit requests

import requests
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy
from vina import Vina

# 1. Unduh struktur protein contoh dari RCSB PDB (mis. 2PRG untuk PPARG)
pdb_id = "2PRG"
pdb_txt = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb").text
open(f"{pdb_id}.pdb", "w").write(pdb_txt)

# 2. Konversi SMILES kandidat jamu (contoh: curcumin / naringenin / berberine) ke 3D PDBQT
smiles = "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c12"  # naringenin
mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol)

prep = MoleculePreparation()
mol_setups = prep.prepare(mol)
pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
open("ligand.pdbqt", "w").write(pdbqt_str)

# 3. Jalankan AutoDock Vina setelah receptor.pdbqt disiapkan & grid box divalidasi (redocking RMSD < 2 A)
# v = Vina(sf_name="vina")
# v.set_receptor("receptor.pdbqt")
# v.set_ligand_from_file("ligand.pdbqt")
# v.compute_vina_maps(center=[x, y, z], box_size=[20, 20, 20])
# v.dock(exhaustiveness=8, n_poses=9)
# print("Afinitas terbaik (kcal/mol):", v.energies()[0][0])
```

> [!WARNING]
> Skor docking (*kcal/mol*) hanyalah perkiraan kasar untuk pemeringkatan awal (*virtual screening*), bukan nilai $K_d$ atau $IC_{50}$ laboratorium yang sebenarnya. Selalu bandingkan dengan **kontrol positif** (obat resmi pada target yang sama) dan hasil **redocking**!
