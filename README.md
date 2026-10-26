# TA-03 Pemodelan dan Simulasi: Aplikasi Web

Simulasi Peluang Target Profit Kanal Penjualan Online vs Toko Fisik (Monte Carlo).

Nama : Wifa Saputra Mahendra
NIM  : 301240063

## Isi folder

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Flask (halaman web dan simulasi) |
| `requirements.txt` | Daftar library yang dibutuhkan (Flask dan NumPy) |
| `README.md` | Panduan ini |

Parameter simulasi (rata-rata transaksi per hari, tabel frekuensi harga dan jumlah) sudah tertanam di `app.py`, hasil ekstraksi dari notebook TA-03. Jadi aplikasi tidak membutuhkan file CSV saat berjalan.

## Menjalankan di komputer sendiri

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Buka http://127.0.0.1:5000 di browser.

## Deploy ke Vercel

Vercel mendeteksi Flask otomatis jika ada variabel `app` di `app.py` pada folder utama.

1. Buat repositori baru di GitHub, lalu unggah tiga file di atas ke folder utama repositori (jangan di dalam subfolder).
2. Login ke vercel.com (bisa memakai akun GitHub).
3. Klik **Add New**, lalu **Project**, pilih repositori tadi, dan klik **Import**.
4. Biarkan pengaturan bawaan (Framework terdeteksi sebagai Flask, tanpa build command), lalu klik **Deploy**.
5. Setelah selesai, Vercel memberi tautan seperti `nama-proyek.vercel.app`. Tautan itu adalah hasil deploy yang dikumpulkan.

Alternatif lewat Vercel CLI: jalankan `npm i -g vercel`, lalu `vercel` (preview) dan `vercel --prod` (produksi) dari folder aplikasi.

## Catatan

- Streamlit tidak dipakai karena tidak berjalan di Vercel. Aplikasi ini memakai Flask.
- Hasil di aplikasi sedikit berbeda dengan notebook karena deretan angka acaknya berbeda. Secara statistik hasilnya setara: peluang toko fisik sekitar 21% dan online sekitar 37%.
- Parameter bisa diubah dari form: jumlah skenario (1.000 sampai 20.000), margin, kenaikan target, dan seed.
