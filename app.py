import numpy as np
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)   # Vercel mencari variabel bernama "app" di file ini

# ---------------------------------------------------------------------------
# DATA HASIL EKSTRAKSI PARAMETER (dari notebook TA-03)
# harga dan jumlah disimpan sebagai tabel frekuensi: (nilai, banyaknya muncul).
# Mengambil acak dari tabel ini sama dengan resampling dari data asli.
# ---------------------------------------------------------------------------
DATA = {
    'In-store': {
        'trx_per_hari': 5.285584,          # rata-rata transaksi per hari
        'rev_bulanan_hist': 20756.99,  # rata-rata pendapatan bulanan historis
        'harga': [(5.0, 204), (6.5, 259), (8.0, 185), (9.5, 199), (11.0, 257), (12.5, 245), (14.0, 214), (15.5, 278), (17.0, 169), (18.5, 133), (20.0, 294), (21.5, 315), (23.0, 253), (24.5, 208), (26.0, 241), (27.5, 242), (29.0, 262), (30.5, 169), (32.0, 210), (33.5, 300), (35.0, 175), (36.5, 220), (38.0, 277), (39.5, 184), (41.0, 300)],
        'jumlah': [(1, 572), (2, 584), (3, 567), (4, 535), (5, 620), (6, 585), (7, 607), (8, 567), (9, 554), (10, 602)],
    },
    'Online': {
        'trx_per_hari': 5.452555,          # rata-rata transaksi per hari
        'rev_bulanan_hist': 21646.42,  # rata-rata pendapatan bulanan historis
        'harga': [(5.0, 217), (6.5, 248), (8.0, 189), (9.5, 198), (11.0, 238), (12.5, 268), (14.0, 216), (15.5, 272), (17.0, 162), (18.5, 151), (20.0, 322), (21.5, 295), (23.0, 257), (24.5, 217), (26.0, 259), (27.5, 248), (29.0, 281), (30.5, 184), (32.0, 254), (33.5, 359), (35.0, 180), (36.5, 221), (38.0, 259), (39.5, 194), (41.0, 287)],
        'jumlah': [(1, 554), (2, 559), (3, 639), (4, 594), (5, 589), (6, 614), (7, 604), (8, 638), (9, 572), (10, 613)],
    },
}

NAMA_KANAL = {'In-store': 'Toko fisik', 'Online': 'Online'}
HARI_PER_BULAN = 365.25 / 12   # rata-rata panjang satu bulan kalender (sekitar 30,44 hari)


def jalankan_simulasi(n_iter, margin, kenaikan, seed):
    """Simulasi Monte Carlo profit bulanan untuk kedua kanal.

    Untuk tiap skenario:
      1. jumlah transaksi sebulan ~ Poisson(transaksi per hari x 30,44)
      2. tiap transaksi: harga dan jumlah diambil acak dari data asli, lalu dikalikan
      3. pendapatan = jumlah semua transaksi, profit = margin x pendapatan
    """
    rng = np.random.default_rng(seed)

    # target profit sama untuk kedua kanal: rata-rata profit historis gabungan, dinaikkan sesuai input
    rata_hist = np.mean([d['rev_bulanan_hist'] for d in DATA.values()])
    target = rata_hist * margin * (1 + kenaikan)

    profit = {}
    for kanal, d in DATA.items():
        harga_nilai = np.array([h[0] for h in d['harga']])
        harga_p = np.array([h[1] for h in d['harga']], dtype=float)
        harga_p /= harga_p.sum()                       # ubah frekuensi menjadi peluang
        jumlah_nilai = np.array([j[0] for j in d['jumlah']])
        jumlah_p = np.array([j[1] for j in d['jumlah']], dtype=float)
        jumlah_p /= jumlah_p.sum()

        jumlah_trx = rng.poisson(d['trx_per_hari'] * HARI_PER_BULAN, n_iter)   # transaksi tiap skenario
        total = int(jumlah_trx.sum())
        harga = rng.choice(harga_nilai, size=total, p=harga_p)        # harga tiap transaksi
        jumlah = rng.choice(jumlah_nilai, size=total, p=jumlah_p)     # jumlah barang tiap transaksi
        nilai = harga * jumlah                                         # nilai tiap transaksi

        # jumlahkan nilai transaksi per skenario memakai jumlah kumulatif
        akhir = np.cumsum(jumlah_trx)
        awal = akhir - jumlah_trx
        kumulatif = np.concatenate(([0.0], np.cumsum(nilai)))
        profit[kanal] = (kumulatif[akhir] - kumulatif[awal]) * margin

    # histogram dengan batas kelas yang sama untuk kedua kanal
    semua = np.concatenate(list(profit.values()))
    batas = np.linspace(semua.min(), semua.max(), 41)

    hasil = []
    for kanal, p in profit.items():
        capai = float((p >= target).mean())
        bawah, atas = np.percentile(p, [2.5, 97.5])
        hasil.append({
            'kanal': kanal,
            'nama': NAMA_KANAL[kanal],
            'rata_rata': float(p.mean()),
            'bawah': float(bawah),
            'atas': float(atas),
            'peluang_capai': capai,
            'peluang_tidak': 1 - capai,
            'counts': np.histogram(p, bins=batas)[0].tolist(),
        })
    return {'target': float(target), 'bins': batas.tolist(), 'hasil': hasil, 'n_iter': n_iter}


@app.route('/api/simulasi', methods=['POST'])
def api_simulasi():
    """Menerima input dari form, menjalankan simulasi, mengirim hasil dalam bentuk JSON."""
    isi = request.get_json(silent=True) or {}
    try:
        n_iter = int(isi.get('n_iter', 5000))
        margin = float(isi.get('margin', 20)) / 100
        kenaikan = float(isi.get('kenaikan', 5)) / 100
        seed = int(isi.get('seed', 301240063))
    except (TypeError, ValueError):
        return jsonify({'error': 'Input harus berupa angka.'}), 400

    # validasi rentang input
    if not 1000 <= n_iter <= 20000:
        return jsonify({'error': 'Jumlah skenario harus antara 1.000 dan 20.000.'}), 400
    if not 0.01 <= margin <= 1:
        return jsonify({'error': 'Margin harus antara 1% dan 100%.'}), 400
    if not -0.5 <= kenaikan <= 1:
        return jsonify({'error': 'Kenaikan target harus antara -50% dan 100%.'}), 400
    if not 0 <= seed < 2 ** 32:
        return jsonify({'error': 'Seed harus bilangan bulat antara 0 dan 4294967295.'}), 400

    return jsonify(jalankan_simulasi(n_iter, margin, kenaikan, seed))


@app.route('/')
def beranda():
    return render_template_string(HALAMAN)


HALAMAN = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Simulasi Peluang Target Profit: Online vs Toko Fisik</title>
<style>
:root{--latar:#f6f7f9;--kartu:#ffffff;--teks:#1d2430;--redup:#5c6678;--garis:#d9dee6;--aksen:#1f5fbf;--bahaya:#d62828}
@media (prefers-color-scheme:dark){:root{--latar:#12161c;--kartu:#1b212b;--teks:#e8ecf2;--redup:#9aa6b8;--garis:#323b49;--aksen:#6aa5ff;--bahaya:#ff6b6b}}
*{box-sizing:border-box}
body{margin:0;background:var(--latar);color:var(--teks);font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55}
main{max-width:860px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.5rem;margin:0 0 4px}
.sub{color:var(--redup);margin:0 0 20px}
.kartu{background:var(--kartu);border:1px solid var(--garis);border-radius:12px;padding:16px;margin-bottom:16px}
form{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
label{display:flex;flex-direction:column;font-size:.85rem;color:var(--redup);gap:4px}
input{font:inherit;padding:8px 10px;border:1px solid var(--garis);border-radius:8px;background:var(--latar);color:var(--teks)}
button{grid-column:1/-1;font:inherit;font-weight:600;padding:10px 16px;border:0;border-radius:8px;background:var(--aksen);color:#fff;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.hasil{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.angka{font-size:2.2rem;font-weight:700;line-height:1.1}
.kecil{color:var(--redup);font-size:.85rem}
canvas{width:100%;height:320px;display:block}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{padding:8px;border-bottom:1px solid var(--garis);text-align:right}
th:first-child,td:first-child{text-align:left}
.galat{color:var(--bahaya);font-weight:600}
details summary{cursor:pointer;font-weight:600}
</style>
</head>
<body>
<main>
<h1>Simulasi Peluang Target Profit</h1>
<p class="sub">Kanal penjualan online vs toko fisik dengan metode Monte Carlo. TA-03 Pemodelan dan Simulasi, Wifa Saputra Mahendra (301240063).</p>

<section class="kartu">
<form id="form">
<label>Jumlah skenario<input type="number" id="n_iter" value="5000" min="1000" max="20000" step="500"></label>
<label>Margin profit (%)<input type="number" id="margin" value="20" min="1" max="100" step="1"></label>
<label>Kenaikan target (%)<input type="number" id="kenaikan" value="5" min="-50" max="100" step="1"></label>
<label>Seed (angka acak awal)<input type="number" id="seed" value="301240063" min="0" max="4294967295"></label>
<button type="submit" id="tombol">Jalankan simulasi</button>
</form>
<p class="kecil" id="pesan"></p>
</section>

<div id="keluaran" hidden>
<section class="hasil" id="kartu-hasil"></section>
<section class="kartu" style="margin-top:16px">
<strong>Distribusi profit bulanan hasil simulasi</strong>
<canvas id="grafik"></canvas>
<p class="kecil">Garis merah putus-putus adalah target profit. Area di sebelah kanan garis berarti target tercapai.</p>
</section>
<section class="kartu">
<table id="tabel"></table>
</section>
</div>

<section class="kartu">
<details>
<summary>Cara kerja model dan keterbatasan</summary>
<p><strong>Model.</strong> Untuk setiap skenario, jumlah transaksi dalam satu bulan (30,44 hari) dibangkitkan dari distribusi Poisson. Tiap transaksi diberi harga per unit dan jumlah barang yang diambil acak dari data asli, lalu dikalikan. Seluruh transaksi dijumlahkan menjadi pendapatan bulanan, dan profit = margin x pendapatan.</p>
<p><strong>Target.</strong> Rata-rata profit bulanan historis kedua kanal, dinaikkan sesuai persentase di form (default 5%). Target yang sama dipakai untuk kedua kanal agar perbandingannya adil. Peluang = jumlah skenario yang profitnya mencapai target dibagi total skenario.</p>
<p><strong>Data.</strong> Dataset Kaggle Retail Store Sales: Dirty for Data Cleaning, setelah dibersihkan (11.769 transaksi, Januari 2022 sampai Desember 2024).</p>
<p><strong>Keterbatasan.</strong> Dataset tidak memiliki kolom profit, sehingga margin adalah asumsi. Datanya sintetis sehingga kedua kanal hampir identik. Model menganggap tiap hari dan transaksi saling bebas, jadi naik turunnya penjualan antarbulan cenderung sedikit lebih kecil daripada data historis.</p>
</details>
</section>
</main>

<script>
const $ = id => document.getElementById(id);
const WARNA = ['#4f9cd6', '#3fae6a'];
const rupiah = v => v.toLocaleString('id-ID', {maximumFractionDigits: 0});
const persen = v => (v * 100).toLocaleString('id-ID', {minimumFractionDigits: 1, maximumFractionDigits: 1}) + '%';
let terakhir = null;

$('form').addEventListener('submit', async e => {
  e.preventDefault();
  $('tombol').disabled = true; $('pesan').textContent = 'Menjalankan simulasi...'; $('pesan').className = 'kecil';
  const isi = {n_iter: $('n_iter').value, margin: $('margin').value, kenaikan: $('kenaikan').value, seed: $('seed').value};
  try {
    const r = await fetch('/api/simulasi', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(isi)});
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Terjadi kesalahan.');
    terakhir = j; tampilkan(j); $('pesan').textContent = '';
  } catch (err) {
    $('pesan').textContent = err.message; $('pesan').className = 'galat';
  }
  $('tombol').disabled = false;
});

function tampilkan(j) {
  $('keluaran').hidden = false;
  $('kartu-hasil').innerHTML = j.hasil.map((h, i) => `
    <div class="kartu" style="border-top:4px solid ${WARNA[i]}">
      <div class="kecil">${h.nama}</div>
      <div class="angka">${persen(h.peluang_capai)}</div>
      <div class="kecil">peluang mencapai target profit (${rupiah(j.target)})</div>
    </div>`).join('');
  $('tabel').innerHTML = '<tr><th>Kanal</th><th>Rata-rata profit</th><th>Rentang 95%</th><th>Peluang capai</th><th>Peluang tidak capai</th></tr>' +
    j.hasil.map(h => `<tr><td>${h.nama}</td><td>${rupiah(h.rata_rata)}</td><td>${rupiah(h.bawah)} - ${rupiah(h.atas)}</td><td>${persen(h.peluang_capai)}</td><td>${persen(h.peluang_tidak)}</td></tr>`).join('');
  gambar(j);
}

function gambar(j) {
  const c = $('grafik'), dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = 320;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext('2d'); g.scale(dpr, dpr);
  const gaya = getComputedStyle(document.documentElement);
  const teks = gaya.getPropertyValue('--teks').trim(), garis = gaya.getPropertyValue('--garis').trim(), bahaya = gaya.getPropertyValue('--bahaya').trim();
  const m = {l: 52, r: 12, t: 12, b: 40}, pw = W - m.l - m.r, ph = H - m.t - m.b;
  const e = j.bins, lo = e[0], hi = e[e.length - 1];
  const maks = Math.max(...j.hasil.flatMap(h => h.counts)) * 1.08;
  const x = v => m.l + (v - lo) / (hi - lo) * pw, y = v => m.t + ph - v / maks * ph;
  g.font = '12px system-ui, sans-serif'; g.fillStyle = teks; g.strokeStyle = garis; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {            // garis bantu horizontal dan label frekuensi
    const v = maks * i / 4;
    g.beginPath(); g.moveTo(m.l, y(v)); g.lineTo(W - m.r, y(v)); g.stroke();
    g.textAlign = 'right'; g.fillText(Math.round(v), m.l - 6, y(v) + 4);
  }
  for (let i = 0; i <= 5; i++) {            // label sumbu x (label pertama dan terakhir digeser agar tidak terpotong)
    const v = lo + (hi - lo) * i / 5;
    g.textAlign = i === 0 ? 'left' : i === 5 ? 'right' : 'center';
    g.fillText(rupiah(v), i === 0 ? m.l : i === 5 ? W - m.r : x(v), H - 20);
  }
  g.textAlign = 'center'; g.fillText('Profit bulanan', m.l + pw / 2, H - 4);
  j.hasil.forEach((h, k) => {               // batang histogram tiap kanal
    g.fillStyle = WARNA[k]; g.globalAlpha = 0.55;
    h.counts.forEach((n, i) => g.fillRect(x(e[i]), y(n), Math.max(x(e[i + 1]) - x(e[i]) - 1, 1), y(0) - y(n)));
  });
  g.globalAlpha = 1;
  g.setLineDash([6, 4]); g.strokeStyle = bahaya; g.lineWidth = 2;   // garis target
  g.beginPath(); g.moveTo(x(j.target), m.t); g.lineTo(x(j.target), m.t + ph); g.stroke(); g.setLineDash([]);
  g.textAlign = 'left'; g.fillStyle = teks;                         // legenda
  j.hasil.forEach((h, k) => { g.fillStyle = WARNA[k]; g.fillRect(m.l + 8, m.t + 6 + k * 18, 12, 12); g.fillStyle = teks; g.fillText(h.nama, m.l + 26, m.t + 16 + k * 18); });
  g.fillStyle = bahaya; g.fillText('Target ' + rupiah(j.target), m.l + 8, m.t + 16 + j.hasil.length * 18);
}
window.addEventListener('resize', () => terakhir && gambar(terakhir));
</script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(debug=True)   # untuk dijalankan di komputer sendiri: python app.py
