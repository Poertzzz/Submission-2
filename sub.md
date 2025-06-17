# Laporan Proyek Machine Learning - Nama Anda

---

## Project Overview

Dalam industri perbankan yang serba cepat, calon nasabah sering kewalahan dengan segudang pilihan bank dan layanan. Kebanjiran informasi ini, atau choice overload, bisa menurunkan kepercayaan diri dan kepuasan mereka dalam memilih.

Proyek ini penting karena sistem rekomendasi bank yang efektif akan menguntungkan baik nasabah maupun bank. Bagi nasabah, 
sistem ini menyederhanakan pencarian, membantu mereka menemukan bank yang relevan secara personal, meningkatkan pengalaman, dan mengungkap layanan baru. 
Untuk bank, algoritma rekomendasi dapat meningkatkan keterlibatan, loyalitas, dan akuisisi nasabah baru, pada akhirnya meningkatkan metrik bisnis seperti jumlah nasabah dan retensi pengguna.

Penelitian menunjukkan bahwa ulasan pelanggan adalah fondasi kuat untuk sistem rekomendasi, karena merefleksikan pengalaman dan preferensi asli pengguna (Adomavicius & Tuzhilin, 2005). Proyek ini memanfaatkan "Banks Reviews Customer Dataset", kumpulan ulasan dan peringkat pelanggan bank yang ekstensif. Dataset kredibel ini memberikan wawasan mendalam tentang kepuasan pelanggan, tren regional, dan faktor pembentuk pengalaman perbankan, memungkinkan kami mengembangkan model rekomendasi yang representatif.

Referensi:

Adomavicius, G., & Tuzhilin, A. (2005). Toward the Next Generation of Recommender Systems: A Survey of the State-of-the-Art and Possible Extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749.
"Banks Reviews Customer Dataset". (n.d.). Sumber internal atau platform data set.

---

## Business Understanding

### Problem Statements

Dalam industri perbankan yang semakin padat, calon nasabah dihadapkan pada **jumlah pilihan bank dan produk yang sangat banyak**, seringkali menyebabkan kebingungan dan kesulitan dalam membuat keputusan yang tepat. 
Fenomena ini, yang dikenal sebagai *choice overload*, dapat mengurangi kepercayaan diri nasabah dan potensi kepuasan mereka. Secara spesifik, masalah yang ingin diatasi proyek ini adalah:

1.  **Kesulitan Calon Nasabah dalam Menemukan Bank yang Sesuai:** Dengan ribuan bank dan beragam layanan yang tersedia, calon nasabah kesulitan untuk menyaring informasi dan mengidentifikasi bank yang paling cocok dengan kebutuhan, preferensi, dan profil risiko mereka. Ini membuang waktu dan dapat berujung pada pilihan yang kurang optimal.
2.  **Kurangnya Personalisasi dalam Penawaran Bank:** Saat ini, banyak penawaran bank bersifat umum, tidak mempertimbangkan preferensi individu nasabah. Hal ini menyebabkan rendahnya relevansi penawaran dan kurangnya keterlibatan nasabah terhadap produk atau layanan bank.

### Goals

Berdasarkan pernyataan masalah di atas, tujuan utama proyek ini adalah mengembangkan sistem rekomendasi yang dapat mengatasi tantangan tersebut dan memberikan nilai tambah bagi calon nasabah maupun industri perbankan. Tujuan yang ingin dicapai adalah:

1.  **Memudahkan Calon Nasabah dalam Mengidentifikasi Bank Ideal:** Menyediakan sistem yang secara otomatis dapat merekomendasikan bank-bank yang paling sesuai dengan preferensi individu calon nasabah, berdasarkan pola ulasan dan rating dari pengguna lain.
2.  **Meningkatkan Relevansi Penawaran Bank Melalui Personalisasi:** Mengembangkan model yang mampu memberikan rekomendasi bank yang sangat personal, sehingga calon nasabah merasa lebih terhubung dengan pilihan yang disajikan dan meningkatkan potensi akuisisi.

### Solution Approach

Untuk mencapai tujuan yang telah ditetapkan, proyek ini akan mengeksplorasi dan membandingkan kinerja tiga pendekatan *machine learning* yang berbeda dalam membangun sistem rekomendasi berbasis *Item-Based Collaborative Filtering*. Evaluasi kinerja model akan menggunakan **Mean Absolute Error (MAE)** untuk mengukur akurasi prediksi.

#### Solution Statements

1.  **Model 1 – User-Based Collaborative Filtering (Cosine Similarity):**
    * **Pendekatan:** Model ini akan mengidentifikasi pengguna dengan preferensi serupa (tetangga terdekat) berdasarkan rating ulasan mereka terhadap bank yang sama. Setelah itu, rekomendasi bank akan diberikan kepada pengguna target berdasarkan bank-bank yang disukai oleh tetangga terdekatnya. Metode *Cosine Similarity* akan digunakan untuk mengukur kemiripan antar pengguna.
    * **Dasar Pemikiran:** Pendekatan ini relatif sederhana dan intuitif, memanfaatkan "wisdom of the crowd" untuk menemukan item yang relevan.

2.  **Model 2 – SVD (Singular Value Decomposition) Matrix Factorization Manual:**
    * **Pendekatan:** SVD adalah teknik dekomposisi matriks yang akan digunakan untuk mengurangi dimensi matriks ulasan pengguna-bank yang jarang (sparse) menjadi representasi laten (faktor tersembunyi). Dengan demikian, preferensi pengguna dan karakteristik bank dapat diwakili oleh sejumlah kecil faktor laten, yang kemudian digunakan untuk memprediksi rating yang hilang.
    * **Dasar Pemikiran:** SVD efektif dalam menangani masalah *sparsity* dan menemukan pola tersembunyi dalam data, seringkali menghasilkan prediksi yang lebih akurat dibandingkan metode berbasis tetangga.

3.  **Model 3 – Neural Collaborative Filtering (NCF):**
    * **Pendekatan:** NCF merupakan pendekatan berbasis *deep learning* yang menggantikan fungsi interaksi matriks tradisional dengan arsitektur jaringan saraf tiruan (neural network). Model ini memungkinkan pemodelan interaksi yang lebih kompleks dan non-linear antara pengguna dan item, berpotensi menangkap pola preferensi yang lebih kaya.
    * **Dasar Pemikiran:** NCF menawarkan fleksibilitas yang lebih besar dalam memodelkan interaksi pengguna-item, berpotensi melampaui linearitas metode tradisional dan mencapai akurasi yang lebih tinggi, terutama dengan dataset yang besar dan kompleks.

Perbandingan ketiga model ini akan memberikan pemahaman mendalam tentang pendekatan mana yang paling efektif dalam merekomendasikan bank di dataset "Banks Reviews Customer Dataset" berdasarkan metrik MAE, serta memberikan dasar untuk memilih model terbaik untuk implementasi.

---

## Data Understanding

Bagian ini membahas informasi mendetail mengenai dataset yang digunakan dalam proyek sistem rekomendasi bank ini. Proyek ini menggunakan **"Banks Reviews Customer Dataset"** yang dapat diunduh melalui platform Kaggle: [https://www.kaggle.com/datasets/dhavalrupapara/banks-customer-reviews-dataset](https://www.kaggle.com/datasets/dhavalrupapara/banks-customer-reviews-dataset). 

Dataset ini merupakan kumpulan ulasan dan peringkat pelanggan bank yang luas, berisi lebih dari **1000 data ulasan dan rating** yang dibuat oleh pengguna untuk berbagai bank. Dataset ini berfungsi sebagai aset berharga bagi ilmuwan data, peneliti, dan profesional perbankan, menyediakan pandangan komprehensif mengenai kepuasan pelanggan, tren perbankan regional, serta faktor-faktor yang membentuk pengalaman perbankan melalui umpan balik pelanggan asli.

### Variabel dalam Dataset

Dataset "Banks Reviews Customer Dataset" umumnya terdiri dari beberapa variabel kunci yang merepresentasikan interaksi antara pelanggan dan bank. Meskipun struktur kolom spesifik mungkin bervariasi, berdasarkan deskripsi, variabel-variabel yang diharapkan ada dan relevan untuk analisis adalah sebagai berikut:

* **`User_ID`**: Identifier unik untuk setiap pelanggan yang memberikan ulasan. Variabel ini krusial untuk mengidentifikasi perilaku dan preferensi pengguna.
* **`Bank_Name`**: Nama bank yang diulas oleh pelanggan. Ini adalah item yang akan direkomendasikan dalam sistem.
* **`Rating`**: Peringkat numerik yang diberikan oleh pelanggan kepada bank (misalnya, skala 1-5). Variabel ini adalah target prediksi utama untuk model rekomendasi.
* **`Review_Text`**: Teks ulasan yang ditulis oleh pelanggan. Meskipun tidak langsung digunakan dalam model *collaborative filtering* berbasis rating, ini bisa menjadi sumber insight tambahan untuk analisis sentimen atau *Natural Language Processing* (NLP) di masa depan.
* **`Date`**: Tanggal ulasan dibuat, berguna untuk analisis tren temporal.
* **`Location/Region`** (Opsional, jika tersedia): Lokasi geografis bank atau pelanggan, yang dapat memberikan wawasan tentang tren regional.


### Struktur Data dan Eksplorasi Awal

Untuk mendapatkan pemahaman yang lebih mendalam tentang data, beberapa langkah eksplorasi awal akan dilakukan:

#### 1. Analisis Deskriptif

Analisis deskriptif akan memberikan gambaran umum tentang statistik kunci dari dataset, seperti jumlah total ulasan, rentang rating, dan distribusi data pada kolom-kolom numerik. Ini membantu mengidentifikasi karakteristik dasar data.
Berikut merupakan info dari dataset bank reviews : 
| # | Column               | Non-Null Count | Dtype   | Keterangan                                                                   |
|---|----------------------|----------------|---------|------------------------------------------------------------------------------|
| 0 | **author** | 996 non-null   | `object` | Nama pengguna yang menulis ulasan. Terdapat 4 nilai yang hilang.            |
| 1 | **date** | 1000 non-null  | `object` | Tanggal ulasan dipublikasikan.                                                |
| 2 | **address** | 1000 non-null  | `object` | Alamat atau lokasi terkait ulasan (kemungkinan lokasi cabang bank).        |
| 3 | **bank** | 1000 non-null  | `object` | Nama bank yang diulas. Ini akan menjadi `Bank_Name` yang kita gunakan.      |
| 4 | **rating** | 1000 non-null  | `float64`| Rating numerik yang diberikan oleh pengguna untuk bank tersebut (skala 1-5). |
| 5 | **review_title_by_user** | 1000 non-null  | `object` | Judul ulasan yang diberikan oleh pengguna.                                    |
| 6 | **review** | 1000 non-null  | `object` | Isi teks ulasan lengkap yang ditulis oleh pengguna.                          |
| 7 | **bank_image** | 1000 non-null  | `object` | URL atau referensi gambar bank.                                              |
| 8 | **rating_title_by_user** | 1000 non-null  | `object` | Judul rating yang diberikan pengguna (seringkali deskripsi singkat rating).  |
| 9 | **useful_count** | 1000 non-null  | `int64`  | Jumlah orang yang menganggap ulasan ini bermanfaat.                         |

Setelah mengetahui kolom-kolom dari dataset tersebut, dilakukan pendeskripsian tentang dataset : 

|index|author|date|address|bank|rating|review\_title\_by\_user|review|bank\_image|rating\_title\_by\_user|useful\_count|
|---|---|---|---|---|---|---|---|---|---|---|
|count|996|1000|1000|1000|1000\.0|1000|1000|1000|1000|1000\.0|
|unique|620|110|107|10|NaN|352|999|10|10|NaN|
|top|ANONYMOUS|Jan 20, 2020|Bangalore|review|NaN|"Good Account"|In SBI customer care, they are not responding properly\.|https://static\.bankbazaar\.com/images/common/bank-logo/ALL\_BANKS\.png|Blown Away\!|NaN|
|freq|117|26|245|285|NaN|105|2|285|550|NaN|
|mean|NaN|NaN|NaN|NaN|4\.3515|NaN|NaN|NaN|NaN|2\.752|
|std|NaN|NaN|NaN|NaN|0\.9407884102351797|NaN|NaN|NaN|NaN|7\.638903641809372|
|min|NaN|NaN|NaN|NaN|0\.5|NaN|NaN|NaN|NaN|0\.0|
|25%|NaN|NaN|NaN|NaN|4\.0|NaN|NaN|NaN|NaN|0\.0|
|50%|NaN|NaN|NaN|NaN|5\.0|NaN|NaN|NaN|NaN|0\.0|
|75%|NaN|NaN|NaN|NaN|5\.0|NaN|NaN|NaN|NaN|2\.0|
|max|NaN|NaN|NaN|NaN|5\.0|NaN|NaN|NaN|NaN|133\.0|

Dataset ini, dengan 1000 ulasan bank, menunjukkan bias positif yang kuat pada rating (rata-rata 4.35 dari 5, dengan median 5.0), didominasi oleh 10 bank unik, 620 penulis unik, dan perlu penanganan pada 4 nilai author yang hilang serta anomali 'review' pada kolom bank dan frekuensi bank_image yang tidak konsisten, sambil juga mencatat distribusi useful_count yang sangat bervariasi.


#### 2. Cek Missing Value dan Duplikasi Data

Penting untuk memeriksa keberadaan *missing value* (nilai yang hilang) di setiap kolom, yang dapat memengaruhi kualitas model. Duplikasi data juga akan diperiksa dan ditangani untuk memastikan setiap entri ulasan adalah unik dan valid.

Pada data yang saya gunakan terdapat 4 missing values pada kolom author dan tidak terdapat duplikasi data:

| # | Column                 | Non-Null Count | Total Rows | Missing Values | Dtype   |
|---|------------------------|----------------|------------|----------------|---------|
| 0 | **author** | 996            | 1000       | 4              | `object`|
| 1 | **date** | 1000           | 1000       | 0              | `object`|
| 2 | **address** | 1000           | 1000       | 0              | `object`|
| 3 | **bank** | 1000           | 1000       | 0              | `object`|
| 4 | **rating** | 1000           | 1000       | 0              | `float64`|
| 5 | **review_title_by_user**| 1000           | 1000       | 0              | `object`|
| 6 | **review** | 1000           | 1000       | 0              | `object`|
| 7 | **bank_image** | 1000           | 1000       | 0              | `object`|
| 8 | **rating_title_by_user**| 1000           | 1000       | 0              | `object`|
| 9 | **useful_count** | 1000           | 1000       | 0              | `int64` |

Duplikasi data : np.int64(0)


#### 3. Jumlah User Unik dan Bank Unik

Menghitung jumlah pengguna unik (`User_ID`) dan bank unik (`Bank_Name`) akan memberikan gambaran tentang skala dataset dari perspektif pengguna dan item. Informasi ini krusial untuk memahami kepadatan (sparsity) matriks interaksi pengguna-item, yang merupakan karakteristik penting dalam sistem rekomendasi.

## Ringkasan Jumlah Pengguna dan Bank Unik

| Kategori      | Jumlah |
|---------------|--------|
| Pengguna Unik | 620    |
| Bank Unik     | 10     |

#### 4. Distribusi Data Rating

Analisis distribusi rating akan menunjukkan bagaimana pelanggan umumnya memberikan peringkat. Apakah ada kecenderungan untuk memberikan rating tinggi (positif) atau rendah (negatif)? Distribusi ini penting untuk memahami bias dalam data dan bagaimana hal itu dapat memengaruhi kinerja model.

| Rating | Jumlah Ulasan |
|:-------|:--------------|
| 0.5    | 10            |
| 1.0    | 13            |
| 1.5    | 1             |
| 2.0    | 30            |
| 2.5    | 2             |
| 3.0    | 71            |
| 3.5    | 21            |
| 4.0    | 257           |
| 4.5    | 45            |
| 5.0    | 550           |

 mengelompokkan data berdasarkan kolom bank : 

| No. | Bank                   | Rata-Rata Rating |
|-----|------------------------|------------------|
| 1   | Citibank               | 4.714286         |
| 2   | Punjab National Bank   | 4.535714         |
| 3   | HDFC Bank              | 4.484043         |
| 4   | Axis Bank              | 4.393130         |
| 5   | Kotak                  | 4.318750         |
| 6   | review                 | 4.305263         |
| 7   | IDBI                   | 4.285714         |
| 8   | SBI                    | 4.280822         |
| 9   | Canara Bank            | 4.257143         |
| 10  | IndusInd Bank          | 4.225000         |

Citibank memperoleh rating tertinggi sebesar 4.71, diikuti oleh Punjab National Bank dan HDFC Bank, sementara seluruh bank menunjukkan tingkat kepuasan pelanggan yang cukup tinggi dengan rating di atas 4, meskipun terdapat entri tidak valid seperti "review" yang sebaiknya dibersihkan dari data.

#### 5. Visualisasi Data

Visualisasi data akan digunakan untuk menyajikan *insight* dari langkah-langkah sebelumnya secara grafis. Berikut merupakn beberapa visualisasi data nya : 

**Distribusi Rating User :**

![Gambar 1 Distribusi_Rating_User](image/Distribusi_Rating_User.png)

   - Dominasi Rating Sempurna (5.0): Terlihat jelas bahwa rating 5.0 mendominasi dataset, dengan jumlah ulasan mencapai lebih dari 550, menunjukkan kecenderungan kuat pengguna untuk memberikan penilaian tertinggi.
   - Kecenderungan Umum Positif: Selain rating 5.0, rating 4.0 juga memiliki frekuensi tinggi (sekitar 250 ulasan). Ini mengindikasikan bahwa sebagian besar ulasan bersifat positif atau sangat positif.
   - Jumlah Rating Negatif/Netral yang Sangat Rendah: Rating di bawah 3.0 (termasuk 0.5, 1.0, 1.5, 2.0, 2.5) memiliki jumlah ulasan yang sangat minim, menunjukkan bahwa ulasan negatif atau netral jarang diberikan dibandingkan dengan ulasan positif.
 
**Jumlah Ulasan Bank :**
![Gambar 2 Jumlah_Ulasan_Bank](image/Jumlah_Ulasan_Bank.png)
   - Dominasi Entri 'review': Terdapat entri anomali bernama 'review' yang memiliki jumlah ulasan tertinggi secara signifikan (sekitar 280 ulasan). Ini sangat tidak wajar untuk nama bank dan mengindikasikan adanya masalah data quality yang serius yang perlu diinvestigasi dan diperbaiki (kemungkinan placeholder atau kesalahan entri data) sebelum pemodelan.
   - Bank-Bank Terpopuler (Selain Anomali): Setelah 'review', SBI (sekitar 220 ulasan) dan HDFC Bank (sekitar 190 ulasan) adalah bank-bank dengan jumlah ulasan terbanyak. Ini menunjukkan bahwa kedua bank ini kemungkinan besar adalah bank besar atau paling banyak digunakan di dataset ini.
   - Distribusi Ulasan yang Tidak Merata: Terdapat ketimpangan yang jelas dalam jumlah ulasan antar bank. Beberapa bank, seperti Punjab National Bank, Citibank, IDBI, dan Indusind Bank, memiliki jumlah ulasan yang sangat sedikit (kurang dari 50), yang dapat menjadi tantangan untuk sistem rekomendasi karena kurangnya data interaksi.

**Heatmap Interaksi User**
![Gambar 3 Heatmap_Interaksi_User](image/Heatmap_Interaksi_User.png)
   - Sparsity (Kepadatan) Data: Heatmap ini dengan jelas menunjukkan bahwa matriks interaksi pengguna-bank sangat sparse (jarang). Sebagian besar sel berwarna kuning pucat (nilai 0 atau tidak ada interaksi), yang berarti mayoritas pengguna hanya berinteraksi (memberikan rating) dengan sedikit bank dari total bank yang tersedia. Ini adalah karakteristik umum dalam sistem rekomendasi dan menjadi tantangan utama.
   - Popularitas Bank (Terlihat dari Jumlah Interaksi): Meskipun ini hanya sampel, terlihat bahwa beberapa bank, seperti 'review' (yang telah kita identifikasi sebagai anomali), 'SBI', dan 'HDFC Bank -', memiliki lebih banyak interaksi (sel berwarna biru gelap) dibandingkan bank lain. Hal ini mengkonfirmasi observasi dari grafik "Jumlah Ulasan per Bank" bahwa bank-bank ini lebih sering diulas.
   - Variasi Pola Interaksi Pengguna: Setiap baris (pengguna) menunjukkan pola interaksi yang berbeda. Beberapa pengguna (misalnya, 'VENGATESAN' atau 'YAD') hanya berinteraksi dengan satu atau dua bank, sementara yang lain (misalnya, 'PRAD') menunjukkan interaksi yang lebih beragam. Pola ini akan menjadi dasar bagi algoritma Collaborative Filtering untuk menemukan kemiripan antar pengguna atau antar bank.

---

## Data Preparation

Berikut adalah versi yang lebih rapi dan informatif dari segmen **Data Preparation**, lengkap dengan penjelasan dan insight pada setiap sub-segmen kode:

---

## Data Preparation

Pada tahap ini, dilakukan serangkaian proses untuk membersihkan dan menyiapkan data agar siap digunakan dalam pemodelan sistem rekomendasi. Langkah-langkah dilakukan secara berurutan untuk memastikan data berkualitas dan relevan. Berikut tahapan data preparation yang diterapkan:

### 1. Penanganan Missing Value dan Duplikasi Data

```python
df = df.dropna(subset=['author'])
df = df.drop_duplicates(subset=['author', 'bank'])
```

* **Tujuan**: Menghapus data yang tidak memiliki nilai pada kolom `author` karena informasi pengguna sangat penting dalam sistem rekomendasi.
* **Insight**: Nilai kosong (`NaN`) pada `author` tidak dapat digunakan untuk memetakan interaksi user dengan item. Duplikasi kombinasi `author` dan `bank` dihapus agar tidak terjadi pembobotan berlebih dalam model terhadap interaksi yang sama.

### 2. Feature Selection

```python
df_clean = df[['author', 'bank', 'rating']]
```

* **Tujuan**: Memilih hanya fitur yang relevan untuk model, yaitu `author` (pengguna), `bank` (item), dan `rating` (nilai interaksi).
* **Insight**: Fitur tambahan yang tidak relevan dapat menimbulkan noise. Dengan menyederhanakan dataset menjadi hanya tiga kolom utama, proses training menjadi lebih efisien dan terfokus.
  
### 3. Encoding Data

```python
from sklearn.preprocessing import LabelEncoder

user_enc = LabelEncoder()
item_enc = LabelEncoder()

df_clean['user'] = user_enc.fit_transform(df_clean['author'])
df_clean['item'] = item_enc.fit_transform(df_clean['bank'])

n_users = df_clean['user'].nunique()
n_items = df_clean['item'].nunique()
```

* **Tujuan**: Mengubah data kategorikal (`author`, `bank`) menjadi representasi numerik (`user`, `item`) agar dapat diproses oleh algoritma machine learning.
* **Insight**: Algoritma seperti collaborative filtering dan matrix factorization tidak dapat memproses string, sehingga perlu dilakukan encoding. Variabel `n_users` dan `n_items` juga dihitung untuk mengetahui ukuran matriks user-item yang akan dibentuk.

### 4. Data Splitting

```python
from sklearn.model_selection import train_test_split

train, test = train_test_split(df_clean, test_size=0.2, random_state=42)
```

* **Tujuan**: Membagi data menjadi data latih (80%) dan data uji (20%) untuk melatih dan mengevaluasi performa model.
* **Insight**: Pemisahan data dilakukan secara acak namun konsisten (melalui `random_state=42`) untuk memastikan hasil eksperimen dapat direproduksi. Evaluasi model akan dilakukan pada data yang belum pernah dilihat model selama proses pelatihan.

 ---

## Modeling
Berikut adalah versi perbaikan dari segmen **Modeling** sesuai permintaan Anda — tanpa menyisipkan ulang kode, lebih ringkas, rapi, namun tetap mencakup penjelasan teknis, parameter, dan evaluasi setiap model. Bagian output Top-N juga ditambahkan sebagai penutup meski belum diimplementasikan.

---

## Modeling

Pada bagian ini, dibangun tiga pendekatan sistem rekomendasi yang berbeda untuk menyelesaikan permasalahan rekomendasi bank kepada pengguna. Evaluasi model dilakukan menggunakan **Mean Absolute Error (MAE)** untuk menilai seberapa dekat prediksi model terhadap nilai rating aktual.

Tiga model yang digunakan adalah:

### Model 1 – User-Based Collaborative Filtering (Cosine Similarity)

Model ini memanfaatkan **kemiripan antar pengguna** (user-user similarity) untuk memprediksi rating pada item yang belum pernah diberi rating oleh pengguna. Kemiripan dihitung menggunakan **Cosine Similarity** pada matriks interaksi user-item.

**Parameter yang digunakan:**

* `k = 5`: jumlah tetangga terdekat (nearest neighbors) yang digunakan dalam prediksi.

**Kelebihan:**

* Mudah diimplementasikan dan dipahami.
* Cocok untuk dataset kecil hingga menengah.

**Kekurangan:**

* Tidak efektif pada data yang sangat sparse.
* Tidak bisa menangani cold-start (pengguna atau item baru).

### Model 2 – SVD Matrix Factorization (Manual)

Model ini menggunakan **Truncated SVD** untuk melakukan dekomposisi matriks interaksi menjadi dimensi laten yang lebih kecil. Matriks kemudian direkonstruksi untuk menghasilkan prediksi rating.

**Parameter yang digunakan:**

* `n_components = 5`: jumlah dimensi laten yang digunakan.

**Kelebihan:**

* Mengurangi kompleksitas data dan mengatasi sparsity.
* Lebih stabil secara matematis dibanding metode manual.

**Kekurangan:**

* Tidak bisa menangani user atau item baru.
* Tidak fleksibel terhadap perubahan data.

### Model 3 – Neural Collaborative Filtering (NCF)

Model ini dibangun menggunakan arsitektur **PyTorch** dan menerapkan pendekatan pembelajaran mendalam untuk memodelkan interaksi kompleks antara user dan item. User dan item diubah menjadi embedding dan diproses melalui layer neural network.

**Parameter yang digunakan:**

* `Embedding dimensi (default)`: berdasarkan input jumlah user dan item.
* `Batch size = 64`
* `Learning rate = 0.005`
* `Loss function = MSELoss`
* `Optimizer = Adam`
* `Epochs = 10`

**Kelebihan:**

* Dapat menangkap pola non-linear dalam data.
* Fleksibel dan mampu belajar representasi fitur tersembunyi dari user dan item.

**Kekurangan:**

* Membutuhkan waktu dan sumber daya komputasi lebih banyak.
* Rentan overfitting jika tidak dilengkapi teknik regularisasi.

---

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
