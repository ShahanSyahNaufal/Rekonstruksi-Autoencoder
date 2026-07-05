# Fashion-MNIST Image Reconstruction using Autoencoder

Project ini mengimplementasikan arsitektur **Autoencoder** menggunakan framework **PyTorch** untuk melakukan kompresi dan rekonstruksi citra menggunakan dataset **Fashion-MNIST**. Eksperimen ini membandingkan pengaruh variasi ukuran dimensi laten (*latent dimension*) terhadap kualitas citra yang dihasilkan kembali oleh model.

---

## 📌 Alur Kerja Proyek
1. **Tahap Training (Cloud):** Model dilatih di Kaggle Notebook memanfaatkan GPU untuk mengekstrak fitur gambar dan menyimpannya ke dalam file bobot model (`.pth`).
2. **Tahap Inferensi (Lokal):** File bobot model diunduh ke lingkungan lokal, kemudian dieksekusi melalui script Python via terminal untuk merekonstruksi satu sampel citra pengujian secara instan.

---

## 🛠️ Persyaratan Sistem & Instalasi

Sebelum menjalankan script rekonstruksi, pastikan Anda telah memasang dependensi pustaka Python yang diperlukan. 

### 1. Kloning Repositori
```bash
git clone [https://github.com/ShahanCodes/fashion-mnist-autoencoder.git](https://github.com/ShahanCodes/fashion-mnist-autoencoder.git)
cd fashion-mnist-autoencoder
2. Instalasi Library

Anda dapat langsung memasang pustaka utama menggunakan pip:
Bash

pip install torch torchvision matplotlib scikit-learn scipy

Atau jika menggunakan Virtual Environment (venv):
Bash

python3 -m venv venv
source venv/bin/activate  # Untuk Linux/macOS
pip install -r requirements.txt

📁 Struktur Direktori Proyek
Plaintext

.
├── autoencoder_dim2.pth   # Bobot model untuk Latent Dimension = 2
├── autoencoder_dim8.pth   # Bobot model untuk Latent Dimension = 8
├── autoencoder_dim32.pth  # Bobot model untuk Latent Dimension = 32
├── data/                  # Folder penyimpanan dataset Fashion-MNIST (auto-generated)
├── reconstruct.py         # Script utama pengujian/inferensi model
└── README.md              # Dokumentasi proyek

🚀 Petunjuk Penggunaan (Terminal)

Script reconstruct.py mendukung argumen terminal (command-line arguments) sehingga fleksibel untuk menguji berbagai file model, ukuran dimensi laten, maupun indeks gambar tertentu tanpa perlu mengubah isi kode program.
Sintaks Perintah:
Bash

python3 reconstruct.py --model [NAMA_FILE_MODEL] --latent_dim [UKURAN_DIMENSI] --index [INDEKS_GAMBAR]

Contoh Eksekusi:

    Menguji Model Dimensi 2 (Kompresi Ekstrem):
    Bash

    python3 reconstruct.py --model autoencoder_dim2.pth --latent_dim 2 --index 0

    Menguji Model Dimensi 8:
    Bash

    python3 reconstruct.py --model autoencoder_dim8.pth --latent_dim 8 --index 0

    Menguji Model Dimensi 32 (Akurasi Terbaik):
    Bash

    python3 reconstruct.py --model autoencoder_dim32.pth --latent_dim 32 --index 0

Parameter Argumen:

    --model (Wajib): Jalur atau nama file model biner .pth yang ingin dimuat.

    --latent_dim (Wajib): Nilai dimensi laten model (harus sesuai dengan arsitektur file .pth terkait).

    --index (Opsional): Indeks baris citra pada dataset pengujian Fashion-MNIST yang ingin direkonstruksi (default: 0).

📊 Hasil Output Eksperimen

Setiap kali script berhasil dieksekusi, program akan otomatis menghasilkan dan memperbarui berkas gambar visualisasi di dalam folder proyek Anda:

    original.png: Gambar asli potongan pakaian yang diambil dari dataset test.

    reconstructed.png: Gambar hasil rekonstruksi ulang oleh komponen Decoder model.

    comparison.png: Gambar visualisasi berdampingan (Side-by-Side) antara citra asli dan hasil rekonstruksi untuk kebutuhan analisis.

📝 Ringkasan Analisis

    Latent Dimension 2: Menghasilkan citra rekonstruksi yang sangat buram karena resolusi asal dipaksa menyusut drastis menjadi hanya 2 fitur numerik (high compression loss).

    Latent Dimension 8: Struktur bentuk dasar pakaian mulai dapat dikenali dengan baik, namun detail pola halus masih tampak kabur.

    Latent Dimension 32: Menghasilkan akurasi rekonstruksi terbaik dengan citra yang tajam dan sangat identik dengan bentuk aslinya karena kapasitas ruang laten yang ideal.

Developed by ShahanSyah
