# README - Klasifikasi Iris dengan Neural Network

Program ini mengimplementasikan klasifikasi dataset **Iris** menggunakan **TensorFlow/Keras** dengan arsitektur feed-forward neural network. Program ini melakukan:
- Memuat dataset Iris dari `sklearn.datasets`
- Melakukan praproses data (encoding label, train-test split)
- Membangun model *deep learning* dengan 4 layer *dense*
- Melatih model selama 50 epoch
- Mengevaluasi performa model dengan *accuracy* dan *loss*
- Menampilkan *confusion matrix* untuk analisis kesalahan klasifikasi
- Menyediakan fungsi interaktif untuk memprediksi kelas bunga Iris baru berdasarkan input pengguna

## Struktur Program

Program ditulis dalam **Jupyter Notebook** (`.ipynb`) dengan sel-sel yang berisi:
1. **Import library** yang diperlukan
2. **Load dan praproses data** Iris
3. **Pembangunan arsitektur model**
4. **Kompilasi dan pelatihan model**
5. **Evaluasi model** pada data uji
6. **Visualisasi riwayat pelatihan** (accuracy/loss per epoch)
7. **Prediksi pada data uji** dan pembuatan confusion matrix
8. **Fungsi untuk prediksi data baru** secara interaktif

## Persyaratan (Requirements)
```
tensorflow
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Instalasi dilakukan dengan:
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

## Cara Menjalankan

1. **Buka notebook** di Jupyter Lab/Notebook atau Google Colab.
2. Jalankan semua sel secara berurutan (Runtime → Run All di Colab).
3. Setelah pelatihan selesai, program akan meminta input **sepal length**, **sepal width**, **petal length**, dan **petal width**.
4. Masukkan nilai-nilai tersebut untuk mendapatkan prediksi kelas Iris (setosa, versicolor, virginica).

### Contoh Input
```
Masukkan sepal length: 5.1
Masukkan sepal width: 3.5
Masukkan petal length: 1.4
Masukkan petal width: 0.2
```
Output:
```
Prediksi kelas: setosa
```

## Arsitektur Model

Model dibangun menggunakan `Sequential` dengan layer:

| Layer (tipe) | Output Shape | Jumlah Parameter |
|--------------|--------------|------------------|
| Dense (input) | (None, 1000) | 5.000 |
| Dense         | (None, 500)  | 500.500 |
| Dense         | (None, 300)  | 150.300 |
| Dense (output) | (None, 3)    | 903 |

- **Aktivasi**: ReLU pada layer tersembunyi, Softmax pada output.
- **Loss function**: `sparse_categorical_crossentropy`
- **Optimizer**: Adam
- **Metrik**: Accuracy

## Hasil 

Setelah 50 epoch, model mencapai akurasi sekitar **86%** pada data uji. Confusion matrix akan ditampilkan, serta grafik riwayat akurasi dan loss selama pelatihan.

output:
```
Loss: 0.2158, Accuracy: 0.8667
```

## Visualisasi

- **Plot riwayat pelatihan**: Menunjukkan kurva `accuracy` dan `loss` untuk data latih dan validasi.
- **Confusion matrix**: Memudahkan melihat kelas mana yang sering salah diklasifikasikan.

**Dibuat untuk**: Praktikum Kecerdasan Buatan (Pertemuan 7)  
**Nama**: [Nama Mahasiswa]  
**NIM**: [NIM]
