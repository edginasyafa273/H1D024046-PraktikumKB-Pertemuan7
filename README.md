# Nama : Edgina Syafa Ayu Wicaksono
# NIM: H1D024046
# Shift : F
# Shift KRS : B

Program ini mengimplementasikan Jaringan Syaraf Tiruan (JST) untuk mengklasifikasikan spesies bunga Iris (Iris setosa, Iris versicolor, Iris virginica) berdasarkan empat fitur morfologi:

Panjang sepal (sepal length)

Lebar sepal (sepal width)

Panjang petal (petal length)

Lebar petal (petal width)

Dataset diambil langsung dari sklearn.datasets.load_iris(). Seluruh kode dirancang untuk dijalankan di Google Colab (dapat juga di Jupyter Notebook lokal).

Program mencakup:

✅ Pemuatan dan praproses data (label encoding, train-test split)

✅ Pembangunan model deep neural network dengan 4 layer dense

✅ Pelatihan model selama 50 epoch (batch size 32)

✅ Evaluasi dan visualisasi (grafik loss/accuracy, confusion matrix)

✅ Prediksi interaktif untuk data bunga baru melalui input pengguna

# Cara Menjalankan di Google Colab
Buka Google Colab dan upload file pertemuan7.ipynb (atau buka langsung dari GitHub).

Runtime → Run all (atau jalankan sel satu per satu).

Setelah pelatihan selesai, program akan meminta input data bunga baru di bagian bawah.

Masukkan keempat nilai fitur, lalu tekan Enter. Program akan menampilkan spesies yang diprediksi.

Catatan: Semua library yang diperlukan sudah tersedia di Colab (TensorFlow, scikit-learn, pandas, numpy, matplotlib, seaborn). Tidak perlu instalasi tambahan.

# Library yang Digunakan
Library	Fungsi
tensorflow & keras	Membangun, melatih, dan mengevaluasi model JST
pandas	Memformat riwayat pelatihan untuk plotting
numpy	Manipulasi array numerik untuk data baru
matplotlib.pyplot	Membuat grafik loss/accuracy
seaborn	Visualisasi confusion matrix (heatmap)
sklearn.datasets	Memuat dataset Iris
sklearn.preprocessing.LabelEncoder	Mengubah label string menjadi numerik (0,1,2)
sklearn.model_selection.train_test_split	Membagi data latih dan uji (80:20)
sklearn.metrics.confusion_matrix	Menghitung confusion matrix
# Dataset Iris
Jumlah sampel: 150

Fitur (4): sepal length, sepal width, petal length, petal width (dalam cm)

Label (3 kelas):

0 → Iris Setosa

1 → Iris Versicolor

2 → Iris Virginica

Dataset dimuat dari load_iris() (baris 16-18 pada kode).

# Praproses Data
1. Label Encoding (baris 22-23)
Label string dikonversi menjadi integer menggunakan LabelEncoder.

2. Train-Test Split (baris 26-28)
Dataset dibagi dengan rasio 80% training, 20% testing menggunakan train_test_split dengan random_state=42 agar hasil dapat direproduksi.

# Arsitektur Model Neural Network
Model dibangun menggunakan Sequential (baris 30-36) dengan arsitektur:

Layer	Tipe	Jumlah Neuron	Aktivasi
Input	Input	4 (fitur)	–
Hidden 1	Dense	1000	ReLU
Hidden 2	Dense	500	ReLU
Hidden 3	Dense	300	ReLU
Output	Dense	3	Softmax
Total parameter: 656.703

Ringkasan model ditampilkan dengan model.summary() (baris 38).

# Kompilasi & Pelatihan
Kompilasi (baris 40-44)
python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
Pelatihan (baris 46-51)
python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)
Epochs = 50 – model melihat seluruh data latih sebanyak 50 kali.

Batch size = 32 – pembaruan bobot setiap 32 sampel.

Validation data = data uji digunakan untuk memantau overfitting.

# Evaluasi & Visualisasi
1. Evaluasi Akhir (baris 53-55)
python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
Contoh hasil:

text
Loss: 0.2158, Accuracy: 0.8667
2. Grafik Loss & Accuracy (baris 57-59)
pd.DataFrame(history.history).plot() menghasilkan grafik:

Sumbu X: Epoch (0–50)

Sumbu Y: Nilai loss dan accuracy (0–1)

Garis biru (accuracy) – akurasi data latih

Garis oranye (loss) – loss data latih

Garis hijau (val_accuracy) – akurasi data validasi

Garis merah (val_loss) – loss data validasi

Grafik menunjukkan penurunan loss dan peningkatan accuracy seiring epoch, namun terdapat sedikit fluktuasi pada val_loss yang mengindikasikan potensi overfitting.

3. Confusion Matrix (baris 67-76)
Prediksi pada data uji: predictions = model.predict(X_test)

Kelas dengan probabilitas tertinggi: predicted_classes = predictions.argmax(axis=1)

Matriks kebingungan: confusion_matrix(y_test, predicted_classes)

True \ Predicted	setosa	versicolor	virginica
setosa	10	0	0
versicolor	0	9	1
virginica	0	4	6
Interpretasi: Model sangat baik dalam mengklasifikasikan setosa (10 benar, 0 salah), cukup baik untuk versicolor (9 benar, 1 salah), dan virginica (6 benar, 4 salah – sering tertukar dengan versicolor).

# Prediksi Data Input Baru (baris 79-94)
Fungsi predict_new_data() memungkinkan pengguna memasukkan empat nilai fitur secara interaktif di terminal Colab.

python
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width  = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width  = float(input("Masukkan petal width: "))
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    print(f"Prediksi kelas: {predicted_label[0]}")
Contoh interaksi:

text
Masukkan sepal length: 5.1
Masukkan sepal width: 3.5
Masukkan petal length: 1.4
Masukkan petal width: 0.2
Prediksi kelas: setosa
# Hasil Akhir Program
Program akan menampilkan:

Ringkasan arsitektur model (jumlah layer, parameter).

Progres pelatihan (loss & accuracy per epoch).

Nilai loss dan accuracy akhir pada data uji.

Grafik loss/accuracy selama pelatihan.

Confusion matrix dalam bentuk heatmap.

Prediksi interaktif untuk data baru.
