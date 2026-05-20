# Nama : Edgina Syafa Ayu Wicaksono
# NIM: H1D024046
Praktikum Kecerdasan Buatan – Pertemuan 7

Program ini merupakan implementasi Jaringan Syaraf Tiruan (JST) menggunakan TensorFlow dan Keras untuk melakukan klasifikasi tiga jenis bunga Iris. Model dilatih menggunakan dataset Iris untuk mengenali spesies Setosa, Versicolor, dan Virginica berdasarkan karakteristik bunga.

1. Import Library
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

Library yang digunakan memiliki fungsi sebagai berikut:

TensorFlow dan Keras → digunakan untuk membangun serta melatih model neural network.
Pandas dan NumPy → membantu pengolahan data dan operasi numerik.
Matplotlib dan Seaborn → digunakan untuk menampilkan visualisasi grafik dan confusion matrix.
Scikit-learn → digunakan untuk memuat dataset, melakukan pembagian data, encoding label, serta evaluasi model.
2. Memuat Dataset
iris = load_iris()
X = iris.data
y = iris.target

Dataset Iris diambil langsung dari library sklearn. Data terdiri dari:

X → data fitur bunga
y → label kelas

Dataset memiliki:

150 data sampel
4 fitur untuk setiap sampel:
sepal length
sepal width
petal length
petal width

Label yang digunakan:

0 = Setosa
1 = Versicolor
2 = Virginica
3. Tahap Preprocessing Data
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

Tahap preprocessing dilakukan sebelum data digunakan dalam proses pelatihan.

Penjelasan:

LabelEncoder() digunakan untuk mengubah label menjadi bentuk numerik apabila data berupa teks.
train_test_split() digunakan untuk membagi dataset menjadi data pelatihan dan data pengujian.

Proporsi data:

Data training = 80%
Data testing = 20%

Parameter random_state=42 digunakan agar pembagian data tetap konsisten setiap program dijalankan.

4. Pembuatan Model JST
model = Sequential([
Input(shape=X_train.shape[1:]),
Dense(1000, activation='relu'),
Dense(500, activation='relu'),
Dense(300, activation='relu'),
Dense(3, activation='softmax')
])

model.summary()

Model dibuat menggunakan beberapa lapisan neural network:

Input Layer

Menerima 4 fitur dari dataset Iris.

Hidden Layer

Model menggunakan tiga hidden layer:

Layer pertama = 1000 neuron
Layer kedua = 500 neuron
Layer ketiga = 300 neuron

Fungsi aktivasi yang digunakan adalah ReLU.

Output Layer

Output layer memiliki:

3 neuron
Aktivasi Softmax

Softmax digunakan untuk menghasilkan probabilitas dari setiap kelas.

Jumlah parameter model mencapai 656.703 parameter.

5. Kompilasi Model
model.compile(
optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy']
)

Konfigurasi model:

Optimizer

adam

Berfungsi memperbarui bobot model selama proses pelatihan.

Loss Function

sparse_categorical_crossentropy

Digunakan karena label masih berbentuk integer.

Metrics

accuracy

Digunakan untuk mengukur tingkat akurasi model.

6. Pelatihan Model
history = model.fit(
X_train,
y_train,
epochs=50,
batch_size=32,
validation_data=(X_test,y_test)
)

Parameter pelatihan:

Epoch = 50
Batch Size = 32

Data uji digunakan sebagai data validasi agar performa model dapat diamati selama proses pelatihan berlangsung.

7. Evaluasi Model
loss,accuracy=model.evaluate(
X_test,
y_test
)

print(
f"Loss:{loss},
Accuracy:{accuracy}"
)

Hasil evaluasi model:

Loss: 0.21582263708114624
Accuracy: 0.8666666746139526

Model memperoleh akurasi sekitar 86,67% pada data pengujian.

8. Visualisasi Hasil Training
pd.DataFrame(
history.history
).plot(figsize=(10,6))

plt.show()

Grafik pelatihan menampilkan beberapa informasi:

Accuracy training
Loss training
Validation accuracy
Validation loss

Berdasarkan grafik, nilai validation loss masih mengalami fluktuasi sehingga terdapat indikasi overfitting, karena model memiliki jumlah parameter yang cukup besar dibandingkan ukuran dataset.

9. Prediksi dan Confusion Matrix
predictions=model.predict(X_test)

predicted_classes=predictions.argmax(axis=1)

cm=confusion_matrix(
y_test,
predicted_classes
)

Prediksi dilakukan terhadap data pengujian, kemudian dibandingkan dengan label asli menggunakan confusion matrix.

Hasil confusion matrix:

True \ Predicted	Setosa	Versicolor	Virginica
Setosa	10	0	0
Versicolor	0	9	1
Virginica	0	4	6

Interpretasi hasil:

Setosa berhasil diklasifikasikan dengan benar seluruhnya.
Versicolor memiliki 1 kesalahan prediksi.
Virginica memiliki 4 data yang diprediksi sebagai Versicolor.
10. Prediksi Data Baru
predict_new_data()

Program menyediakan fitur prediksi data secara interaktif dengan memasukkan:

sepal length
sepal width
petal length
petal width

Contoh:

Masukkan sepal length: 5.1
Masukkan sepal width: 3.5
Masukkan petal length: 1.4
Masukkan petal width: 0.2

Prediksi kelas: setosa
11. Cara Menjalankan Program
Buka Google Colab.
Upload file notebook.
Jalankan seluruh cell.
Tunggu proses training selesai.
Masukkan data bunga untuk melakukan prediksi.

12. Kesimpulan

Program JST berhasil melakukan klasifikasi bunga Iris dengan akurasi sekitar 86,67%. Meskipun hasilnya cukup baik, model masih menunjukkan indikasi overfitting karena jumlah neuron yang digunakan cukup besar untuk dataset yang relatif kecil.

Beberapa perbaikan yang dapat dilakukan:

Mengurangi jumlah neuron.
Menambahkan layer Dropout.
Menggunakan EarlyStopping.
Menyesuaikan arsitektur model agar lebih sederhana.
