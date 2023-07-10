from keras.models import Model 
from keras.layers import Input, Dense  
import pandas as pd 
from tensorflow.keras.utils import to_categorical
import numpy as np

data = pd.read_csv('iris.data').values
#digunakan untuk membaca file "iris.data" yang merupakan file CSV

rand = np.arange(149)
#Pernyataan ini membuat array NumPy yang berisi angka dari 0 hingga 148 secara berurutan. 
#Fungsi #np.arange() digunakan untuk membuat array berurutan dengan langkah 1.
np.random.shuffle(rand)
#untuk mengacak data atau indeks dalam array. 
# Setelah pernyataan ini dieksekusi, elemen-elemen dalam array rand akan berada dalam urutan acak.
c = 0
#menginisialisasi variabel c dengan nilai 0. 
#Variabel ini dapat digunakan untuk tujuan tertentu dalam kode yang belum ditampilkan.

copy = data.copy()
#membuat salinan dari DataFrame data dan menyimpannya dalam variabel copy

for i in rand:
	data[c] = copy[i]
	c = c+1
	#dengan menggunakan perulangan tersebut, nilai-nilai dari elemen copy[i] akan disimpan dalam kolom-kolom data secara berurutan 
	#sesuai dengan nilai c. 
	#Ini menghasilkan pengacakan data dalam DataFrame data berdasarkan urutan acak dalam array rand.

print(data)

X = data[:, :4]
y = data[:, 4]
print(y)
labels = {}
cnt = 0 
#X akan berisi matriks fitur yang terdiri dari kolom-kolom 0-3 dari DataFrame data, 
#dan y akan berisi array label dari kolom terakhir DataFrame data. Anda dapat menggunakan X dan y untuk melatih model atau 
#melakukan analisis lebih lanjut pada data Anda. Variabel labels dapat digunakan untuk memetakan label numerik ke label kelas yang sesuai, 
#dan cnt dapat digunakan untuk menghitung jumlah kelas unik dalam data.

for i in y:
	#Pernyataan ini memulai perulangan for yang akan mengiterasi melalui setiap elemen dalam array y. 
	# Setiap elemen dalam y akan disimpan dalam variabel i saat melakukan iterasi.
	if i not in labels:
		labels[i] = cnt  
		#Jika nilai i belum ada dalam kamus labels, maka nilai i akan ditambahkan ke kamus sebagai kunci dengan nilai cnt. 
		#Ini memetakan nilai i ke nilai cnt dalam kamus labels.
		cnt = cnt + 1 
		#Setelah nilai i ditambahkan ke kamus labels, nilai cnt akan ditingkatkan sebesar 1. Ini memastikan bahwa pada iterasi berikutnya, 
		# nilai unik selanjutnya dalam y akan diberikan nilai cnt yang baru dalam kamus labels.

print("="*50)
print(labels)
for i in range(y.shape[0]):
	y[i] = labels[y[i]]
print(y)
y = to_categorical(y)
print(y)
#array y akan mengalami beberapa perubahan: pemetaan label numerik ke label kelas menggunakan kamus labels,
#kemudian konversi ke one-hot encoding menggunakan fungsi to_categorical. Kita dapat menggunakan array y yang sudah diubah ini untuk 
#melatih model klasifikasi atau tujuan lainnya yang membutuhkan representasi kelas biner.ss

X= np.array(X, dtype="float64")
# Pada langkah ini, fungsi np.array() digunakan untuk mengonversi variabel X menjadi array NumPy. 
# Dalam hal ini, X merupakan matriks fitur yang sebelumnya telah diambil dari DataFrame data. 
# Dengan memberikan argumen dtype="float64", kita menentukan bahwa tipe data array NumPy yang dihasilkan adalah float64.
y= np.array(y, dtype="float64")
# Pada langkah ini, fungsi np.array() digunakan untuk mengonversi variabel y menjadi array NumPy. 
# Dalam hal ini, y merupakan array label yang sebelumnya telah diambil dari DataFrame data. Dengan memberikan argumen dtype="float64",
#  kita menentukan bahwa tipe data array NumPy yang dihasilkan adalah float64.

print(X.shape, y.shape)

inp = Input(shape=(4))
#Pernyataan ini mendefinisikan lapisan input untuk model jaringan saraf tiruan. shape=(4) 
# menunjukkan bahwa input memiliki dimensi (4,), yang sesuai dengan jumlah fitur dalam set data Anda.

x = Dense(32, activation="relu")(inp)
#Pernyataan ini mendefinisikan lapisan Dense dengan 32 unit dan fungsi aktivasi ReLU. 
# Lapisan ini menerima input dari lapisan sebelumnya, yaitu lapisan input inp.

op = Dense(3, activation="softmax")(x)
# Pernyataan ini mendefinisikan lapisan output Dense dengan 3 unit dan fungsi aktivasi softmax. 
# Lapisan ini menerima input dari lapisan sebelumnya, yaitu lapisan x.

model=  Model(inputs=inp, outputs=op)
#Pernyataan ini mendefinisikan model dengan menggunakan objek Model dari modul keras. 
# Model ini memiliki input inp dan output op yang telah didefinisikan sebelumnya.

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['acc'])
#mengompilasi model dengan menentukan pengoptimal, fungsi loss, dan metrik evaluasi. Dalam hal ini, 
# kita menggunakan optimizer "rmsprop" (metode pengoptimalisasi), loss function "categorical_crossentropy" 
# (fungsi kerugian untuk klasifikasi multikelas), dan metrik evaluasi akurasi ("acc").
model.fit(X, y, epochs=30)
#melatih model menggunakan metode fit(). Model diberikan data latihan X dan label y yang telah diproses sebelumnya. 
# Jumlah epochs ditentukan sebagai 30, 
# yang menunjukkan berapa kali model akan melihat seluruh set data latihan dalam proses pelatihan.
model.save('model.h5')
# menyimpan model yang telah dilatih dalam file "model.h5". 
# File ini berisi arsitektur model, bobot, dan konfigurasi lainnya yang diperlukan untuk merekonstruksi model.
arr = []
#menginisialisasi variabel arr sebagai sebuah list kosong. 
# Variabel ini bisa digunakan untuk tujuan tertentu dalam kode yang belum ditampilkan.
for k in labels.keys():
	#memulai perulangan for yang akan mengiterasi melalui kunci-kunci dalam kamus labels. labels.keys() 
	# mengembalikan objek dict_keys yang berisi kunci-kunci dalam kamus labels. 
	# Setiap kunci akan disimpan dalam variabel k saat melakukan iterasi.
	arr.append(k)
	#Pada setiap iterasi, nilai kunci k akan ditambahkan ke dalam list arr menggunakan metode .append().
	#  Ini akan menambahkan kunci ke dalam list sebagai elemen baru.
	
print(arr)
print(labels)
np.save("labels.npy", np.array(arr))