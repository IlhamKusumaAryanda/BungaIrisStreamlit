import streamlit as st
#modul Streamlit diimpor dan diatribusikan ke alias st, yang digunakan untuk mengakses fungsi-fungsi Streamlit.
from keras.models import load_model 
#digunakan untuk memuat model yang telah disimpan sebelumnya dan mengembalikan model tersebut.
import numpy as np 
#NumPy adalah modul yang sangat berguna untuk komputasi numerik dalam Python. Dalam kode ini, 
# modul NumPy diimpor sehingga kita dapat menggunakan fungsi-fungsi dan metode-metode yang disediakan oleh NumPy.


model = load_model("model.h5")
labels = np.load("labels.npy") 
#digunakan untuk memuat model yang telah disimpan sebelumnya dalam format HDF5 (dengan ekstensi .h5). 
#Model yang dimuat akan ditugaskan ke variabel model, dan dapat digunakan untuk melakukan prediksi atau melatih ulang.

st.title("Klasifikasi Bunga Iris Streamlit")

a = float(st.number_input("panjang sepal dalam cm"))
b = float(st.number_input("lebar sepal dalam cm"))
c = float(st.number_input("panjang petal dalam cm"))
d = float(st.number_input("lebar petal dalam cm"))

btn = st.button("prediksi Bunga")

if btn:
	pred = model.predict(np.array([a,b,c,d]).reshape(1,-1))
	pred = labels[np.argmax(pred)]
	st.subheader(pred)

	if pred=="Iris Setosa":
		st.image("setosa.jpg")
	elif pred=="Iris Versicolour":
		st.image("versicolor.jpg")
	else:	
		st.image("verginca.jpg")

