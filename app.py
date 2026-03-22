from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model

# Jangan lupa load model ML Anda (misal keras/tensorflow atau scikit-learn)
# from keras.models import load_model 

app = Flask(__name__)

# Memuat model secara global sangat penting agar model 
# tidak membebani komputasi dengan me-load ulang setiap kali ada request prediksi.
# model = load_model('model_prediksi_saham.h5')
@st.cache_resource
def get_model():
    return load_model("my_model.keras")

@app.route('/', methods=['GET', 'POST'])
def predict_stock():
    if request.method == 'POST':
        # Mengambil data yang diketikkan pengguna dari form
        kode_input = request.form.get('kode_saham').strip().upper()
        
        # Tambahkan suffix .JK secara otomatis untuk saham bursa Indonesia
        if not kode_input.endswith('.JK'):
            kode_input = f"{kode_input}.JK"
            
        try:
            # 1. Pemrosesan data yang fleksibel
            df_prepared = prepare_stock_data(kode_input)
            
            # 2. Ekstraksi fitur untuk input model 
            # Misal disesuaikan dengan bagian 'xtrain, ytrain = train[:, :-5], train[:,-5:]'
            # x_input = df_prepared.values[-1:, :-5] # Ambil baris data terakhir
            
            # 3. Masukkan ke dalam fungsi prediksi
            # y_pred = model.predict(x_input)
            
            # (Hanya teks simulasi, tampilkan y_pred untuk implementasi aslinya)
            return f"<h3>Data untuk saham {kode_input} berhasil diproses!</h3><p>Hasil prediksi akan muncul di sini.</p>"
            
        except Exception as e:
            return f"<h3>Terjadi kesalahan: {str(e)}</h3>"

    # Halaman interface pengguna sederhana menggunakan HTML Form
    return '''
        <h2>Aplikasi Prediksi Harga Saham</h2>
        <form method="POST">
            <label>Masukkan Kode Emiten (contoh: BBRI, TLKM, BBNI):</label>
            <input type="text" name="kode_saham" required>
            <button type="submit">Prediksi</button>
        </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)