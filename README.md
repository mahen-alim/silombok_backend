# Panduan Penggunaan Backend Server SiLombok

Server ini digunakan sebagai penghubung antara hasil model latih yang sudah dilakukan proses training/pelatihan secara komprehensif dengan server website SiLombok. Server ini diaktifkan ketika pengguna ingin menggunakan fitur prediksi kesehatan buah cabai di website SiLombok, dengan sistem akan mengambil output prediksi berupa model latih yang didefinisikan pada sebuah API Flask (Framework Backend Python) kemudian server website akan melakukan permintaan/request pengambilan model latih dari alamat API Flask yang sudah didefinisikan sebelumnya. 

## Instalasi & Penggunaan

1. Clone repository ini ke dalam folder lokal Anda :
    ```bash
    git clone https://github.com/mahen-alim/silombok_backend.git
    ```
    
2. Masuk kedalam folder proyek yang sudah diclone :
   ```bash
   cd silombok_backend
   ```
    
3. Buat Virtual Environment Python :
   - Pastikan python dan pip sudah diinstal :
     ```bash
     python --version
     pip --version
     ```
   - Buat virtual environment :
     ```bash
     python -m venv nama_env
     ```
   - Gantilah nama_env dengan nama yang diinginkan untuk virtual environment Anda. Misalnya :
     ```bash
     python -m venv myenv
     ```
   - Aktifkan virtual environment :
     - Windows
       ```bash
       .\myenv\Scripts\activate
       ```
     - macOS/Linux
       ```bash
       source myenv/bin/activate
       ``` 

4. Instal dependensi yang diperlukan :
    ```bash
    pip install -r requirements.txt
    ```
    
5. Latih model, dengan mengetikkan perintah berikut :
   ```bash
   python training2.py
   ```

6. Jalankan file yang berisi api flask, dengan mengetikkan perintah berikut :
   ```bash
   python detect.py
   ```

7. Jika ingin menonaktifkan virtual environment, Anda bisa mengetikkan ini pada terminal :
   ```bash
   deactivate
   ```

8. Akses dan panduan penggunaan website SiLombok dapat diakses pada link berikut :
   https://github.com/mahen-alim/silombok

## Teknologi yang Digunakan

- **Backend**: Python, Flask (Framework Python).
- **Algoritma Data Latih**: Naive Bayes.
- **Version Control**: Git, GitHub.
- **Text Editor**: Visual Studio Code.
- **API Testing**: HTTPie.

## Kontribusi

Jika Anda ingin berkontribusi dalam pengembangan proyek ini, silakan lakukan **fork** repository ini dan kirimkan **pull request** dengan deskripsi perubahan yang jelas.

## Kontak

Jika Anda memiliki pertanyaan lebih lanjut, silakan hubungi kami di:  
- Email: mahennekkers27@gmail.com
- No. WA: 0895807400305

---
