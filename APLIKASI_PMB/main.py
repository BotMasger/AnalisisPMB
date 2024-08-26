import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

st.set_page_config(page_title="Universitas Komputer Indonesia")

class MainClass:

    def __init__(self):
        self.data = Data()
        self.preprocessing = Preprocessing()
        self.clustering = Clustering()
    
    def run(self):
        st.markdown("<h2><center>APLIKASI PENENTUAN PROMOSI BERDASARKAN KARAKTERISTIK MAHASISWA </center></h2>", unsafe_allow_html=True)
        with st.sidebar:
            selected = option_menu('Menu', ['Import Data', 'Preprocessing & Transformasi', 'Clustering & Visualisasi'], default_index=0)

        if selected == 'Import Data':
            self.data.menu_data()

        elif selected == 'Preprocessing & Transformasi':
            if 'uploaded_files' in st.session_state and len(st.session_state.uploaded_files) == 2:
                self.preprocessing.menu_preprocessing()
            else:
                st.error("Data tidak lengkap atau belum diunggah. Silakan unggah 2 file pada menu Import Data terlebih dahulu.")

        elif selected == 'Clustering & Visualisasi':
            if 'df_selected' in st.session_state:
                self.clustering.menu_clustering()
            else:
                st.error("Data tidak tersedia atau belum diproses. Silakan lakukan Preprocessing & Transformasi Data terlebih dahulu.")

class Data:

    def __init__(self):
        self.preprocessing = Preprocessing()

    def menu_data(self):
        uploaded_files = st.file_uploader("Upload Data PMB dan Data Promosi Files", type=["xlsx"], accept_multiple_files=True)

        if uploaded_files and len(uploaded_files) == 2:
            st.session_state.uploaded_files = uploaded_files
            data_pmb = None
            data_promosi = None
            pmb_uploaded = False
            promosi_uploaded = False

            for uploaded_file in uploaded_files:
                st.write(f"- {uploaded_file.name}")
                df = pd.read_excel(uploaded_file)

                # Convert numeric columns to string
                numeric_cols = df.select_dtypes(include=np.number).columns
                df[numeric_cols] = df[numeric_cols].astype(str)

                # Identify file based on name and store in respective variable
                if "pmb" in uploaded_file.name.lower():
                    data_pmb = df
                    pmb_uploaded = True
                    st.dataframe(data_pmb)
                elif "promosi" in uploaded_file.name.lower():
                    data_promosi = df
                    promosi_uploaded = True
                    st.dataframe(data_promosi)

            if data_pmb is not None and data_promosi is not None:
                st.success("Files uploaded successfully.")
            else:
                if not pmb_uploaded:
                    st.warning("File PMB tidak sesuai atau belum diupload.")
                if not promosi_uploaded:
                    st.warning("File Promosi tidak sesuai atau belum diupload.")
                
class Preprocessing:
    
    def menu_preprocessing(self):
        if 'uploaded_files' in st.session_state:
            uploaded_files = st.session_state.uploaded_files
            
            # Load data
            try:
                data_PMB = pd.read_excel(uploaded_files[0])
                data_Promosi = pd.read_excel(uploaded_files[1])
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memuat file: {e}")
                st.stop()
            
            # Convert integer columns to string
            integer_columns_pmb = data_PMB.select_dtypes(include=[np.int64]).columns
            integer_columns_promosi = data_Promosi.select_dtypes(include=[np.int64]).columns
            
            data_PMB[integer_columns_pmb] = data_PMB[integer_columns_pmb].astype(str)
            data_Promosi[integer_columns_promosi] = data_Promosi[integer_columns_promosi].astype(str)

            # List of required columns
            required_columns_pmb = ['NIM','NAMA','ASAL SEKOLAH','PROVINSI','PROGRAM STUDI']
            required_columns_promosi = ['NAMA', 'PROGRAM STUDI','NOMOR UJIAN','WEBSITE','INSTAGRAM', 'BROSUR', 'TWITTER' ,'YOUTUBE', 'TIKTOK']

            # Check for missing columns in dataPMB
            missing_columns_pmb = [col for col in required_columns_pmb if col not in data_PMB.columns]
            missing_columns_promosi = [col for col in required_columns_promosi if col not in data_Promosi.columns]

            if missing_columns_pmb:
                st.warning(f"Data yang Anda masukkan salah. Kolom yang hilang di dataPMB: {', '.join(missing_columns_pmb)}")
                st.stop()
            
            if missing_columns_promosi:
                st.warning(f"Data yang Anda masukkan salah. Kolom yang hilang di dataPromosi: {', '.join(missing_columns_promosi)}")
                st.stop()
            st.subheader("Penggabungan Data")
            try:
                df_merged = pd.merge(data_PMB, data_Promosi, on=['NAMA', 'PROGRAM STUDI'], how='inner')
                st.write(df_merged)
            except KeyError as e:
                st.error(f"Terjadi kesalahan saat menggabungkan data: Kolom tidak ditemukan - {e}")
                st.stop()
            
            df_cleaned = df_merged.dropna().copy()
            st.subheader("Pembersihan Data")
            st.write(df_cleaned)
            
            def get_jenis_sekolah(asal_sekolah):
                if 'SMA' in asal_sekolah:
                    return 'SMA'
                elif 'SMK' in asal_sekolah:
                    return 'SMK'
                elif 'MA' in asal_sekolah:
                    return 'MA'
                elif 'PESANTREN' in asal_sekolah:
                    return 'PESANTREN'
                elif 'HOMESCHOOLING' in asal_sekolah:
                    return 'HOMESCHOOLING'
                elif 'PKBM' in asal_sekolah:
                    return 'PKBM'
                elif 'UNIVERSITAS' in asal_sekolah:
                    return 'UNIVERSITAS'
                else:
                    return 'Lainnya'
            
            df_cleaned['JENIS SEKOLAH'] = df_cleaned['ASAL SEKOLAH'].apply(get_jenis_sekolah)
            st.subheader("Penambahan Atribut Jenis Sekolah")
            st.write(df_cleaned)
            
            jenis_sekolah_mapping = {
                'SMA': 1,
                'SMK': 2,
                'MA': 3,
                'PESANTREN': 4,
                'UNIVERSITAS': 5,
                'PKBM': 6,
                'HOMESCHOOLING': 7
            }
            df_cleaned['JENIS SEKOLAH'] = df_cleaned['JENIS SEKOLAH'].map(jenis_sekolah_mapping)
            
            df_cleaned['NIM'] = df_cleaned['NIM'].astype(str)
            df_cleaned['PROGRAM STUDI'] = df_cleaned['NIM'].str[1:3].astype(int)
            
            provinsi_counts = df_cleaned['PROVINSI'].value_counts()
            provinsi_mapping = {provinsi: i+1 for i, provinsi in enumerate(provinsi_counts.index)}
            df_cleaned['PROVINSI'] = df_cleaned['PROVINSI'].map(provinsi_mapping)
            
            columns_to_replace = ['WEBSITE', 'INSTAGRAM', 'BROSUR', 'TWITTER', 'YOUTUBE', 'TIKTOK']
            df_cleaned[columns_to_replace] = df_cleaned[columns_to_replace].replace({'Ya': 1, '-': 0})

            st.subheader("Proses Transformasi")
            st.write(df_cleaned)

            selected_attributes = ['PROVINSI', 'PROGRAM STUDI', 'WEBSITE', 'TWITTER', 'INSTAGRAM', 'BROSUR', 'YOUTUBE', 'TIKTOK', 'JENIS SEKOLAH']
            df_selected = df_cleaned[selected_attributes]
            st.subheader("Atribut yang Dipilih")
            st.write(df_selected)

            # Simpan df_selected ke session state
            st.session_state.df_selected = df_selected
        else:
            st.warning("Data belum diunggah dengan benar. Silakan unggah data terlebih dahulu di menu 'Import Data'.")


class Clustering:

    def menu_clustering(self):
        if 'df_selected' in st.session_state:
            df_selected = st.session_state.df_selected
            clustering_data = df_selected.select_dtypes(include=[np.number])

            num_clusters = st.number_input('Enter Number of Clusters for Data Mining', min_value=2, max_value=10, value=2, step=1)

            if st.button('Mulai Clustering'):
                kmeans = KMeans(n_clusters=num_clusters, init='random', random_state=42)
                kmeans.fit(clustering_data)

                cluster_labels = kmeans.labels_ + 1
                df_selected['Cluster'] = cluster_labels

                st.subheader("Hasil Clustering")
                st.dataframe(df_selected)
                    
                dbi = davies_bouldin_score(clustering_data, kmeans.labels_)
                st.write(f"Davies-Bouldin Index (DBI): {dbi}")
                st.write("Interpretasi DBI:")
                if dbi < 1:
                    st.success("Hasil clustering sangat baik.")
                else:
                    st.error("Hasil clustering buruk.")

                # Proses Pie Chart
                prodi_mapping = {
                    1: 'TEKNIK INFORMATIKA-S1',
                    2: 'SISTEM KOMPUTER-S1',
                    3: 'TEKNIK INDUSTRI-S1',
                    4: 'TEKNIK ARSITEKTUR-S1',
                    5: 'SISTEM INFORMASI-S1',
                    6: 'PERENCANAAN WILAYAH DAN KOTA-S1',
                    8: 'TEKNIK KOMPUTER-D3',
                    9: 'MANAJEMEN INFORMATIKA-D3',
                    10: 'KOMPUTERISASI AKUNTANSI-D3',
                    30: 'TEKNIK SIPIL-S1',
                    31: 'TEKNIK ELEKTRO-S1',
                    11: 'AKUNTANSI-S1',
                    12: 'MANAJEMEN-S1',
                    13: 'AKUNTANSI-D3',
                    14: 'MANAJEMEN PEMASARAN-D3',
                    15: 'KEUANGAN DAN PERBANKAN-D3',
                    16: 'ILMU HUKUM-S1',
                    17: 'ILMU PEMERINTAHAN-S1',
                    18: 'ILMU KOMUNIKASI-S1',
                    43: 'HUBUNGAN INTERNASIONAL-S1',
                    19: 'DESAIN KOMUNIKASI VISUAL-S1',
                    20: 'DESAIN INTERIOR-S1',
                    21: 'DESAIN GRAFIS-D3',
                    37: 'SASTRA INGGRIS-S1',
                    38: 'SASTRA JEPANG-S1'}

                provinsi_mapping = {
                    1: 'Jawa Barat', 
                    2: 'Banten', 
                    3: 'DKI Jakarta',
                    4: 'Jawa Tengah',
                    5: 'Sumatera Selatan',
                    6: 'Bangka Belitung',
                    7: 'Sumatera Utara',
                    8: 'Sumatera Barat',
                    9: 'Riau',
                    10: 'Jawa Timur',
                    11: 'Papua Barat',
                    12: 'Bengkulu',
                    13: 'Jambi',
                    14: 'Kalimantan Timur',
                    15: 'Sulawesi Selatan',
                    16: 'Sulawesi Utara',
                    17: 'Kalimantan Barat',
                    18: 'Nanggroe Aceh Darussalam (NAD)',
                    19: 'Lampung',
                    20: 'Gorontalo',
                    21: 'Sulawesi Tengah',
                    22: 'Kalimantan Tengah',
                    23: 'Nusa Tenggara Timur',
                    24: 'Kepulauan Riau',
                    25: 'Nusa Tenggara Barat',
                    26: 'Sulawesi Tenggara',
                    27: 'Kalimantan Selatan',
                    28: 'Maluku Utara',
                    29: 'Maluku'}
                
                jenis_sekolah_mapping = {
                    1: 'SMA', 
                    2: 'SMK', 
                    3: 'MA',
                    4: 'PESANTREN',
                    5: 'UNIVERSITAS',
                    6: 'PKBM',
                    7: 'HOMESCHOOLING'}
                
                media_columns = ['WEBSITE', 'INSTAGRAM', 'TWITTER', 'BROSUR', 'YOUTUBE', 'TIKTOK']
                df_selected[media_columns] = df_selected[media_columns].replace({1: 'Ya', 0: '-'})

                df_selected['PROGRAM STUDI'] = df_selected['PROGRAM STUDI'].map(prodi_mapping)
                df_selected['PROVINSI'] = df_selected['PROVINSI'].map(provinsi_mapping)
                df_selected['JENIS SEKOLAH'] = df_selected['JENIS SEKOLAH'].map(jenis_sekolah_mapping)

                st.title('Repesentasi Kelompok')

                # Display the number of clusters
                st.subheader(f'Jumlah Kelompok Yang Dipilih: {df_selected["Cluster"].nunique()}')

                # Loop through each cluster
                for cluster in sorted(df_selected['Cluster'].unique()):
                    st.write(f"Kelompok: {cluster}, Memiliki Anggota Sebanyak: {len(df_selected[df_selected['Cluster'] == cluster])}")

                    with st.expander(f'Karakteristik Anggota - Kelompok {cluster}:'):
                
                        # Plot pie charts for the current cluster
                        df_cluster = df_selected[df_selected['Cluster'] == cluster]
                        prodi_counts = df_cluster['PROGRAM STUDI'].value_counts()
                        jenis_sekolah_counts = df_cluster['JENIS SEKOLAH'].value_counts()
                        provinsi_counts = df_cluster['PROVINSI'].value_counts()
                        media_promosi_counts = df_cluster[media_columns].apply(lambda x: (x == 'Ya').sum())

                        # Pie chart for top 3 Program Studi
                        # Pie chart for top 3 Program Studi
                        top3_prodi = prodi_counts
                        bottom3_prodi = prodi_counts

                        # Plot pie chart for top 3 program studi
                        fig, ax = plt.subplots()
                        ax.pie(top3_prodi, labels=top3_prodi.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f'Program Studi Dengan Peminat Tertinggi - Kelompok {cluster}')
                        st.pyplot(fig)

                        # Kalimat representasi untuk top 3 Program Studi
                        st.subheader('Representasi Program Studi dengan Peminat tertinggi ')
                        st.write(f"- Promosi ini bertujuan untuk menarik minat siswa {', '.join(jenis_sekolah_counts.head(3).index)} yang berada di {', '.join(provinsi_counts.head(3).index)} agar memilih {', '.join(top3_prodi.index)}. Strategi promosi akan difokuskan pada platform digital seperti {', '.join(media_promosi_counts.nlargest(3).index)}.")

                        # Plot pie chart for bottom 3 program studi
                        fig, ax = plt.subplots()
                        ax.pie(bottom3_prodi, labels=bottom3_prodi.index, autopct='%1.1f%%', startangle=90)
                        ax.set_title(f'Program Studi Dengan Peminat Terendah - Kelompok {cluster}')
                        st.pyplot(fig)

                        # Kalimat representasi untuk bottom 3 Program Studi
                        st.subheader('Representasi Program Studi dengan Peminat terendah')
                        st.write(f"- Promosi yang dilakukan untuk mengenalkan {', '.join(bottom3_prodi.index)} kepada calon mahasiswa baru dari {', '.join(jenis_sekolah_counts.head(3).index)} di {', '.join(provinsi_counts.head(3).index)}. Promosi ini akan memanfaatkan media sosial populer seperti {', '.join(media_promosi_counts.nlargest(3).index)} universitas.")
            else:
                st.warning("Tidak ada data yang tersedia untuk clustering. Silakan lakukan preprocessing terlebih dahulu.")

if __name__ == "__main__":
    app = MainClass()
    app.run()
