import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data training untuk validasi range input
df = pd.read_csv('gym_members_exercise_tracking.csv')

# Load model KMeans
with open('kmeans_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.title('Gym Member Clustering Prediction Kelompok 6')
st.write('Aplikasi ini membantu memprediksi cluster member gym berdasarkan karakteristik mereka')

# Form input 
st.header('Input Karakteristik Member')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', 
                         min_value=int(df['Age'].min()),
                         max_value=int(df['Age'].max()),
                         value=int(df['Age'].mean()))
    
    gender = st.selectbox('Gender', ['Male', 'Female'])
    
    weight = st.number_input('Weight (kg)', 
                           min_value=float(df['Weight (kg)'].min()),
                           max_value=float(df['Weight (kg)'].max()),
                           value=float(df['Weight (kg)'].mean()))
    
    height = st.number_input('Height (m)', 
                           min_value=float(df['Height (m)'].min()),
                           max_value=float(df['Height (m)'].max()),
                           value=float(df['Height (m)'].mean()))
    
    max_bpm = st.number_input('Max BPM',
                             min_value=int(df['Max_BPM'].min()),
                             max_value=int(df['Max_BPM'].max()),
                             value=int(df['Max_BPM'].mean()))
    
    avg_bpm = st.number_input('Average BPM',
                             min_value=int(df['Avg_BPM'].min()),
                             max_value=int(df['Avg_BPM'].max()),
                             value=int(df['Avg_BPM'].mean()))
    
    resting_bpm = st.number_input('Resting BPM',
                                 min_value=int(df['Resting_BPM'].min()),
                                 max_value=int(df['Resting_BPM'].max()),
                                 value=int(df['Resting_BPM'].mean()))
    
with col2:
    session_duration = st.number_input('Session Duration (hours)',
                                     min_value=float(df['Session_Duration (hours)'].min()),
                                     max_value=float(df['Session_Duration (hours)'].max()),
                                     value=float(df['Session_Duration (hours)'].mean()))
    
    calories_burned = st.number_input('Calories Burned',
                                    min_value=float(df['Calories_Burned'].min()),
                                    max_value=float(df['Calories_Burned'].max()),
                                    value=float(df['Calories_Burned'].mean()))
    
    workout_type = st.selectbox('Workout Type', ['Cardio', 'Strength', 'HIIT', 'Yoga'])
    
    fat_percentage = st.number_input('Fat Percentage',
                                   min_value=float(df['Fat_Percentage'].min()),
                                   max_value=float(df['Fat_Percentage'].max()),
                                   value=float(df['Fat_Percentage'].mean()))
    
    water_intake = st.number_input('Water Intake (liters)',
                                 min_value=float(df['Water_Intake (liters)'].min()),
                                 max_value=float(df['Water_Intake (liters)'].max()),
                                 value=float(df['Water_Intake (liters)'].mean()))
    
    workout_frequency = st.slider('Workout Frequency (days/week)',
                                min_value=int(df['Workout_Frequency (days/week)'].min()),
                                max_value=int(df['Workout_Frequency (days/week)'].max()),
                                value=int(df['Workout_Frequency (days/week)'].mean()))
    
    experience_level = st.selectbox('Experience Level', [1, 2, 3],
                                  help='1: Beginner, 2: Intermediate, 3: Advanced')
    
    bmi = st.number_input('BMI',
                         min_value=float(df['BMI'].min()),
                         max_value=float(df['BMI'].max()),
                         value=float(df['BMI'].mean()))

# Encode categorical variables
gender_encoded = 1 if gender == 'Male' else 0
workout_type_mapping = {'Cardio': 0, 'Strength': 1, 'HIIT': 2, 'Yoga': 3}
workout_type_encoded = workout_type_mapping[workout_type]

# Tombol prediksi
if st.button('Predict Cluster'):
    try:
        # Siapkan data untuk prediksi
        input_data = np.array([[
            age, gender_encoded, weight, height, max_bpm, avg_bpm, 
            resting_bpm, session_duration, calories_burned, workout_type_encoded,
            fat_percentage, water_intake, workout_frequency, experience_level, bmi
        ]])
        
        # Lakukan prediksi
        cluster = model.predict(input_data)[0]
        
        # Tampilkan hasil
        st.header('Hasil Prediksi')
        
        if cluster == 0:
            st.success('Member termasuk dalam Cluster 0: Member High Performance')
            st.write("""
            Karakteristik Member High Performance:
            - Memiliki rata-rata kalori terbakar sangat tinggi (>1000 kalori)
            - Durasi latihan konsisten dan lebih lama (>1.5 jam)
            - Heart rate optimal (Max BPM 170-190, Avg BPM 150-170)
            - Fat percentage rendah (<20%)
            - Water intake tinggi (>3 liter)
            - Workout frequency tinggi (4-5 hari/minggu)
            - Experience level advanced (level 3)
            
            Rekomendasi:
            - Tingkatkan intensitas dengan program HIIT atau strength training
            - Tambahkan variasi latihan untuk mencegah plateau
            - Fokus pada target spesifik (muscle gain/endurance)
            - Pertahankan nutrisi dan hidrasi optimal
            - Ikuti program kompetitif atau challenge
            """)
        else:
            st.warning('Member termasuk dalam Cluster 1: Member Development')
            st.write("""
            Karakteristik Member Development:
            - Kalori terbakar moderate (500-800 kalori)
            - Durasi latihan lebih pendek (<1 jam)
            - Heart rate moderate (Max BPM 150-170, Avg BPM 120-140) 
            - Fat percentage lebih tinggi (>25%)
            - Water intake moderate (1.5-2.5 liter)
            - Workout frequency moderate (2-3 hari/minggu)
            - Experience level pemula-menengah (level 1-2)
            
            Rekomendasi:
            - Mulai dengan program dasar fokus pada form dan teknik
            - Tingkatkan durasi latihan secara bertahap
            - Kombinasikan cardio ringan dengan strength training dasar
            - Tetapkan target mingguan yang realistis
            - Tingkatkan frekuensi latihan secara bertahap
            - Edukasi nutrisi dan pentingnya hidrasi
            - Sediakan personal trainer untuk guidance
            """)
            
    except Exception as e:
        st.error(f'Terjadi kesalahan dalam prediksi: {str(e)}')

st.header('Cluster Distribution')
if st.checkbox('Show Cluster Distribution'):
    try:
        cluster_dist = pd.Series(model.labels_).value_counts()
        st.bar_chart(cluster_dist)
        st.write('Cluster Centers:', model.cluster_centers_)
    except Exception as e:
        st.error(f'Terjadi kesalahan dalam menampilkan distribusi cluster: {str(e)}')
