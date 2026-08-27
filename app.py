import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Sleep Track",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_preprocess_dataset(data_filename: str = "sleep_health_dataset.csv"):
    if not os.path.exists(data_filename):
        raise FileNotFoundError(f"الملف غير موجود : {data_filename}")
        
    df_raw = pd.read_csv(data_filename)

    df = df_raw[df_raw['gender'].isin(['Male', 'Female'])].copy()
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).astype(int)
    df['sleep_disorder_risk'] = (df['sleep_disorder_risk'] != 'Healthy').astype(int)
    
    columns_to_drop = [
        "person_id", "occupation", "country", "sleep_quality_score",
        "alcohol_units_before_bed", "work_hours_that_day", "felt_rested",
        "chronotype", "mental_health_condition", "sleep_aid_used",
        "room_temperature_celsius", "season", "day_type",
        "weekend_sleep_diff_hrs", "cognitive_performance_score"
    ]
    
    existing_to_drop = [c for c in columns_to_drop if c in df.columns]
    df_model = df.drop(columns=existing_to_drop).copy()
    
    return df_raw, df_model

df_raw, df_model = load_and_preprocess_dataset()

X = df_model.drop(columns=['sleep_disorder_risk'])
y = df_model['sleep_disorder_risk']

# ---------------------------------------------------------
# 3. Model Training Engine
# ---------------------------------------------------------
@st.cache_resource
def train_all_models(X_df, y_df):
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.2, random_state=42, stratify=y_df
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=12),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8),
        'KNN': KNeighborsClassifier(n_neighbors=7)
    }
    
    results = {}
    fitted_models = {}
    
    for name, model in models.items():
        if name in ['Logistic Regression', 'KNN']:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
            probs = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
        results[name] = {
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds, zero_division=0),
            'Recall': recall_score(y_test, preds, zero_division=0),
            'F1-Score': f1_score(y_test, preds, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, probs),
            'y_test': y_test,
            'preds': preds,
            'probs': probs
        }
        fitted_models[name] = model
        
    return results, fitted_models, scaler, X_test, y_test

model_results, fitted_models, scaler, X_test, y_test = train_all_models(X, y)

# ---------------------------------------------------------
# 4. Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.markdown("## 🌙 Diagnostic Menu")
st.sidebar.markdown("---")

menu_choice = st.sidebar.radio(
    "Select View:",
    [
        "📊 Executive Summary & EDA",
        "🤖 AI Models Benchmark",
        "🔮 Real-Time Patient Risk Predictor",
        "📂 Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Target:** `sleep_disorder_risk`\n\n"
    "• **Class 0:** Healthy\n\n"
    "• **Class 1:** At Risk (Mild/Mod/Severe)\n\n"
    f"**Total Patients:** {len(df_model):,}"
)

# ---------------------------------------------------------
# 5. Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-title-card">
    <h1>🌙 Sleep Track</h1>
    <p>For tracking and analyzing sleep health data</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 1: EDA & Executive Summary
# ---------------------------------------------------------
if menu_choice == "📊 Executive Summary & EDA":
    st.markdown("### 📈 Key Clinical Metrics")
    
    m1, m2, m3, m4, m5 = st.columns(5)
    total_records = len(df_model)
    at_risk_count = (df_model['sleep_disorder_risk'] == 1).sum()
    healthy_count = (df_model['sleep_disorder_risk'] == 0).sum()
    avg_sleep = df_model['sleep_duration_hrs'].mean()
    avg_stress = df_model['stress_score'].mean()
    
    m1.metric("Total Cohort", f"{total_records:,}")
    m2.metric("Healthy Subjects", f"{healthy_count:,}", delta=f"{healthy_count/total_records*100:.1f}%")
    m3.metric("At Risk Subjects", f"{at_risk_count:,}", delta=f"-{at_risk_count/total_records*100:.1f}%", delta_color="inverse")
    m4.metric("Avg Sleep Duration", f"{avg_sleep:.2f} hrs")
    m5.metric("Avg Stress Level", f"{avg_stress:.1f} / 10")
    
    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### 🎯 Target Risk Class Balance")
        fig_donut = px.pie(
            names=['Healthy (0)', 'At Risk (1)'],
            values=[healthy_count, at_risk_count],
            hole=0.5,
            color_discrete_sequence=['#10b981', '#ef4444']
        )
        fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=350)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c2:
        st.markdown("#### 💤 Sleep Duration vs Stress Level")
        fig_scatter = px.scatter(
            df_model.sample(min(2000, len(df_model))),
            x='sleep_duration_hrs',
            y='stress_score',
            color='sleep_disorder_risk',
            color_discrete_map={0: '#10b981', 1: '#ef4444'},
            opacity=0.7
        )
        fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

    c3, c4 = st.columns([1, 1])
    
    with c3:
        st.markdown("#### ⚖️ BMI Distribution Across Risk Profiles")
        fig_box = px.box(
            df_model,
            x='sleep_disorder_risk',
            y='bmi',
            color='sleep_disorder_risk',
            color_discrete_map={0: '#3b82f6', 1: '#ec4899'}
        )
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=350)
        st.plotly_chart(fig_box, use_container_width=True)

    with c4:
        st.markdown("#### 🧠 Screen Time vs Risk")
        fig_hist = px.histogram(
            df_model,
            x='screen_time_before_bed_mins',
            color='sleep_disorder_risk',
            barmode='overlay',
            color_discrete_map={0: '#06b6d4', 1: '#f59e0b'}
        )
        fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: AI Models Benchmark
# ---------------------------------------------------------
elif menu_choice == "🤖 AI Models Benchmark":
    st.markdown("### 🏆 Machine Learning Model Leaderboard")
    
    metrics_df = pd.DataFrame({
        m: {
            'Accuracy': f"{model_results[m]['Accuracy']*100:.2f}%",
            'Precision': f"{model_results[m]['Precision']*100:.2f}%",
            'Recall': f"{model_results[m]['Recall']*100:.2f}%",
            'F1-Score': f"{model_results[m]['F1-Score']*100:.2f}%",
            'ROC-AUC': f"{model_results[m]['ROC-AUC']:.4f}"
        } for m in model_results
    }).T
    
    st.dataframe(metrics_df, use_container_width=True)
    
    st.markdown("---")
    col_sel, _ = st.columns([1, 1])
    selected_model_name = col_sel.selectbox("Select Model for Detailed Diagnostic Evaluation:", list(model_results.keys()))
    
    res = model_results[selected_model_name]
    
    col_cm, col_roc = st.columns([1, 1])
    
    with col_cm:
        st.markdown(f"#### 🧩 Confusion Matrix — {selected_model_name}")
        cm = confusion_matrix(res['y_test'], res['preds'])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale='Purples',
            x=['Healthy (0)', 'At Risk (1)'], y=['Healthy (0)', 'At Risk (1)']
        )
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with col_roc:
        st.markdown(f"#### 📉 ROC Curve Analysis")
        fpr, tpr, _ = roc_curve(res['y_test'], res['probs'])
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC = {res['ROC-AUC']:.3f}", line=dict(color='#8b5cf6', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='#94a3b8')))
        fig_roc.update_layout(xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=380)
        st.plotly_chart(fig_roc, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Real-Time Patient Risk Predictor
# ---------------------------------------------------------
elif menu_choice == "🔮 Real-Time Patient Risk Predictor":
    st.markdown("### 🩺 Patient Interactive Health Assessment")
    st.write("Enter physiological and lifestyle parameters to generate immediate AI risk probability.")
    
    with st.form("patient_form"):
        p_col1, p_col2, p_col3 = st.columns(3)
        
        with p_col1:
            age = st.number_input("Age", 18, 100, 32)
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            bmi = st.number_input("BMI", 15.0, 50.0, 24.5, step=0.1)
            sleep_hrs = st.slider("Sleep Duration (Hours)", 3.0, 12.0, 7.0, step=0.1)
            rem_pct = st.slider("REM Sleep %", 5.0, 35.0, 21.0, step=0.5)
            deep_sleep_pct = st.slider("Deep Sleep %", 5.0, 35.0, 20.0, step=0.5)
            
        with p_col2:
            sleep_lat = st.number_input("Sleep Latency (Minutes)", 0, 120, 15)
            wake_ep = st.number_input("Wake Episodes per Night", 0, 15, 2)
            caffeine = st.number_input("Caffeine Before Bed (mg)", 0, 500, 50)
            screen_time = st.number_input("Screen Time Before Bed (mins)", 0, 300, 45)
            exercise = st.radio("Exercised Today?", ["Yes", "No"], horizontal=True)
            steps = st.number_input("Daily Steps Count", 0, 30000, 8000)

        with p_col3:
            nap = st.number_input("Nap Duration (mins)", 0, 180, 0)
            stress = st.slider("Perceived Stress Score (1-10)", 1.0, 10.0, 4.0, step=0.1)
            heart_rate = st.number_input("Resting Heart Rate (BPM)", 40, 120, 68)
            shift_work_val = st.radio("Shift Work Employee?", ["No", "Yes"], horizontal=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            submit_btn = st.form_submit_button("⚡check now")
            
    if submit_btn:
        gender_val = 0 if gender_label == "Female" else 1
        exercise_val = 1 if exercise == "Yes" else 0
        shift_val = 1 if shift_work_val == "Yes" else 0
        
        # Creating a dataframe that exactly matches the training data columns
        user_data = pd.DataFrame([{
            'age': age,
            'gender': gender_val,
            'bmi': bmi,
            'sleep_duration_hrs': sleep_hrs,
            'rem_percentage': rem_pct,
            'deep_sleep_percentage': deep_sleep_pct,
            'sleep_latency_mins': sleep_lat,
            'wake_episodes_per_night': wake_ep,
            'caffeine_mg_before_bed': caffeine,
            'screen_time_before_bed_mins': screen_time,
            'exercise_day': exercise_val,
            'steps_that_day': steps,
            'nap_duration_mins': nap,
            'stress_score': stress,
            'heart_rate_resting_bpm': heart_rate,
            'shift_work': shift_val
        }])
        
        # We use Random Forest as the default predictor here
        best_model = fitted_models['Random Forest']
        pred = best_model.predict(user_data)[0]
        prob = best_model.predict_proba(user_data)[0][1]
        
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown("#### 🎯 Clinical Prediction Outcome")
            if pred == 1:
                st.error(f"⚠️ **AT RISK** — High Probability of Sleep Disorder")
            else:
                st.success(f"✅ **HEALTHY** — Low Probability of Sleep Disorder")
                
            st.metric("Predicted Disorder Probability", f"{prob*100:.1f}%")

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Risk Index Gauge"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#ef4444" if prob > 0.5 else "#10b981"},
                    'steps': [
                        {'range': [0, 40], 'color': "rgba(16, 185, 129, 0.2)"},
                        {'range': [40, 70], 'color': "rgba(245, 158, 11, 0.2)"},
                        {'range': [70, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                    ]
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#ffffff', height=260)
            st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: Raw Dataset Explorer
# ---------------------------------------------------------
elif menu_choice == "📂 Dataset Explorer":
    st.markdown("### 🔍 Interactive Cohort Data Explorer")
    
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        risk_filter = st.multiselect("Filter Target Status", options=[0, 1], default=[0, 1], format_func=lambda x: "Healthy (0)" if x==0 else "At Risk (1)")
    with f_col2:
        age_range = st.slider("Filter Age Range", int(df_model['age'].min()), int(df_model['age'].max()), (18, 65))
        
    filtered_df = df_model[
        (df_model['sleep_disorder_risk'].isin(risk_filter)) &
        (df_model['age'].between(age_range[0], age_range[1]))
    ]
    
    st.write(f"Showing **{len(filtered_df):,}** matching patient records:")
    st.dataframe(filtered_df, use_container_width=True)