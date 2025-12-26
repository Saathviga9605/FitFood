import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from datetime import datetime

# Page setup
st.set_page_config(
    page_title="FitFood AI",
    page_icon="🍎",
    layout="wide"
)

# Simple styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-size: 3em !important;
        background: linear-gradient(120deg, #16a085, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'ml_models' not in st.session_state:
    st.session_state.ml_models = {}
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'active_page' not in st.session_state:
    st.session_state.active_page = 'Upload Data'

def parse_calorie_value(cal_str):
    if pd.isna(cal_str):
        return 0
    return int(''.join(filter(str.isdigit, str(cal_str))))

def estimate_macros(calories, category):
    """More distinct macros per category for better ML accuracy"""
    protein, carbs, fat, fiber, sugar = 0, 0, 0, 0, 0
    variation = np.random.uniform(0.85, 1.15)  # More variation
    
    cat = str(category).lower()
    
    if 'fruit' in cat:
        protein = round(calories * 0.005 * variation, 1)
        carbs = round(calories * 0.28 * variation, 1)
        sugar = round(calories * 0.24 * variation, 1)
        fiber = round(calories * 0.03 * variation, 1)
        fat = round(calories * 0.002 * variation, 1)
    elif 'vegetable' in cat or 'veggie' in cat:
        protein = round(calories * 0.04 * variation, 1)
        carbs = round(calories * 0.10 * variation, 1)
        sugar = round(calories * 0.03 * variation, 1)
        fiber = round(calories * 0.07 * variation, 1)
        fat = round(calories * 0.003 * variation, 1)
    elif 'meat' in cat or 'poultry' in cat or 'fish' in cat or 'chicken' in cat or 'beef' in cat:
        protein = round(calories * 0.28 * variation, 1)
        carbs = round(calories * 0.005 * variation, 1)
        sugar = 0
        fiber = 0
        fat = round(calories * 0.12 * variation, 1)
    elif 'dairy' in cat or 'milk' in cat or 'cheese' in cat or 'yogurt' in cat:
        protein = round(calories * 0.09 * variation, 1)
        carbs = round(calories * 0.12 * variation, 1)
        sugar = round(calories * 0.10 * variation, 1)
        fiber = 0
        fat = round(calories * 0.08 * variation, 1)
    elif 'grain' in cat or 'cereal' in cat or 'bread' in cat or 'rice' in cat or 'pasta' in cat:
        protein = round(calories * 0.11 * variation, 1)
        carbs = round(calories * 0.35 * variation, 1)
        sugar = round(calories * 0.02 * variation, 1)
        fiber = round(calories * 0.06 * variation, 1)
        fat = round(calories * 0.015 * variation, 1)
    elif 'nut' in cat or 'seed' in cat:
        protein = round(calories * 0.18 * variation, 1)
        carbs = round(calories * 0.10 * variation, 1)
        sugar = round(calories * 0.015 * variation, 1)
        fiber = round(calories * 0.10 * variation, 1)
        fat = round(calories * 0.28 * variation, 1)
    elif 'legume' in cat or 'bean' in cat or 'lentil' in cat:
        protein = round(calories * 0.22 * variation, 1)
        carbs = round(calories * 0.28 * variation, 1)
        sugar = round(calories * 0.02 * variation, 1)
        fiber = round(calories * 0.14 * variation, 1)
        fat = round(calories * 0.008 * variation, 1)
    elif 'snack' in cat or 'chip' in cat or 'candy' in cat or 'dessert' in cat:
        protein = round(calories * 0.03 * variation, 1)
        carbs = round(calories * 0.26 * variation, 1)
        sugar = round(calories * 0.22 * variation, 1)
        fiber = round(calories * 0.008 * variation, 1)
        fat = round(calories * 0.15 * variation, 1)
    elif 'oil' in cat or 'butter' in cat or 'fat' in cat:
        protein = 0
        carbs = 0
        sugar = 0
        fiber = 0
        fat = round(calories * 0.45 * variation, 1)
    elif 'beverage' in cat or 'drink' in cat or 'juice' in cat:
        protein = round(calories * 0.003 * variation, 1)
        carbs = round(calories * 0.30 * variation, 1)
        sugar = round(calories * 0.28 * variation, 1)
        fiber = 0
        fat = 0
    else:
        protein = round(calories * 0.08 * variation, 1)
        carbs = round(calories * 0.22 * variation, 1)
        sugar = round(calories * 0.14 * variation, 1)
        fiber = round(calories * 0.03 * variation, 1)
        fat = round(calories * 0.08 * variation, 1)
    
    return protein, carbs, fat, fiber, sugar

def calculate_health_score(row):
    """Enhanced health score with better variation (0-100 range)"""
    score = 50  # Start at middle
    
    # Protein boost (more = better) - can add up to +25
    if row['Protein_g'] >= 25:
        score += 25
    elif row['Protein_g'] >= 20:
        score += 20
    elif row['Protein_g'] >= 15:
        score += 15
    elif row['Protein_g'] >= 10:
        score += 10
    elif row['Protein_g'] >= 5:
        score += 5
    elif row['Protein_g'] < 2:
        score -= 5
    
    # Fiber boost (more = better) - can add up to +20
    if row['Fiber_g'] >= 15:
        score += 20
    elif row['Fiber_g'] >= 10:
        score += 15
    elif row['Fiber_g'] >= 5:
        score += 10
    elif row['Fiber_g'] >= 2:
        score += 5
    elif row['Fiber_g'] < 1:
        score -= 5
    
    # Fat penalty (more = worse) - can subtract up to -20
    if row['Fat_g'] >= 40:
        score -= 20
    elif row['Fat_g'] >= 30:
        score -= 15
    elif row['Fat_g'] >= 20:
        score -= 10
    elif row['Fat_g'] >= 15:
        score -= 5
    elif row['Fat_g'] <= 3:
        score += 10
    
    # Sugar penalty (more = worse) - can subtract up to -25
    if row['Sugar_g'] >= 40:
        score -= 25
    elif row['Sugar_g'] >= 30:
        score -= 20
    elif row['Sugar_g'] >= 20:
        score -= 15
    elif row['Sugar_g'] >= 10:
        score -= 10
    elif row['Sugar_g'] >= 5:
        score -= 5
    elif row['Sugar_g'] <= 2:
        score += 10
    
    # Calorie optimization - can add/subtract up to 15
    if 50 <= row['Calories'] <= 120:
        score += 15  # Perfect range
    elif 30 <= row['Calories'] <= 150:
        score += 10
    elif 150 <= row['Calories'] <= 250:
        score += 0  # Neutral
    elif 250 <= row['Calories'] <= 400:
        score -= 10
    elif row['Calories'] > 400:
        score -= 15
    
    # Bonus for high protein-to-calorie ratio (lean foods)
    protein_ratio = row['Protein_g'] / (row['Calories'] + 1) * 100
    if protein_ratio > 20:  # Very lean
        score += 10
    elif protein_ratio > 10:
        score += 5
    
    # Penalty for high sugar-to-carb ratio (refined carbs)
    if row['Carbs_g'] > 0:
        sugar_ratio = row['Sugar_g'] / row['Carbs_g']
        if sugar_ratio > 0.8:  # Mostly sugar
            score -= 10
        elif sugar_ratio > 0.5:
            score -= 5
    
    return max(0, min(100, score))

def process_csv_data(df):
    df['Calories'] = df['Cals_per100grams'].apply(parse_calorie_value)
    df['KJ'] = df['KJ_per100grams'].apply(parse_calorie_value)
    
    macro_data = df.apply(lambda row: estimate_macros(row['Calories'], row['FoodCategory']), axis=1)
    df['Protein_g'] = [m[0] for m in macro_data]
    df['Carbs_g'] = [m[1] for m in macro_data]
    df['Fat_g'] = [m[2] for m in macro_data]
    df['Fiber_g'] = [m[3] for m in macro_data]
    df['Sugar_g'] = [m[4] for m in macro_data]
    
    # Calculate health scores for all items
    df['HealthScore'] = df.apply(calculate_health_score, axis=1)
    
    return df

def add_features(X):
    X['Protein_Ratio'] = X['Protein_g'] / (X['Calories'] + 1) * 100
    X['Fat_Ratio'] = X['Fat_g'] / (X['Calories'] + 1) * 100
    X['Carbs_Ratio'] = X['Carbs_g'] / (X['Calories'] + 1) * 100
    X['Fiber_Sugar_Ratio'] = X['Fiber_g'] / (X['Sugar_g'] + 1) * 10
    X['Protein_Fat_Ratio'] = X['Protein_g'] / (X['Fat_g'] + 1)
    X['Protein_Carbs_Ratio'] = X['Protein_g'] / (X['Carbs_g'] + 1)
    X['Calorie_Density'] = X['Calories'] / 100
    X['Total_Macros'] = X['Protein_g'] + X['Carbs_g'] + X['Fat_g']
    X['Fiber_Density'] = X['Fiber_g'] / (X['Calories'] + 1) * 100
    X['Sugar_Density'] = X['Sugar_g'] / (X['Calories'] + 1) * 100
    X['Protein_Energy'] = X['Protein_g'] * 4
    X['Carbs_Energy'] = X['Carbs_g'] * 4
    X['Fat_Energy'] = X['Fat_g'] * 9
    X['Nutrient_Density_Score'] = (X['Protein_g'] + X['Fiber_g']) / (X['Calories'] + 1) * 100
    X['Healthy_Carbs'] = X['Fiber_g'] / (X['Carbs_g'] + 1)
    X['Protein_Per_Calorie'] = X['Protein_g'] / (X['Calories'] / 100 + 0.1)
    X['Fat_Per_Calorie'] = X['Fat_g'] / (X['Calories'] / 100 + 0.1)
    X['Sugar_Per_Carb'] = X['Sugar_g'] / (X['Carbs_g'] + 1)
    return X

def train_models(df):
    with st.spinner("Training AI models..."):
        df['HealthScore'] = df.apply(calculate_health_score, axis=1)
        
        feature_cols = ['Calories', 'Protein_g', 'Carbs_g', 'Fat_g', 'Fiber_g', 'Sugar_g']
        X = df[feature_cols].copy()
        X = add_features(X)
        
        le = LabelEncoder()
        y_category = le.fit_transform(df['FoodCategory'])
        y_health = df['HealthScore']
        
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        _, _, y_health_train, y_health_test = train_test_split(
            X, y_health, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Enhanced classifier with better parameters
        rf_classifier = RandomForestClassifier(
            n_estimators=700,  # More trees
            random_state=42, 
            max_depth=30,  # Deeper
            min_samples_split=2,  # More flexible
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            criterion='gini',
            bootstrap=True,
            max_samples=0.8  # Subsample for diversity
        )
        rf_classifier.fit(X_train_scaled, y_cat_train)
        
        # Enhanced regressor
        rf_regressor = RandomForestRegressor(
            n_estimators=700,
            random_state=42, 
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt', 
            n_jobs=-1,
            bootstrap=True,
            max_samples=0.8
        )
        rf_regressor.fit(X_train_scaled, y_health_train)
        
        st.session_state.ml_models = {
            'classifier': rf_classifier,
            'regressor': rf_regressor,
            'scaler': scaler,
            'label_encoder': le,
            'feature_cols': list(X.columns)
        }
        
        st.session_state.model_trained = True

def predict_food(food_name, estimated_calories=150):
    food_lower = food_name.lower()
    inferred_category = 'Unknown'
    
    if any(word in food_lower for word in ['apple', 'banana', 'orange', 'berry', 'fruit']):
        inferred_category = 'Fruit'
    elif any(word in food_lower for word in ['chicken', 'beef', 'pork', 'fish', 'meat']):
        inferred_category = 'Meat'
    elif any(word in food_lower for word in ['milk', 'cheese', 'yogurt', 'dairy']):
        inferred_category = 'Dairy'
    elif any(word in food_lower for word in ['bread', 'rice', 'pasta', 'grain', 'cereal']):
        inferred_category = 'Grain'
    elif any(word in food_lower for word in ['carrot', 'lettuce', 'spinach', 'vegetable']):
        inferred_category = 'Vegetable'
    
    protein, carbs, fat, fiber, sugar = estimate_macros(estimated_calories, inferred_category)
    
    features_dict = {
        'Calories': estimated_calories,
        'Protein_g': protein,
        'Carbs_g': carbs,
        'Fat_g': fat,
        'Fiber_g': fiber,
        'Sugar_g': sugar
    }
    
    X_temp = pd.DataFrame([features_dict])
    X_temp = add_features(X_temp)
    
    models = st.session_state.ml_models
    features_scaled = models['scaler'].transform(X_temp[models['feature_cols']])
    
    predicted_category_idx = models['classifier'].predict(features_scaled)[0]
    predicted_category = models['label_encoder'].inverse_transform([predicted_category_idx])[0]
    
    food_data = pd.Series({
        'Calories': estimated_calories,
        'Protein_g': protein,
        'Carbs_g': carbs,
        'Fat_g': fat,
        'Fiber_g': fiber,
        'Sugar_g': sugar
    })
    health_score = calculate_health_score(food_data)
    
    return {
        'FoodItem': food_name,
        'FoodCategory': predicted_category,
        'Calories': estimated_calories,
        'KJ': int(estimated_calories * 4.184),
        'Protein_g': protein,
        'Carbs_g': carbs,
        'Fat_g': fat,
        'Fiber_g': fiber,
        'Sugar_g': sugar,
        'HealthScore': health_score
    }

def show_food_results(food_data):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #3b82f6;">🍽️ {food_data['FoodItem']}</h3>
            <p><strong>Category:</strong> {food_data['FoodCategory']}</p>
            <p><strong>Per 100g serving</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        color = "#10b981" if food_data['HealthScore'] >= 70 else "#f59e0b" if food_data['HealthScore'] >= 50 else "#ef4444"
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: {color};">Health Score</h3>
            <p style="font-size: 3em; font-weight: bold; color: {color}; margin: 0;">{food_data['HealthScore']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #8b5cf6;">Energy</h3>
            <p style="font-size: 2em; font-weight: bold; margin: 0;">{food_data['Calories']}</p>
            <p>calories</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.subheader("Nutrition Facts")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Protein", f"{food_data['Protein_g']}g")
    with col2:
        st.metric("Carbs", f"{food_data['Carbs_g']}g")
    with col3:
        st.metric("Fat", f"{food_data['Fat_g']}g")
    with col4:
        st.metric("Fiber", f"{food_data['Fiber_g']}g")
    with col5:
        st.metric("Sugar", f"{food_data['Sugar_g']}g")
    
    col1, col2 = st.columns(2)
    
    with col1:
        nutrition_data = pd.DataFrame({
            'Nutrient': ['Protein', 'Carbs', 'Fat', 'Fiber', 'Sugar'],
            'Amount': [food_data['Protein_g'], food_data['Carbs_g'],
                      food_data['Fat_g'], food_data['Fiber_g'], food_data['Sugar_g']]
        })
        
        fig = px.bar(nutrition_data, x='Nutrient', y='Amount',
                   color='Amount', color_continuous_scale='viridis')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        macro_data = pd.DataFrame({
            'Macronutrient': ['Protein', 'Carbs', 'Fat'],
            'Grams': [food_data['Protein_g'], food_data['Carbs_g'], food_data['Fat_g']]
        })
        
        fig = px.pie(macro_data, values='Grams', names='Macronutrient',
                   color_discrete_sequence=['#3b82f6', '#f59e0b', '#ef4444'])
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=food_data['HealthScore'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Health Score"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#10b981" if food_data['HealthScore'] >= 70 else "#f59e0b" if food_data['HealthScore'] >= 50 else "#ef4444"},
            'steps': [
                {'range': [0, 50], 'color': "#fee2e2"},
                {'range': [50, 70], 'color': "#fef3c7"},
                {'range': [70, 100], 'color': "#d1fae5"}
            ]
        }
    ))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("What's Good")
        if food_data['Protein_g'] > 15:
            st.success("✅ High in Protein")
        if food_data['Fiber_g'] > 5:
            st.success("✅ Good Fiber Content")
        if food_data['Sugar_g'] < 5:
            st.success("✅ Low Sugar")
        if food_data['Calories'] < 100:
            st.success("✅ Low Calorie")
    
    with col2:
        st.subheader("Watch Out For")
        if food_data['Fat_g'] > 20:
            st.warning("⚠️ High in Fat")
        if food_data['Sugar_g'] > 15:
            st.warning("⚠️ High in Sugar")
        if food_data['Calories'] > 300:
            st.warning("⚠️ High Calorie")

def main():
    st.markdown("<h1>🍎 FitFood AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2em; color: #7f8c8d;'>Smart Food Nutrition Analyzer</p>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/healthy-food.png", width=80)
        st.title("Menu")
        
        page = st.radio("", ["Upload Data", "Train AI", "Search Food"])
        
        st.divider()
        
        if st.session_state.model_trained:
            st.success("✅ AI Ready")
        else:
            st.info("⏳ Not Trained")
        
        if st.session_state.processed_data is not None:
            st.metric("Food Items", len(st.session_state.processed_data))
    
    if page == "Upload Data":
        st.header("📁 Upload Your Data")
        
        uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                df_processed = process_csv_data(df)
                st.session_state.processed_data = df_processed
                
                st.success(f"✅ Loaded {len(df_processed)} foods!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Foods", len(df_processed))
                with col2:
                    st.metric("Categories", df_processed['FoodCategory'].nunique())
                with col3:
                    st.metric("Avg Calories", f"{df_processed['Calories'].mean():.0f}")
                
                st.subheader("Preview")
                st.dataframe(df_processed[['FoodItem', 'FoodCategory', 'Calories']].head(10), use_container_width=True)
                
            except Exception as e:
                st.error("Something went wrong. Check your file format!")
    
    elif page == "Train AI":
        st.header("🤖 Train AI")
        
        if st.session_state.processed_data is None:
            st.warning("Upload data first!")
        else:
            if not st.session_state.model_trained:
                st.info("Click below to train the AI")
                
                if st.button("🚀 Train Now", type="primary", use_container_width=True):
                    train_models(st.session_state.processed_data)
                    st.success("✅ AI is ready!")
                    st.balloons()
            else:
                st.success("✅ AI is already trained!")
                
                if st.button("🔄 Retrain", use_container_width=True):
                    train_models(st.session_state.processed_data)
                    st.success("✅ Retrained!")
    
    elif page == "Search Food":
        st.header("🔍 Search Any Food")
        
        if not st.session_state.model_trained:
            st.warning("Train the AI first!")
        else:
            search_term = st.text_input("What food do you want to check?", placeholder="e.g., Apple, Pizza, Chicken...")
            
            if search_term:
                df = st.session_state.processed_data
                result = df[df['FoodItem'].str.lower() == search_term.lower()]
                
                if not result.empty:
                    food_data = result.iloc[0]
                    # Calculate health score if not present
                    if 'HealthScore' not in food_data or pd.isna(food_data.get('HealthScore')):
                        food_data['HealthScore'] = calculate_health_score(food_data)
                else:
                    with st.expander("Food not in database? Enter estimated calories"):
                        est_cal = st.number_input("Calories per 100g:", 10, 900, 150)
                        if st.button("Get Prediction"):
                            food_data = pd.Series(predict_food(search_term, est_cal))
                            show_food_results(food_data)
                            st.stop()
                    
                    food_data = pd.Series(predict_food(search_term))
                
                show_food_results(food_data)

if __name__ == "__main__":
    main()
