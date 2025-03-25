import streamlit as st 
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import json
import time
import base64

# Set page title and icon
st.set_page_config(
    page_title="Advanced BMI Calculator",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.2), rgba(0, 0, 0, 0.2)),
                        url('https://i.pinimg.com/736x/45/63/32/456332c909f732da36000118c44943ab.jpg');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .title-text {
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
        font-size: 48px;
        text-shadow: 2px 2px 4px #000000;
        padding: 20px;
        text-align: center;
        background: linear-gradient(90deg, rgba(63,94,251,0.5) 0%, rgba(252,70,107,0.5) 100%);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .subtitle-text {
        color: #ffffff;
        font-family: 'Arial', sans-serif;
        font-size: 24px;
        text-shadow: 1px 1px 2px #000000;
        text-align: center;
        margin-bottom: 20px;
    }
    .content-section {
        background: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.3s, box-shadow 0.3s;
    }
    .content-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.3);
    }
    .stButton button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        cursor: pointer;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #45a049 0%, #3d8a41 100%);
        color: white;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .stNumberInput {
        transition: all 0.3s ease;
    }
    .stNumberInput:focus {
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
    }
    .stSlider {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        padding: 20px;
        color: white;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        margin-top: 20px;
    }
    .results-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(240, 240, 240, 0.9) 100%);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin-top: 20px;
        text-align: center;
    }
    .bmi-category {
        font-size: 24px;
        font-weight: bold;
        padding: 10px;
        border-radius: 8px;
        display: inline-block;
        margin: 10px 0;
    }
    .health-tip {
        margin-top: 15px;
        padding: 15px;
        border-radius: 8px;
        font-size: 18px;
    }
    .plotly-container {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Lottie animation function
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Load animations
health_animation = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_5njp3vgg.json")
weight_animation = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_Xf9mNu.json")

# Title and description with custom styling
st.markdown("<h1 class='title-text'>⚖️ Advanced BMI Calculator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Calculate your Body Mass Index (BMI) and discover personalized health insights.</p>", unsafe_allow_html=True)

# Create two columns for layout
col1, col2 = st.columns([1, 1])

with col1:
    # Animation
    if health_animation:
        st_lottie(health_animation, height=350, key="health")
    
    # Unit selection with improved styling
    st.markdown("<div class='content-section'>", unsafe_allow_html=True)
    st.subheader("📏 Choose Your Unit System")
    unit = st.selectbox(
        "Select measurement system:",
        ("Metric (kg, cm)", "Imperial (lbs, inches)", "Mixed (kg, feet/inches)")
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Input fields with improved styling
    st.markdown("<div class='content-section'>", unsafe_allow_html=True)
    st.subheader("📊 Enter Your Measurements")
    
    if unit == "Metric (kg, cm)":
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, key="weight_kg")
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1, key="height_cm")
        height_in_meters = height / 100
        
    elif unit == "Imperial (lbs, inches)":
        weight = st.number_input("Weight (lbs)", min_value=1.0, max_value=700.0, value=154.0, step=0.1, key="weight_lbs")
        height = st.number_input("Height (inches)", min_value=20.0, max_value=100.0, value=67.0, step=0.1, key="height_in")
        weight = weight * 0.453592  # Convert lbs to kg
        height_in_meters = height * 0.0254  # Convert inches to meters
        
    else:  # Mixed
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, step=0.1, key="weight_kg_mixed")
        col_ft, col_in = st.columns(2)
        with col_ft:
            feet = st.number_input("Feet", min_value=1, max_value=8, value=5, step=1)
        with col_in:
            inches = st.number_input("Inches", min_value=0, max_value=11, value=7, step=1)
        height_in_meters = (feet * 30.48 + inches * 2.54) / 100
    
    # Additional metrics
    age = st.slider("Age", min_value=2, max_value=120, value=30, step=1)
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    
    # Activity level
    activity_level = st.select_slider(
        "Activity Level",
        options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
        value="Moderately Active"
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Weight animation
    if weight_animation:
        st_lottie(weight_animation, height=300, key="weight")
    
    # Calculate BMI function
    def calculate_bmi(weight, height):
        bmi = weight / (height ** 2)
        return round(bmi, 2)
    
    # BMI Categories
    def get_bmi_category(bmi):
        if bmi < 18.5:
            return "Underweight", "#3366cc", "Your BMI indicates you're underweight. This may suggest insufficient calorie intake or other health issues."
        elif 18.5 <= bmi < 25:
            return "Normal Weight", "#4CAF50", "Your BMI is within the normal range. Keep maintaining a balanced diet and regular exercise."
        elif 25 <= bmi < 30:
            return "Overweight", "#ff9800", "Your BMI indicates you're overweight. Consider focusing on healthy dietary changes and increasing physical activity."
        elif 30 <= bmi < 35:
            return "Obese (Class I)", "#f44336", "Your BMI indicates Class I obesity. Consider consulting a healthcare professional for personalized advice."
        elif 35 <= bmi < 40:
            return "Obese (Class II)", "#e91e63", "Your BMI indicates Class II obesity. It's recommended to consult with healthcare professionals for a tailored weight management plan."
        else:
            return "Obese (Class III)", "#9c27b0", "Your BMI indicates Class III obesity. Please consult with healthcare professionals for medical guidance and support."
    
    # Calculate BMR (Basal Metabolic Rate)
    def calculate_bmr(weight, height_in_meters, age, gender):
        if gender == "Male":
            bmr = 88.362 + (13.397 * weight) + (4.799 * height_in_meters * 100) - (5.677 * age)
        else:
            bmr = 447.593 + (9.247 * weight) + (3.098 * height_in_meters * 100) - (4.330 * age)
        return round(bmr)
    
    # Calculate daily calorie needs
    def calculate_calories(bmr, activity):
        activity_factors = {
            "Sedentary": 1.2,
            "Lightly Active": 1.375,
            "Moderately Active": 1.55,
            "Very Active": 1.725,
            "Extremely Active": 1.9
        }
        return round(bmr * activity_factors[activity])
    
    # Calculate ideal weight range
    def calculate_ideal_weight(height_in_meters, gender):
        if gender == "Male":
            lower = round(18.5 * (height_in_meters ** 2), 1)
            upper = round(24.9 * (height_in_meters ** 2), 1)
        else:
            lower = round(18.5 * (height_in_meters ** 2), 1)
            upper = round(24.9 * (height_in_meters ** 2), 1)
        return lower, upper
    
    # Create BMI scale visualization with Plotly
    def create_bmi_gauge(bmi_value):
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bmi_value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BMI Scale"},
            gauge = {
                'axis': {'range': [10, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [10, 18.5], 'color': '#3366cc'},
                    {'range': [18.5, 25], 'color': '#4CAF50'},
                    {'range': [25, 30], 'color': '#ff9800'},
                    {'range': [30, 35], 'color': '#f44336'},
                    {'range': [35, 40], 'color': '#e91e63'},
                    {'range': [40, 50], 'color': '#9c27b0'}],
                'threshold': {
                    'line': {'color': "white", 'width': 6},
                    'thickness': 0.75,
                    'value': bmi_value}
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        
        return fig
    
    # Button to calculate with loading animation
    st.markdown("<div class='content-section'>", unsafe_allow_html=True)
    if st.button("📊 Calculate Health Metrics"):
        with st.spinner("Calculating your health metrics..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            
            # Calculate BMI
            bmi = calculate_bmi(weight, height_in_meters)
            category, color, description = get_bmi_category(bmi)
            
            # Calculate BMR and daily calorie needs
            bmr = calculate_bmr(weight, height_in_meters, age, gender)
            daily_calories = calculate_calories(bmr, activity_level)
            
            # Calculate ideal weight range
            lower_weight, upper_weight = calculate_ideal_weight(height_in_meters, gender)
            
            # Display results
            st.success("Calculations complete!")
            
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            st.subheader("🔍 Your BMI Results")
            
            # Display BMI with large font
            st.markdown(f"<h1 style='font-size:60px; color:{color};'>{bmi}</h1>", unsafe_allow_html=True)
            
            # Display category with styling
            st.markdown(f"<div class='bmi-category' style='background-color:{color}; color:white;'>{category}</div>", unsafe_allow_html=True)
            
            # Description
            st.markdown(f"<p style='font-size:18px;'>{description}</p>", unsafe_allow_html=True)
            
            # Visual feedback with Plotly gauge chart
            st.markdown("<div class='plotly-container'>", unsafe_allow_html=True)
            fig = create_bmi_gauge(bmi)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional health metrics
            st.subheader("📈 Additional Health Insights")
            
            # Create three columns for metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align:center;'>BMR</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align:center;'>{bmr} kcal/day</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;'>Calories your body needs at rest</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_b:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align:center;'>Daily Calories</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='text-align:center;'>{daily_calories} kcal/day</h2>", unsafe_allow_html=True)
                st.markdown("<p style='text-align:center;'>Calories to maintain current weight</p>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col_c:
                st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align:center;'>Ideal Weight Range</h3>", unsafe_allow_html=True)
                
                if unit == "Metric (kg, cm)":
                    st.markdown(f"<h2 style='text-align:center;'>{lower_weight} - {upper_weight} kg</h2>", unsafe_allow_html=True)
                elif unit == "Imperial (lbs, inches)":
                    lower_lbs = round(lower_weight / 0.453592, 1)
                    upper_lbs = round(upper_weight / 0.453592, 1)
                    st.markdown(f"<h2 style='text-align:center;'>{lower_lbs} - {upper_lbs} lbs</h2>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h2 style='text-align:center;'>{lower_weight} - {upper_weight} kg</h2>", unsafe_allow_html=True)
            
            st.markdown("<p style='text-align:center;'>Healthy weight range for your height</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Personalized health tips
        st.subheader("💡 Personalized Health Tips")
        
        if category == "Underweight":
            st.markdown("<div class='health-tip' style='background-color:rgba(51, 102, 204, 0.1); border-left:5px solid #3366cc;'>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Consider increasing your calorie intake with nutrient-dense foods</li>
                <li>Include healthy fats like avocados, nuts, and olive oil in your diet</li>
                <li>Incorporate strength training to build muscle mass</li>
                <li>Consult with a healthcare provider or dietitian for personalized advice</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif category == "Normal Weight":
            st.markdown("<div class='health-tip' style='background-color:rgba(76, 175, 80, 0.1); border-left:5px solid #4CAF50;'>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Maintain your balanced diet and regular exercise routine</li>
                <li>Aim for 150 minutes of moderate exercise per week</li>
                <li>Stay hydrated and get adequate sleep</li>
                <li>Continue regular health check-ups</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        elif category == "Overweight":
            st.markdown("<div class='health-tip' style='background-color:rgba(255, 152, 0, 0.1); border-left:5px solid #ff9800;'>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Focus on portion control and mindful eating</li>
                <li>Increase physical activity gradually (aim for 30 minutes daily)</li>
                <li>Reduce processed foods and added sugars</li>
                <li>Consider consulting a healthcare provider for personalized advice</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        else:  # Obese categories
            st.markdown("<div class='health-tip' style='background-color:rgba(244, 67, 54, 0.1); border-left:5px solid #f44336;'>", unsafe_allow_html=True)
            st.markdown("""
            <ul>
                <li>Consult with healthcare professionals for a comprehensive weight management plan</li>
                <li>Consider working with a registered dietitian for personalized nutrition guidance</li>
                <li>Start with gentle, low-impact exercises like walking or swimming</li>
                <li>Focus on sustainable lifestyle changes rather than quick fixes</li>
                <li>Monitor other health metrics like blood pressure and cholesterol</li>
            </ul>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Weight management plan
        st.subheader("🗓️ Weight Management Plan")
        
        # Weight goal
        if category == "Underweight":
            goal = "gain"
            calories_change = 500
        elif category in ["Normal Weight"]:
            goal = "maintain"
            calories_change = 0
        else:
            goal = "lose"
            calories_change = -500
        
        if goal != "maintain":
            target_calories = daily_calories + calories_change
            st.markdown(f"""
            <div style='background-color:rgba(33, 150, 243, 0.1); padding:15px; border-radius:8px; margin-top:10px;'>
                <h4>Recommended daily calorie intake to {goal} weight:</h4>
                <h2 style='text-align:center;'>{target_calories} kcal/day</h2>
                <p>This would result in approximately {abs(calories_change/500)} lb ({abs(calories_change/500 * 0.453592):.1f} kg) {goal} per week.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Create a weight projection chart
        if goal != "maintain":
            weeks = list(range(0, 13))  # 12 weeks projection
            if goal == "lose":
                current_weight_kg = weight
                projected_weights = [current_weight_kg + (i * (calories_change/500 * 0.453592)) for i in range(0, 13)]
                title = "12-Week Weight Loss Projection"
            else:
                current_weight_kg = weight
                projected_weights = [current_weight_kg + (i * abs(calories_change/500 * 0.453592)) for i in range(0, 13)]
                title = "12-Week Weight Gain Projection"
            
            # Create the plot with Plotly Express
            fig = px.line(
                x=weeks,
                y=projected_weights,
                labels={'x': 'Weeks', 'y': 'Weight (kg)'},
                title=title
            )
            
            # Customize the plot
            fig.update_traces(line=dict(color=color, width=3))
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                hovermode="x"
            )
            
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
        
        # Download button for report
        st.download_button(
            label="📥 Download Health Report",
            data=f"""
            BMI HEALTH REPORT
            -----------------
            Date: {time.strftime("%Y-%m-%d")}
            
            MEASUREMENTS
            Height: {height_in_meters*100:.1f} cm
            Weight: {weight:.1f} kg
            BMI: {bmi}
            Category: {category}
            
            HEALTH METRICS
            BMR: {bmr} kcal/day
            Daily Calorie Needs: {daily_calories} kcal/day
            Ideal Weight Range: {lower_weight} - {upper_weight} kg
            
            RECOMMENDATIONS
            {description}
            """,
            file_name="bmi_health_report.txt",
            mime="text/plain",
        )
        
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown("Made with ❤️ by Hafsa Kamali", unsafe_allow_html=True)
st.markdown("© 2025 BMI Calculator | Privacy Policy | Terms of Service", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)