import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import base64
from pathlib import Path
import os

# Set page configuration
st.set_page_config(
    page_title="Hafsa Kamali | Professional Dashboard", 
    page_icon="👩‍💻", 
    layout="wide"
)

# Improved image handling function with multiple extension support
def img_to_base64(img_path):
    try:
        img_path = Path(img_path).absolute()
        if not img_path.exists():
            return None
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Error loading image {img_path}: {str(e)}")
        return None

# Get current directory with multiple asset location checks
def get_assets_dir():
    possible_locations = [
        Path(__file__).parent.absolute() / "assets",
        Path(__file__).parent.parent.absolute() / "assets",
        Path.cwd().absolute() / "assets"
    ]
    
    for location in possible_locations:
        if location.exists():
            return location
    
    assets_dir = Path(__file__).parent.absolute() / "assets"
    assets_dir.mkdir(exist_ok=True)
    return assets_dir

# Initialize assets directory
assets_dir = get_assets_dir()

# Custom CSS with reduced overlay effect
def get_custom_css(bg_image=None):
    if bg_image:
        bg_style = f'url("data:image/jpeg;base64,{bg_image}")'
    else:
        bg_style = '#0A192F'  # Default dark background color
    
    return f"""
    <style>
    .stApp {{
        background: {bg_style};
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    
    .profile-img-container {{
        position: relative;
        width: 300px;
        height: 300px;
        margin: 0 auto;
    }}
    
    .profile-img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 50%;
        border: 4px solid #64FFDA;
        box-shadow: 0 0 20px rgba(100, 255, 218, 0.4);
        animation: float 6s ease-in-out infinite;
    }}
    
    .gradient-text {{
        background: linear-gradient(90deg, #64FFDA, #8892B0);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-10px); }}
    }}
    </style>
    """

def show_fallback_image():
    st.markdown("""
    <div class="profile-img-container">
        <div style="width:100%; height:100%; border-radius:50%;; 
                    display:flex; align-items:center; justify-content:center;
                    border:4px solid #64FFDA; color:#64FFDA;">
            Profile Image
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Try multiple possible image names and extensions
    bg_image = None
    profile_image = None
    
    # Try different background image variations
    for bg_name in ['bg3', 'background', 'bg']:
        for ext in ['.jpeg', '.jpg', '.png']:
            bg_path = assets_dir / f"{bg_name}{ext}"
            if bg_path.exists():
                bg_image = img_to_base64(bg_path)
                break
        if bg_image:
            break
    
    # Try different profile image variations
    for profile_name in ['hafsa', 'profile', 'hafsa_kamali']:
        for ext in ['.png', '.jpg', '.jpeg']:
            profile_path = assets_dir / f"{profile_name}{ext}"
            if profile_path.exists():
                profile_image = img_to_base64(profile_path)
                break
        if profile_image:
            break
    
    # Apply custom CSS with reduced overlay
    st.markdown(get_custom_css(bg_image), unsafe_allow_html=True)
    
    # Profile Section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if profile_image:
            st.markdown(f"""
            <div class="profile-img-container">
                <img src="data:image/png;base64,{profile_image}" class="profile-img" alt="Profile">
            </div>
            """, unsafe_allow_html=True)
        else:
            show_fallback_image()
            st.warning("""
            Using fallback profile image. Please ensure:
            1. Image exists in the assets folder
            2. File is named one of: hafsa.png, profile.jpg, etc.
            3. File extension matches (.png, .jpg, .jpeg)
            """)
    
    with col2:
        st.markdown("# <span class='gradient-text'>Hafsa Kamali</span>", unsafe_allow_html=True)
        st.markdown("## Passionate Technologist", unsafe_allow_html=True)
        st.markdown("""
        Currently part of the Governor House IT Initiative, 
        blending technology with creativity to build impactful solutions.
        """)

    # Rest of your dashboard code remains unchanged...
    # Skills Visualization
    st.markdown("## 💻 Core Skills", unsafe_allow_html=True)
    
    # Language Skills
    languages_data = {
        'Skill': ['Python', 'JavaScript', 'TypeScript', 'HTML/CSS', 'React'],
        'Proficiency': [95, 85, 80, 90, 85]
    }
    df_languages = pd.DataFrame(languages_data)
    
    fig_languages = px.bar(
        df_languages, 
        x='Skill', 
        y='Proficiency', 
        title='Programming Languages & Frameworks',
        color='Skill',
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig_languages.update_layout(
        plot_bgcolor='rgba(10, 25, 47, 0.8)',
        paper_bgcolor='rgba(10, 25, 47, 0.5)',
        font_color='white'
    )
    st.plotly_chart(fig_languages, use_container_width=True)

    # Frameworks and Tools
    st.markdown("### 🛠 Frameworks & Tools", unsafe_allow_html=True)
    tools_data = {
        'Category': ['Web Frameworks', 'ML Frameworks', 'Databases', 'DevOps'],
        'Tools': [
            'Streamlit, Flask, React, Next.js',
            'TensorFlow, PyTorch, Jupyter',
            'SQL, NoSQL, Pandas',
            'Git, Docker',
        ]
    }
    df_tools = pd.DataFrame(tools_data)
    st.table(df_tools.style.set_properties(**{
        'background-color': 'rgba(10, 25, 47, 0.7)',
        'color': 'white',
        'border-color': '#64FFDA'
    }))

    # Project Complexity Radar Chart
    project_skills = {
        'Skill': ['Python Dev', 'ML & AI', 'Web Apps', 'Data Analysis'],
        'Expertise': [90, 85, 80, 75]
    }
    
    fig_radar = go.Figure(data=go.Scatterpolar(
        r=project_skills['Expertise'],
        theta=project_skills['Skill'],
        fill='toself',
        line_color='#64FFDA',
        fillcolor='rgba(100, 255, 218, 0.2)'
    ))
    fig_radar.update_layout(
        title='Professional Expertise Radar',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                color='white'
            ),
            bgcolor='rgba(10, 25, 47, 0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    st.plotly_chart(fig_radar, use_container_width=True)

if __name__ == "__main__":
    main()