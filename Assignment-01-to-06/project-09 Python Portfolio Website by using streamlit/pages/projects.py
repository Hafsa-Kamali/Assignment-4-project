import streamlit as st
import base64
from pathlib import Path
import plotly.graph_objs as go

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

def get_custom_css(bg_image=None, profile_image=None):
    return f"""
    <style>
    .stApp {{
        background: {f'url("data:image/jpeg;base64,{bg_image}")' if bg_image else '#0A192F'};
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        color: white;
    }}
    
    .profile-img {{
        width: 250px;
        height: 250px;
        object-fit: cover;
        border-radius: 50%;
        border: 5px solid #64FFDA;
        box-shadow: 0 0 30px rgba(100, 255, 218, 0.4);
        display: block;
        margin: 0 auto 20px;
    }}
    
    .profile-name {{
        text-align: center;
        font-size: 3em;
       background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
       -webkit-background-clip: text;
       background-clip: text;
       color: transparent;
        margin-bottom: 30px;
        font-weight: bold;
         text-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.4), /* Soft drop shadow */
        0 0 10px rgba(139, 92, 246, 0.4); /* Neon glow */
    }}
    
    .profile-bio {{
        text-align: center;
        max-width: 900px;
        margin: 0 auto 40px;
        color: #8892B0;
        line-height: 1.6;
    }}
    
    .project-card {{
        background-color: rgba(17, 34, 64, 0.7);
        border-radius: 15px;
        padding: 35px;
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        height: 100%;
        width: 100%;
    }}
    
    .project-card:hover {{
        transform: scale(1.03);
        box-shadow: 0 10px 20px rgba(100, 255, 218, 0.2);
    }}
    
    .tech-badge {{
        background-color: rgba(100, 255, 218, 0.1);
        color: #64FFDA;
        padding: 5px 10px;
        border-radius: 20px;
        margin-right: 10px;
        margin-bottom: 10px;
        display: inline-block;
    }}
    </style>
    """

def main():
    # Load images
    bg_image = img_to_base64(get_assets_dir() / "bg5.jpg")
    profile_image = img_to_base64(get_assets_dir() / "hafsa.png")
    
    # Apply custom CSS
    st.markdown(get_custom_css(bg_image, profile_image), unsafe_allow_html=True)

    # Profile Section
    if profile_image:
        st.markdown(f"""
        <img src="data:image/png;base64,{profile_image}" class="profile-img" alt="Hafsa Kamali">
        <div class="profile-name">Hafsa Kamali</div>
        <div class="profile-bio">
        A passionate Python developer and technology enthusiast with a keen interest in creating innovative software solutions. 
        My journey in programming is driven by a curiosity to solve real-world problems through code. 
        From health tech applications to utility tools, I strive to develop projects that make a meaningful impact. 
        Each project is an opportunity to learn, grow, and push the boundaries of what's possible with technology.
        </div>
        """, unsafe_allow_html=True)

    # Comprehensive project list
    projects = [
        {
            "name": "BMI Calculator",
            "description": "Comprehensive health tracking tool that calculates Body Mass Index with detailed health insights.",
            "technologies": ["Python", "Health Tech", "Data Analysis"],
            "link": "project-08-streamlit-bmi-calculator",
            "icon": "⚖️"
        },
        {
            "name": "Library Management System",
            "description": "Robust digital library management application for tracking books, members, and lending records.",
            "technologies": ["Python", "Database Management", "CRUD Operations"],
            "link": "project-01-madlib-python-project",
            "icon": "📚"
        },
        {
            "name": "Password Generator",
            "description": "Advanced secure password generation tool with customizable complexity and strength metrics.",
            "technologies": ["Python", "Cryptography", "Security"],
            "link": "project-07-password-generator-project",
            "icon": "🔐"
        },
        {
            "name": "Face Emotion Detector",
            "description": "AI-powered emotion recognition system using computer vision and machine learning techniques.",
            "technologies": ["OpenCV", "Machine Learning", "Computer Vision"],
            "link": "#face-emotion-detector",
            "icon": "😊"
        },
        {
            "name": "Unit Converter",
            "description": "Flexible unit conversion tool supporting multiple measurement systems and categories.",
            "technologies": ["Python", "Utility Tools", "Calculation"],
            "link": "#unit-converter",
            "icon": "📏"
        },
        {
            "name": "Growth Mind Challenge",
            "description": "Interactive personal development platform with daily challenges and progress tracking.",
            "technologies": ["Python", "Self-Improvement", "Goal Setting"],
            "link": "#growth-mind-challenge",
            "icon": "🌱"
        }
    ]

    # Project Grid Layout with Two Columns
    cols = st.columns(2)
    for i, project in enumerate(projects):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="project-card">
                <h3>{project['icon']} {project['name']}</h3>
                <p>{project['description']}</p>
                <div>
                    {"".join([f'<span class="tech-badge">{tech}</span>' for tech in project['technologies']])}
                </div>
                <br>
                <a href="{project['link']}" target="_blank" style="color: #64FFDA; text-decoration: none;">
                    🔗 Explore Project
                </a>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()