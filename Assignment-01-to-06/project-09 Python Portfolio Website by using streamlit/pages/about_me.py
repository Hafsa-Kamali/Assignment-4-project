import streamlit as st
import base64
import os
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="About Me - Hafsa Kamali",
    page_icon="👨‍💻",
    layout="wide"
)

def get_image_base64(image_path):
    """Safe image loading with detailed error handling"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        st.error(f"🚨 Image not found at: {image_path}")
    except PermissionError:
        st.error(f"🔒 Permission denied for: {image_path}")
    except Exception as e:
        st.error(f"❌ Error loading {image_path}: {str(e)}")
    return None

def get_custom_css():
    return """
    <style>
   .profile-image {
        border-radius: 50%;
        border: 4px solid #8b5cf6;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
        transition: all 0.3s ease;
        width: 300px;
        height: 300px;
        object-fit: cover;
      /* Choose ONE of these animation effects below */
        
        /* Option 1: Gentle pulse animation */
        animation: pulse 3s ease-in-out infinite;
        
        /* Option 2: Subtle float animation */
        /* animation: float 6s ease-in-out infinite; */
    }
    
    /* Keyframes for pulse animation */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.03); }
        100% { transform: scale(1); }
    }
    
    /* Keyframes for float animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    
    .profile-image:hover {
        /* Preserve your existing hover effects */
        transform: scale(1.03);
        box-shadow: 0 8px 25px rgba(139, 92, 246, 0.5);
        /* Add a little extra to the hover */
        animation-play-state: paused;
    }
    /* About Me Container */
    .about-container {
        background-color: rgba(0,0,0,0.7);
        border-radius: 15px;
        padding: 2.5rem;
        color: white;
        backdrop-filter: blur(10px);
    }
    .main-heading {
    background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: bold;
    font-size: 1em;
    text-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.2), /* Soft drop shadow */
        0 0 10px rgba(139, 92, 246, 0.4); /* Neon glow */
 }
  /* Contact Form Styling */
    .stForm {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 20px !important;
        border: 2px solid transparent !important;
        background-clip: padding-box !important;
        position: relative !important;
    }
    
    .stForm::before {
        content: "";
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        z-index: -1;
        border-radius: 12px;
        background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
    }
    
    /* Input Fields */
    .stTextInput>div>div>input, 
    .stTextArea>div>textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Labels */
    .stTextInput>label, 
    .stTextArea>label,
    .stMarkdown>div>div>h2 {
        color: white !important;
    }
    
    /* Submit Button */
    .stForm button {
        background: linear-gradient(90deg, #8b5cf6, #ec4899) !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 8px 16px !important;
        transition: all 0.3s ease !important;
    }
    
    .stForm button:hover {
        transform: scale(1.02) !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5) !important;
    }
 
    </style>
    """

def main():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Get correct assets path (main project folder)
    current_dir = Path(__file__).parent.parent
    assets_dir = current_dir / "assets"
    
    # Debug sidebar
    with st.sidebar.expander("🛠 Debug Info"):
        st.write(f"**Assets Directory:** `{assets_dir}`")
        if assets_dir.exists():
            image_files = [f for f in assets_dir.glob("*") if f.suffix.lower() in ('.png', '.jpg', '.jpeg')]
            st.write(f"**Found {len(image_files)} images:**")
            for img in image_files:
                st.write(f"- `{img.name}`")
        else:
            st.error("Assets directory doesn't exist!")

    # PROFILE IMAGE - Robust loading
    profile_image = None
    profile_path = None
    
    # Try possible profile image names in order of priority
    possible_names = [
        "hafsa.png", "hafi.jpg", "hafsa.jpeg",  # Exact matches
        "profile.png", "profile.jpg", "profile.jpeg",  # Common names
        "user.png", "avatar.png"  # Fallbacks
    ]
    
    for name in possible_names:
        test_path = assets_dir / name
        if test_path.exists():
            profile_path = test_path
            break
    
    if profile_path:
        st.sidebar.success(f"✅ Using profile image: `{profile_path.name}`")
        profile_image = get_image_base64(profile_path)
    else:
        st.sidebar.error("❌ No suitable profile image found!")
    
    # BACKGROUND IMAGE
    bg_image = None
    bg_path = assets_dir / "bg2.jpg"  # Primary
    if not bg_path.exists():
        bg_path = next((f for f in assets_dir.glob("*") 
                       if "background" in f.name.lower() and f.suffix.lower() in ('.jpg', '.jpeg', '.png')), None)
    
    if bg_path:
        bg_image = get_image_base64(bg_path)
        st.sidebar.success(f"✅ Using background: `{bg_path.name}`")
    
    # Apply background
    if bg_image:
        st.markdown(f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{bg_image}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        }
        </style>
        """, unsafe_allow_html=True)
        st.sidebar.warning("⚠️ Using gradient background - no image found")

    # MAIN CONTENT
    with st.container():
        st.markdown('<div class="about-container">', unsafe_allow_html=True)

# With this cleaner version:
        st.markdown("""
      <style>
       .main-heading {
        background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        color: transparent !important;
        font-weight: bold !important;
        font-size: 2em !important;  /* Increased from 1em */
        text-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.2),
            0 0 10px rgba(139, 92, 246, 0.4) !important;
        
    }
     
    
</style>
<h1>👩🏻‍💼 <span class="main-heading">Hafsa Kamali</span></h1>
""", unsafe_allow_html=True)
        
        # Profile + Summary Columns
        col1, col2 = st.columns([1, 2], gap="large")
        
        with col1:
            if profile_image:
                st.markdown(
                    f'<img src="data:image/png;base64,{profile_image}" class="profile-image">',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '''<div class="profile-image" style="
                        display: flex; 
                        align-items: center; 
                        justify-content: center;
                        color: white;
                        font-weight: bold;
                        background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%);
                    ">Hafsa Kamali</div>''',
                    unsafe_allow_html=True
                )
        
        with col2:
            st.markdown("""
            ## Professional Summary
            
            I'm a passionate technologist with expertise in:
            - **Python Development**
            - **Machine Learning & AI**
            - **Web Applications**
            - **Data Analysis**
            - **Certified Web Developer**
            
            ### Core Skills
            - **Languages:** Python, JavaScript, TypeScript, HTML, CSS, React 
            - **Frameworks:** Streamlit, Flask, React, Next.js , Jupyter, Django , Node.js, Firebase
            - **Tools:** Git, Docker, TensorFlow, PyTorch
            - **Data:** SQL, NoSQL, Pandas, NumPy, Matplotlib, Seaborn
            
            Currently part of the **Governor House IT Initiative**,
            I blend technology with creativity to build impactful solutions.
            """)
  # Contact Form
        st.markdown("## 📬 Get in Touch")
        with st.form("contact_form"):
            cols = st.columns(2)
            with cols[0]:
                name = st.text_input("Your Name")
            with cols[1]:
                email = st.text_input("Your Email")
            message = st.text_area("Your Message")
            submit_button = st.form_submit_button("Send Message")
            
            if submit_button:
                st.success("Thank you! I'll respond soon.")
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
