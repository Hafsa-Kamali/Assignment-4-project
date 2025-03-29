import streamlit as st
import base64
import os
import random
import time

# Set page configuration
st.set_page_config(
    page_title="Hafsa Kamali's Portfolio",
    page_icon="👨‍💻",
    layout="wide"
)

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except Exception as e:
        st.error(f"Error encoding image {image_path}: {e}")
        return None

# Custom CSS for styling and animations
def get_custom_css():
    return """
    <style>
    @keyframes backgroundSlide {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .stApp {
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        transition: background-image 1s ease-in-out;
        animation: backgroundSlide 5s ease infinite;
        opacity: 1;
    }
    .custom-profile-image {
        border-radius: 50% !important;
        border: 4px solid white !important;
        transition: all 0.5s ease !important;
        animation: float 2s ease-in-out infinite !important;
        object-fit: cover;
        max-width: 300px !important;
        max-height: 300px !important;
        margin-top: 40px !important;
    }
    .nav-link {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 15px;
        background-color: rgba(255,255,255,0.1);
        border-radius: 10px;
        transition: all 0.3s ease;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .nav-link:hover {
        background-color: rgba(255,255,255,0.2);
        transform: scale(1.05);
    }
.main-heading {
    background: linear-gradient(90deg, #8b5cf6, #ec4899, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: bold;
    font-size: 1.5em;
    text-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.2), /* Soft drop shadow */
        0 0 10px rgba(139, 92, 246, 0.4); /* Neon glow */
}
.sub-heading {
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
    </style>
    """

# Fetch images with full error handling
def get_images(image_list):
    valid_images = []
    for img in image_list:
        full_path = os.path.abspath(img)
        if os.path.exists(full_path):
            valid_images.append(full_path)
        else:
            st.warning(f"Image not found: {full_path}")
    return valid_images

# Get image base64 with detailed error logging
def get_image_base64(image_path):
    try:
        # Use absolute path
        full_path = os.path.abspath(image_path)
        with open(full_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        st.error(f"Failed to load background image {full_path}: {e}")
        return None

# Main function
def main():
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    # Set background
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Comprehensive background image paths
    background_paths = [
        os.path.join(current_dir, "assets", "freepik__upload__26918.jpeg"),
        os.path.join(current_dir, "assets", "bg1.jpg"),
        os.path.join(current_dir,  "assets", "bg2.jpg"),
        os.path.join(current_dir, "assets", "bg3.jpeg"),
        os.path.join(current_dir, "assets", "bg4.jpg"),
        os.path.join(current_dir, "assets", "bg5.jpg"),
    ]
    
    # Load background images
    background_images = []
    for path in background_paths:
        abs_path = os.path.abspath(path)
        bg_image = get_image_base64(abs_path)
        if bg_image:
            background_images.append(bg_image)
    
    # Set background or use gradient
    if background_images:
        # Choose first image to start
        first_image = background_images[0]
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{first_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            animation: backgroundSlide 5s ease infinite;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            animation: backgroundSlide 5s ease infinite;
        }
        </style>
        """, unsafe_allow_html=True)

    # Define profile image paths with absolute paths
    profile_images = get_images([
        os.path.join(current_dir, "assets", "hafsa.png"),
        os.path.join(current_dir, "assets","hafsa.png")
    ])

    # Create a container for content with semi-transparent background
    st.markdown("""
    <div style="background-color: rgba(0,0,0,0.6); padding: 30px; border-radius: 15px; color: white; margin: 20px; ">
    """, unsafe_allow_html=True)

    # Hero section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <h1 style="color: white; font-size: 48px; margin-bottom: 20px;">Hello! I'm <span class="main-heading">Hafsa Kamali.</span></h1>
        <p style="color: white; font-size: 20px; margin-bottom: 30px;">
        I'm currently working on <strong class="sub-heading">training Machine Learning and AI models</strong> while also exploring the world of <strong class="sub-heading">Data Science</strong>. I'm a <strong class="sub-heading">certified Web Developer</strong>, specializing in <strong>Frontend technologies</strong> like <strong class="sub-heading">HTML, CSS, JavaScript, TypeScript, and Next.js</strong>.
        </p>
        <p style="color: white; font-size: 20px; margin-bottom: 20px;">
        Beyond web development, I have a strong passion for <strong class="sub-heading">graphic design and video editing</strong>, bringing creativity into my projects. Additionally, I'm a <strong class="sub-heading">Python Developer</strong>, continuously expanding my expertise in AI and data-driven technologies.
        </p>
        <p style="color: white; font-size: 20px;">
        Currently, I am part of the <strong class="sub-heading">Governor House IT Initiative</strong>, where I am enhancing my skills under the guidance of incredible mentors. My journey is all about blending technology with creativity to build impactful solutions. 🚀✨
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        if profile_images:
            profile_image = random.choice(profile_images)
            encoded_image = encode_image(profile_image)
            if encoded_image:
                st.markdown(f'<img src="data:image/png;base64,{encoded_image}" class="custom-profile-image" width="300" style="max-width: 100%;">', unsafe_allow_html=True)
            else:
                st.error("Failed to encode profile image")
        else:
            st.error("No profile images found")

    # Navigation section
    st.markdown("## Explore My Portfolio")
    
    # Use custom CSS for navigation links
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="nav-link">', unsafe_allow_html=True)
        st.page_link("pages/dashboard.py", label="Dashboard", icon="📊")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="nav-link">', unsafe_allow_html=True)
        st.page_link("pages/chatbot.py", label="Chatbot", icon="💬")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="nav-link">', unsafe_allow_html=True)
        st.page_link("pages/projects.py", label="Projects", icon="🚀")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="nav-link">', unsafe_allow_html=True)
        st.page_link("pages/about_me.py", label="About Me", icon="👩🏻‍💻")
        st.markdown('</div>', unsafe_allow_html=True)


    # Close the content container
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()