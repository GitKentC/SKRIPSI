import streamlit as st
import os
import base64

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

# Path to the local image
image_path = r"./asset/landingpg.png"
btn1_path = r"./asset/githubbt.png"
btn2_path = r"./asset/getstartedbt.png"

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Get the base64 string for the image
img_base64 = image_to_base64(image_path)
btn1_base64 = image_to_base64(btn1_path)
btn2_base64 = image_to_base64(btn2_path)

# Background and page style configuration
page_bg_img = f"""
<style>
[data-testid="stSidebar"] {{
    display: none
}}

[data-testid="collapsedControl"] {{
    display: none
}}

[data-testid="stAppViewContainer"] {{
    background-image: url(data:image/png;base64,{img_base64});
    background-size: cover;
    background-position: top center;
    background-repeat: no-repeat;
    background-attachment: local;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stSidebar"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stFooter"] {{
    background: rgba(0,0,0,0);
}}

.center-buttons {{
    display: flex;
    flex-direction: column;  /* Stack the buttons vertically */
    justify-content: center;
    align-items: center;
    position: absolute;
    top: 400px;
    width: 100%;
    z-index: 9999;
}}

.button-container img {{
    cursor: pointer;
    width: 200px;  /* Adjust the size of the buttons */
    height: auto;
    margin: 10px;  /* Add spacing between the buttons */
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the URLs to redirect to
url1 = "https://github.com/GitKentC/SKRIPSI"
url2 = "/App" # redirects to page_title="App"

st.markdown(f"""
<div class="center-buttons">
    <div class="button-container">
        <a href="{url1}" target="_blank">
            <img src="data:image/png;base64,{btn1_base64}" alt="Button 1">
        </a>
        <a href="{url2}" target="_self">
            <img src="data:image/png;base64,{btn2_base64}" alt="Button 2">
        </a>
    </div>
</div>
""", unsafe_allow_html=True)