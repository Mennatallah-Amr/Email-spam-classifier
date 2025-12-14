import streamlit as st
import pickle
import base64
from preprocess import preprocess_text

import nltk
nltk.data.path.append("./nltk_data")



st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()

    st.markdown(
        f"""
        <style>
        /* App background */
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Main content panel */
        .block-container {{
            background-color: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 16px;
        }}

        /* Sidebar background */
        section[data-testid="stSidebar"] {{
            background-color: #f5f0e6; /* beige */
        }}

        /* Sidebar info / boxes */
        section[data-testid="stSidebar"] .stAlert,
        section[data-testid="stSidebar"] .stButton > button {{
            background-color: #e6d3b1 !important; /* light brown */
            color: #6b4f2d !important; /* brown text */
            border-radius: 8px;
        }}

        /* All text color */
        h1, h2, h3, p, label, span, div {{
            color: #6b4f2d !important;
        }}

        /* Predict button */
        div.stButton > button {{
            background-color: #e6d3b1 !important; /* light brown */
            color: #6b4f2d !important; /* brown text */
            border-radius: 8px;
            border: none;
        }}

        /* Text area */
        textarea {{
            background-color: #fffaf0;
            color: #6b4f2d;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background
set_background("background1.jpg")

# Load model & vectorizer

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


# Sidebar

st.sidebar.title("About")
st.sidebar.info(
    "This app uses a Logistic Regression model "
    "to classify emails as **Spam** or **Ham**."
)

st.sidebar.title("Try Examples")

spam_example = """Congratulations!
You have won a $1000 gift card.
Click the link to claim your reward now."""

ham_example = """Hi team,
Please remember our meeting tomorrow at 10 AM.
Let me know if you have any questions."""

if st.sidebar.button("Load Spam Example"):
    st.session_state.email_text = spam_example

if st.sidebar.button("Load Ham Example"):
    st.session_state.email_text = ham_example

# Main UI

st.title("📧 Email Spam Classifier")
st.write("Paste an email below and check whether it is **Spam** or **Ham**.")

email_text = st.text_area(
    "Email content",
    height=200,
    key="email_text"
)

# Prediction

if st.button("Predict"):
    if email_text.strip() == "":
        st.warning("Please enter email text.")
    else:
        with st.spinner("Analyzing email..."):
            processed = preprocess_text(email_text)
            vector = vectorizer.transform([processed])
            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

        if prediction == 1:
            st.error(
                f"🚨 **Spam Email**\n\n"
                f"Confidence: **{prob[1] * 100:.2f}%**"
            )
        else:
            st.success(
                f"✅ **Ham Email**\n\n"
                f"Confidence: **{prob[0] * 100:.2f}%**"
            )
