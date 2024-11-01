import streamlit as st
import classify_page


st.set_page_config(
    page_title="Xmodel Transformers",
    page_icon="🤖",
    layout="wide")

PAGES = {
    "Classify Image": classify_page
}
st.sidebar.title("Xmodel Transformers")

st.sidebar.write("Breast Cancer Prediction System: Benign and Malignant")

st.sidebar.subheader('Navigation:')
selection = st.sidebar.radio("", list(PAGES.keys()))

page = PAGES[selection]

page.app()

