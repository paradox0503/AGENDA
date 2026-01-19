import streamlit as st

# ---------------------- Streamlit App ----------------------
st.set_page_config(page_title="AGENDA: A General Deep Approximation Framework for Data Series Similarity Search", layout="wide")
st.markdown(
    """
    <h1 style='font-size:2.2rem; font-weight:700; margin-bottom: 1rem;'>AGENDA: A General Deep Approximation Framework for Data Series Similarity Search</h1>
    """,
    unsafe_allow_html=True
)
st.image("architecture.png", caption="Framework Architecture", use_container_width=True)
st.markdown("---")

# st.write("Please use the navigation bar on the left to select a feature page.")

# Set default workspace
if 'workspace' not in st.session_state:
    st.session_state.workspace = "./"

