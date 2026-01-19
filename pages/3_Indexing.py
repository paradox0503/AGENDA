"""
Search Page
"""
from pathlib import Path
import streamlit as st
import os
import json
from pathlib import Path
import zipfile
import rarfile
import shutil
from utils import ensure_workspace, run_shell_command_index
def modify_nth_line(file_path, n, new_content, line_start=1):

    line_index = n - line_start

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if line_index < 0 or line_index >= len(lines):
        print(f"error: line number {n} out of range (1-{len(lines)})")
        return False

    lines[line_index] = new_content + '\n' if not new_content.endswith('\n') else new_content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True

st.markdown(
    """
<style>
.stApp { background-color: #f7f9fc; }
.center-btn { display:flex; justify-content:center; margin: 18px 0 12px 0; }

/* black bordered boxes: st.container(border=True) */
div[data-testid="stContainer"][data-border="true"]{
  border: 2px solid #222 !important;
  border-radius: 0px !important;
  background: white !important;
  padding: 12px !important;
  min-height: 240px;
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown("""
<style>
.st-emotion-cache-1tkb1dl {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
#
st.markdown("""
<style>
.st-emotion-cache-ysq8gg {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Indexing")
#### -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
st.markdown("---")
st.header("Load target data series collection")


indexing_data_dir = Path("./app/data/indexing_data/")
indexing_data_dir.mkdir(parents=True, exist_ok=True)

st.markdown("""
<style>
.st-emotion-cache-fa6x4z {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


uploaded_files = st.file_uploader(
    "Upload the dataset (supporting .zip and .rar and .bin formats)",
    type=["zip", "rar", "bin"],
    accept_multiple_files=True,
    key="indexing_upload_dataset"
)

if uploaded_files:
    process_status = st.empty()
    process_status.info(f"Processing {len(uploaded_files)} file(s)...")

    uploaded_file_names = []

    for uploaded_file in uploaded_files:
        local_file_path = indexing_data_dir / uploaded_file.name

        try:
            with open(local_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.endswith('.zip'):
                try:
                    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                        zip_ref.extractall(indexing_data_dir)
                    if local_file_path.exists():
                        local_file_path.unlink()
                    st.success(f"✓ Extracted: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to extract {uploaded_file.name}: {e}")

            elif uploaded_file.name.endswith('.rar'):
                try:
                    with rarfile.RarFile(local_file_path, 'r') as rar_ref:
                        rar_ref.extractall(indexing_data_dir)
                    if local_file_path.exists():
                        local_file_path.unlink()
                    st.success(f"✓ Extracted: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to extract {uploaded_file.name}: {e}")

            elif uploaded_file.name.endswith('.bin'):
                st.success(f"✓ Saved binary file: {uploaded_file.name}")

            uploaded_file_names.append(uploaded_file.name)

        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")


    process_status.empty()

else:
    pass

st.session_state["indexing_data_dir"] = str(indexing_data_dir)
#### -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
st.markdown("---")
st.header("Select approximation method/model")#Select indexing method
Select_approximation_method_or_model=["PAA","DFT","SPARTAN","SEANet","TimeLLM","UniTime","AutoTimes","S2IPLLM", "AGENDA"]
Select_approximation_method_or_model_value= st.selectbox(
    "",
    options=Select_approximation_method_or_model,
    key="Select approximation method or model"
)
#### -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
st.markdown("---")
st.header("Select indexing method")#

index_method = st.selectbox(
    "",
    options=['iSAX', 'DIDS', 'Dumpy'],
    key="index_method"
)
#### -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0

if Select_approximation_method_or_model_value in ["SEANet","GPT4TS","TimeLLM","UniTime","AutoTimes","S2IPLLM", "AGENDA"]:
    v_model_path = st.selectbox(
        label="Model path",
        key="Model path",
        options=["/data/AGENDA/app/fine/fine_20260115_044416/50.pickle",
            "/data/AGENDA/app/fine/fine_20260115_213141/10.pickle"]
    )
if index_method=='iSAX':
    txt_path="/data/AGENDA/isax/index.txt"
    ####-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0
    st.markdown("---")
    st.header("Configuration")#
    with st.expander("", expanded=False):
        v = st.text_input("ref_objs_size",      key="ref_objs_size",value="1000")
        modify_nth_line(txt_path, 8, v)
        v = st.text_input("approximate_leaf_size",     key="approximate_leaf_size",value="10000")
        modify_nth_line(txt_path, 9, v)
        v = st.text_input("ts_buffer_size_for_read",
        key="ts_buffer_size_for_read",value="10000")
        modify_nth_line(txt_path, 10, v)
        v = st.text_input("ts_buffer_size_per_ref_obj",key="ts_buffer_size_per_ref_obj",value="100")
        modify_nth_line(txt_path, 11, v)

else:
    pass

st.markdown("---")
if st.button("Start Indexing", key="start_search", type="primary"):
    if index_method=='iSAX':

        cmd = (
            f"cd /data/AGENDA/isax/build && "
            f"make && "
            f"./index "
        )
        run_shell_command_index(cmd, workdir="./")
    else:
        print("error")
else:
    pass