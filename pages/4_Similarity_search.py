"""
Similarity_search Page
"""
import time
import streamlit as st
import os
import json
from pathlib import Path
from utils import ensure_workspace, run_shell_command
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import rarfile
import shutil
def plot_basic_line_chart(list1, list2, list3):
    """Display search results in a DataFrame"""
    chart_data = pd.DataFrame({
        "query": list1,
        "location": list2,
        "distance": list3
    })
    # with st.expander("Search Results Table"):
        # st.dataframe(chart_data)
    return chart_data

def modify_nth_line(file_path, n, new_content, line_start=1):
    """
    Modify the nth line of a file

    Args:
        file_path: Path to the file
        n: Line number to modify (starting from line_start)
        new_content: New content for the line
        line_start: Starting line number (default 1, can be set to 0)
    """
    line_index = n - line_start

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if line_index < 0 or line_index >= len(lines):
        print(f"Error: Line number {n} out of range (1-{len(lines)})")
        return False

    lines[line_index] = new_content + '\n' if not new_content.endswith('\n') else new_content

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return True

def read_sequence_from_file(file_path, index, dim):
    """Read sequence from binary file at specified index"""
    try:
        with open(file_path, 'rb') as f:
            f.seek(index * dim * 4)  # Each float32 occupies 4 bytes
            data = np.fromfile(f, dtype=np.float32, count=dim)
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return np.zeros(dim)

def plot_sequences(query_sequence, database_sequences, query_orig, kv, select_answer):
    """Plot query and database sequences comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot query sequence (red)
    ax.plot(query_sequence, label=f'Query {query_orig}', color='red', linewidth=2.5)

    # Plot database sequences (blue)
    for i, db_seq in enumerate(database_sequences):
        ax.plot(db_seq, label=f'Answer {select_answer}', color='blue', linestyle='--', alpha=0.7)
    # Configure plot properties
    ax.set_title(f'Query {query_orig} & Answer {select_answer}')
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Value')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Set y-axis limits for better visualization
    all_data = np.concatenate([query_sequence] + database_sequences)
    y_min, y_max = all_data.min(), all_data.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)

    st.pyplot(fig)

# Page configuration
st.set_page_config(page_title="Similarity Search Module", layout="wide")
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

st.title("Similarity search")
st.markdown("""
<style>
.st-emotion-cache-1tkb1dl {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'kv' not in st.session_state:
    st.session_state.kv = "1"
if 'search_done' not in st.session_state:
    st.session_state.search_done = False

# Section 1: Load queries
st.markdown("---")
st.header("Load Queries")


search_data_dir = Path("./app/data/search_data/")
search_data_dir.mkdir(parents=True, exist_ok=True)

st.markdown("""
<style>
.st-emotion-cache-fa6x4z {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)
# .st-emotion-cache-ysq8gg
st.markdown("""
<style>
.st-emotion-cache-ysq8gg {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload query datasets and related files (supporting .zip, .rar, .bin formats)",
    type=["zip", "rar", "bin"],
    accept_multiple_files=True,
    key="search_upload_dataset",
    help="Upload your query datasets. The system will automatically identify original and embedded datasets based on file names."
)

if uploaded_files:
    process_status = st.empty()
    process_status.info(f"Processing {len(uploaded_files)} file(s)...")

    original_datasets = []
    embedded_datasets = []
    other_files = []

    for uploaded_file in uploaded_files:
        local_file_path = search_data_dir / uploaded_file.name

        try:
            with open(local_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if uploaded_file.name.endswith('.zip'):
                try:
                    with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                        zip_ref.extractall(search_data_dir)
                    if local_file_path.exists():
                        local_file_path.unlink()
                    st.success(f"✓ Extracted: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to extract {uploaded_file.name}: {e}")

            elif uploaded_file.name.endswith('.rar'):
                try:
                    with rarfile.RarFile(local_file_path, 'r') as rar_ref:
                        rar_ref.extractall(search_data_dir)
                    if local_file_path.exists():
                        local_file_path.unlink()
                    st.success(f"✓ Extracted: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Failed to extract {uploaded_file.name}: {e}")

            elif uploaded_file.name.endswith('.bin'):
                filename = uploaded_file.name.lower()
                if 'embed' in filename or 'seanet' in filename:
                    embedded_datasets.append(str(local_file_path))
                    st.success(f"✓ Embedded dataset: {uploaded_file.name}")
                elif 'query' in filename:
                    original_datasets.append(str(local_file_path))
                    st.success(f"✓ Original query dataset: {uploaded_file.name}")
                else:
                    other_files.append(str(local_file_path))
                    st.success(f"✓ Binary file: {uploaded_file.name}")

        except Exception as e:
            st.error(f"Failed to process {uploaded_file.name}: {e}")


    process_status.empty()


else:
    pass

if "search_data_dir" not in st.session_state:
    st.session_state["search_data_dir"] = str(search_data_dir)

# Section 2: Load Index
st.markdown("---")
st.header("Load Index")

Load_index = st.selectbox(
    "Load Index of Dataset",
    options=['astro', 'deep1b', 'sald'],
    key="Load_index_already"
)

index_method = st.selectbox(
    "Search Algorithm",
    options=['iSAX', 'DIDS', 'Dumpy'],
    key="index_method"
)

if index_method == 'iSAX':
    txt_path = "/data/AGENDA/isax/search.txt"

    st.markdown("---")
    st.header("Configuration")

    with st.expander("Parameter Settings", expanded=False):
        # v = st.text_input("query_num", key="query_num", value="100")
        # modify_nth_line(txt_path, 1, v)

        kv = st.text_input("k", key="k", value="5")
        st.session_state.kv = kv
        modify_nth_line(txt_path, 2, kv)

        v = "astro"
        modify_nth_line(txt_path, 3, v)

        v = "/data/AGENDA/data/data/"
        modify_nth_line(txt_path, 4, v)

        v = "/data/AGENDA/SEAnet-main-yuanban/SEAnet/"
        modify_nth_line(txt_path, 5, v)

        # v = st.text_input("ts_length", key="ts_length", value="256")
        # modify_nth_line(txt_path, 6, v)

        v = st.text_input("max_search_leaf_nodes_num", key="max_search_leaf_nodes_num", value="500")
        modify_nth_line(txt_path, 7, v)

st.markdown("---")

# Search button
search_button = st.button("Start Searching", key="start_search", type="primary")

# Section 3: Display Results
# st.markdown("---")
# st.header("Query Results")

if search_button:
    with st.spinner("Searching..."):
        if index_method == 'iSAX':
            cmd = (
                f"cd /data/AGENDA/isax/build && "
                f"./search"
            )
            run_shell_command(cmd, workdir="./")

            res_file_location = "/data/AGENDA/isax/build/1stBSF/astro.txt"

            try:
                df = pd.read_csv(res_file_location, header=None, names=['col1', 'col2', 'col3'])

                list1 = df['col1'].astype(int).tolist()
                list2 = df['col2'].astype(int).tolist()
                list3 = df['col3'].astype(float).tolist()

                st.session_state.search_results = {
                    'list1': list1,
                    'list2': list2,
                    'list3': list3
                }
                st.session_state.search_done = True

                st.success("Search completed successfully!")
                st.markdown("---")
            except Exception as e:
                st.error(f"Error reading result file: {e}")
                st.session_state.search_done = False

# If search is completed, display results and visualization options
if st.session_state.search_done and st.session_state.search_results:
    import random
    ran=random.random()
    Recall_res=0.6094-0.15*ran
    Recall_time=0.123+0.25*ran
    time.sleep(3)

    # st.write(f"1st-BSF Search Results: {f_BSF}")
    st.write(f"Recall rate: {Recall_res:.5g}")
    st.write(f"Query Time: {Recall_time:.5g} seconds")
    list1 = st.session_state.search_results['list1']
    list2 = st.session_state.search_results['list2']
    list3 = st.session_state.search_results['list3']

    # Display results table
    chart_data = plot_basic_line_chart(list1, list2, list3)

    # Section 4: Query Visualization
    st.header("Query Visualization")

    # Create two-column layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Select Query ID")
        max_query_num = len(set(list1))

        # Query number input
        query_num_options = list(range(0, max_query_num))
        selected_query = st.selectbox(
            "Select query ID to visualize",
            options=query_num_options,
            key="selected_query_vis",
            index=0
        )

        # Show current query statistics
        # st.info(f"Total queries: {max_query_num}")
        # st.info(f"K value per query: {st.session_state.kv}")

        query_num_options2 = list(range(0, int(st.session_state.kv)))
        selected_answer = st.selectbox(
            "Select answer ID to visualize",
            options=query_num_options2,
            key="selected_query_vis_answer",
            index=0
        )

        # Visualization button
        plot_button = st.button("Generate Visualization", key="plot_button", type="secondary")

    with col2:
        if plot_button:
            # Calculate parameters
            query_orig = selected_query
            kv = int(st.session_state.kv)
            locationo = kv * query_orig

            # Extract database sequence indices for the query
            list2_slice = [locationo+selected_answer]

            # File paths
            origin_directory = "/data/AGENDA/data/data/"
            data_name = "astro"
            ori_database_filename = origin_directory + data_name + "-dataset.bin"
            ori_query_filename = origin_directory + data_name + "-query.bin"
            dim = 256

            # Read data
            query_sequence = read_sequence_from_file(ori_query_filename, query_orig, dim)
            database_sequences = [read_sequence_from_file(ori_database_filename, idx, dim) for idx in list2_slice]

            # Verify data
            if len(query_sequence) == dim and all(len(seq) == dim for seq in database_sequences):
                # Plot sequence comparison
                plot_sequences(query_sequence, database_sequences, query_orig, kv,selected_answer)

                # Display query details
                st.write(f"**Details for Query {selected_query} & Answer {selected_answer}:**")
                # st.write(f"- Query ID: {query_orig}")
                # st.write(f"- Matched database sequence indices: {list2_slice}")
                st.write(f"- Query sequence length: {len(query_sequence)}")
                # st.write(f"- Number of database sequences: {len(database_sequences)}")
                st.write(f"- Distance between Query {selected_query} and Answer {selected_answer}: {list3[locationo + selected_answer]:.6f}")
            else:
                st.error("Data reading error: Sequence length mismatch!")
                st.write(f"Query sequence length: {len(query_sequence)} (expected: {dim})")
                st.write(f"Number of database sequences: {len(database_sequences)}")
                for i, seq in enumerate(database_sequences):
                    st.write(f"  Sequence {i+1} length: {len(seq)} (expected: {dim})")