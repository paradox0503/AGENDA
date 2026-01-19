"""
Utilities for the Embedding + Searching Framework
"""

import os
import subprocess
import streamlit as st
from pathlib import Path
from typing import Optional
from datetime import datetime
import threading

DEFAULT_WORKSPACE = "./"


DEFAULT_OUTPUT_FILE = "./app/data"


def ensure_workspace(path: str) -> str:
    """Ensure the workspace directory exists and return its absolute path."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def display_directory_tree(path: str, level: int = 0) -> None:
    """Display a directory tree in Streamlit."""
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            indent = "  " * level
            if os.path.isdir(item_path):
                st.write(f"{indent}📁 {item}")
                display_directory_tree(item_path, level + 1)
            else:
                st.write(f"{indent}📄 {item}")
    except PermissionError:
        st.write(f"{'  ' * level}No access permission")


def select_directories(path: str, level: int = 0, selected: Optional[list] = None) -> list:
    """Display directory tree with checkboxes for selection and return selected directories."""
    if selected is None:
        selected = []
    try:
        items = sorted(os.listdir(path))
        for item in items:
            item_path = os.path.join(path, item)
            indent = "  " * level
            if os.path.isdir(item_path):
                checkbox_key = f"select_{item_path}"
                is_selected = st.checkbox(f"{indent}📁 {item}", key=checkbox_key)
                if is_selected:
                    selected.append(item_path)
                with st.expander(f"{indent}📁 {item} (Expand to view sub-items)"):
                    select_directories(item_path, level + 1, selected)
    except PermissionError:
        st.write(f"{indent}No access permission")
    return selected


# =========================
# 修改后的 run_shell_command 函数
# =========================
def run_shell_command(cmd: str, workdir: Optional[str] = None) -> None:
    # """Run a shell command asynchronously and return immediately."""
    # st.write(f"Running command: `{cmd}`")

        # Save command to session state for later inspection
    if 'running_commands' not in st.session_state:
        st.session_state.running_commands = []

    try:
        # Start process but do not wait for completion
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # save process info
        process_info = {
            'cmd': cmd,
            'process': process,
            'start_time': datetime.now(),
            'workdir': workdir
        }
        st.session_state.running_commands.append(process_info)

        # Start background thread to handle output (optional)
        def read_output(proc, cmd_str):
            output_lines = []
            try:
                # Read output without blocking main thread
                while True:
                    line = proc.stdout.readline()
                    if not line and proc.poll() is not None:
                        break
                    if line:
                        output_lines.append(line)
            except Exception as e:
                pass

        # Start output reading thread
        output_thread = threading.Thread(
            target=read_output,
            args=(process, cmd),
            daemon=True
        )
        output_thread.start()

        # st.success(f"Command started in background. Process ID: {process.pid}")

    except Exception as e:
        st.error(f"Error starting command: {e}")


def run_shell_command_index(cmd: str, workdir: Optional[str] = None) -> None:
    # """Run a shell command and display output in Streamlit."""
    try:
        process = subprocess.Popen(
            cmd,
            shell=True,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            executable="/bin/bash"
        )
        output_container = st.empty()
        logs = ""
        for line in process.stdout:
            logs += line
            output_container.text_area("log output", logs, height=300)
        process.wait()
        if process.returncode == 0:
            st.success("success")
            pass
        else:
            st.error(f"failed:{process.returncode}")
            pass
    except Exception as e:
        pass
