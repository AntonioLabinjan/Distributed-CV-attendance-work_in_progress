import tkinter as tk
import subprocess
import os
import signal

# Globalni dict za procese
node_processes = {}

# Traženje node fajlova
def get_all_nodes():
    return sorted([f for f in os.listdir() if f.startswith("node_") and f.endswith(".py")])

# Pokretanje nodea
def start_node(node_file):
    if node_file not in node_processes or node_processes[node_file].poll() is not None:
        process = subprocess.Popen(["python", node_file])
        node_processes[node_file] = process
        print(f"[STARTED] {node_file}")
    else:
        print(f"[ALREADY RUNNING] {node_file}")

# Gašenje nodea
def stop_node(node_file):
    process = node_processes.get(node_file)
    if process and process.poll() is None:
        os.kill(process.pid, signal.SIGTERM)
        print(f"[STOPPED] {node_file}")
    else:
        print(f"[NOT RUNNING] {node_file}")

# GUI sučelje
def create_gui():
    window = tk.Tk()
    window.title("Node Starter")
    window.geometry("300x400")

    tk.Label(window, text="Distribuirani Nodeovi", font=("Helvetica", 14)).pack(pady=10)

    for node_file in get_all_nodes():
        frame = tk.Frame(window)
        frame.pack(pady=5)

        label = tk.Label(frame, text=node_file, width=15)
        label.pack(side=tk.LEFT)

        start_btn = tk.Button(frame, text="Start", command=lambda nf=node_file: start_node(nf))
        start_btn.pack(side=tk.LEFT, padx=5)

        stop_btn = tk.Button(frame, text="Stop", command=lambda nf=node_file: stop_node(nf))
        stop_btn.pack(side=tk.LEFT)

    window.mainloop()

if __name__ == "__main__":
    create_gui()
