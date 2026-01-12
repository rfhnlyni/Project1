# IMPORTS
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from main import Converter  

# CONSTANTS & CONFIGURATION
# Style
COLORS = {
    "primary": "#0553A1",
    "bg": "#F0F0F0",
    "card_bg": "#FFFFFF",
    "border": "#CCCCCC",
    "text_primary": "#000000",
    "text_secondary": "#666666",
    "success": "#00AA00",
    "error": "#CC0000",
}

FONTS = {
    "title": ("Arial", 16, "bold"),
    "heading": ("Arial", 12, "bold"),
    "body": ("Arial", 10),
    "small": ("Arial", 9),
    "monospace": ("Courier", 9),
}

# LiDAR mapping (Display names to internal names)
LIDAR_MAP = {
    "ID 0 Top": "lidar_point_cloud_top_lidar",
    "ID 1 Top Rear": "lidar_point_cloud_top_rear_lidar",
    "ID 2 Left": "lidar_point_cloud_left_lidar",
    "ID 3 Rear": "lidar_point_cloud_rear_lidar",
    "ID 4 Right": "lidar_point_cloud_right_lidar",
    "ID 5 Front": "lidar_point_cloud_front_lidar",
}

# Create converter instance to access FILTER_MAP
converter_instance = Converter()

# MAIN PAGE INITIALIZATION
root = tk.Tk()
root.title("LiDAR Intensity Extraction Tool")
root.geometry("1000x800")

# Smaller font for all messageboxes
root.option_add("*Font", "Arial 9")

# Center window on screen
root.update_idletasks()
width = root.winfo_width()
height = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (width // 2)
y = (root.winfo_screenheight() // 2) - (height // 2)
root.geometry(f'{width}x{height}+{x}+{y}')

# Set the theme
try:
    style = ttk.Style()
    style.theme_use('clam')
except:
    pass
    
# START SCREEN
def show_start_screen():
    """Show the start screen"""
    # Clear window
    for widget in root.winfo_children():
        widget.destroy()
    
    # Create start frame
    start_frame = tk.Frame(root, bg=COLORS["bg"])
    start_frame.pack(fill="both", expand=True)
    
    # Title
    tk.Label(start_frame, text="LiDAR Intensity Extraction Tool", font=FONTS["title"], bg=COLORS["bg"], fg=COLORS["primary"]).pack(pady=(295, 10))
    
    # Description
    tk.Label(start_frame, text="Filter, extract and merge LiDAR intensity data from PCD to BIN files", font=FONTS["body"], bg=COLORS["bg"], fg=COLORS["text_secondary"]).pack(pady=(0, 10))
    
    # Container for buttons
    button_container = tk.Frame(start_frame, bg=COLORS["bg"])
    button_container.pack()
    
    # Start button
    tk.Button(button_container, text="Start", command=show_configuration_screen, bg=COLORS["primary"], fg="white", font=FONTS["body"], relief="raised", bd=2, padx=20, pady=8, width=20).pack()
    
    # View Data Organization button
    view_structure_btn = tk.Button(button_container, text="View required data organization", command=show_data_organization, font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["primary"], relief="flat", bd=0, highlightthickness=0)
    view_structure_btn.pack(pady=(10, 0))
    
    # Underline effect on hover
    def on_enter(e):
        view_structure_btn.config(font=("Arial", 9, "underline"), fg=COLORS["primary"])
    
    def on_leave(e):
        view_structure_btn.config(font=FONTS["small"], fg=COLORS["primary"])
    
    view_structure_btn.bind("<Enter>", on_enter)
    view_structure_btn.bind("<Leave>", on_leave)
    
    # Version
    tk.Label(start_frame, text="v1.0", font=("Arial", 8), bg=COLORS["bg"], fg=COLORS["text_secondary"]).pack(side="bottom", pady=20)

# DATA ORGANIZATION INFORMATION
def show_data_organization():
    """Show the data organization information in a messagebox"""
    data_organization = """Your folders should follow this structure:

project_root/
├── data/
│   ├── sequences/
│   │   ├── 00/                        
│   │   │   ├── cameras/
│   │   │   ├── image_2/
│   │   │   ├── labels/
│   │   │   ├── velodyne/              
│   │   │   ├── calib.txt
│   │   │   ├── instances.txt
│   │   │   └── poses.txt
│   │   ├── 01/
│   │   ├── 02/
│   │   └── ...
│   │
│   └── pcd_data/
│       ├── 00/                        
│       │   ├── lidar_point_cloud_top_lidar/
│       │   │   ├── 000.pcd
│       │   │   ├── 001.pcd
│       │   │   └── ...
│       │   ├── lidar_point_cloud_top_rear_lidar/
│       │   ├── lidar_point_cloud_front_lidar/
│       │   ├── lidar_point_cloud_rear_lidar/
│       │   ├── lidar_point_cloud_left_lidar/
│       │   └── lidar_point_cloud_right_lidar/
│       ├── 01/
│       ├── 02/
│       └── ...
│
├── output/
│   ├── extracted_intensity/
│   │   ├── lidar_point_cloud_top_lidar/                        
│   │   │   ├── 00/
│   │   │   │    ├── velodyne/
│   │   │   │    └── labels/
│   │   │   ├── 01/
│   │   │   └── ...
│   │   └── ...
│   └── converted_data/
│       ├── 00/                        
│       │   ├── cameras/
│       │   ├── image_2/
│       │   ├── labels/
│       │   ├── velodyne/
│       │   ├── calib.txt
│       │   ├── instances.txt
│       │   └── poses.txt
│       ├── 01/
│       └── ...    
│
├── lidar_filter.py
├── lidar_merger.py
├── intensity_extractor.py
├── main.py
├── requirements.txt
└── README.md

OUTPUT FOLDER:
    (Will be created automatically if it doesn't exist)"""
    
    # Create a dialog with better formatting
    dialog = tk.Toplevel(root)
    dialog.title("Data Structure Information")
    dialog.geometry("600x400")
    dialog.configure(bg=COLORS["bg"])
    dialog.transient(root)
    dialog.grab_set()
    
    # Center the dialog
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f'600x500+{x}+{y}')
    
    # Container
    container = tk.Frame(dialog, bg=COLORS["bg"], padx=20, pady=20)
    container.pack(fill="both", expand=True)
    
    # Title
    tk.Label(container, text="Required Data Structure", 
            font=FONTS["heading"], bg=COLORS["bg"], fg=COLORS["primary"]).pack(pady=(0, 15))
    
    # Text frame with scrollbar
    text_frame = tk.Frame(container, bg=COLORS["card_bg"], bd=1, relief="solid")
    text_frame.pack(fill="both", expand=True, pady=(0, 15))
    
    # Scrollbar
    scrollbar = tk.Scrollbar(text_frame)
    scrollbar.pack(side="right", fill="y")
    
    # Text widget
    text_widget = tk.Text(text_frame, wrap="word", font=FONTS["monospace"], bg=COLORS["card_bg"], fg=COLORS["text_primary"], yscrollcommand=scrollbar.set, padx=10, pady=10, height=20, width=60)
    text_widget.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text_widget.yview)
    
    # Insert the data organization
    text_widget.insert("1.0", data_organization)
    text_widget.config(state="disabled")
    
    # Close button
    tk.Button(container, text="Close", command=dialog.destroy, bg=COLORS["primary"], fg="white", font=FONTS["body"], padx=20, pady=5).pack()
    
    # Focus the dialog
    dialog.focus_set()
    
# CONFIGURATION AND FOLDER SELECTION SCREEN 
def show_configuration_screen():
    """Show the main configuration screen"""
    # Clear window
    for widget in root.winfo_children():
        widget.destroy()
    
    root.config(bg=COLORS["bg"])
    root.title("LiDAR Intensity Extraction Tool")

    # Main container
    main_container = tk.Frame(root, bg=COLORS["bg"], padx=20, pady=20)
    main_container.pack(fill="both", expand=True)

    # Header
    header_frame = tk.Frame(main_container, bg=COLORS["bg"])
    header_frame.pack(fill="x", pady=(0, 20))
    
    tk.Label(header_frame, text="LiDAR Intensity Extraction Tool", font=FONTS["title"], bg=COLORS["bg"], fg=COLORS["primary"]).pack()
    
    tk.Label(header_frame, text="Filter, extract and merge LiDAR intensity data from PCD to BIN files", font=FONTS["small"], bg=COLORS["bg"], fg=COLORS["text_secondary"]).pack(pady=(5, 0))

    # Variables for folder paths
    folder_path_seq = tk.StringVar()
    folder_path_pcd = tk.StringVar()
    output_folder = tk.StringVar()
    
    config_frame = tk.Frame(main_container, bg=COLORS["card_bg"], bd=1, relief="solid", padx=15, pady=15)
    config_frame.pack(fill="x", pady=(0, 15))
    
    tk.Label(config_frame, text="Folder Configuration", font=FONTS["heading"], bg=COLORS["card_bg"], fg=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(0, 10))
    
    # Create select folder sections
    create_folder_section(config_frame, "SemanticKITTI Sequences Folder:", folder_path_seq, validate_semantickitti_structure)
    create_folder_section(config_frame, "PCD Data Folder:", folder_path_pcd, validate_pcd_structure)
    create_folder_section(config_frame, "Output Folder:", output_folder, validate_output_folder)
    
    lidar_frame = tk.Frame(main_container, bg=COLORS["card_bg"], bd=1, relief="solid", padx=15, pady=15)
    lidar_frame.pack(fill="x", pady=(0, 15))
    
    tk.Label(lidar_frame, text="LiDAR Selection", font=FONTS["heading"], bg=COLORS["card_bg"], fg=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(0, 10))
    
    tk.Label(lidar_frame, text="Select LiDAR sensors to process:", font=FONTS["body"], bg=COLORS["card_bg"], fg=COLORS["text_primary"]).pack(anchor="w", pady=(0, 10))
    
    # Checkboxes for LiDAR selection
    checkboxes_frame = tk.Frame(lidar_frame, bg=COLORS["card_bg"])
    checkboxes_frame.pack(fill="x", pady=(0, 5))
    
    # Columns for checkboxes
    col1 = tk.Frame(checkboxes_frame, bg=COLORS["card_bg"])
    col1.pack(side="left", fill="both", expand=True, padx=(0, 10))
    col2 = tk.Frame(checkboxes_frame, bg=COLORS["card_bg"])
    col2.pack(side="left", fill="both", expand=True)
    
    # Checkbox variables
    lidar_vars = {}
    lidar_items = list(LIDAR_MAP.items())
    mid_point = len(lidar_items) // 2
    
    # Checkboxes in two columns
    for i, (display_name, internal_name) in enumerate(lidar_items):
        var = tk.BooleanVar(value=True)  # All selected by default
        lidar_vars[display_name] = var
        
        # Choose which column based on position
        parent_frame = col1 if i < mid_point else col2
        
        cb = tk.Checkbutton(parent_frame, text=display_name, variable=var, font=FONTS["body"], bg=COLORS["card_bg"], anchor="w", padx=5, pady=2)
        cb.pack(fill="x")
    
    control_frame = tk.Frame(lidar_frame, bg=COLORS["card_bg"])
    control_frame.pack(fill="x", pady=(10, 0))
    
    def select_all_lidars():
        """Select all LiDAR checkboxes"""
        for var in lidar_vars.values():
            var.set(True)
        update_selection_display()
    
    def deselect_all_lidars():
        """Deselect all LiDAR checkboxes"""
        for var in lidar_vars.values():
            var.set(False)
        update_selection_display()
    
    def update_selection_display():
        """Update the selection display label"""
        selected = get_selected_lidars()
        if selected:
            display_text = f"Selected: {len(selected)} LiDAR sensor(s)"
        else:
            display_text = "No LiDARs selected"
        selection_display.config(text=display_text)
    
    # Select/deselect buttons
    tk.Button(control_frame, text="Select All", command=select_all_lidars, font=FONTS["small"], padx=10, pady=2).pack(side="left", padx=(0, 5))
    
    tk.Button(control_frame, text="Deselect All", command=deselect_all_lidars, font=FONTS["small"], padx=10, pady=2).pack(side="left")
    
    # Selection display label
    selection_display = tk.Label(lidar_frame, text="Selected: 6 LiDAR sensor(s)", font=FONTS["small"], bg=COLORS["card_bg"], fg=COLORS["text_secondary"])
    selection_display.pack(fill="x", pady=(10, 0))
    
    # Function to get selected LiDARs
    def get_selected_lidars():
        """Get list of selected LiDAR display names"""
        return [name for name, var in lidar_vars.items() if var.get()]
    
    # Bind checkbox changes to update display
    for var in lidar_vars.values():
        var.trace_add("write", lambda *args: update_selection_display())
    
    # Action buttons frame
    action_frame = tk.Frame(main_container, bg=COLORS["bg"])
    action_frame.pack(fill="x", pady=(20, 0))
    
    def run_extraction():
        """Validate and start extraction process"""
        seq_dir = folder_path_seq.get()
        pcd_dir = folder_path_pcd.get()
        out_dir = output_folder.get()
        
        # Get selected LiDARs
        selected_lidar_names = get_selected_lidars()
        
        if not seq_dir or not pcd_dir or not out_dir:
            messagebox.showerror("Error", "Please select all required folders!")
            return
        
        if not selected_lidar_names:
            messagebox.showerror("Error", "Please select at least one LiDAR sensor!")
            return
        
        # Validate folders before proceeding
        seq_valid, seq_messages = validate_semantickitti_structure(seq_dir)
        pcd_valid, pcd_messages = validate_pcd_structure(pcd_dir)
        out_valid, out_messages = validate_output_folder(out_dir)
        
        if not (seq_valid and pcd_valid and out_valid):
            error_msg = "Folder validation failed:\n\n"
            if not seq_valid:
                error_msg += f"SemanticKITTI: {seq_messages[0]}\n"
            if not pcd_valid:
                error_msg += f"PCD: {pcd_messages[0]}\n"
            if not out_valid:
                error_msg += f"Output: {out_messages[0]}\n"
            messagebox.showerror("Validation Error", error_msg)
            return

        # Show progress
        for widget in root.winfo_children():
            widget.destroy()
        
        show_progress_screen(seq_dir, pcd_dir, out_dir, selected_lidar_names)
    
    def go_back():
        """Go back to start screen"""
        for widget in root.winfo_children():
            widget.destroy()
        show_start_screen()

    # Create a frame for the two buttons to be side by side
    button_container = tk.Frame(action_frame, bg=COLORS["bg"])
    button_container.pack()
    
    # Back button
    back_button = tk.Button(button_container, text="Back", command=go_back, font=FONTS["body"], padx=20, pady=8, width=10)
    back_button.pack(side="left", padx=(0, 10))
    
    # Run button
    run_button = tk.Button(button_container, text="Start Extraction", command=run_extraction, bg=COLORS["primary"], fg="white", font=FONTS["body"], relief="raised", bd=2, padx=20, pady=8, width=15)
    run_button.pack(side="left")

def create_folder_section(parent, label_text, variable, validation_func):
    """Create a folder selection section"""
    frame = tk.Frame(parent, bg=COLORS["card_bg"])
    frame.pack(fill="x", pady=8)
    
    # Label
    tk.Label(frame, text=label_text, font=FONTS["body"], bg=COLORS["card_bg"], fg=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(0, 5))
    
    # Entry and button frame
    entry_frame = tk.Frame(frame, bg=COLORS["card_bg"])
    entry_frame.pack(fill="x")
    
    # Entry
    entry = tk.Entry(entry_frame, textvariable=variable, font=FONTS["body"], bg="white", fg=COLORS["text_primary"], relief="solid", bd=1)
    entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
    
    def browse():
        folder = filedialog.askdirectory()
        if folder:
            variable.set(folder)
            if validation_func:
                is_valid, messages = validation_func(folder)
                status_label.config(
                    text=f"{'✓' if is_valid else '✗'} {messages[0]}",
                    fg=COLORS["success"] if is_valid else COLORS["error"])
    
    # Browse button
    browse_btn = tk.Button(entry_frame, text="Browse", command=browse, font=FONTS["body"], padx=10)
    browse_btn.pack(side="right")
    
    # Status label
    status_label = tk.Label(frame, text="", font=FONTS["small"], bg=COLORS["card_bg"], fg=COLORS["text_secondary"], anchor="w")
    status_label.pack(fill="x", pady=(5, 0))
    
    return entry, status_label
         
# PROGRESS AND PROCESSING SCREEN
def show_progress_screen(seq_dir, pcd_dir, out_dir, selected_lidar_names):
    """Show the progress window"""
    root.config(bg=COLORS["bg"])
    
    # Main progress container
    main_progress = tk.Frame(root, bg=COLORS["bg"], padx=20, pady=20)
    main_progress.pack(fill="both", expand=True)
    
    # Progress header
    tk.Label(main_progress, text="Extraction in Progress", font=FONTS["title"], bg=COLORS["bg"], fg=COLORS["primary"]).pack(pady=(0, 20))
    
    # Progress frame
    progress_frame = tk.Frame(main_progress, bg=COLORS["card_bg"], bd=1, relief="solid", padx=15, pady=15)
    progress_frame.pack(fill="x", pady=(0, 20))
    
    # Status label
    status_label = tk.Label(progress_frame, text="Starting conversion...", font=FONTS["heading"], bg=COLORS["card_bg"], fg=COLORS["text_primary"])
    status_label.pack(pady=(0, 15))
    
    # Progress bar
    progress = ttk.Progressbar(progress_frame, orient="horizontal", length=600, mode="determinate")
    progress.pack(fill="x", pady=(0, 10))
    
    # Percentage label
    percent_label = tk.Label(progress_frame, text="0%", font=FONTS["heading"], bg=COLORS["card_bg"], fg=COLORS["primary"])
    percent_label.pack()
    
    # Logs frame
    logs_frame = tk.Frame(main_progress, bg=COLORS["card_bg"], bd=1, relief="solid", padx=15, pady=15)
    logs_frame.pack(fill="both", expand=True)
    
    tk.Label(logs_frame, text="Processing Logs", font=FONTS["heading"], bg=COLORS["card_bg"], fg=COLORS["text_primary"], anchor="w").pack(fill="x", pady=(0, 10))
    
    # Create scrolled text for logs
    log_text = scrolledtext.ScrolledText(logs_frame, height=12, font=FONTS["monospace"], bg="white", fg=COLORS["text_primary"], relief="solid", bd=1)
    log_text.pack(fill="both", expand=True)
    log_text.config(state=tk.DISABLED)
    
    # Button frame at bottom
    button_frame = tk.Frame(main_progress, bg=COLORS["bg"])
    button_frame.pack(fill="x", pady=(15, 0))
    
    def update_progress(text, value):
        """Update progress"""
        status_label.config(text=text)
        progress["value"] = value
        percent_label.config(text=f"{int(value)}%")
        root.update_idletasks()
    
    def log_to_gui(log_message):
        """Add message to log display"""
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, log_message + "\n")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)
        root.update_idletasks()
    
    def copy_logs():
        """Copy logs to clipboard"""
        log_text.config(state=tk.NORMAL)
        logs = log_text.get("1.0", tk.END)
        root.clipboard_clear()
        root.clipboard_append(logs)
        log_text.config(state=tk.DISABLED)
        messagebox.showinfo("Copied", "Logs copied to clipboard!")
    
    def save_logs():
        """Save logs to file"""
        log_text.config(state=tk.NORMAL)
        logs = log_text.get("1.0", tk.END)
        log_text.config(state=tk.DISABLED)
        
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_path:
            with open(file_path, "w") as f:
                f.write(logs)
            messagebox.showinfo("Saved", f"Logs saved to:\n{file_path}")
    
    def back_to_main():
        """Return to start screen"""
        for widget in root.winfo_children():
            widget.destroy()
        show_start_screen()
    
    # Add action buttons
    tk.Button(button_frame, text="Copy Logs", command=copy_logs, font=FONTS["body"]).pack(side="left", padx=(0, 10))
    
    tk.Button(button_frame, text="Save Logs", command=save_logs, font=FONTS["body"]).pack(side="left", padx=(0, 10))
    
    tk.Button(button_frame, text="New Extraction", command=back_to_main,  bg=COLORS["primary"], fg="white", font=FONTS["body"]).pack(side="left")
    
    # Convert display names to internal names
    lidars = [LIDAR_MAP[name] for name in selected_lidar_names]
    
    # Get filter values from Converter instance (no duplicate FILTER_MAP)
    filters = [converter_instance.EXPECTED_MAPPING[l] for l in lidars]

    # Reset progress bar
    progress["maximum"] = 100
    update_progress("Starting conversion...", 0)
    
    # Clear log text
    log_text.config(state=tk.NORMAL)
    log_text.delete("1.0", tk.END)
    log_text.config(state=tk.DISABLED)
    
    # Log validation results
    log_to_gui("=" * 50)
    log_to_gui("FOLDER VALIDATION RESULTS")
    log_to_gui("=" * 50)
    
    # Validation summary
    seq_valid, seq_messages = validate_semantickitti_structure(seq_dir)
    pcd_valid, pcd_messages = validate_pcd_structure(pcd_dir)
    out_valid, out_messages = validate_output_folder(out_dir)
    
    log_to_gui(f"SemanticKITTI: {seq_dir}")
    log_to_gui(f"  Status: {'Valid' if seq_valid else 'Invalid'}")
    if seq_messages:
        log_to_gui(f"  Message: {seq_messages[0]}")
    
    log_to_gui(f"\nPCD Data: {pcd_dir}")
    log_to_gui(f"  Status: {'Valid' if pcd_valid else 'Invalid'}")
    if pcd_messages:
        log_to_gui(f"  Message: {pcd_messages[0]}")
    
    log_to_gui(f"\nOutput: {out_dir}")
    log_to_gui(f"  Status: {'Valid' if out_valid else 'Invalid'}")
    if out_messages:
        log_to_gui(f"  Message: {out_messages[0]}")
    
    log_to_gui(f"\nSelected LiDARs: {', '.join(selected_lidar_names)}")
    log_to_gui(f"Processing {len(lidars)} LiDAR sensor(s)")
    log_to_gui(f"Internal names: {', '.join(lidars)}")
    log_to_gui(f"Filter values: {', '.join(map(str, filters))}\n")
    
    def run_conversion_task():
        """Run the conversion"""
        try:
            class GUILogConverter(Converter):
                def log(self, message):
                    try:
                        super().log(message)
                    except:
                        pass
                    root.after(0, lambda: log_to_gui(message))
            
            converter = GUILogConverter()
            success = converter.run_conversion(input_seq=seq_dir, input_pcd=pcd_dir, output_dir=out_dir, lidars=lidars, filter_values=filters, progress_callback=update_progress)

            if success:
                root.after(0, lambda: update_progress("Conversion Complete!", 100))
                root.after(0, lambda: messagebox.showinfo(
                    "Success", 
                    "Conversion finished successfully!\nResults have been saved to the output folder."))
            else:
                root.after(0, lambda: update_progress("Conversion Failed", 100))
                root.after(0, lambda: log_to_gui("\n" + "=" * 50))
                root.after(0, lambda: log_to_gui("CONVERSION FAILED"))
                root.after(0, lambda: log_to_gui("=" * 50))
                root.after(0, lambda: messagebox.showerror(
                    "Failed", 
                    "Conversion failed. Please check the logs above for details."))

        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            root.after(0, lambda: log_to_gui(f"\n{error_msg}"))
            root.after(0, lambda: update_progress("Error Occurred", 100))
            root.after(0, lambda: messagebox.showerror(
                "Error", 
                f"An error occurred during conversion:\n\n{str(e)}"))
    
    # Run conversion
    conversion_thread = threading.Thread(target=run_conversion_task, daemon=True)
    conversion_thread.start()

# VALIDATION FUNCTIONS
def validate_semantickitti_structure(folder_path):
    """Validate SemanticKITTI folder structure"""
    if not os.path.exists(folder_path):
        return False, ["Folder does not exist"]
    
    validation_errors = []
    
    # Check if folder contains sequences directly or has sequences subfolder
    if "sequences" in os.listdir(folder_path):
        base_path = os.path.join(folder_path, "sequences")
    else:
        base_path = folder_path
    
    if not os.path.exists(base_path):
        return False, [f"No valid structure found in {folder_path}"]
    
    # Find sequence folders 
    sequence_folders = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 2:
            sequence_folders.append(item)
    
    if not sequence_folders:
        return False, [f"No sequence folders (00, 01, etc.) found in {base_path}"]
    
    # Check first few sequences
    checked_count = 0
    for seq in sequence_folders[:3]:
        seq_path = os.path.join(base_path, seq)
        
        # Check velodyne folder
        velodyne_path = os.path.join(seq_path, "velodyne")
        if not os.path.exists(velodyne_path):
            validation_errors.append(f"Sequence {seq}: Missing 'velodyne' folder")
            continue
        
        # Check for .bin files
        bin_files = [f for f in os.listdir(velodyne_path) if f.endswith('.bin')]
        if not bin_files:
            validation_errors.append(f"Sequence {seq}: No .bin files in velodyne folder")
        else:
            checked_count += 1
    
    if checked_count == 0:
        validation_errors.insert(0, "No valid sequences with .bin files found")
    
    if validation_errors:
        return False, validation_errors
    
    return True, [f"Found {len(sequence_folders)} sequence folders with .bin files"]
    
def validate_pcd_structure(folder_path):
    """Validate PCD folder structure"""
    if not os.path.exists(folder_path):
        return False, ["Folder does not exist"]
    
    validation_errors = []
    
    # Check if folder contains sequence folders
    sequence_folders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path) and item.isdigit() and len(item) == 2:
            sequence_folders.append(item)
    
    if not sequence_folders:
        # Check if it's a single sequence folder with LiDAR folders
        lidar_folders_found = False
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path) and item.startswith("lidar_point_cloud_"):
                lidar_folders_found = True
                break
        
        if lidar_folders_found:
            sequence_folders = ["00"]
        else:
            return False, ["No sequence folders (00, 01, etc.) or LiDAR folders found"]
    
    expected_lidars = list(converter_instance.EXPECTED_MAPPING.keys())
    
    # Check first sequence
    checked_lidars = 0
    seq = sequence_folders[0]
    seq_path = os.path.join(folder_path, seq)
    
    if not os.path.exists(seq_path):
        return False, [f"Sequence folder {seq} does not exist"]
    
    for lidar in expected_lidars:
        lidar_path = os.path.join(seq_path, lidar)
        if os.path.exists(lidar_path):
            # Check for .pcd files
            pcd_files = [f for f in os.listdir(lidar_path) if f.endswith('.pcd')]
            if pcd_files:
                checked_lidars += 1
            else:
                validation_errors.append(f"LiDAR {lidar}: No .pcd files found")
        else:
            validation_errors.append(f"LiDAR {lidar}: Folder not found")
    
    if checked_lidars == 0:
        validation_errors.insert(0, "No LiDAR folders with .pcd files found")
    
    if validation_errors:
        return False, validation_errors
    
    return True, [f"Found {len(sequence_folders)} sequence folders with PCD data"]

def validate_output_folder(folder_path):
    """Validate output folder"""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
            return True, ["Output folder created successfully"]
        except Exception as e:
            return False, [f"Cannot create output folder: {str(e)}"]
    
    if not os.access(folder_path, os.W_OK):
        return False, ["No write permission for output folder"]
    
    return True, ["Output folder is ready"]

# START SCREEN INITIALLY
show_start_screen()

# Run main loop
root.mainloop()
