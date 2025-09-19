import cv2
import numpy as np
import threading
import datetime
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Label, Button, Frame, Tk, StringVar, Entry, Text, WORD, END
import pandas as pd
import xgboost as xgb
import os
from tkinter import font as tkfont
import json
from tqdm import tqdm

class EyeTrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Application")
        self.root.geometry('700x650')
        
        self.user = {
            "name": "",
            "student_id": ""
        }
        
        self.camera_fps = 30
        self.cap = None

        self.all_tasks_eye_positions_with_timestamps = {}
        self.current_task_eye_positions = []
        self.tracking_active = False
        self.tracking_thread = None
        
        self.eye_x = 0
        self.eye_y = 0
        self.eye_width = 0
        self.eye_height = 0
        self.eye_detected = False
        
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        self.timer_label = None
        self.start_button = None
        self.text_area = None
        self.guide_label = None
        self.task_label = None
        
        # Lower threshold to detect smaller saccades
        self.velocity_threshold_px_per_ms = 0.005 
        # Shorter minimum duration to detect brief saccades
        self.min_event_duration_ms = 20 
        
        self.calibrated_pupil_offset_x = 0.0
        self.calibrated_pupil_offset_y = 0.0
        self.calibration_raw_pupil_data = []
        self.is_calibrating = False
        
        self.tasks = [
            {'name': 'Text 1', 'file': 'text_t1_syllables.txt', 'words': 71},
            {'name': 'Text 2', 'file': 'text_t4_meaningful.txt', 'words': 85},
            {'name': 'Text 3', 'file': 'text_t5_pseudo.txt', 'words': 83}
        ]
        self.current_task_index = 0
        self.current_task_start_time = 0
        self.current_task_duration_seconds = 0
        
        self.task_word_counts = {
            'T1': 71,  # Syllables
            'T4': 85,  # Meaningful Text
            'T5': 83   # Pseudo Text
        }

        self.dyslexia_model = None
        self.load_prediction_model()
        
        self.create_input_form()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        """Handle window closing event to ensure webcam cleanup."""
        if self.tracking_active:
            self.tracking_active = False
            if self.tracking_thread and self.tracking_thread.is_alive():
                self.tracking_thread.join(timeout=1)
        self.cleanup_webcam()
        self.root.destroy()

    def load_prediction_model(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "dyslexia_xgboost_model.json")
        try:
            self.dyslexia_model = xgb.XGBClassifier()
            self.dyslexia_model.load_model(model_path)
            print(f"XGBoost model loaded successfully from {model_path}")
        except xgb.core.XGBoostError as e:
            messagebox.showwarning("Model Load Error", f"Could not load XGBoost model from {model_path}. Please ensure it is trained and saved: {e}")
            self.dyslexia_model = None
        except FileNotFoundError:
            messagebox.showwarning("Model Not Found", f"XGBoost model file not found at {model_path}. Please train the model first.")
            self.dyslexia_model = None
        except Exception as e:
            messagebox.showwarning("Model Load Error", f"An unexpected error occurred while loading the model: {e}")
            self.dyslexia_model = None

    def create_input_form(self):
        self.clear_screen()
        self.root.geometry('500x400')
        
        self.title_label = Label(self.root, text="Enter Student Information", width=25, font=("bold", 20))
        self.title_label.place(x=60, y=53)
        
        self.name_var = StringVar()
        self.name_label = Label(self.root, text="Full Name", width=20, font=("bold", 10))
        self.name_label.place(x=80, y=130)
        self.name_entry = Entry(self.root, textvariable=self.name_var)
        self.name_entry.place(x=240, y=130)
        
        self.id_var = StringVar()
        self.id_label = Label(self.root, text="Student ID", width=20, font=("bold", 10))
        self.id_label.place(x=68, y=180)
        self.id_entry = Entry(self.root, textvariable=self.id_var)
        self.id_entry.place(x=240, y=180)
        
        self.submit_btn = Button(self.root, text="Submit", command=self.save_user_info, font=("bold", 14))
        self.submit_btn.place(x=220, y=240)
    
    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def save_user_info(self):
        self.user["name"] = self.name_var.get()
        self.user["student_id"] = self.id_var.get()
        
        if not self.user["name"] or not self.user["student_id"]:
            messagebox.showerror("Input Error", "Please enter both Full Name and Student ID.")
            return

        if not self.initialize_webcam():
            messagebox.showerror("Webcam Initialization Failed", "Could not start webcam. Please check your camera setup.")
            self.cleanup_webcam()
            return
        
        self.detect_initial_eye_position()
        if not self.eye_detected:
             messagebox.showerror("Eye Detection Error", "Eyes not detected. Please ensure your face is well-lit and directly facing the camera.")
             self.cleanup_webcam()
             return

        self.show_calibration_screen()

    def initialize_webcam(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)  
            if not self.cap.isOpened():
                print("Error: Could not open webcam")
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.camera_fps <= 0:
                self.camera_fps = 30
            return True
        return True

    def cleanup_webcam(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def show_calibration_screen(self):
        self.clear_screen()
        self.root.geometry('700x650')

        Label(self.root, text="Calibration Phase", font=("bold", 24)).pack(pady=20)
        Label(self.root, text="Please look steadily at the red circle for a few seconds.", font=("bold", 14)).pack(pady=10)
        
        self.calibration_canvas = tk.Canvas(self.root, width=200, height=200, bg="lightgray", bd=2, relief="groove")
        self.calibration_canvas.pack(pady=20)
        self.calibration_dot = self.calibration_canvas.create_oval(90, 90, 110, 110, fill="red", outline="red")
        
        self.guide_label = Label(self.root, text="Starting calibration in 3 seconds...", font=("bold", 12), fg="blue")
        self.guide_label.pack(pady=10)

        self.start_button = Button(self.root, text="Start Calibration", command=self.start_calibration_sequence, font=("bold", 12), padx=15, pady=5)
        self.start_button.pack(pady=10)

    def start_calibration_sequence(self):
        self.start_button.config(state=tk.DISABLED)
        self.guide_label.config(text="Calibration in progress...", fg="green")
        self.calibration_raw_pupil_data = []
        self.is_calibrating = True
        self.tracking_active = True
        self.current_task_start_time = time.time()

        self.tracking_thread = threading.Thread(target=self.track_eyes_for_calibration_and_tasks)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()

        self.root.after(3000, self.end_calibration_sequence)

    def end_calibration_sequence(self):
        self.tracking_active = False
        self.is_calibrating = False
        self.process_calibration_data()
        messagebox.showinfo("Calibration Complete", "Calibration finished successfully!")
        self.show_reading_screen()

    def track_eyes_for_calibration_and_tasks(self):
        """Unified tracking loop for both calibration and reading tasks."""
        if self.cap is None or not self.cap.isOpened() or not self.eye_detected:
            messagebox.showerror("Webcam/Detection Error", "Webcam not ready or eyes not detected. Cannot track.")
            self.tracking_active = False
            return
        
        while self.tracking_active:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            current_time_ms = (time.time() - self.current_task_start_time) * 1000
            
            if self.eye_detected:
                cv2.rectangle(frame, (self.eye_x, self.eye_y), 
                                 (self.eye_x + self.eye_width, self.eye_y + self.eye_height), 
                                 (0, 255, 0), 2)
                
                roi = frame[self.eye_y:self.eye_y + self.eye_height, 
                               self.eye_x:self.eye_x + self.eye_width]
                
                if roi.size == 0:
                    continue
                    
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
                
                _, threshold = cv2.threshold(gray_roi, 70, 255, cv2.THRESH_BINARY_INV)
                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                
                if contours:
                    cnt = contours[0]
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    pupil_center_x_rel_roi = x + int(w/2)
                    pupil_center_y_rel_roi = y + int(h/2)
                    
                    if self.is_calibrating:
                        self.calibration_raw_pupil_data.append((pupil_center_x_rel_roi, pupil_center_y_rel_roi))
                    else:
                        abs_x = self.eye_x + pupil_center_x_rel_roi 
                        abs_y = self.eye_y + pupil_center_y_rel_roi
                        
                        self.current_task_eye_positions.append((current_time_ms, abs_x, abs_y))
                else:
                    if self.is_calibrating:
                        self.calibration_raw_pupil_data.append((-1, -1))
                    elif self.current_task_eye_positions:
                        _, last_x, last_y = self.current_task_eye_positions[-1]
                        self.current_task_eye_positions.append((current_time_ms, last_x, last_y))
                    else:
                        self.current_task_eye_positions.append((current_time_ms, -1, -1))

            cv2.imshow('Eye Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.tracking_active = False
                break
        
    def process_calibration_data(self):
        valid_pupil_data = [p for p in self.calibration_raw_pupil_data if p != (-1, -1)]

        if not valid_pupil_data:
            messagebox.showwarning("Calibration Error", "No valid pupil data collected during calibration. Check webcam and lighting.")
            return

        pupil_x_offsets = []
        pupil_y_offsets = []

        eye_roi_center_x = self.eye_width / 2
        eye_roi_center_y = self.eye_height / 2

        for rel_x, rel_y in valid_pupil_data:
            pupil_x_offsets.append(rel_x - eye_roi_center_x)
            pupil_y_offsets.append(rel_y - eye_roi_center_y)
        
        self.calibrated_pupil_offset_x = np.mean(pupil_x_offsets)
        self.calibrated_pupil_offset_y = np.mean(pupil_y_offsets)
        print(f"Calibration successful. Pupil offsets: X={self.calibrated_pupil_offset_x:.2f}, Y={self.calibrated_pupil_offset_y:.2f}")

    def show_reading_screen(self):
        self.clear_screen()
        self.root.geometry('700x650')
        
        current_task = self.tasks[self.current_task_index]
        
        heading = Label(self.root, text=f"Task {self.current_task_index + 1}: Please read the {current_task['name']}", font=("bold", 20))
        heading.pack(pady=20)
        
        self.task_label = Label(self.root, text=f"Current Task: {current_task['name']}", font=("bold", 14))
        self.task_label.pack(pady=5)
        
        timer_frame = tk.Frame(self.root)
        timer_frame.pack(pady=10)
        Label(timer_frame, text="Time: ", font=("bold", 17)).pack(side=tk.LEFT)
        self.timer_label = Label(timer_frame, text="0:00:00", font=("bold", 17))
        self.timer_label.pack(side=tk.LEFT)
        
        self.reading_text_content = ""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        text_file_path = os.path.join(script_dir, current_task['file'])
        try:
            with open(text_file_path, 'r', encoding="utf8") as text_file:
                self.reading_text_content = text_file.read()
        except FileNotFoundError:
            self.reading_text_content = f"Sample text for {current_task['name']}. File '{current_task['file']}' not found. Please ensure the file exists for proper analysis."
            messagebox.showwarning("File Not Found", f"Text file '{current_task['file']}' not found. Using sample text for this task.")
        
        self.text_area = Text(self.root, width=61, height=15, font=("bold", 14), wrap=WORD, bd=2, relief="groove")
        self.text_area.pack(pady=10)
        self.text_area.insert(END, self.reading_text_content)
        self.text_area.config(state=tk.DISABLED)
        
        self.guide_label = Label(self.root, text="Please position your eyes within the camera view before pressing Start", font=("bold", 12), fg="blue")
        self.guide_label.pack(pady=10)
        
        self.start_button = Button(self.root, text="Start Task", command=self.toggle_tracking, font=("bold", 12), padx=15, pady=5)
        self.start_button.pack(pady=10)
    
    def toggle_tracking(self):
        if not self.tracking_active:
            self.tracking_active = True
            self.start_button.config(text="Stop Task", command=self.toggle_tracking)
            if self.guide_label:
                self.guide_label.config(text="Tracking active...", fg="green")
            
            self.current_task_eye_positions = []
            self.current_task_start_time = time.time()
            
            if self.tracking_thread is None or not self.tracking_thread.is_alive():
                 self.tracking_thread = threading.Thread(target=self.track_eyes_for_calibration_and_tasks)
                 self.tracking_thread.daemon = True
                 self.tracking_thread.start()
            
            self.update_timer()
        else:
            self.tracking_active = False
            self.current_task_duration_seconds = int(time.time() - self.current_task_start_time)
            
            current_task_name = self.tasks[self.current_task_index]['name']
            self.all_tasks_eye_positions_with_timestamps[current_task_name] = list(self.current_task_eye_positions)
            
            messagebox.showinfo("Task Complete", f"Task '{current_task_name}' completed. Duration: {self.current_task_duration_seconds} seconds.")

            self.current_task_index += 1
            if self.current_task_index < len(self.tasks):
                self.tracking_active = False
                self.show_reading_screen()
            else:
                self.tracking_active = False
                self.show_results()

    def detect_initial_eye_position(self):
        if not self.cap or not self.cap.isOpened():
            self.eye_detected = False
            return False

        ret, frame = self.cap.read()
        if not ret:
            self.eye_detected = False
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            (fx, fy, fw, fh) = faces[0]
            face_roi = gray[fy:fy+fh, fx:fx+fw]
            
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            
            if len(eyes) > 0:
                eye_dict = {}
                for (ex, ey, ew, eh) in eyes:
                    area = ew * eh
                    eye_dict[area] = (fx + ex, fy + ey, ew, eh)
                
                sorted_eyes = sorted(eye_dict.items(), reverse=True)
                if sorted_eyes:
                    _, (self.eye_x, self.eye_y, self.eye_width, self.eye_height) = sorted_eyes[0]
                    self.eye_detected = True
                    
                    self.eye_x = max(0, self.eye_x - 5)
                    self.eye_y = max(0, self.eye_y - 5)
                    self.eye_width = min(frame.shape[1] - self.eye_x, self.eye_width + 10)
                    self.eye_height = min(frame.shape[0] - self.eye_y, self.eye_height + 10)
        else:
            eyes = self.eye_cascade.detectMultiScale(gray)
            
            if len(eyes) > 0:
                eye_dict = {}
                for (ex, ey, ew, eh) in eyes:
                    area = ew * eh
                    eye_dict[area] = (ex, ey, ew, eh)
                
                sorted_eyes = sorted(eye_dict.items(), reverse=True)
                if sorted_eyes:
                    _, (self.eye_x, self.eye_y, self.eye_width, self.eye_height) = sorted_eyes[0]
                    self.eye_detected = True
                    
                    self.eye_x = max(0, self.eye_x - 5)
                    self.eye_y = max(0, self.eye_y - 5)
                    self.eye_width = min(frame.shape[1] - self.eye_x, self.eye_width + 10)
                    self.eye_height = min(frame.shape[0] - self.eye_y, self.eye_height + 10)
        
        return self.eye_detected

    def update_timer(self):
        if self.tracking_active and not self.is_calibrating:
            elapsed_time = int(time.time() - self.current_task_start_time)
            time_str = str(datetime.timedelta(seconds=elapsed_time))
            self.timer_label.config(text=time_str)
            self.root.after(1000, self.update_timer)
    
    def detect_events_ivt(self, positions_with_timestamps, velocity_threshold, min_event_duration):
        if len(positions_with_timestamps) < 2:
            return [], []

        fixations = []
        saccades = []

        data = np.array(positions_with_timestamps)
        timestamps_ms = data[:, 0]
        x_coords = data[:, 1]
        y_coords = data[:, 2]

        dt_ms = np.diff(timestamps_ms)
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        velocities = np.sqrt(dx**2 + dy**2) / np.where(dt_ms > 0, dt_ms, 1e-9)

        event_types = (velocities > velocity_threshold).astype(int) 

        current_event_type = None
        current_event_start_time = 0.0 
        current_event_start_idx = 0
        
        for i in range(len(event_types)):
            current_point_time = timestamps_ms[i+1]
            
            if current_event_type is None:
                current_event_type = event_types[i]
                current_event_start_time = timestamps_ms[i]
                current_event_start_idx = i
                continue

            if event_types[i] != current_event_type:
                event_duration = current_point_time - current_event_start_time
                if event_duration >= min_event_duration:
                    event_data = {
                        'start_time': current_event_start_time,
                        'end_time': current_point_time,
                        'duration': event_duration
                    }
                    if current_event_type == 0:
                        fixations.append(event_data)
                    else:
                        saccades.append(event_data)
                
                current_event_type = event_types[i]
                current_event_start_time = current_point_time
                current_event_start_idx = i + 1
        
        if current_event_type is not None:
            last_point_time = timestamps_ms[-1]
            event_duration = last_point_time - current_event_start_time
            if event_duration >= min_event_duration:
                event_data = {
                    'start_time': current_event_start_time,
                    'end_time': last_point_time,
                    'duration': event_duration
                }
                if current_event_type == 0:
                    fixations.append(event_data)
                else:
                    saccades.append(event_data)

        return fixations, saccades
    
    def calculate_task_features(self, positions_with_timestamps, reading_duration_seconds, word_count):
        if not positions_with_timestamps or len(positions_with_timestamps) < 2:
            return {
                'n_sacc_trial': 0,
                'mean_fix_dur_trial': 0,
                'std_fix_dur_trial': 0,
                'max_fix_dur_trial': 0,
                'n_fix_trial': 0
            }

        # Get raw eye movement data for analysis
        data = np.array(positions_with_timestamps)
        timestamps_ms = data[:, 0]
        x_coords = data[:, 1]
        y_coords = data[:, 2]
        
        # Calculate velocities for debugging
        dt_ms = np.diff(timestamps_ms)
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        distances = np.sqrt(dx**2 + dy**2)
        velocities = distances / np.where(dt_ms > 0, dt_ms, 1e-9)
        
        # Detect events with current parameters
        fixations, saccades = self.detect_events_ivt(
            positions_with_timestamps,
            self.velocity_threshold_px_per_ms,
            self.min_event_duration_ms
        )
        
        # Debug information
        print(f"Total data points: {len(positions_with_timestamps)}")
        print(f"Average velocity: {np.mean(velocities):.4f} px/ms")
        print(f"Max velocity: {np.max(velocities):.4f} px/ms")
        print(f"Detected saccades: {len(saccades)}")
        print(f"Detected fixations: {len(fixations)}")
        
        total_saccades = len(saccades)
        total_fixations = len(fixations)
        
        # Calculate fixation statistics
        fixation_durations_ms = [f['duration'] for f in fixations]
        if fixation_durations_ms:
            mean_fix_dur_ms = np.mean(fixation_durations_ms)
            std_fix_dur_ms = np.std(fixation_durations_ms)
            max_fix_dur_ms = np.max(fixation_durations_ms)
        else:
            mean_fix_dur_ms = 0.0
            std_fix_dur_ms = 0.0
            max_fix_dur_ms = 0.0

        return {
            'n_sacc_trial': total_saccades,
            'mean_fix_dur_trial': mean_fix_dur_ms,
            'std_fix_dur_trial': std_fix_dur_ms,
            'max_fix_dur_trial': max_fix_dur_ms,
            'n_fix_trial': total_fixations
        }
        
    def aggregate_features_for_prediction(self, all_task_features, tasks_info):
        # Initialize task-specific feature lists
        task_features = {
            'T1': {'n_sacc': [], 'mean_fix_dur': [], 'std_fix_dur': [], 'max_fix_dur': [], 'n_fix': []},
            'T4': {'n_sacc': [], 'mean_fix_dur': [], 'std_fix_dur': [], 'max_fix_dur': [], 'n_fix': []},
            'T5': {'n_sacc': [], 'mean_fix_dur': [], 'std_fix_dur': [], 'max_fix_dur': [], 'n_fix': []}
        }
        
        # Map task names to task keys (T1, T4, T5)
        task_mapping = {
            'Text 1': 'T1',
            'Text 2': 'T4',
            'Text 3': 'T5'
        }
        
        # Collect features for each task
        for task_name, features in all_task_features.items():
            # Determine task key
            task_key = None
            for key in task_mapping:
                if key in task_name:
                    task_key = task_mapping[key]
                    break
            
            if task_key:
                task_features[task_key]['n_sacc'].append(features.get('n_sacc_trial', 0))
                task_features[task_key]['mean_fix_dur'].append(features.get('mean_fix_dur_trial', 0) / 1000.0)  # Convert to seconds
                task_features[task_key]['std_fix_dur'].append(features.get('std_fix_dur_trial', 0) / 1000.0)    # Convert to seconds
                task_features[task_key]['max_fix_dur'].append(features.get('max_fix_dur_trial', 0) / 1000.0)   # Convert to seconds
                task_features[task_key]['n_fix'].append(features.get('n_fix_trial', 0))
        
        # Calculate mean values for each task and metric
        processed_features = {}
        for task in ['T1', 'T4', 'T5']:
            for metric in ['n_sacc', 'mean_fix_dur', 'std_fix_dur', 'max_fix_dur', 'n_fix']:
                key = f'{task}_{metric}'
                values = task_features[task][metric]
                processed_features[key] = np.mean(values) if values else 0.0
        
        # Calculate the final features used for prediction
        text_characteristics = {
            'T1': {'words': 71},  # Syllables
            'T4': {'words': 85},  # Meaningful Text
            'T5': {'words': 83}   # Pseudo Text
        }
        
        task_results = {
            'T1': {'n_sacc_trial': processed_features['T1_n_sacc'], 'n_fix_trial': processed_features['T1_n_fix']},
            'T4': {'n_sacc_trial': processed_features['T4_n_sacc'], 'n_fix_trial': processed_features['T4_n_fix']},
            'T5': {'n_sacc_trial': processed_features['T5_n_sacc'], 'n_fix_trial': processed_features['T5_n_fix']},
        }
        
        # Calculate features to match the model's expected scale
        # Get word counts for each task
        word_counts = {
            'T1': 71,  # Syllables
            'T4': 85,  # Meaningful Text
            'T5': 83   # Pseudo Text
        }
        
        # Calculate saccades and fixations per word for each task
        saccades_per_word = []
        fixations_per_word = []
        
        for task in task_results:
            word_count = word_counts.get(task, 1)
            saccades = task_results[task]['n_sacc_trial']
            fixations = task_results[task]['n_fix_trial']
            
            if word_count > 0:
                saccades_per_word.append(saccades / word_count)
                fixations_per_word.append(fixations / word_count)
        
        # Calculate mean saccades per word across tasks (scale to match model's expected range)
        n_sacc_trial_mean = np.mean(saccades_per_word)
        n_sacc_trial_sum = sum(task_results[task]['n_sacc_trial'] for task in task_results)
        n_fix_mean = np.mean(fixations_per_word) 
        
        # Calculate duration features (convert to seconds for model)
        duration_ms_mean = np.mean([
            processed_features[f'{task}_mean_fix_dur'] 
            for task in ['T1', 'T4', 'T5']
        ])
        
        duration_ms_std = np.mean([
            processed_features[f'{task}_std_fix_dur'] 
            for task in ['T1', 'T4', 'T5']
        ])
        
        duration_ms_max = max([
            processed_features[f'{task}_max_fix_dur'] 
            for task in ['T1', 'T4', 'T5']
        ])
        
        return {
            'n_sacc_trial_mean': n_sacc_trial_mean,
            'duration_ms_std': duration_ms_std,
            'n_sacc_trial_sum': n_sacc_trial_sum,
            'duration_ms_max': duration_ms_max,
            'duration_ms_mean': duration_ms_mean,
            'n_fix_mean': n_fix_mean
        }
    
    def show_results(self):
        self.clear_screen()
        self.root.geometry('700x700')
        
        # Calculate features for each task
        all_task_calculated_features = {}
        total_reading_duration = 0
        
        for task_info in self.tasks:
            task_name = task_info['name']
            word_count = task_info['words']
            eye_positions = self.all_tasks_eye_positions_with_timestamps.get(task_name, [])
            
            # Calculate task duration in seconds
            task_duration_sec = 1  # Default to 1 second to avoid division by zero
            if len(eye_positions) > 1:
                task_duration_sec = (eye_positions[-1][0] - eye_positions[0][0]) / 1000

            # Calculate features for this task
            all_task_calculated_features[task_name] = self.calculate_task_features(
                eye_positions, task_duration_sec, word_count
            )
            total_reading_duration += task_duration_sec

        # Get aggregated features for prediction
        final_model_input_features = self.aggregate_features_for_prediction(
            all_task_calculated_features, self.tasks
        )
        
        # Create main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Add title
        heading = tk.Label(main_frame, text="Final Eye Tracking Results & Prediction", 
                          font=("Arial", 16, "bold"))
        heading.pack(pady=(0, 20))
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Task-specific features frame
        task_frame = ttk.LabelFrame(scrollable_frame, text="Task-Specific Features", 
                                   padding="10")
        task_frame.pack(fill=tk.X, pady=(0, 15))
        
        for task_info in self.tasks:
            task_name = task_info['name']
            features = all_task_calculated_features.get(task_name, {})
            
            # Task header
            ttk.Label(task_frame, text=f"{task_name}", 
                     font=("Arial", 10, "bold")).pack(anchor='w', pady=(5, 2))
            
            # Task features
            for feature, value in features.items():
                display_name = feature.replace('_trial', '').replace('_', ' ').title()
                if 'fix_dur' in feature:
                    value_str = f"{value:.2f} ms"
                else:
                    value_str = f"{value:.2f}"
                ttk.Label(task_frame, 
                         text=f"  {display_name}: {value_str}", 
                         font=("Arial", 9)).pack(anchor='w', padx=10)
        
        # Aggregated features frame
        agg_frame = ttk.LabelFrame(scrollable_frame, 
                                 text="Aggregated Features for Prediction",
                                 padding="10")
        agg_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Format and display each aggregated feature
        feature_descriptions = {
            'n_sacc_trial_mean': 'Mean Saccades per Word',
            'duration_ms_std': 'Avg. Fixation Duration Std Dev (s)',
            'n_sacc_trial_sum': 'Total Saccades',
            'duration_ms_max': 'Max Fixation Duration (s)',
            'duration_ms_mean': 'Mean Fixation Duration (s)',
            'n_fix_mean': 'Mean Fixation Count per Word'
        }
        
        for feature, value in final_model_input_features.items():
            display_name = feature_descriptions.get(feature, feature.replace('_', ' ').title())
            # Format the value based on its type and magnitude
            if isinstance(value, (int, float)):
                if abs(value) >= 1000 or (abs(value) < 0.0001 and value != 0):
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
                
            ttk.Label(agg_frame, 
                     text=f"{display_name}: {value_str}", 
                     font=("Arial", 10)).pack(anchor='w', pady=2)
        
        # Prediction result
        prediction_frame = ttk.LabelFrame(scrollable_frame, 
                                        text="Dyslexia Prediction",
                                        padding="10")
        prediction_frame.pack(fill=tk.X, pady=(0, 15))
        
        prediction_label = ttk.Label(prediction_frame, 
                                   text="", 
                                   font=("Arial", 12, "bold"))
        prediction_label.pack(pady=5)
        
        # Make prediction if model is available
        if self.dyslexia_model:
            try:
                # Prepare input features in the correct order
                model_input_df = pd.DataFrame([final_model_input_features])[
                    ['n_sacc_trial_mean', 'duration_ms_std', 'n_sacc_trial_sum', 
                     'duration_ms_max', 'duration_ms_mean', 'n_fix_mean']
                ]
                
                # Get prediction probabilities
                prediction_proba = self.dyslexia_model.predict_proba(model_input_df)[0]
                
                # Apply threshold (0.4) to determine final prediction
                threshold = 0.25 
                prediction = 1 if prediction_proba[1] >= threshold else 0
                
                # Format results
                label = "Dyslexic" if prediction == 1 else "Non-Dyslexic"
                confidence = prediction_proba[1] * 100 
                
                # Update UI with prediction
                prediction_label.config(
                    text=f"Prediction: {label}",
                    foreground="red" if prediction == 1 else "green"
                )
                
            except Exception as e:
                prediction_label.config(
                    text=f"Prediction Error: {str(e)}", 
                    foreground="red"
                )
        else:
            prediction_label.config(
                text="Prediction: Model not loaded or available.", 
                foreground="orange"
            )
        
        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Add save button at the bottom
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        save_btn = ttk.Button(
            button_frame,
            text="Save Results",
            command=lambda: self.save_features_to_file(final_model_input_features, total_reading_duration)
        )
        save_btn.pack(side=tk.RIGHT, padx=5)
        
        back_btn = ttk.Button(
            button_frame,
            text="Back to Start",
            command=self.create_input_form
        )
        back_btn.pack(side=tk.RIGHT, padx=5)
        
        # Configure canvas scrolling
        canvas.bind_all("<MouseWheel>", 
                       lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
    
    def save_features_to_file(self, features, total_duration):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"eye_tracking_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write(f"Eye Tracking Results\n")
            f.write(f"Date: {datetime.datetime.now()}\n")
            f.write(f"Name: {self.user.get('name', 'N/A')}\n")
            f.write(f"Student ID: {self.user.get('student_id', 'N/A')}\n")
            f.write(f"Tracking Duration: {int(total_duration)} seconds\n\n")
            
            f.write("Extracted Features:\n")
            f.write("-" * 40 + "\n")
            for feature, value in features.items():
                f.write(f"{feature}: {value:.4f}\n" if isinstance(value, float) else f"{feature}: {value}\n")
        
        messagebox.showinfo("Success", f"Results saved to {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingApp(root)
    root.mainloop()