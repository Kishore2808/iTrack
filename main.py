import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import mediapipe as mp
import time
import util
from simple_facerec import SimpleFacerec
import matplotlib.pyplot as plt

class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.text_lab = util.get_text_label(self.main_window, 'iTrack\nATTENTION\nMONITORING\nSYSTEM')
        self.text_lab.place(x=765, y=30)

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'black', self.login, fg='black')
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = '/Users/kishorekumarr/Documents/SW/photos'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = '/Users/kishorekumarr/Documents/SW/loge.txt'
        self.logg_path = '/Users/kishorekumarr/Documents/SW/logge.txt'

    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_arr = frame
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

        self._label.after(20, self.process_webcam)

    def login(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)
        print(name)
        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
            with open(self.log_path, 'a') as f:
                f.write('{},{},in'.format(name, datetime.datetime.now()))
            self.cap.release()
            self.main_window.destroy()
            self.face_detection_loop()

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user,fg='black')
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user,fg='black')
        self.try_again_button_register_new_user_window.place(x=750, y=400)

        self.capture_label = util.get_img_label(self.register_new_user_window)
        self.capture_label.place(x=10, y=0, width=700, height=500)

        self.add_img_to_label(self.capture_label)

        self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
        self.entry_text_register_new_user.place(x=750, y=150)

        self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
        self.text_label_register_new_user.place(x=750, y=70)

    def try_again_register_new_user(self):
        self.register_new_user_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def accept_register_new_user(self):
        name = self.entry_text_register_new_user.get(1.0, "end-1c")

        embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

        file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
        pickle.dump(embeddings, file)

        util.msg_box('Success!', 'User was registered successfully !')

        self.register_new_user_window.destroy()

    # def accept_register_new_user(self):
    #     name = self.entry_text_register_new_user.get(1.0, "end-1c")

    #     cv2.imwrite(os.path.join(self.db_dir, '{}.jpeg'.format(name)),self.register_new_user_capture)

    #     util.msg_box('Success!', 'User was registered successfully !')

    #     self.register_new_user_window.destroy()

    def graph(self):


        log_file_path = '/Users/kishorekumarr/Documents/SW/logge.txt'

        
        all_values = []

       
        with open(log_file_path, 'r') as file:
            for line in file:
                
                values = list(map(float, line.split()))
                
                
                all_values.extend(values)


        plt.figure(figsize=(10, 5))
        plt.plot(all_values, marker='o', linestyle='-', color='b')

        
        plt.xlabel('Days')
        plt.ylabel('Hours')
        plt.title('Graph of Time Spent On Screen')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    # def face_detection_loop(self):
    #     sfr = SimpleFacerec()
    #     sfr.load_encoding_images("images/")

    #     cap = cv2.VideoCapture(0)

    #     maximum_time = 15

    #     face_detection = mp.solutions.face_detection.FaceDetection()

    #     starting_time = time.time()
    #     initial = time.time()

    #     total_time = 0
    #     elapsed_time = 0
    #     last_time_face_detected = time.time()

    #     max_duration = 5

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         height, width, channels = frame.shape
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #         cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)

    #         results = face_detection.process(rgb_frame)

    #         if results.detections:
    #             current_time = time.time()
    #             elapsed_time = current_time - last_time_face_detected

    #             total_time += elapsed_time

    #             # cv2.putText(frame, "Face Detected: {:.2f} seconds".format(elapsed_time), (10, 50), cv2.FONT_HERSHEY_PLAIN,
    #                         # 3, (15, 225, 215), 2)
                
    #             last_time_face_detected = current_time

    #             print(f"Elapsed Time: {elapsed_time:.2f} seconds")
    #             print("Face looking at the screen")

    #         else:
    #             last_time_face_detected = time.time()

    #             print("NO FACE")
    #             print(f"Total time so far: {total_time:.2f} seconds")

    #         cv2.imshow("Frame", frame)
            
    #         if int(time.time() - initial) >= max_duration:
    #             break

    #         key = cv2.waitKey(1)
    #         if key == 10:
    #             break

    #     final = int(time.time() - initial)
    #     print(f"Total time elapsed: {final} seconds")
    #     percentage = ((total_time * 100) / final)- 0.97
    #     print(f"Percentage of time looking at screen: {percentage:.2f}%")
    #     total_timee= total_time-0.05
    #     print(f"Total time spent looking at screen: {total_timee:.2f} seconds")
    #     with open(self.logg_path, 'a') as f:
    #         f.write(f"{total_time:.2f} ") 



    #     cap.release()
    #     cv2.destroyAllWindows()
    #     self.graph()





############################################EXISTING VERSION##########################################




    # def face_detection_loop(self):
    #     cap = cv2.VideoCapture(0)

    #     eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    #     max_duration = 5  # seconds
    #     starting_time = time.time()
    #     initial = time.time()
    #     total_time = 0
    #     elapsed_time = 0
    #     last_time_eye_detected = time.time()

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    #         height, width, channels = frame.shape
    #         cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)

    #         if len(eyes) > 0:
    #             current_time = time.time()
    #             elapsed_time = current_time - last_time_eye_detected
    #             total_time += elapsed_time

    #             last_time_eye_detected = current_time

    #             print(f"Elapsed Time (eyes): {elapsed_time:.2f} seconds")
    #             print("Eyes detected (looking at screen)")

    #             for (ex, ey, ew, eh) in eyes:
    #                 cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    #         else:
    #             last_time_eye_detected = time.time()
    #             print("NO EYES")
    #             print(f"Total time so far: {total_time:.2f} seconds")

    #         cv2.imshow("Eye Detection", frame)

    #         if int(time.time() - initial) >= max_duration:
    #             break

    #         key = cv2.waitKey(1)
    #         if key == 10:
    #             break

    #     final = int(time.time() - initial)
    #     print(f"Total time elapsed: {final} seconds")
    #     percentage = ((total_time * 100) / final) - 0.97
    #     print(f"Percentage of time looking at screen (eyes): {percentage:.2f}%")
    #     total_timee = total_time - 0.05
    #     print(f"Total time spent looking at screen (eyes): {total_timee:.2f} seconds")

    #     with open(self.logg_path, 'a') as f:
    #         f.write(f"{total_time:.2f} ")

    #     cap.release()
    #     cv2.destroyAllWindows()
    #     self.graph()




###############################################GEMINI VERSION########################################## - ONLY IRIS-

    # def face_detection_loop(self):
    #     import numpy as np
    #     cap = cv2.VideoCapture(0)

    #     mp_face_mesh = mp.solutions.face_mesh
    #     face_mesh = mp_face_mesh.FaceMesh(
    #         max_num_faces=1,
    #         refine_landmarks=True,
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5
    #     )

    #     max_duration = 15
    #     initial = time.time()
    #     total_focus_time = 0
    #     last_time_focused = time.time()
        
    #     # Landmark indices
    #     LEFT_IRIS_CENTER = 473
    #     LEFT_EYE_INNER_CORNER = 33
    #     LEFT_EYE_OUTER_CORNER = 133
        
    #     RIGHT_IRIS_CENTER = 468
    #     RIGHT_EYE_INNER_CORNER = 263
    #     RIGHT_EYE_OUTER_CORNER = 362

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: break

    #         frame = cv2.flip(frame, 1)
    #         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #         img_h, img_w = frame.shape[:2]
    #         results = face_mesh.process(rgb_frame)

    #         is_focused = False
    #         if results.multi_face_landmarks:
    #             landmarks = results.multi_face_landmarks[0].landmark

    #             # Get pixel coordinates for all necessary landmarks
    #             left_iris_pt = np.array([landmarks[LEFT_IRIS_CENTER].x * img_w, landmarks[LEFT_IRIS_CENTER].y * img_h])
    #             left_inner_corner_pt = np.array([landmarks[LEFT_EYE_INNER_CORNER].x * img_w, landmarks[LEFT_EYE_INNER_CORNER].y * img_h])
    #             left_outer_corner_pt = np.array([landmarks[LEFT_EYE_OUTER_CORNER].x * img_w, landmarks[LEFT_EYE_OUTER_CORNER].y * img_h])

    #             right_iris_pt = np.array([landmarks[RIGHT_IRIS_CENTER].x * img_w, landmarks[RIGHT_IRIS_CENTER].y * img_h])
    #             right_inner_corner_pt = np.array([landmarks[RIGHT_EYE_INNER_CORNER].x * img_w, landmarks[RIGHT_EYE_INNER_CORNER].y * img_h])
    #             right_outer_corner_pt = np.array([landmarks[RIGHT_EYE_OUTER_CORNER].x * img_w, landmarks[RIGHT_EYE_OUTER_CORNER].y * img_h])

    #             # Calculate the Euclidean distance
    #             dist_left_inner = np.linalg.norm(left_iris_pt - left_inner_corner_pt)
    #             dist_left_outer = np.linalg.norm(left_iris_pt - left_outer_corner_pt)
                
    #             dist_right_inner = np.linalg.norm(right_iris_pt - right_inner_corner_pt)
    #             dist_right_outer = np.linalg.norm(right_iris_pt - right_outer_corner_pt)

    #             # Avoid division by zero
    #             if dist_left_outer > 0 and dist_right_outer > 0:
    #                 left_eye_ratio = dist_left_inner / dist_left_outer
    #                 right_eye_ratio = dist_right_inner / dist_right_outer

    #                 # --- FINAL CALIBRATED IF CONDITION ---
    #                 # Checks if the calculated ratios are within a window around your personal average of ~1.6
    #                 if 1.50 < left_eye_ratio < 1.70 and 1.50 < right_eye_ratio < 1.70:
    #                     is_focused = True

    #         # --- Update and Display Logic ---
    #         current_time = time.time()
    #         if is_focused:
    #             focus_duration = current_time - last_time_focused
    #             total_focus_time += focus_duration
    #             status_text = "FOCUSED"
    #             status_color = (0, 255, 0)
    #         else:
    #             status_text = "NOT FOCUSED"
    #             status_color = (0, 0, 255)
    #         last_time_focused = current_time

    #         cv2.rectangle(frame, (0, 0), (img_w, 70), (10, 10, 10), -1)
    #         cv2.putText(frame, f"STATUS: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    #         cv2.putText(frame, f"Focus Time: {total_focus_time:.2f} s", (img_w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #         cv2.putText(frame, f"Time Left: {max_duration - (current_time - initial):.0f} s", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #         cv2.imshow("Attention Monitoring", frame)

    #         if int(time.time() - initial) >= max_duration: break
    #         if cv2.waitKey(1) == 27: break

    #     # --- Cleanup and Graphing ---
    #     final_duration = time.time() - initial
    #     percentage = (total_focus_time / final_duration) * 100 if final_duration > 0 else 0
    #     print(f"Total time elapsed: {final_duration:.2f} seconds")
    #     print(f"Total time spent focused on screen: {total_focus_time:.2f} seconds")
    #     print(f"Percentage of time focused: {percentage:.2f}%")
    #     with open(self.logg_path, 'a') as f: f.write(f"{total_focus_time:.2f} ")
    #     cap.release()
    #     cv2.destroyAllWindows()
    #     self.graph()



    def face_detection_loop(self):
        import numpy as np
        cap = cv2.VideoCapture(0)

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        def eye_aspect_ratio(eye_points):
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            ear = (A + B) / (2.0 * C)
            return ear

        # --- Time, Threshold, and Data Collection Variables ---
        max_duration = 15
        initial = time.time()
        total_focus_time = 0
        last_time_focused = time.time()
        EYE_AR_THRESHOLD = 0.25
        time_eyes_closed_start = 0
        EYES_CLOSED_DURATION_THRESHOLD = 3
        
        focus_sequence = []
        last_sample_time = time.time()
        
        # --- NEW: Frame counters for scoring ---
        focused_frames_in_second = 0
        total_frames_in_second = 0

        # Landmark indices
        LEFT_IRIS_CENTER = 473
        LEFT_EYE_INNER_CORNER = 33
        LEFT_EYE_OUTER_CORNER = 133
        RIGHT_IRIS_CENTER = 468
        RIGHT_EYE_INNER_CORNER = 263
        RIGHT_EYE_OUTER_CORNER = 362
        LEFT_EYE_LID = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_LID = [362, 385, 387, 263, 373, 380]

        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)

            is_focused = False
            are_eyes_open = False
            
            # --- NEW: Increment frame counter on every frame ---
            total_frames_in_second += 1

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Check EAR
                left_eye_lid_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in LEFT_EYE_LID])
                right_eye_lid_pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in RIGHT_EYE_LID])
                avg_ear = (eye_aspect_ratio(left_eye_lid_pts) + eye_aspect_ratio(right_eye_lid_pts)) / 2.0
                if avg_ear > EYE_AR_THRESHOLD:
                    are_eyes_open = True

                # Check Gaze Ratio
                left_iris_pt = np.array([landmarks[LEFT_IRIS_CENTER].x * img_w, landmarks[LEFT_IRIS_CENTER].y * img_h])
                left_inner_corner_pt = np.array([landmarks[LEFT_EYE_INNER_CORNER].x * img_w, landmarks[LEFT_EYE_INNER_CORNER].y * img_h])
                left_outer_corner_pt = np.array([landmarks[LEFT_EYE_OUTER_CORNER].x * img_w, landmarks[LEFT_EYE_OUTER_CORNER].y * img_h])
                right_iris_pt = np.array([landmarks[RIGHT_IRIS_CENTER].x * img_w, landmarks[RIGHT_IRIS_CENTER].y * img_h])
                right_inner_corner_pt = np.array([landmarks[RIGHT_EYE_INNER_CORNER].x * img_w, landmarks[RIGHT_EYE_INNER_CORNER].y * img_h])
                right_outer_corner_pt = np.array([landmarks[RIGHT_EYE_OUTER_CORNER].x * img_w, landmarks[RIGHT_EYE_OUTER_CORNER].y * img_h])

                dist_left_outer = np.linalg.norm(left_iris_pt - left_outer_corner_pt)
                dist_right_outer = np.linalg.norm(right_iris_pt - right_outer_corner_pt)

                if dist_left_outer > 0 and dist_right_outer > 0:
                    dist_left_inner = np.linalg.norm(left_iris_pt - left_inner_corner_pt)
                    dist_right_inner = np.linalg.norm(right_iris_pt - right_inner_corner_pt)
                    left_eye_ratio = dist_left_inner / dist_left_outer
                    right_eye_ratio = dist_right_inner / dist_right_outer
                    if 1.40 < left_eye_ratio < 1.80 and 1.40 < right_eye_ratio < 1.80:
                        if are_eyes_open:
                            is_focused = True

            if is_focused:
                # --- NEW: Increment focused frame counter ---
                focused_frames_in_second += 1

            # --- Timer and Status Logic ---
            current_time = time.time()
            if is_focused:
                focus_duration = current_time - last_time_focused
                total_focus_time += focus_duration
                status_text = "FOCUSED"
                status_color = (0, 255, 0)
                time_eyes_closed_start = 0
            else:
                status_color = (0, 0, 255)
                if are_eyes_open:
                    status_text = "LOOKING AWAY"
                    time_eyes_closed_start = 0
                else:
                    if time_eyes_closed_start == 0:
                        time_eyes_closed_start = current_time
                        status_text = "BLINKING"
                    else:
                        duration_closed = current_time - time_eyes_closed_start
                        if duration_closed > EYES_CLOSED_DURATION_THRESHOLD:
                            status_text = "EYES CLOSED (DROWSY)"
                        else:
                            status_text = "BLINKING"
            last_time_focused = current_time
            
            # --- NEW: Data sampling logic with scoring ---
            if current_time - last_sample_time >= 1.0:
                # Calculate focus score for the last second
                if total_frames_in_second > 0:
                    focus_score = focused_frames_in_second / total_frames_in_second
                else:
                    focus_score = 0
                
                focus_sequence.append(focus_score)
                
                # Reset counters for the next second
                total_frames_in_second = 0
                focused_frames_in_second = 0
                last_sample_time = current_time

            # --- Display logic (no changes) ---
            cv2.rectangle(frame, (0, 0), (img_w, 70), (10, 10, 10), -1)
            cv2.putText(frame, f"STATUS: {status_text}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Focus Time: {total_focus_time:.2f} s", (img_w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Time Left: {max_duration - (current_time - initial):.0f} s", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("Attention Monitoring", frame)

            if int(time.time() - initial) > max_duration: break
            if cv2.waitKey(1) == 27: break

        # --- Cleanup, Final Calculations, and Data Saving ---
        cap.release()
        cv2.destroyAllWindows()
        
        session_dir = 'sessions_data'
        if not os.path.exists(session_dir):
            os.mkdir(session_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        session_filename = os.path.join(session_dir, f"session_{timestamp}.txt")
        
        with open(session_filename, 'w') as f:
            # Format floats to 2 decimal places and join with commas
            f.write(",".join(["{:.2f}".format(s) for s in focus_sequence]))
            
        print(f"Session data saved to: {session_filename}")
        # Format the print output for readability
        formatted_data = ["{:.2f}".format(s) for s in focus_sequence]
        print(f"Data: {formatted_data}")

        # --- Graphing (no changes) ---
        final_duration = time.time() - initial
        percentage = (total_focus_time / final_duration) * 100 if final_duration > 0 else 0
        print(f"Total time elapsed: {final_duration:.2f} seconds")
        print(f"Total time spent focused on screen: {total_focus_time:.2f} seconds")
        print(f"Percentage of time focused: {percentage:.2f}%")
        with open(self.logg_path, 'a') as f: f.write(f"{total_focus_time:.2f} ")
        self.graph()













































    # import numpy as np

    # def face_detection_loop(self):
    #     sfr = SimpleFacerec()
    #     sfr.load_encoding_images("images/")
    #     cap = cv2.VideoCapture(0)
    #     mp_face_mesh = mp.solutions.face_mesh
    #     face_mesh = mp_face_mesh.FaceMesh(
    #         max_num_faces=1,
    #         refine_landmarks=True,
    #         min_detection_confidence=0.5,
    #         min_tracking_confidence=0.5
    #     )

    #     # Eye landmarks indices (MediaPipe Face Mesh)
    #     LEFT_EYE = [33, 160, 158, 133, 153, 144]
    #     RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
    #     # Time tracking variables
    #     total_focus_time = 0
    #     last_focus_time = time.time()
    #     EYE_AR_THRESHOLD = 0.3  # Eye Aspect Ratio threshold

    #     def eye_aspect_ratio(eye_points):
    #         # Vertical distances
    #         A = np.linalg.norm(eye_points[1] - eye_points[5])
    #         B = np.linalg.norm(eye_points[2] - eye_points[4])
    #         # Horizontal distance
    #         C = np.linalg.norm(eye_points[0] - eye_points[3])
    #         return (A + B) / (2.0 * C)

    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break

    #         frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    #         results = face_mesh.process(frame)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    #         if results.multi_face_landmarks:
    #             for face_landmarks in results.multi_face_landmarks:
    #                 # Extract eye coordinates
    #                 left_eye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
    #                                     face_landmarks.landmark[i].y * frame.shape[0]) 
    #                                     for i in LEFT_EYE], dtype=np.int32)
                    
    #                 right_eye = np.array([(face_landmarks.landmark[i].x * frame.shape[1], 
    #                                     face_landmarks.landmark[i].y * frame.shape[0]) 
    #                                     for i in RIGHT_EYE], dtype=np.int32)

    #                 # Calculate Eye Aspect Ratio (EAR)
    #                 left_ear = eye_aspect_ratio(left_eye)
    #                 right_ear = eye_aspect_ratio(right_eye)
    #                 avg_ear = (left_ear + right_ear) / 2.0

    #                 if avg_ear > EYE_AR_THRESHOLD:
    #                     current_time = time.time()
    #                     total_focus_time += current_time - last_focus_time
    #                     last_focus_time = current_time
    #                 else:
    #                     last_focus_time = time.time()

    #         # Display tracking info
    #         cv2.putText(frame, f"Focus Time: {total_focus_time:.2f}s", 
    #                 (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
    #         cv2.imshow('Eye Tracking', frame)
            
    #         if cv2.waitKey(5) & 0xFF == 27:
    #             break

    #     cap.release()
    #     cv2.destroyAllWindows()
    #     self.graph()


if __name__ == "__main__":
    app = App()
    app.start()
