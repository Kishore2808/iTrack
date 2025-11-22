# import os.path
# import datetime
# import pickle

# import tkinter as tk
# import cv2
# from PIL import Image, ImageTk
# import face_recognition

# import util
# #from test import test


# class App:
#     def __init__(self):
#         self.main_window = tk.Tk()
#         self.main_window.geometry("1200x520+350+100")

#         self.text_lab = util.get_text_label(self.main_window, 'ONLINE\nMONITORING\nSYSTEM')
#         self.text_lab.place(x=775, y=70)

#         # self.login_button_main_window = util.get_button(self.main_window, 'login', 'green', self.login)
#         # self.login_button_main_window.place(x=750, y=200)

#         self.login_button_main_window = util.get_button(self.main_window, 'login','black', self.login,fg='black')
#         self.login_button_main_window.place(x=750, y=300)

#         # self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',
#         #                                                             self.register_new_user, fg='black')
#         # self.register_new_user_button_main_window.place(x=750, y=400)

#         self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray',self.register_new_user, fg='black')
#         self.register_new_user_button_main_window.place(x=750, y=400)

#         self.webcam_label = util.get_img_label(self.main_window)
#         self.webcam_label.place(x=10, y=0, width=700, height=500)

#         self.add_webcam(self.webcam_label)

#         self.db_dir = '/Users/ramadas/Documents/SW/photos'
#         if not os.path.exists(self.db_dir):
#             os.mkdir(self.db_dir)

#         self.log_path = './loge.txt'

#     def add_webcam(self, label):
#         if 'cap' not in self.__dict__:
#             self.cap = cv2.VideoCapture(0)

#         self._label = label
#         self.process_webcam()

#     def process_webcam(self):
#         ret, frame = self.cap.read()
#         self.most_recent_capture_arr = frame
#         img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
#         self.most_recent_capture_pil = Image.fromarray(img_)
#         imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
#         self._label.imgtk = imgtk
#         self._label.configure(image=imgtk)

#         self._label.after(20, self.process_webcam)

#         # self.most_recent_capture_arr = frame
#         # img_ = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
#         # self.most_recent_capture_pil = Image.fromarray(img_)
#         # imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
#         # self._label.imgtk = imgtk
#         # self._label.configure(image=imgtk)

#         # self._label.after(20, self.process_webcam)

#     def login(self):


#         name = util.recognize(self.most_recent_capture_arr, self.db_dir)

#         if name in ['unknown_person', 'no_persons_found']:
#             util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
#         else:
#             util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
#             with open(self.log_path, 'a') as f:
#                 f.write('{},{},in\n'.format(name, datetime.datetime.now()))
#                 f.close()


#     def register_new_user(self):
#         self.register_new_user_window = tk.Toplevel(self.main_window)
#         self.register_new_user_window.geometry("1200x520+370+120")

#         self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
#         self.accept_button_register_new_user_window.place(x=750, y=300)

#         self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
#         self.try_again_button_register_new_user_window.place(x=750, y=400)

#         self.capture_label = util.get_img_label(self.register_new_user_window)
#         self.capture_label.place(x=10, y=0, width=700, height=500)

#         self.add_img_to_label(self.capture_label)

#         self.entry_text_register_new_user = util.get_entry_text(self.register_new_user_window)
#         self.entry_text_register_new_user.place(x=750, y=150)

#         self.text_label_register_new_user = util.get_text_label(self.register_new_user_window, 'Please, \ninput username:')
#         self.text_label_register_new_user.place(x=750, y=70)

#     def try_again_register_new_user(self):
#         self.register_new_user_window.destroy()

#     def add_img_to_label(self, label):
#         imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
#         label.imgtk = imgtk
#         label.configure(image=imgtk)

#         self.register_new_user_capture = self.most_recent_capture_arr.copy()

#     def start(self):
#         self.main_window.mainloop()

#     def accept_register_new_user(self):
#         name = self.entry_text_register_new_user.get(1.0, "end-1c")

#         embeddings = face_recognition.face_encodings(self.register_new_user_capture)[0]

#         file = open(os.path.join(self.db_dir, '{}.pickle'.format(name)), 'wb')
#         pickle.dump(embeddings, file)

#         util.msg_box('Success!', 'User was registered successfully !')

#         self.register_new_user_window.destroy()


# if __name__ == "__main__":
#     app = App()
#     app.start()

import os
import datetime
import pickle
import tkinter as tk
import cv2
from PIL import Image, ImageTk
import face_recognition
import mediapipe as mp
import time
import util  # Ensure this contains the relevant utility functions


from simple_facerec import SimpleFacerec


class App:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")

        self.text_lab = util.get_text_label(self.main_window, 'ONLINE\nMONITORING\nSYSTEM')
        self.text_lab.place(x=775, y=70)

        self.login_button_main_window = util.get_button(self.main_window, 'login', 'black', self.login, fg='black')
        self.login_button_main_window.place(x=750, y=300)

        self.register_new_user_button_main_window = util.get_button(self.main_window, 'register new user', 'gray', self.register_new_user, fg='black')
        self.register_new_user_button_main_window.place(x=750, y=400)

        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=0, width=700, height=500)

        self.add_webcam(self.webcam_label)

        self.db_dir = '/Users/ramadas/Documents/SW/photos'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './loge.txt'

        

    def add_webcam(self, label):
        # Create a VideoCapture object for webcam input, if not already created
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Convert to RGB for Tkinter label update
        img_ = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_arr = frame
        self.most_recent_capture_pil = Image.fromarray(img_)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        self._label.imgtk = imgtk
        self._label.configure(image=imgtk)

       

        # Continue updating every 20 ms
        self._label.after(20, self.process_webcam)

    def login(self):
        name = util.recognize(self.most_recent_capture_arr, self.db_dir)

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('Ups...', 'Unknown user. Please register new user or try again.')
        else:
            util.msg_box('Welcome back !', 'Welcome, {}.'.format(name))
            with open(self.log_path, 'a') as f:
                f.write('{},{},in\n'.format(name, datetime.datetime.now()))
            self.main_window.destroy()
            # Call face detection loop
            self.face_detection_loop()

    def register_new_user(self):
        self.register_new_user_window = tk.Toplevel(self.main_window)
        self.register_new_user_window.geometry("1200x520+370+120")

        self.accept_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Accept', 'green', self.accept_register_new_user)
        self.accept_button_register_new_user_window.place(x=750, y=300)

        self.try_again_button_register_new_user_window = util.get_button(self.register_new_user_window, 'Try again', 'red', self.try_again_register_new_user)
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

    def face_detection_loop(self):

        # # Initialize face detection (for tracking if user is looking at the screen)
        # self.face_detection = mp.solutions.face_detection.FaceDetection()
        # self.starting_time = time.time()
        # self.total_time = 0
        # self.elapsed_time = 0
        # self.last_time_face_detected = time.time()

        # # Set the duration for webcam feed
        # self.max_duration = 5  # seconds (example)

        # # Face Detection (to track if the user is looking at the screen)
        # frame = self.most_recent_capture_arr

        # if frame is None:
        #     return

        # height, width, channels = frame.shape
        # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # # Draw rectangle
        # cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)

        # # Face Detection
        # results = self.face_detection.process(rgb_frame)

        # # If a face is detected
        # if results.detections:
        #     current_time = time.time()
        #     # Calculate the elapsed time since the face was last detected
        #     self.elapsed_time = current_time - self.last_time_face_detected

        #     # Add elapsed time to the total_time
        #     self.total_time += self.elapsed_time

        #     # Draw elapsed time on screen
        #     cv2.putText(frame, "Face Detected: {:.2f} seconds".format(self.elapsed_time), (10, 50), cv2.FONT_HERSHEY_PLAIN,
        #                 3, (15, 225, 215), 2)

        #     # Update last time face was detected
        #     self.last_time_face_detected = current_time

        #     print(f"Elapsed Time: {self.elapsed_time:.2f} seconds")
        #     print("Face looking at the screen")

        # else:
        #     # Reset elapsed time when no face is detected
        #     self.last_time_face_detected = time.time()

        #     print("NO FACE")
        #     print(f"Total time so far: {self.total_time:.2f} seconds")

        # # Display frame
        # cv2.imshow("Frame", frame)

        # # Check if max_duration has passed (e.g., stop after 5 seconds for demonstration)
        # if int(time.time() - self.starting_time) >= self.max_duration:
        #     final = int(time.time() - self.starting_time)
        #     print(f"Total time elapsed: {final} seconds")
        #     percentage = (self.total_time * 100) / final
        #     print(f"Percentage of time looking at screen: {percentage:.2f}%")
        #     print(f"Total time spent looking at screen: {self.total_time:.2f} seconds")
        #     self.cap.release()
        #     cv2.destroyAllWindows()
        #     return


        # Encode faces from a folder
        sfr = SimpleFacerec()
        sfr.load_encoding_images("images/")

        # Load Camera
        cap = cv2.VideoCapture(0)

        # Settings
        maximum_time = 15  # Seconds

        # Load Face Detector
        face_detection = mp.solutions.face_detection.FaceDetection()

        # Track TIME
        starting_time = time.time()
        initial = time.time()

        total_time = 0  # Total time spent with face detected
        elapsed_time = 0  # Time spent with face detected in each cycle
        last_time_face_detected = time.time()  # To track time since last face detection

        # Set the duration after which the loop will stop
        max_duration = 5  # seconds (For demonstration, set to 5 seconds)

        while True:
            # Take frame from camera
            ret, frame = cap.read()
            if not ret:
                break
            
            height, width, channels = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw rectangle
            cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)

            # Face Detection
            results = face_detection.process(rgb_frame)

            # If a face is detected
            if results.detections:
                current_time = time.time()
                # Calculate the elapsed time since the face was last detected
                elapsed_time = current_time - last_time_face_detected

                # Add elapsed time to the total_time
                total_time += elapsed_time

                # Draw elapsed time on screen
                cv2.putText(frame, "Face Detected: {:.2f} seconds".format(elapsed_time), (10, 50), cv2.FONT_HERSHEY_PLAIN,
                            3, (15, 225, 215), 2)
                
                # Update last time face was detected
                last_time_face_detected = current_time

                print(f"Elapsed Time: {elapsed_time:.2f} seconds")
                print("Face looking at the screen")

            else:
                # Reset elapsed time when no face is detected
                last_time_face_detected = time.time()

                print("NO FACE")
                print(f"Total time so far: {total_time:.2f} seconds")

            # Display frame
            cv2.imshow("Frame", frame)
            
            # Check if 30 seconds have passed (adjust `max_duration` to 5 for this example)
            if int(time.time() - initial) >= max_duration:
                break

            key = cv2.waitKey(1)
            if key == 10:  # Break the loop if Enter is pressed
                break

        # Calculate final time elapsed
        final = int(time.time() - initial)
        print(f"Total time elapsed: {final} seconds")
        percentage = (total_time * 100) / final
        print(f"Percentage of time looking at screen: {percentage:.2f}%")
        print(f"Total time spent looking at screen: {total_time:.2f} seconds")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.start()
