# import streamlit as st
# import cv2
# import mediapipe as mp
# import tempfile
# import numpy as np

# st.set_page_config(page_title="Eye Contact Analyzer", layout="centered")
# st.title("Eye Contact Analyzer")
# st.write("Upload a recorded video. It will play normally, and after processing, you will get the eye contact percentage.")

# # --- Upload section ---
# uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# if uploaded_file is not None:
#     # Save uploaded video to temp file
#     temp_file = tempfile.NamedTemporaryFile(delete=False)
#     temp_file.write(uploaded_file.read())
#     video_path = temp_file.name

#     # --- Play uploaded video in Streamlit (short height) ---
#     st.markdown(
#         """
#         <style>
#         video {
#             max-height: 240px;  /* adjust height here */
#             width: auto;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
#     st.video(video_path)

#     # --- Initialize MediaPipe ---
#     mp_face_mesh = mp.solutions.face_mesh
#     mp_drawing = mp.solutions.drawing_utils

#     # --- Prepare to save backend processed video (facemesh overlay only in backend) ---
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
#     backend_video_path = "processed_backend.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(backend_video_path, fourcc, fps, (width, height))

#     frame_count = 0
#     eye_contact_frames = 0
#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     with mp_face_mesh.FaceMesh(
#         max_num_faces=1,
#         refine_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as face_mesh:

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_count += 1
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = face_mesh.process(rgb_frame)

#             if results.multi_face_landmarks:
#                 for face_landmarks in results.multi_face_landmarks:
#                     # --- Eye contact logic ---
#                     left_eye = np.mean([[lm.x, lm.y] for lm in [
#                         face_landmarks.landmark[33],
#                         face_landmarks.landmark[133]
#                     ]], axis=0)
#                     right_eye = np.mean([[lm.x, lm.y] for lm in [
#                         face_landmarks.landmark[362],
#                         face_landmarks.landmark[263]
#                     ]], axis=0)
#                     nose_tip = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])
#                     eye_center = np.mean([left_eye, right_eye], axis=0)
#                     eye_direction = nose_tip - eye_center

#                     if abs(eye_direction[0]) < 0.02:
#                         eye_contact_frames += 1

#                     # --- Draw facemesh only in backend video ---
#                     mp_drawing.draw_landmarks(
#                         frame,
#                         face_landmarks,
#                         mp_face_mesh.FACEMESH_TESSELATION,
#                         landmark_drawing_spec=None,
#                         connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
#                     )

#             # Write frame to backend video
#             out.write(frame)

#             # Update progress
#             progress_bar.progress(frame_count / total_frames)
#             status_text.text(f"Processing frame {frame_count}/{total_frames}")

#     cap.release()
#     out.release()

#     # --- Compute eye contact results ---
#     eye_contact_percentage = (eye_contact_frames / total_frames) * 100
#     st.success("✅ Analysis complete!")
#     st.subheader("📊 Eye Contact Report")
#     st.write(f"**Eye Contact Percentage:** {eye_contact_percentage:.2f}%")

# eye_contact_logic.py
# combined_eye_contact_logic.py
import cv2
import mediapipe as mp
import numpy as np
import os

def process_eye_contact(video_path, output_path, progress_callback=None):
    """
    Processes video to calculate eye contact percentage
    and saves a processed video with facemesh overlay.
    Returns: eye_contact_percentage, output_video_path
    """

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    eye_contact_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_eye = np.mean([[lm.x, lm.y] for lm in [
                        face_landmarks.landmark[33],
                        face_landmarks.landmark[133]
                    ]], axis=0)
                    right_eye = np.mean([[lm.x, lm.y] for lm in [
                        face_landmarks.landmark[362],
                        face_landmarks.landmark[263]
                    ]], axis=0)
                    nose_tip = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])
                    eye_center = np.mean([left_eye, right_eye], axis=0)
                    eye_direction = nose_tip - eye_center

                    # Eye contact threshold
                    if abs(eye_direction[0]) < 0.02:
                        eye_contact_frames += 1

                    # Draw facemesh
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
                    )

            out.write(frame)

            if progress_callback:
                progress_callback(frame_count, total_frames)

    cap.release()
    out.release()

    eye_contact_percentage = (eye_contact_frames / total_frames) * 100
    return eye_contact_percentage, output_path
