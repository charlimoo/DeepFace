# app.py

import streamlit as st
import os
import tempfile
import glob
import cv2
import shutil
import time
import face_match_backend as backend  # Import the backend logic

# --- Constants for Database Paths ---
FACE_DATABASE_ROOT = os.path.join(os.getcwd(), "face_database")
PROCESSED_DB_PATH = os.path.join(FACE_DATABASE_ROOT, "_cropped_faces")
os.makedirs(FACE_DATABASE_ROOT, exist_ok=True) # Ensure the source directory exists

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Face Match Dashboard",
    page_icon="🤖",
    layout="wide"
)

# --- CSS to control the max height of the media display ---
st.markdown("""
<style>
    /* Target the image and video elements specifically */
    .stImage > img, .stVideo > video {
        max-height: 550px;
        object-fit: contain; /* Keeps aspect ratio */
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)


# --- State Management ---
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processed_media' not in st.session_state:
    st.session_state.processed_media = None
if 'source_file_info' not in st.session_state:
    st.session_state.source_file_info = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# --- UI Helper Functions ---
def handle_db_upload():
    if 'db_uploader' in st.session_state and st.session_state.db_uploader is not None:
        uploaded_files = st.session_state.db_uploader
        saved_count = 0
        for uploaded_file in uploaded_files:
            file_path = os.path.join(FACE_DATABASE_ROOT, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_count += 1
        if saved_count > 0:
            st.toast(f"Successfully saved {saved_count} new image(s).")

def manage_source_database_ui(db_path):
    with st.container(height=350):
        image_files = glob.glob(os.path.join(db_path, '*.jpg')) + \
                      glob.glob(os.path.join(db_path, '*.jpeg')) + \
                      glob.glob(os.path.join(db_path, '*.png'))

        if not image_files:
            st.info("Database is empty. Upload images to begin.")
            return

        st.markdown(f"**{len(image_files)} source image(s) found.**")
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(image_files)):
            with cols[i % 4]:
                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                if st.button("Delete", key=f"del_src_{img_path}", use_container_width=True):
                    try:
                        os.remove(img_path)
                        st.toast(f"Deleted {os.path.basename(img_path)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting file: {e}")

def display_cropped_faces_ui(db_path):
    st.info(f"**Active Database for Matching:** `{os.path.basename(os.path.normpath(db_path))}`")
    if not os.path.isdir(db_path):
        st.warning("Cropped database has not been built yet.")
        return

    with st.container(height=350):
        image_files = glob.glob(os.path.join(db_path, '*.jpg')) + \
                      glob.glob(os.path.join(db_path, '*.jpeg')) + \
                      glob.glob(os.path.join(db_path, '*.png'))

        if not image_files:
            st.info("No cropped faces found. Click 'Build/Update' to process source images.")
            return

        st.markdown(f"**{len(image_files)} cropped face(s) found.**")
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(image_files)):
            with cols[i % 4]:
                st.image(img_path, caption=os.path.splitext(os.path.basename(img_path))[0], use_container_width=True)

def display_results_ui(results, verification_threshold):
    if not results: return
    st.markdown("---")
    st.subheader("✅ Match Results")
    unmatched_faces, matched_faces_count = [], 0

    for result in results:
        is_strong_match = result.get('has_strong_match', False)
        if is_strong_match and result['matches'] is not None and not result['matches'].empty:
            matched_faces_count += 1
            match = result['matches'].iloc[0]
            st.markdown(f"##### Person `#{result['person_index']}` is a Strong Match!")
            res_cols = st.columns([1, 1, 2])
            with res_cols[0]:
                st.image(result['ref_crop'], channels="BGR", caption=f"Detected Face #{result['person_index']}", use_container_width=True)
            with res_cols[1]:
                if os.path.exists(match['identity']):
                    st.image(match['identity'], caption=f"Best Match: {os.path.splitext(os.path.basename(match['identity']))[0]}", use_container_width=True)
            with res_cols[2]:
                st.metric(label="Similarity Score", value=f"{match['similarity']:.2f}%")
                st.metric(label="Distance (Lower is Better)", value=f"{match['distance']:.4f}", delta=f"Threshold: {verification_threshold}", delta_color="off")
        else:
            unmatched_faces.append(str(result['person_index']))

    if unmatched_faces:
        st.info(f"ℹ️ No strong matches found for Person #{', '.join(unmatched_faces)}.")
    if matched_faces_count == 0 and not unmatched_faces:
        st.warning("Faces were detected, but no strong matches were found in the database.")


# --- Main UI Layout ---
st.title("🤖 Face Match Dashboard")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    MODEL_NAME = st.selectbox("Model", ("ArcFace", "VGG-Face", "Facenet", "SFace"), 0, help="ArcFace is recommended.")
    DETECTOR_BACKEND = st.selectbox("Detector", ('retinaface', 'mtcnn', 'yolov8', 'opencv'), 0, help="RetinaFace is recommended.")
    DISTANCE_METRIC = st.selectbox("Metric", ('cosine', 'euclidean', 'euclidean_l2'), 0)
    VERIFICATION_THRESHOLD = st.slider("Verification Threshold", 0.0, 2.0, backend.get_threshold(MODEL_NAME, DISTANCE_METRIC), 0.01, help="Lower = stricter matching.")
    st.markdown("---")
    st.header("🎬 Video Options")
    FRAME_SKIP = st.number_input("Frame Skip", 1, 300, 15, help="Process 1 frame every N frames.")
    st.markdown("---")
    
    # --- NEW: Model Status and Download Section ---
    st.header("📦 Model Status")
    model_is_ready = backend.check_model_exists(MODEL_NAME)
    if model_is_ready:
        st.success(f"✅ Model '{MODEL_NAME}' is ready.")
    else:
        st.warning(f"Model '{MODEL_NAME}' not found.")
        st.info("The model will be downloaded automatically on first use, or you can download it now.")
        if st.button(f"Download '{MODEL_NAME}' Now"):
            with st.spinner(f"Downloading {MODEL_NAME} weights... This may take a few minutes and will freeze the app."):
                success, error_msg = backend.download_model(MODEL_NAME)
                if success:
                    st.success(f"{MODEL_NAME} downloaded successfully!")
                else:
                    st.error(f"Download failed: {error_msg}")
            # Brief pause to allow user to see the message before rerun
            time.sleep(2)
            st.rerun()
    # --- END OF NEW SECTION ---


# --- Main Page Layout ---
col_center, col_right = st.columns([2, 3])

with col_center:
    st.header("📤 Upload Media for Analysis")
    source_file = st.file_uploader("Upload an image or video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'], label_visibility="collapsed")
    if source_file and st.session_state.source_file_info != source_file.name:
        st.session_state.source_file_info = source_file.name
        st.session_state.processed_media = None
        st.session_state.results = None
        st.session_state.analysis_complete = False

    st.markdown("---")
    st.header("👤 Face Database Management")

    with st.expander("1. Manage Source Images", expanded=True):
        st.file_uploader(
            "Upload new images to the database",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="db_uploader",
            on_change=handle_db_upload
        )
        manage_source_database_ui(FACE_DATABASE_ROOT)

    with st.expander("2. Process Database", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Build / Update Database", type="primary", use_container_width=True, help="Detects faces in source images and saves them to a separate folder for fast matching."):
                with st.spinner(f"Processing images with '{DETECTOR_BACKEND}'..."):
                    _, count, failures = backend.crop_and_prepare_db(FACE_DATABASE_ROOT, DETECTOR_BACKEND)
                    st.success(f"Successfully created {count} cropped faces.")
                    if failures: st.warning(f"Could not find faces in: {', '.join(failures)}")
                    st.rerun()
        with col2:
            if st.button("⚠️ Delete Cropped Faces", use_container_width=True, help="Deletes the entire cropped face database. Does not affect source images."):
                if os.path.exists(PROCESSED_DB_PATH):
                    shutil.rmtree(PROCESSED_DB_PATH)
                    st.success("Cropped face database deleted.")
                    st.rerun()
                else:
                    st.info("Cropped database does not exist.")

    with st.expander("3. View Active Matching Database", expanded=True):
        display_cropped_faces_ui(PROCESSED_DB_PATH)


with col_right:
    st.header("🔬 Analysis & Results")

    media_placeholder = st.empty()
    if st.session_state.processed_media is not None:
        media_placeholder.image(st.session_state.processed_media, channels="BGR")
    elif st.session_state.analysis_complete:
        if st.session_state.results:
            media_placeholder.success(f"✅ Analysis complete. Found {len(st.session_state.results)} unique person(s). See results below.")
        else:
            media_placeholder.warning("⚠️ Analysis complete, but no unique individuals were identified.")
    elif source_file:
        file_ext = os.path.splitext(source_file.name)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            media_placeholder.image(source_file)
        else:
            media_placeholder.video(source_file)
    else:
        media_placeholder.info("Upload media to begin analysis.")

    is_db_ready = os.path.exists(PROCESSED_DB_PATH) and any(f.endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(PROCESSED_DB_PATH))
    analyze_button = st.button(
        "🚀 Analyze and Find Matches",
        type="primary",
        use_container_width=True,
        disabled=(source_file is None or not is_db_ready)
    )
    if not is_db_ready and source_file:
        st.warning("The cropped face database is empty. Please build it first before running an analysis.")

    log_placeholder = st.empty()
    results_container = st.container()

    if analyze_button:
        # First, ensure the model is ready before proceeding
        if not backend.check_model_exists(MODEL_NAME):
            log_placeholder.error(f"Model '{MODEL_NAME}' is not downloaded. Please download it from the sidebar first.")
        else:
            debug_placeholder = st.expander("Show Verbose Debug Logs", expanded=False)
            st.session_state.analysis_complete = False
            st.session_state.results = []
            st.session_state.processed_media = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_file.name)[1]) as tmp_file:
                tmp_file.write(source_file.getvalue())
                tmp_file_path = tmp_file.name

            file_ext = os.path.splitext(source_file.name)[1].lower()
            with st.spinner('Analyzing... Please wait.'):
                if file_ext in ['.jpg', '.jpeg', '.png']:
                    processed_img, results, error = backend.process_image(
                        tmp_file_path, PROCESSED_DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_METRIC, VERIFICATION_THRESHOLD
                    )
                    if error:
                        log_placeholder.error(error)
                    else:
                        for res in results:
                            if res.get('has_strong_match'):
                                fa = res['facial_area']
                                cv2.rectangle(processed_img, (fa['x'], fa['y']), (fa['x'] + fa['w'], fa['y'] + fa['h']), (0, 255, 0), 3)
                                cv2.putText(processed_img, f"MATCH: #{res['person_index']}", (fa['x'], fa['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        st.session_state.processed_media = processed_img
                        st.session_state.results = results
                else: # Video
                    media_placeholder.empty()
                    progress_bar = st.progress(0, "Initializing...")
                    processor = backend.process_video(tmp_file_path, PROCESSED_DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_METRIC, VERIFICATION_THRESHOLD, FRAME_SKIP)
                    for update_type, data in processor:
                        if update_type == 'progress': progress_bar.progress(data['value'], text=data['text'])
                        elif update_type == 'frame_update': media_placeholder.image(data, channels="BGR", caption="Processing...")
                        elif update_type == 'result': st.session_state.results.append(data)
                        elif update_type == 'error': log_placeholder.error(data)
                        elif update_type == 'debug':
                            with debug_placeholder:
                                st.info(data)
                    progress_bar.empty()
                    log_placeholder.success("Video processing complete!")

            st.session_state.analysis_complete = True
            os.remove(tmp_file_path)
            st.rerun()

    if st.session_state.results is not None:
        with results_container:
            display_results_ui(st.session_state.results, VERIFICATION_THRESHOLD)