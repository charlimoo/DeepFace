# app_persian_ltr.py

import streamlit as st
import os
import tempfile
import glob
import cv2
import shutil
import time
import json
import face_match_backend as backend

# --- ترجمه‌ها (Translations) ---
T = {
    "page_title": "داشبورد تطبیق چهره",
    "page_icon": "🤖",
    "dashboard_title": "داشبورد تطبیق چهره",
    "edit_person_info_title": "ویرایش اطلاعات شخص",
    "name_label": "نام",
    "nat_code_label": "کد ملی",
    "save_changes_button": "ذخیره تغییرات",
    "saved_info_toast": "اطلاعات برای {} ذخیره شد",
    "db_empty_info": "پایگاه داده خالی است. برای شروع، تصاویری را بارگذاری کنید.",
    "db_images_found": "{} تصویر منبع یافت شد.",
    "edit_button_icon": "✏️",
    "delete_button_icon": "❌",
    "edit_button_help": "ویرایش نام و کد ملی این شخص",
    "delete_button_help": "حذف این تصویر و اطلاعات آن از پایگاه داده",
    "deleted_toast": "{} حذف شد",
    "delete_error": "خطا در حذف فایل: {}",
    "active_db_info": "پایگاه داده فعال برای تطبیق:",
    "cropped_db_not_built": "پایگاه داده چهره‌های برش‌خورده هنوز ساخته نشده است.",
    "no_cropped_faces_info": "هیچ چهره برش‌خورده‌ای یافت نشد. برای پردازش تصاویر منبع، روی «ساخت/به‌روزرسانی» کلیک کنید.",
    "cropped_faces_found": "{} چهره برش‌خورده یافت شد.",
    "match_results_header": "✅ نتایج تطبیق",
    "strong_match_header": "شخص «#{}» یک تطابق قوی است!",
    "matched_identity_label": "هویت تطبیق داده شده:",
    "detected_face_caption": "چهره شناسایی‌شده #{}",
    "best_match_caption": "بهترین تطابق در پایگاه داده",
    "similarity_metric": "امتیاز شباهت",
    "distance_metric": "فاصله (کمتر بهتر است)",
    "threshold_delta": "آستانه: {}",
    "no_strong_match_info": "ℹ️ تطابق قوی برای شخص/اشخاص #{} یافت نشد.",
    "no_strong_matches_warning": "چهره‌ها شناسایی شدند، اما هیچ تطابق قوی در پایگاه داده یافت نشد.",
    "config_header": "⚙️ پیکربندی",
    "model_label": "مدل",
    "model_help": (
        "مدل یادگیری عمیق برای استخراج ویژگی‌های چهره را انتخاب کنید:\n\n"
        "- **ArcFace:** پیشرفته و بسیار دقیق، بهترین انتخاب برای اکثر موارد. عملکرد عالی در تشخیص چهره‌های سخت و زوایای مختلف.\n\n"
        "- **VGG-Face:** مدلی کلاسیک و شناخته‌شده. دقت خوبی دارد اما ممکن است در برابر تغییرات چهره (مانند عینک یا ماسک) به اندازه ArcFace قوی نباشد.\n\n"
        "- **Facenet:** مدل محبوب از گوگل. دقت بالایی دارد اما معمولاً کمی کندتر از ArcFace است.\n\n"
        "- **SFace:** مدلی سبک و بسیار سریع، مناسب برای کاربردهایی که سرعت اولویت دارد و دقت فوق‌العاده بالا نیاز نیست (مانند دستگاه‌های کم‌توان)."
    ),
    "detector_label": "شناساگر",
    "detector_help": (
        "الگوریتم تشخیص چهره در تصویر را انتخاب کنید:\n\n"
        "- **retinaface:** شناساگر مدرن و بسیار دقیق. بهترین گزینه برای یافتن چهره‌های کوچک، تار یا در زوایای دشوار. انتخاب پیشنهادی.\n\n"
        "- **mtcnn:** یک شناساگر قوی و محبوب که علاوه بر چهره، نقاط کلیدی (چشم، بینی، دهان) را نیز مشخص می‌کند. دقت بالایی دارد.\n\n"
        "- **yolov8:** نسخه جدید و سریع YOLO. برای تشخیص سریع چهره‌ها در ویدیوهای زنده عالی است، اما ممکن است به اندازه retinaface دقیق نباشد.\n\n"
        "- **opencv:** شناساگر داخلی کتابخانه OpenCV. سریع و ساده است اما دقت آن در مقایسه با مدل‌های دیگر کمتر است و ممکن است چهره‌های سخت را از دست بدهد."
    ),
    "metric_label": "معیار فاصله",
    "threshold_label": "آستانه تایید",
    "threshold_help": "مقدار کمتر = تطبیق سخت‌گیرانه‌تر.",
    "video_options_header": "🎬 تنظیمات ویدیو",
    "frame_skip_label": "پرش از فریم",
    "frame_skip_help": "پردازش ۱ فریم از هر تعداد فریم.",
    "model_status_header": "📦 وضعیت مدل",
    "model_ready_success": "✅ مدل «{}» آماده است.",
    "model_not_found_warning": "مدل «{}» یافت نشد.",
    "model_download_info": "مدل در اولین استفاده به طور خودکار دانلود می‌شود، یا می‌توانید اکنون آن را دانلود کنید.",
    "download_model_button": "دانلود مدل «{}»",
    "downloading_spinner": "در حال دانلود وزن‌های مدل {}... این فرآیند ممکن است چند دقیقه طول بکشد و برنامه را موقتا متوقف کند.",
    "download_success": "{} با موفقیت دانلود شد!",
    "download_failed": "دانلود ناموفق بود: {}",
    "upload_media_header": "📤 بارگذاری رسانه برای تحلیل",
    "file_uploader_label": "یک تصویر یا ویدیو بارگذاری کنید",
    "db_management_header": "👤 مدیریت پایگاه داده چهره‌ها",
    "expander_source_images": "۱. مدیریت تصاویر منبع",
    "db_uploader_label": "بارگذاری تصاویر جدید در پایگاه داده",
    "saved_new_images": "با موفقیت {} تصویر جدید ذخیره شد.",
    "expander_process_db": "۲. پردازش پایگاه داده",
    "build_db_button": "ساخت / به‌روزرسانی پایگاه داده",
    "build_db_help": "چهره‌ها را در تصاویر منبع شناسایی کرده و برای تطبیق سریع در یک پوشه جداگانه ذخیره می‌کند.",
    "processing_db_spinner": "در حال پردازش تصاویر با «{}»...",
    "db_build_success": "{} چهره برش‌خورده با موفقیت ایجاد شد.",
    "db_build_warning": "چهره‌ای در این تصاویر یافت نشد: {}",
    "delete_cropped_button": "⚠️ حذف چهره‌های برش‌خورده",
    "delete_cropped_help": "کل پایگاه داده چهره‌های برش‌خورده را حذف می‌کند. تصاویر منبع دست‌نخورده باقی می‌مانند.",
    "delete_cropped_success": "پایگاه داده چهره‌های برش‌خورده حذف شد.",
    "delete_cropped_info": "پایگاه داده چهره‌های برش‌خورده وجود ندارد.",
    "expander_view_active_db": "۳. مشاهده پایگاه داده فعال تطبیق",
    "analysis_results_header": "🔬 تحلیل و نتایج",
    "analysis_complete_success": "✅ تحلیل کامل شد. {} شخص منحصربه‌فرد یافت شد. نتایج را در زیر ببینید.",
    "analysis_complete_warning": "⚠️ تحلیل کامل شد، اما هیچ فرد منحصربه‌فردی شناسایی نشد.",
    "upload_prompt": "برای شروع تحلیل، رسانه‌ای را بارگذاری کنید.",
    "db_not_ready_warning": "پایگاه داده چهره‌های برش‌خورده خالی است. لطفاً قبل از اجرای تحلیل، آن را بسازید.",
    "analyze_button": "🚀 تحلیل و یافتن تطابق‌ها",
    "model_not_downloaded_error": "مدل «{}» دانلود نشده است. لطفاً ابتدا آن را از نوار کناری دانلود کنید.",
    "verbose_logs_expander": "نمایش گزارش‌های دقیق",
    "analyzing_spinner": "در حال تحلیل... لطفاً صبر کنید.",
    "progress_bar_init": "در حال آماده‌سازی...",
    "progress_bar_processing": "در حال پردازش...",
    "video_process_complete": "پردازش ویدیو کامل شد!",
    "unknown_person": "ناشناس",
    "not_applicable": "N/A",
}


# --- Constants for Database Paths ---
FACE_DATABASE_ROOT = os.path.join(os.getcwd(), "face_database")
PROCESSED_DB_PATH = os.path.join(FACE_DATABASE_ROOT, "_cropped_faces")
METADATA_FILE = os.path.join(FACE_DATABASE_ROOT, "metadata.json")
os.makedirs(FACE_DATABASE_ROOT, exist_ok=True)

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title=T["page_title"],
    page_icon=T["page_icon"],
    layout="wide"
)

# --- Custom Persian Font Styling (LTR Layout) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    html, body, [class*="st-"], .st-emotion-cache-10trblm, .st-emotion-cache-1kyxreq, .st-emotion-cache-1c7y2kd {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    * {
        font-family: 'Vazirmatn', sans-serif !important;
    }
    .stImage > img, .stVideo > video {
        max-height: 550px;
        object-fit: contain;
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
if 'edit_modal_for' not in st.session_state:
    st.session_state.edit_modal_for = None

# --- Metadata Helper Functions (with UTF-8 support) ---
def load_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_metadata(data):
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

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
            st.toast(T["saved_new_images"].format(saved_count))
            del st.session_state.db_uploader
            st.rerun()

@st.dialog(T["edit_person_info_title"])
def edit_info_dialog(img_filename, current_meta):
    st.image(os.path.join(FACE_DATABASE_ROOT, img_filename))
    with st.form("edit_form"):
        name = st.text_input(T["name_label"], value=current_meta.get("name", ""))
        nat_code = st.text_input(T["nat_code_label"], value=current_meta.get("national_code", ""))
        
        submitted = st.form_submit_button(T["save_changes_button"])
        if submitted:
            metadata = load_metadata()
            metadata[img_filename] = {"name": name, "national_code": nat_code}
            save_metadata(metadata)
            st.toast(T["saved_info_toast"].format(img_filename), icon="✅")
            st.session_state.edit_modal_for = None
            st.rerun()

def manage_source_database_ui(db_path):
    metadata = load_metadata()
    
    if st.session_state.edit_modal_for:
        img_to_edit = st.session_state.edit_modal_for
        current_meta = metadata.get(img_to_edit, {})
        edit_info_dialog(img_to_edit, current_meta)

    with st.container(height=350):
        image_files = glob.glob(os.path.join(db_path, '*.jpg')) + \
                      glob.glob(os.path.join(db_path, '*.jpeg')) + \
                      glob.glob(os.path.join(db_path, '*.png'))

        if not image_files:
            st.info(T["db_empty_info"])
            return

        st.markdown(f"**{T['db_images_found'].format(len(image_files))}**")
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(image_files)):
            with cols[i % 4]:
                img_filename = os.path.basename(img_path)
                st.image(img_path, caption=img_filename, use_container_width=True)
                
                button_cols = st.columns(2)
                with button_cols[0]: # Edit Button on the left
                    if st.button(T["edit_button_icon"], key=f"edit_{img_filename}", use_container_width=True, help=T["edit_button_help"]):
                        st.session_state.edit_modal_for = img_filename
                        st.rerun()

                with button_cols[1]: # Delete Button on the right
                    if st.button(T["delete_button_icon"], key=f"del_src_{img_path}", use_container_width=True, type="secondary", help=T["delete_button_help"]):
                        try:
                            os.remove(img_path)
                            if img_filename in metadata:
                                del metadata[img_filename]
                                save_metadata(metadata)
                            st.toast(T["deleted_toast"].format(img_filename))
                            st.rerun()
                        except Exception as e:
                            st.error(T["delete_error"].format(e))

def display_cropped_faces_ui(db_path):
    st.info(f"**{T['active_db_info']}** `{os.path.basename(os.path.normpath(db_path))}`")
    if not os.path.isdir(db_path):
        st.warning(T["cropped_db_not_built"])
        return

    with st.container(height=350):
        image_files = glob.glob(os.path.join(db_path, '*.jpg')) + \
                      glob.glob(os.path.join(db_path, '*.jpeg')) + \
                      glob.glob(os.path.join(db_path, '*.png'))

        if not image_files:
            st.info(T["no_cropped_faces_info"])
            return

        st.markdown(f"**{T['cropped_faces_found'].format(len(image_files))}**")
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(image_files)):
            with cols[i % 4]:
                st.image(img_path, caption=os.path.splitext(os.path.basename(img_path))[0], use_container_width=True)

def display_results_ui(results, verification_threshold):
    if not results: return
    metadata = load_metadata()
    
    st.markdown("---")
    st.subheader(T["match_results_header"])
    unmatched_faces, matched_faces_count = [], 0

    for result in results:
        is_strong_match = result.get('has_strong_match', False)
        if is_strong_match and result['matches'] is not None and not result['matches'].empty:
            matched_faces_count += 1
            match = result['matches'].iloc[0]
            
            matched_identity_path = match['identity']
            matched_cropped_filename = os.path.basename(matched_identity_path)
            source_file_stem = matched_cropped_filename.rsplit('_face_', 1)[0]
            
            source_filename = None
            for key in metadata.keys():
                if os.path.splitext(key)[0] == source_file_stem:
                    source_filename = key
                    break
            
            person_info = metadata.get(source_filename, {})
            person_name = person_info.get("name", T["unknown_person"])
            person_code = person_info.get("national_code", T["not_applicable"])
            
            st.markdown(f"##### {T['strong_match_header'].format(result['person_index'])}")
            st.markdown(f"**{T['matched_identity_label']}** `{person_name}`")
            
            res_cols = st.columns([1, 1, 2])
            with res_cols[0]:
                st.image(result['ref_crop'], channels="BGR", caption=T["detected_face_caption"].format(result['person_index']), use_container_width=True)
            with res_cols[1]:
                if os.path.exists(matched_identity_path):
                    st.image(matched_identity_path, caption=T["best_match_caption"], use_container_width=True)
            with res_cols[2]:
                st.text(f"{T['name_label']}: {person_name}")
                st.text(f"{T['nat_code_label']}: {person_code}")
                st.metric(label=T["similarity_metric"], value=f"{match['similarity']:.2f}%")
                st.metric(label=T["distance_metric"], value=f"{match['distance']:.4f}", delta=T["threshold_delta"].format(verification_threshold), delta_color="off")
        else:
            unmatched_faces.append(str(result['person_index']))

    if unmatched_faces:
        st.info(T["no_strong_match_info"].format(', '.join(unmatched_faces)))
    if matched_faces_count == 0 and not unmatched_faces:
        st.warning(T["no_strong_matches_warning"])

# --- Main UI Layout ---
st.title(f"{T['page_icon']} {T['dashboard_title']}")

# --- Sidebar ---
with st.sidebar:
    st.header(T["config_header"])
    MODEL_NAME = st.selectbox(T["model_label"], ("ArcFace", "VGG-Face", "Facenet", "SFace"), 0, help=T["model_help"])
    DETECTOR_BACKEND = st.selectbox(T["detector_label"], ('retinaface', 'mtcnn', 'yolov8', 'opencv'), 0, help=T["detector_help"])
    DISTANCE_METRIC = st.selectbox(T["metric_label"], ('cosine', 'euclidean', 'euclidean_l2'), 0)
    VERIFICATION_THRESHOLD = st.slider(T["threshold_label"], 0.0, 2.0, backend.get_threshold(MODEL_NAME, DISTANCE_METRIC), 0.01, help=T["threshold_help"])
    st.markdown("---")
    st.header(T["video_options_header"])
    FRAME_SKIP = st.number_input(T["frame_skip_label"], 1, 300, 15, help=T["frame_skip_help"])
    st.markdown("---")
    
    st.header(T["model_status_header"])
    model_is_ready = backend.check_model_exists(MODEL_NAME)
    if model_is_ready:
        st.success(T["model_ready_success"].format(MODEL_NAME))
    else:
        st.warning(T["model_not_found_warning"].format(MODEL_NAME))
        st.info(T["model_download_info"])
        if st.button(T["download_model_button"].format(MODEL_NAME)):
            with st.spinner(T["downloading_spinner"].format(MODEL_NAME)):
                success, error_msg = backend.download_model(MODEL_NAME)
                if success:
                    st.success(T["download_success"].format(MODEL_NAME))
                else:
                    st.error(T["download_failed"].format(error_msg))
            time.sleep(2)
            st.rerun()

# --- Main Page Layout ---
col_center, col_right = st.columns([2, 3])

with col_center:
    st.header(T["upload_media_header"])
    source_file = st.file_uploader(T["file_uploader_label"], type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'], label_visibility="collapsed")
    if source_file and st.session_state.source_file_info != source_file.name:
        st.session_state.source_file_info = source_file.name
        st.session_state.processed_media = None
        st.session_state.results = None
        st.session_state.analysis_complete = False

    st.markdown("---")
    st.header(T["db_management_header"])

    with st.expander(T["expander_source_images"], expanded=True):
        st.file_uploader(
            T["db_uploader_label"],
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="db_uploader",
            on_change=handle_db_upload
        )
        manage_source_database_ui(FACE_DATABASE_ROOT)

    with st.expander(T["expander_process_db"], expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            if st.button(T["build_db_button"], type="primary", use_container_width=True, help=T["build_db_help"]):
                with st.spinner(T["processing_db_spinner"].format(DETECTOR_BACKEND)):
                    _, count, failures = backend.crop_and_prepare_db(FACE_DATABASE_ROOT, DETECTOR_BACKEND)
                    st.success(T["db_build_success"].format(count))
                    if failures: st.warning(T["db_build_warning"].format(', '.join(failures)))
                    st.rerun()
        with col2:
            if st.button(T["delete_cropped_button"], use_container_width=True, help=T["delete_cropped_help"]):
                if os.path.exists(PROCESSED_DB_PATH):
                    shutil.rmtree(PROCESSED_DB_PATH)
                    st.success(T["delete_cropped_success"])
                    st.rerun()
                else:
                    st.info(T["delete_cropped_info"])

    with st.expander(T["expander_view_active_db"], expanded=True):
        display_cropped_faces_ui(PROCESSED_DB_PATH)

with col_right:
    st.header(T["analysis_results_header"])

    media_placeholder = st.empty()
    if st.session_state.processed_media is not None:
        media_placeholder.image(st.session_state.processed_media, channels="BGR")
    elif st.session_state.analysis_complete:
        if st.session_state.results:
            media_placeholder.success(T["analysis_complete_success"].format(len(st.session_state.results)))
        else:
            media_placeholder.warning(T["analysis_complete_warning"])
    elif source_file:
        file_ext = os.path.splitext(source_file.name)[1].lower()
        if file_ext in ['.jpg', '.jpeg', '.png']:
            media_placeholder.image(source_file)
        else:
            media_placeholder.video(source_file)
    else:
        media_placeholder.info(T["upload_prompt"])

    is_db_ready = os.path.exists(PROCESSED_DB_PATH) and any(f.endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(PROCESSED_DB_PATH))
    analyze_button = st.button(
        T["analyze_button"],
        type="primary",
        use_container_width=True,
        disabled=(source_file is None or not is_db_ready)
    )
    if not is_db_ready and source_file:
        st.warning(T["db_not_ready_warning"])

    log_placeholder = st.empty()
    results_container = st.container()

    if analyze_button:
        if not backend.check_model_exists(MODEL_NAME):
            log_placeholder.error(T["model_not_downloaded_error"].format(MODEL_NAME))
        else:
            debug_placeholder = st.expander(T["verbose_logs_expander"], expanded=False)
            st.session_state.analysis_complete = False
            st.session_state.results = []
            st.session_state.processed_media = None
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(source_file.name)[1]) as tmp_file:
                tmp_file.write(source_file.getvalue())
                tmp_file_path = tmp_file.name

            file_ext = os.path.splitext(source_file.name)[1].lower()
            with st.spinner(T["analyzing_spinner"]):
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
                                # The text on the image remains English for now as cv2 doesn't handle Persian well
                                cv2.putText(processed_img, f"MATCH: #{res['person_index']}", (fa['x'], fa['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        st.session_state.processed_media = processed_img
                        st.session_state.results = results
                else: # Video
                    media_placeholder.empty()
                    progress_bar = st.progress(0, T["progress_bar_init"])
                    processor = backend.process_video(tmp_file_path, PROCESSED_DB_PATH, MODEL_NAME, DETECTOR_BACKEND, DISTANCE_METRIC, VERIFICATION_THRESHOLD, FRAME_SKIP)
                    for update_type, data in processor:
                        if update_type == 'progress': progress_bar.progress(data['value'], text=data['text'])
                        elif update_type == 'frame_update': media_placeholder.image(data, channels="BGR", caption=T["progress_bar_processing"])
                        elif update_type == 'result': st.session_state.results.append(data)
                        elif update_type == 'error': log_placeholder.error(data)
                        elif update_type == 'debug':
                            with debug_placeholder:
                                st.info(data)
                    progress_bar.empty()
                    log_placeholder.success(T["video_process_complete"])

            st.session_state.analysis_complete = True
            os.remove(tmp_file_path)
            st.rerun()

    if st.session_state.results is not None:
        with results_container:
            display_results_ui(st.session_state.results, VERIFICATION_THRESHOLD)