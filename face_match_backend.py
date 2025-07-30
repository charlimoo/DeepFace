# face_match_backend.py

import os
import cv2
import glob
import numpy as np
import pandas as pd
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import multiprocessing


def get_model_path(model_name):
    """Gets the expected path for a model's weight file."""
    model_weights = {
        "VGG-Face": "vgg_face_weights.h5",
        "Facenet": "facenet_weights.h5",
        "Facenet512": "facenet512_weights.h5",
        "ArcFace": "arcface_weights.h5",
        "Dlib": "dlib_face_recognition_resnet_model_v1.dat",
        "SFace": "sface_weights.h5",
        "OpenFace": "openface_weights.h5",
    }
    filename = model_weights.get(model_name)
    if not filename:
        return None
    
    # Path is ~/.deepface/weights/filename
    home = os.path.expanduser("~")
    return os.path.join(home, ".deepface", "weights", filename)

def check_model_exists(model_name):
    """Checks if a model's weights file has been downloaded."""
    model_path = get_model_path(model_name)
    if model_path and os.path.exists(model_path):
        return True
    return False

def download_model(model_name):
    """
    Triggers the download of a specific model by calling build_model.
    This is a blocking operation.
    """
    try:
        # DeepFace.build_model is the function that handles the download
        DeepFace.build_model(model_name)
        return True, None
    except Exception as e:
        return False, str(e)
    
    
def represent_in_process(frame, model_name, detector_backend):
    """
    A wrapper to run DeepFace.represent in a separate process
    to prevent memory leaks.
    """
    try:
        # This function runs in its own memory space
        embedding_objs = DeepFace.represent(
            img_path=frame,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True
        )
        return embedding_objs
    except Exception as e:
        # Return the error to the main process if something goes wrong
        return e
    
# --- Helper Functions (No Streamlit here) ---

def get_threshold(model_name, distance_metric):
    """Returns the official verification threshold for a given model and metric."""
    thresholds = {
        "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
        "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
        "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
        "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
    }
    return thresholds.get(model_name, {}).get(distance_metric, 0.40)

def convert_distance_to_similarity(distance, threshold):
    """Converts a distance score to a similarity percentage relative to the threshold."""
    if distance > threshold * 2: return 0.0
    return 100 * max(0, 1 - (distance / (threshold * 2)))

def crop_and_prepare_db(source_db_path, detector_backend):
    """
    Finds faces in all images, crops them, resizes to a width of 400px
    while maintaining aspect ratio, and saves them to '_cropped_faces'.
    """
    cropped_db_path = os.path.join(source_db_path, "_cropped_faces")
    os.makedirs(cropped_db_path, exist_ok=True)

    for f in glob.glob(os.path.join(cropped_db_path, "*")):
        os.remove(f)

    image_files = glob.glob(os.path.join(source_db_path, '*.jpg')) + \
                  glob.glob(os.path.join(source_db_path, '*.jpeg')) + \
                  glob.glob(os.path.join(source_db_path, '*.png'))

    faces_count, failed_files = 0, []

    for img_path in image_files:
        try:
            face_objs = DeepFace.extract_faces(
                img_path=img_path,
                detector_backend=detector_backend,
                align=True,
                enforce_detection=True
            )
            for i, face_obj in enumerate(face_objs):
                face_crop_raw = face_obj['face']
                face_crop_bgr = cv2.cvtColor((face_crop_raw * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                # --- CHANGE HERE: Resize while maintaining aspect ratio ---
                h, w, _ = face_crop_bgr.shape
                target_width = 400
                scale = target_width / w
                target_height = int(h * scale)
                resized_face = cv2.resize(face_crop_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)
                # --- END CHANGE ---

                original_filename = os.path.splitext(os.path.basename(img_path))[0]
                new_filename = f"{original_filename}_face_{i+1}.jpg"
                save_path = os.path.join(cropped_db_path, new_filename)
                cv2.imwrite(save_path, resized_face)
                faces_count += 1
        except Exception:
            failed_files.append(os.path.basename(img_path))

    for pkl_file in glob.glob(os.path.join(cropped_db_path, "*.pkl")):
        os.remove(pkl_file)

    return cropped_db_path, faces_count, failed_files


# --- Core Image Processing Backend ---

def process_image(img_path, db_path, model_name, detector_backend, distance_metric, verification_threshold):
    """
    Processes an image to find faces and matches.
    Returns:
        - (np.array) The image with RED bounding boxes drawn on it. Green boxes are added by the frontend.
        - (list) A list of structured dictionaries containing results for each face.
        - (str or None) An error message if something went wrong.
    """
    try:
        # Step 1: Extract faces to get their exact coordinates for drawing boxes later.
        face_objects = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            align=True
        )
        if not face_objects:
            return None, None, "No faces were detected in the uploaded image."

        # Step 2: Find matches for all detected faces in the image at once.
        matches_df_list = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            distance_metric=distance_metric,
            detector_backend=detector_backend,
            silent=True,
            align=True
        )

        original_img = cv2.imread(img_path)
        img_with_boxes = original_img.copy()
        results_list = []

        # Zip the face objects (with coordinates) and their corresponding match results.
        for i, (face_obj, df) in enumerate(zip(face_objects, matches_df_list)):
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img_with_boxes, f"#{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            ref_face_img = (face_obj['face'] * 255).astype(np.uint8)
            ref_face_img = cv2.cvtColor(ref_face_img, cv2.COLOR_RGB2BGR)

            has_strong_match = False
            if not df.empty:
                df.rename(columns={f'{model_name}_{distance_metric}': 'distance'}, inplace=True, errors='ignore')
                df['similarity'] = df['distance'].apply(lambda d: convert_distance_to_similarity(d, verification_threshold))
                if not df.empty and df.iloc[0]['distance'] <= verification_threshold:
                    has_strong_match = True

            results_list.append({
                'person_index': i + 1,
                'matches': df,
                'ref_crop': ref_face_img,
                'facial_area': facial_area,
                'has_strong_match': has_strong_match
            })

        return img_with_boxes, results_list, None

    except Exception as e:
        error_msg = f"An error occurred during image processing: {str(e)}"
        if "Face could not be detected" in str(e):
            error_msg = "No faces were detected in the uploaded image."
        return None, None, error_msg

# --- Core Video Processing Backend (as a Generator) ---

# In face_match_backend.py

def process_video(video_path, db_path, model_name, detector_backend, distance_metric, verification_threshold, frame_skip):
    """
    Processes a video to find unique individuals and their matches using a
    multiprocessing pool and improved detection logic.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        yield ('error', "Cannot read video file or video has no frames.")
        return

    all_detections, frame_count = [], 0
    ctx = multiprocessing.get_context('spawn')
    pool = ctx.Pool(processes=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        progress_text = f"Analyzing frame {frame_count}/{total_frames}..."
        yield ('progress', {'value': frame_count / total_frames, 'text': progress_text})

        try:
            yield ('debug', f"Frame {frame_count}: Submitting to processing pool...")
            async_result = pool.apply_async(represent_in_process, (frame, model_name, detector_backend))
            embedding_objs = async_result.get()

            if isinstance(embedding_objs, Exception):
                raise embedding_objs

            yield ('debug', f"Frame {frame_count}: Received {len(embedding_objs)} result(s) from pool.")

            if not embedding_objs:
                yield ('debug', f"Frame {frame_count}: No faces found by .represent().")
                yield ('frame_update', frame)
                continue

            frame_with_boxes = frame.copy()
            
            # --- START OF MODIFIED LOGIC ---
            for obj in embedding_objs:
                # Log the confidence of EVERY detected face for better debugging.
                confidence = obj.get('confidence', 0)
                yield ('debug', f"  - Face detected with confidence: {confidence:.4f}")

                # Remove the strict 0.90 threshold. Let's process all faces
                # and let the clustering algorithm sort them out.
                all_detections.append({
                    'embedding': obj['embedding'],
                    'facial_area': obj['facial_area'],
                    'frame': frame.copy()
                })

                # Draw a box for visualization regardless of confidence.
                fa = obj['facial_area']
                cv2.rectangle(frame_with_boxes, (fa['x'], fa['y']), (fa['x'] + fa['w'], fa['y'] + fa['h']), (0, 0, 255), 2)
            # --- END OF MODIFIED LOGIC ---
            
            yield ('frame_update', frame_with_boxes)

        except Exception as e:
            yield ('debug', f"Frame {frame_count}: ERROR during face representation - {str(e)}")
            yield ('frame_update', frame)
            pass

    pool.close()
    pool.join()
    cap.release()

    yield ('debug', f"Video scan complete. Total face instances found (all confidences): {len(all_detections)}")

    if not all_detections:
        # This message will now only appear if DeepFace truly finds zero faces in the whole video.
        yield ('error', "No faces were detected in any frame of the video.")
        return

    yield ('progress', {'value': 1.0, 'text': f"Detected {len(all_detections)} face instances. Clustering..."})
    all_embeddings = np.array([d['embedding'] for d in all_detections])

    yield ('debug', f"Clustering {len(all_embeddings)} embeddings with eps={verification_threshold}...")
    clusters = DBSCAN(metric=distance_metric, eps=verification_threshold, min_samples=2, n_jobs=-1).fit_predict(all_embeddings)
    unique_cluster_ids = set(clusters) - {-1}
    yield ('debug', f"Clustering complete. Found {len(unique_cluster_ids)} unique clusters (people). Labels: {clusters}")

    if not unique_cluster_ids:
        yield ('error', "Could not identify any unique individuals (clusters). All faces were considered unique. Try increasing the Verification Threshold or reducing Frame Skip.")
        return

    yield ('progress', {'value': 1.0, 'text': f"Found {len(unique_cluster_ids)} unique individual(s). Finding matches..."})

    for cluster_id in unique_cluster_ids:
        indices = np.where(clusters == cluster_id)[0]
        rep_detection = all_detections[indices[0]]
        fa = rep_detection['facial_area']
        rep_crop = rep_detection['frame'][fa['y']:fa['y'] + fa['h'], fa['x']:fa['x'] + fa['w']]

        try:
            yield ('debug', f"Finding matches for cluster #{cluster_id + 1}...")
            matches_df_list = DeepFace.find(
                img_path=rep_crop,
                db_path=db_path,
                model_name=model_name,
                distance_metric=distance_metric,
                enforce_detection=True,
                align=True,
                silent=True
            )
            yield ('debug', f"Match search for cluster #{cluster_id + 1} complete.")
            matches_df = matches_df_list[0] if matches_df_list and not matches_df_list[0].empty else pd.DataFrame()

            has_strong_match = False
            if not matches_df.empty:
                matches_df.rename(columns={f'{model_name}_{distance_metric}': 'distance'}, inplace=True, errors='ignore')
                matches_df['similarity'] = matches_df['distance'].apply(lambda d: convert_distance_to_similarity(d, verification_threshold))
                if matches_df.iloc[0]['distance'] <= verification_threshold:
                    has_strong_match = True

            yield ('result', {
                'person_index': f"{cluster_id + 1}",
                'matches': matches_df,
                'ref_crop': rep_crop,
                'has_strong_match': has_strong_match
            })
        except Exception as e:
            yield ('error', f"CRITICAL ERROR on find() for person #{cluster_id + 1}: {e}")