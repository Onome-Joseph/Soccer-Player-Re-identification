"""Detection process"""

import torch
import gdown
import numpy as np
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from deep_sort_realtime.deepsort_tracker import DeepSort

# Download from Google Drive using file ID
url = 'https://drive.google.com/uc?id=1XyzAbCDEFG123456789'
output = 'best.pt'
gdown.download(url, output, quiet=False)    # Load model from the downloaded file

# Initialize YOLO model
model = YOLO('best.pt')

video_path = '/15sec_input_720p.mp4'
target_video_path = '/15sec_input_720p_reid_9.mp4'

ball_id = 0  # Class ID for ball

# Custom color palette
custom_palette = sv.ColorPalette.from_hex([
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00",
    "#FF00FF", "#00FFFF", "#FFA500", "#800080", "#008000"
])

# Initialize annotators
ellipse_annotator = sv.EllipseAnnotator(color=custom_palette, color_lookup=sv.ColorLookup.TRACK)
label_annotator = sv.LabelAnnotator(
    color=custom_palette,
    color_lookup=sv.ColorLookup.TRACK,
    text_color=sv.Color.from_hex("000000"),
    text_position=sv.Position.BOTTOM_CENTER
)
triangle_annotator = sv.TriangleAnnotator(color=sv.Color.from_hex("#0009B6"), base=20, height=17)

class TorchReIDEmbedderWrapper:
    """Proper wrapper for TorchReID models to work with DeepSort"""
    def __init__(self, model_name='osnet_ain_x1_0', device='cuda'):
        self.extractor = FeatureExtractor(
            model_name=model_name,
            device=device,
            verbose=True
        )
        print(f"Loaded TorchReID model: {model_name}")
        self.device = device
        self.model_name = model_name

    def __call__(self, image, bboxes):
        crops = []
        valid_indices = []

        for i, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            if w <= 10 or h <= 10:  # Skip too small boxes
                crops.append(np.zeros((1,1,3), dtype=np.uint8)) # Placeholder for invalid box
                valid_indices.append(i)
                continue

            # Convert to integers and clip to image bounds
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(image.shape[1], x1 + int(w))
            y2 = min(image.shape[0], y1 + int(h))

            if x2 <= x1 or y2 <= y1:  # Skip invalid boxes
                crops.append(np.zeros((1,1,3), dtype=np.uint8)) # Placeholder for invalid box
                valid_indices.append(i)
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
                valid_indices.append(i)
            else:
                crops.append(np.zeros((1,1,3), dtype=np.uint8)) # Placeholder for invalid box
                valid_indices.append(i)


        if not crops:
            return [None] * len(bboxes)

        # Extract features in batch
        try:
            features = self.extractor(crops)
            features = features.cpu().numpy()

            # Create full feature list with None for invalid boxes
            full_features = [None] * len(bboxes)
            feature_idx = 0
            for i in range(len(bboxes)):
              if i in valid_indices:
                full_features[i] = features[feature_idx].tolist()
                feature_idx += 1
              else:
                full_features[i] = None

            return full_features
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return [None] * len(bboxes)

# TorchReID models
MODEL_NAME = 'osnet_ain_x1_0'  # Best balance of accuracy/speed


# Initialize tracker with TorchReID embedder
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedder = TorchReIDEmbedderWrapper(model_name=MODEL_NAME, device=device)

tracker = DeepSort(
    max_age=140,
    n_init=3,
    max_cosine_distance=0.4,  # Stricter matching threshold
    nn_budget=210,
    override_track_class=None,
    embedder_gpu=True if device == 'cuda' else False,
    half=True if device == 'cuda' else False,
    bgr=True
)

# Process video
video_info = sv.VideoInfo.from_video_path(video_path)
with sv.VideoSink(target_video_path, video_info=video_info) as video_sink:
    for frame in tqdm(sv.get_video_frames_generator(video_path), total=video_info.total_frames):
        # Run YOLO inference
        result = model.track(frame, conf=0.3, persist=True, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Process ball detections
        ball_mask = detections.class_id == ball_id
        ball_detections = detections[ball_mask]
        if len(ball_detections) > 0:
            padded_boxes = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            ball_detections.xyxy = padded_boxes

        # Process player detections
        player_mask = detections.class_id != ball_id
        player_detections = detections[player_mask]
        player_detections = player_detections.with_nms(threshold=0.5, class_agnostic=True)

        # Prepare DeepSORT inputs with embeddings
        detections_for_deepsort = []
        bboxes = []
        for xyxy, confidence in zip(player_detections.xyxy, player_detections.confidence):
            x1, y1, x2, y2 = xyxy
            w, h = x2 - x1, y2 - y1
            bboxes.append([x1, y1, w, h])

        # Get embeddings for the current frame
        embeddings = embedder(frame, bboxes)

        # Create detections list for DeepSORT, including embeddings
        detections_for_deepsort = []
        for i, (bbox, confidence) in enumerate(zip(bboxes, player_detections.confidence)):
            if embeddings[i] is not None: # Only add detections with valid embeddings
              detections_for_deepsort.append(([bbox[0], bbox[1], bbox[2], bbox[3]], float(confidence), embeddings[i]))

        # Update tracker with detections and embeddings
        tracks = tracker.update_tracks(detections_for_deepsort, frame=frame)

        # Convert tracks to Detections format
        track_xyxy = []
        track_ids = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1: # Filter out unconfirmed or lost tracks
                continue
            ltrb = track.to_ltrb()
            track_xyxy.append(ltrb)
            track_ids.append(int(track.track_id))

        # Create player detections with re-ID
        player_detections = sv.Detections(
            xyxy=np.array(track_xyxy) if track_xyxy else np.empty((0, 4)),
            tracker_id=np.array(track_ids, dtype=int) if track_ids else np.empty(0, dtype=int)
        )

        # Annotate frame
        annotated_frame = frame.copy()
        if len(player_detections) > 0:
            labels = [f"#{tid}" for tid in player_detections.tracker_id]
            annotated_frame = ellipse_annotator.annotate(annotated_frame, player_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, player_detections, labels=labels)

        if len(ball_detections) > 0:
            annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

        video_sink.write_frame(annotated_frame)

print("Processing complete!")
