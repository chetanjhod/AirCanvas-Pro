import os
import time
import threading
import io
import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from streamlit_autorefresh import st_autorefresh
from login import require_login


st.set_page_config(
    page_title="AirCanvas Pro | AI Drawing Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 0%, rgba(91, 33, 182, 0.22), transparent 35%),
            radial-gradient(circle at 100% 10%, rgba(14, 165, 233, 0.12), transparent 30%),
            #080b14;
        color: #f8fafc;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.97);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .hero {
        padding: 25px 28px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.17), rgba(15, 23, 42, 0.75));
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: clamp(27px, 4vw, 42px);
        font-weight: 800;
        letter-spacing: -1.4px;
        margin-bottom: 8px;
        color: #ffffff;
    }

    .hero-subtitle {
        color: #b7c3d6;
        font-size: 14px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 19px;
        font-weight: 700;
        margin: 12px 0;
        color: #f8fafc;
    }

    .info-card {
        background: rgba(17, 24, 39, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .info-title {
        color: #c4b5fd;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 7px;
    }

    .info-text {
        color: #cbd5e1;
        font-size: 13px;
        line-height: 1.7;
    }

    .badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.17);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.35);
        padding: 5px 11px;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 700;
        margin-bottom: 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        width: 100%;
        border-radius: 11px;
        font-weight: 600;
        min-height: 42px;
        border: 1px solid rgba(148, 163, 184, 0.18);
        transition: 0.2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #a78bfa;
        color: #ddd6fe;
    }

    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 12px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

require_login()

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "hand_landmarker.task",
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


class AirCanvasEngine:

    def __init__(self):
        self.lock = threading.RLock()
        self.canvas = np.zeros(
            (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
            dtype=np.uint8,
        )
        self.previous_point = None
        self.smoothed_point = None
        self.smoothing_factor = 0.65
        self.color = (0, 255, 120)
        self.brush_size = 7
        self.eraser = False
        self.drawing_enabled = True
        self.status = "start the camera"
        self.model_error = None
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 15
        self.landmarker = None
        self.last_timestamp = 0
        self.frames_processed = 0
        self._initialize_model()

    def _initialize_model(self):
        if not os.path.exists(MODEL_PATH):
            self.model_error = "model file not found: " + MODEL_PATH
            self.status = "Model file missing"
            return

        try:
            options = HandLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=MODEL_PATH
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                options
            )
            self.model_error = None
            self.status = "Camera start karo"

        except Exception as e:
            self.model_error = str(e)
            self.status = "model initialization failed"

    def save_undo(self):
        self.undo_stack.append(self.canvas.copy())

        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)

        self.redo_stack.clear()

    def undo(self):
        with self.lock:
            if self.undo_stack:
                self.redo_stack.append(self.canvas.copy())
                self.canvas = self.undo_stack.pop()
                self.previous_point = None
                self.smoothed_point = None

    def redo(self):
        with self.lock:
            if self.redo_stack:
                self.undo_stack.append(self.canvas.copy())
                self.canvas = self.redo_stack.pop()
                self.previous_point = None
                self.smoothed_point = None

    def clear(self):
        with self.lock:
            self.save_undo()
            self.canvas = np.zeros(
                (CANVAS_HEIGHT, CANVAS_WIDTH, 3),
                dtype=np.uint8,
            )
            self.previous_point = None
            self.smoothed_point = None
            self.status = "Canvas cleared"

    def set_color(self, rgb_color):
        with self.lock:
            rgb = tuple(
                int(rgb_color[i:i + 2], 16)
                for i in (1, 3, 5)
            )
            self.color = (rgb[2], rgb[1], rgb[0])

    def set_brush_size(self, size):
        with self.lock:
            self.brush_size = int(size)

    def set_eraser(self, enabled):
        with self.lock:
            if self.eraser != bool(enabled):
                self.previous_point = None
            self.eraser = bool(enabled)

    def set_drawing_enabled(self, enabled):
        with self.lock:
            if self.drawing_enabled != bool(enabled):
                self.previous_point = None
                self.smoothed_point = None
            self.drawing_enabled = bool(enabled)

    @staticmethod
    def finger_is_up(tip, pip):
        return tip.y < pip.y

    def draw_on_canvas(self, point):
        x, y = point

        if self.eraser:
            draw_color = (0, 0, 0)
            thickness = max(18, self.brush_size * 3)
        else:
            draw_color = self.color
            thickness = self.brush_size

        if self.previous_point is None:
            self.save_undo()
            cv2.circle(
                self.canvas,
                (x, y),
                max(1, thickness // 2),
                draw_color,
                -1,
                cv2.LINE_AA,
            )
        else:
            cv2.line(
                self.canvas,
                self.previous_point,
                (x, y),
                draw_color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.circle(
                self.canvas,
                (x, y),
                max(1, thickness // 2),
                draw_color,
                -1,
                cv2.LINE_AA,
            )

        self.previous_point = (x, y)

    def overlay_canvas(self, image):
        height, width = image.shape[:2]

        artwork = cv2.resize(
            self.canvas,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = np.any(artwork > 0, axis=2)

        if np.any(mask):
            image[mask] = artwork[mask]

        return image

    def process_frame(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = cv2.flip(image, 1)

        height, width = image.shape[:2]
        camera_point = None

        try:
            if self.landmarker is None:
                cv2.putText(
                    image,
                    "MediaPipe model unavailable",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                return av.VideoFrame.from_ndarray(
                    image,
                    format="bgr24",
                )

            rgb_image = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb_image),
            )

            timestamp = max(
                int(time.monotonic() * 1000),
                self.last_timestamp + 1,
            )
            self.last_timestamp = timestamp

            results = self.landmarker.detect_for_video(
                mp_image,
                timestamp,
            )

            with self.lock:
                if results.hand_landmarks:
                    landmarks = results.hand_landmarks[0]

                    index_tip = landmarks[8]
                    index_pip = landmarks[6]
                    middle_tip = landmarks[12]
                    middle_pip = landmarks[10]

                    index_up = self.finger_is_up(
                        index_tip,
                        index_pip,
                    )
                    middle_up = self.finger_is_up(
                        middle_tip,
                        middle_pip,
                    )

                    raw_x = int(
                        np.clip(index_tip.x, 0, 1)
                        * (CANVAS_WIDTH - 1)
                    )
                    raw_y = int(
                        np.clip(index_tip.y, 0, 1)
                        * (CANVAS_HEIGHT - 1)
                    )

                    if self.smoothed_point is None:
                        smooth_x, smooth_y = raw_x, raw_y
                    else:
                        smooth_x = int(
                            self.smoothing_factor * raw_x
                            + (1 - self.smoothing_factor)
                            * self.smoothed_point[0]
                        )
                        smooth_y = int(
                            self.smoothing_factor * raw_y
                            + (1 - self.smoothing_factor)
                            * self.smoothed_point[1]
                        )

                    self.smoothed_point = (smooth_x, smooth_y)

                    camera_point = (
                        int(smooth_x * (width - 1) / (CANVAS_WIDTH - 1)),
                        int(smooth_y * (height - 1) / (CANVAS_HEIGHT - 1)),
                    )

                    should_draw = (
                        not middle_up
                        and self.drawing_enabled
                    )

                    if should_draw:
                        self.draw_on_canvas((smooth_x, smooth_y))
                        self.status = "Drawing"
                    elif index_up and middle_up:
                        self.previous_point = None
                        self.status = "Drawing paused"
                    else:
                        self.previous_point = None

                        if not self.drawing_enabled:
                            self.status = "Drawing disabled"
                        else:
                            self.status = "Hand detected"

                else:
                    self.previous_point = None
                    self.smoothed_point = None
                    self.status = "Show your hand to camera"

                self.frames_processed += 1
                image = self.overlay_canvas(image)

                if camera_point is not None:
                    cv2.circle(
                        image,
                        camera_point,
                        9,
                        (0, 255, 120),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        image,
                        camera_point,
                        14,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                status_text = self.status

            cv2.rectangle(
                image,
                (0, 0),
                (width, 42),
                (12, 16, 30),
                -1,
            )

            cv2.putText(
                image,
                "AIR CANVAS PRO",
                (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (200, 180, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                image,
                status_text,
                (max(10, width - 270), 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (120, 255, 190),
                1,
                cv2.LINE_AA,
            )

        except Exception:
            with self.lock:
                self.status = "Processing frame..."

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )

    def get_canvas(self):
        with self.lock:
            return self.canvas.copy()


if "aircanvas_engine" not in st.session_state:
    st.session_state.aircanvas_engine = AirCanvasEngine()

engine = st.session_state.aircanvas_engine


with st.sidebar:
    st.markdown(
        """
        <div class="hero-title" style="font-size:25px;">
            🎨 AirCanvas
        </div>
        <div class="hero-subtitle">
            AI-powered gesture drawing studio
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🖌️ Drawing Tools</div>',
        unsafe_allow_html=True,
    )

    selected_color = st.color_picker(
        "Brush Color",
        value="#00FF78",
        key="aircanvas_color",
    )
    engine.set_color(selected_color)

    selected_size = st.slider(
        "Brush Size",
        min_value=2,
        max_value=30,
        value=7,
        key="aircanvas_brush_size",
    )
    engine.set_brush_size(selected_size)

    eraser_enabled = st.toggle(
        "🧽 Eraser Mode",
        value=False,
        key="aircanvas_eraser",
    )
    engine.set_eraser(eraser_enabled)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">⚙️ Controls</div>',
        unsafe_allow_html=True,
    )

    drawing_enabled = st.toggle(
        "✍️ Enable Drawing",
        value=True,
        key="aircanvas_drawing_enabled",
    )
    engine.set_drawing_enabled(drawing_enabled)

    undo_col, redo_col = st.columns(2)

    with undo_col:
        if st.button("↩️ Undo", use_container_width=True):
            engine.undo()
            st.rerun()

    with redo_col:
        if st.button("↪️ Redo", use_container_width=True):
            engine.redo()
            st.rerun()

    if st.button("🗑️ Clear Canvas", use_container_width=True):
        engine.clear()
        st.rerun()

    st.markdown("---")

    st.markdown(
        '<div class="section-title">🖐️ Gesture Guide</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">☝️ DRAW</div>
            <div class="info-text">
                Index finger UP<br>
                Middle finger DOWN
            </div>
        </div>

        <div class="info-card">
            <div class="info-title">✌️ PAUSE</div>
            <div class="info-text">
                Index finger UP<br>
                Middle finger UP
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("AirCanvas Pro • MediaPipe AI")


st.markdown(
    """
    <div class="hero">
        <div class="badge">✨ AI GESTURE DRAWING</div>
        <div class="hero-title">Your Hands. Your Canvas.</div>
        <div class="hero-subtitle">
            Draw in the air using your index finger.
            No mouse, no touchscreen — just your hand
            and real-time computer vision.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


if engine.model_error:
    st.error("MediaPipe model error: " + engine.model_error)
    st.info(
        "make sure the"
        "hand_landmark.task file is present inside the models folder."
    )


camera_col, canvas_col = st.columns([1.05, 1], gap="large")


with camera_col:
    st.markdown(
        '<div class="section-title">📷 Live Camera</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Allow camera access and place your index finger in front of the camera."
    )

    camera_ctx = webrtc_streamer(
        key="aircanvas-pro-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=engine.process_frame,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        async_processing=True,
    )

    if camera_ctx.state.playing:
        st.success("🟢 Camera connected")

        st_autorefresh(
            interval=1000,
            key="aircanvas_canvas_refresh",
        )
    else:
        st.info(
            "click the play button to start the camera."
            "and  allow camera access in your browser."
        )

    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">💡 Drawing Tips</div>
            <div class="info-text">
                keep your hand in front of the camera. raise your index finger to draw, and
                Raise both fingers to pause.

        </div>
        """,
        unsafe_allow_html=True,
    )


with canvas_col:
    st.markdown(
        '<div class="section-title">🖼️ Drawing Canvas</div>',
        unsafe_allow_html=True,
    )

    st.caption("your artwork will continue to appear here.")

    canvas = engine.get_canvas()

    canvas_rgb = cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB,)

    st.image(canvas_rgb,caption="AirCanvas Drawing Surface",use_container_width=True,)

    png_image = Image.fromarray(canvas_rgb)
    buffer = io.BytesIO()
    png_image.save(buffer, format="PNG")

    st.download_button(
        label="⬇️ Download Drawing (PNG)",
        data=buffer.getvalue(),
        file_name="aircanvas_drawing.png",
        mime="image/png",
        use_container_width=True,
    )


st.markdown("---")

st.markdown(
    '<div class="section-title">📊 Studio Status</div>',
    unsafe_allow_html=True,
)

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.metric("Canvas Size",f"{CANVAS_WIDTH} × {CANVAS_HEIGHT}",)

with stat2:
    st.metric("Brush Size",f"{engine.brush_size} px",)

with stat3:
    with engine.lock:
        current_status = engine.status
    st.metric("Current Status",current_status,)

st.markdown(
    """
    <div class="footer">
        AirCanvas Pro · Built with Streamlit, OpenCV
        and MediaPipe · Gesture-based AI Drawing
    </div>
    """,
    unsafe_allow_html=True,
)