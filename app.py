import os
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from keras_facenet import FaceNet


# ============================================================
# CONFIGURATION
# ============================================================

PROTOTYPE_PATH = "outputs/player_prototypes.npy"
NAMES_PATH = "outputs/player_names.npy"
YUNET_MODEL = "models/face_detector/face_detection_yunet_2023mar.onnx"

st.set_page_config(
    page_title="CricketFace AI",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */
    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(30, 105, 210, 0.16),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #020b1d 0%,
                #061632 50%,
                #020b1d 100%
            );
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #06152f 0%,
                #031027 100%
            );
        border-right: 1px solid rgba(70, 140, 230, 0.25);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* Sidebar brand */
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #f4f8ff;
        margin-bottom: 0;
    }

    .brand-ai {
        color: #3185ff;
    }

    .brand-subtitle {
        color: #7092bd;
        font-size: 0.72rem;
        letter-spacing: 2px;
        margin-top: 2px;
        margin-bottom: 25px;
    }

    .sidebar-section {
        color: #4e79a9;
        font-size: 0.67rem;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 8px;
    }

    .engine-item {
        color: #7596bd;
        font-size: 0.78rem;
        margin: 7px 0;
    }

    /* Navigation radio */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 5px;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent;
        border-radius: 9px;
        padding: 8px 10px;
        color: #a9c2e4;
        transition: 0.2s;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background: rgba(42, 116, 220, 0.14);
    }

    /* Main content */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        color: #f5f8ff;
        margin-bottom: 0;
    }

    .main-title span {
        color: #2580ff;
    }

    .main-subtitle {
        color: #d6e4f7;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 3px;
    }

    .main-description {
        color: #88a8d0;
        font-size: 0.9rem;
        line-height: 1.6;
        max-width: 650px;
    }

    /* Cards */
    .card {
        background:
            linear-gradient(
                145deg,
                rgba(8, 29, 61, 0.95),
                rgba(4, 20, 45, 0.96)
            );
        border: 1px solid rgba(54, 121, 211, 0.45);
        border-radius: 16px;
        padding: 24px;
        min-height: 100%;
        box-shadow:
            0 12px 35px rgba(0, 0, 0, 0.18);
    }

    .card-result {
        border: 1px solid rgba(38, 210, 180, 0.65);
    }

    .card-title {
        color: #f1f6ff;
        font-size: 1.12rem;
        font-weight: 750;
        margin-bottom: 4px;
    }

    .card-description {
        color: #7598c2;
        font-size: 0.78rem;
        margin-bottom: 20px;
    }

    /* Upload area */
    div[data-testid="stFileUploader"] {
        background: rgba(5, 20, 45, 0.7);
        border: 1px dashed rgba(74, 143, 230, 0.65);
        border-radius: 14px;
        padding: 15px;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
        border: none;
    }

    div[data-testid="stFileUploader"] small {
        color: #7598c2;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(
            90deg,
            #1268ed,
            #2c82ff
        );
        color: white;
        font-weight: 700;
        min-height: 44px;
        box-shadow: 0 8px 22px rgba(20, 100, 230, 0.22);
    }

    .stButton > button:hover {
        background: linear-gradient(
            90deg,
            #0e5fdc,
            #2575ed
        );
        color: white;
    }

    /* Match badge */
    .match-badge {
        display: inline-block;
        background: rgba(34, 211, 169, 0.13);
        border: 1px solid rgba(34, 211, 169, 0.55);
        color: #39e2b8;
        border-radius: 20px;
        padding: 6px 13px;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.8px;
        margin-bottom: 15px;
    }

    /* Player result */
    .identified-label {
        color: #6f91bc;
        font-size: 0.68rem;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-top: 8px;
    }

    .player-name {
        color: #f5f8ff;
        font-size: 2rem;
        font-weight: 800;
        margin-top: 3px;
        margin-bottom: 15px;
    }

    .score-number {
        color: #36ddb5;
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1;
    }

    .score-label {
        color: #8ba9cc;
        font-size: 0.78rem;
        margin-top: 6px;
    }

    .technical-note {
        color: #55779f;
        font-size: 0.68rem;
        margin-top: 5px;
    }

    /* Top match rows */
    .match-row {
        background: rgba(10, 34, 68, 0.65);
        border: 1px solid rgba(50, 103, 171, 0.35);
        border-radius: 9px;
        padding: 10px 12px;
        margin: 7px 0;
    }

    .match-row-first {
        border-color: rgba(34, 211, 169, 0.6);
        background: rgba(20, 100, 92, 0.18);
    }

    .match-rank {
        color: #7396bf;
        font-size: 0.73rem;
        font-weight: 700;
    }

    .match-name {
        color: #e8f0fb;
        font-size: 0.82rem;
        font-weight: 650;
    }

    .match-score {
        color: #72aaf3;
        font-size: 0.76rem;
        font-weight: 700;
    }

    /* Stats */
    .stat-box {
        background: rgba(7, 27, 56, 0.72);
        border: 1px solid rgba(55, 116, 193, 0.32);
        border-radius: 12px;
        padding: 13px;
        text-align: center;
    }

    .stat-number {
        color: #f2f6fd;
        font-size: 1.15rem;
        font-weight: 800;
    }

    .stat-label {
        color: #6689b5;
        font-size: 0.65rem;
        letter-spacing: 0.8px;
        margin-top: 3px;
    }

    /* Pipeline */
    .pipeline-card {
        background:
            linear-gradient(
                145deg,
                rgba(7, 27, 56, 0.95),
                rgba(3, 16, 37, 0.96)
            );
        border: 1px solid rgba(50, 113, 195, 0.4);
        border-radius: 16px;
        padding: 20px;
        margin-top: 22px;
    }

    .pipeline-title {
        color: #f0f5fd;
        font-size: 0.95rem;
        font-weight: 750;
        letter-spacing: 0.7px;
        margin-bottom: 15px;
    }

    .pipeline-step {
        text-align: center;
        padding: 8px;
    }

    .pipeline-number {
        color: #3d91ff;
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .pipeline-name {
        color: #e5edf9;
        font-weight: 700;
        font-size: 0.82rem;
        margin-top: 4px;
    }

    .pipeline-desc {
        color: #6689b5;
        font-size: 0.65rem;
        line-height: 1.4;
        margin-top: 3px;
    }

    /* Page headings */
    .page-heading {
        color: #f4f8ff;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .page-description {
        color: #7698c0;
        margin-bottom: 25px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #405e83;
        font-size: 0.65rem;
        margin-top: 28px;
        padding-top: 16px;
        border-top: 1px solid rgba(53, 101, 162, 0.2);
    }

    /* Hide Streamlit branding */
    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_facenet():
    return FaceNet()


@st.cache_resource
def load_face_detector():
    return cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )


@st.cache_data
def load_prototypes():
    prototypes = np.load(PROTOTYPE_PATH)
    player_names = np.load(NAMES_PATH)
    return prototypes, player_names


embedder = load_facenet()
face_detector = load_face_detector()
player_prototypes, player_names = load_prototypes()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# FACE DETECTION
# ============================================================

def detect_and_crop_face(image):
    """
    Detect the strongest face using YuNet and return:
    - cropped face
    - original BGR image
    - detection confidence
    """

    image_bgr = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    height, width = image_bgr.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )

    _, faces = face_detector.detect(
        image_bgr
    )

    if faces is None or len(faces) == 0:
        return None, image_bgr, None

    best_face = max(
        faces,
        key=lambda face: face[-1]
    )

    x, y, w, h = best_face[:4]
    detection_score = float(best_face[-1])

    padding = 0.20

    x1 = max(
        0,
        int(x - w * padding)
    )

    y1 = max(
        0,
        int(y - h * padding)
    )

    x2 = min(
        width,
        int(x + w * (1 + padding))
    )

    y2 = min(
        height,
        int(y + h * (1 + padding))
    )

    face_crop = image_bgr[
        y1:y2,
        x1:x2
    ]

    face_crop = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    face_crop = cv2.resize(
        face_crop,
        (160, 160)
    )

    return (
        face_crop,
        image_bgr,
        detection_score
    )


# ============================================================
# FACE EMBEDDING
# ============================================================

def get_embedding(face_image):
    embedding = embedder.embeddings(
        np.expand_dims(
            face_image,
            axis=0
        )
    )[0]

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:
        raise ValueError(
            "Could not generate a valid face embedding."
        )

    embedding = embedding / norm

    return embedding


# ============================================================
# PLAYER PREDICTION
# ============================================================

def predict_player(face_image):

    embedding = get_embedding(
        face_image
    )

    similarities = np.dot(
        player_prototypes,
        embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1]

    best_index = top_indices[0]

    predicted_player = str(
        player_names[best_index]
    )

    best_score = float(
        similarities[best_index]
    )

    return (
        predicted_player,
        best_score,
        top_indices,
        similarities
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-title">'
        'CRICKETFACE <span class="brand-ai">AI</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-subtitle">AI VISION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Identify Player",
            "History",
            "Supported Players",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        '<div class="sidebar-section">AI ENGINE</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="engine-item">YuNet Face Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="engine-item">FaceNet Embeddings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="engine-item">Cosine Similarity</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="engine-item">10 Supported Players</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<br><br>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="text-align:center; color:#55789f; '
        'font-size:0.67rem; line-height:1.6;">'
        'INDIAN CRICKETER<br>'
        'IDENTIFICATION SYSTEM'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">'
        'CRICKETFACE <span>AI</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Indian Cricketer Identification System'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-description">'
        'Identify one of 10 supported Indian cricketers '
        'using face detection, deep face embeddings, '
        'and similarity-based matching.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    stat1, stat2, stat3 = st.columns(3)

    with stat1:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-number">10</div>'
            '<div class="stat-label">SUPPORTED PLAYERS</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat2:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-number">512D</div>'
            '<div class="stat-label">FACE EMBEDDING</div>'
            '</div>',
            unsafe_allow_html=True
        )

    with stat3:
        st.markdown(
            '<div class="stat-box">'
            '<div class="stat-number">AI</div>'
            '<div class="stat-label">SIMILARITY MATCHING</div>'
            '</div>',
            unsafe_allow_html=True
        )

    st.write("")

    if st.button(
        "Identify a Player",
        use_container_width=True
    ):
        st.session_state.page_redirect = "Identify Player"
        st.rerun()

    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-title">
                IDENTIFICATION PIPELINE
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    p1, p2, p3, p4, p5 = st.columns(5)

    pipeline = [
        (
            p1,
            "01",
            "Face Detection",
            "Locate and crop the face"
        ),
        (
            p2,
            "02",
            "FaceNet",
            "Generate deep embedding"
        ),
        (
            p3,
            "03",
            "Embedding",
            "Create 512D feature vector"
        ),
        (
            p4,
            "04",
            "Matching",
            "Compare with player prototypes"
        ),
        (
            p5,
            "05",
            "Player Identified",
            "Return best similarity match"
        )
    ]

    for column, number, name, description in pipeline:
        with column:
            st.markdown(
                f"""
                <div class="pipeline-step">
                    <div class="pipeline-number">{number}</div>
                    <div class="pipeline-name">{name}</div>
                    <div class="pipeline-desc">{description}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ============================================================
# IDENTIFY PLAYER
# ============================================================

elif page == "Identify Player":

    st.markdown(
        '<div class="main-title">'
        'CRICKETFACE <span>AI</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-subtitle">'
        'Identify Indian Cricketer'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-description">'
        'Upload a clear image containing a visible face '
        'to find the closest supported player match.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    upload_col, result_col = st.columns(
        [1, 1],
        gap="large"
    )

    # --------------------------------------------------------
    # UPLOAD CARD
    # --------------------------------------------------------

    with upload_col:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    Upload Player Image
                </div>
                <div class="card-description">
                    Drop an image or browse your files.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Drop image here or browse",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="visible"
        )

        st.caption(
            "JPG / PNG • Maximum 200MB • "
            "Clear frontal face recommended"
        )

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            identify_button = st.button(
                "Identify Player",
                use_container_width=True
            )

        else:

            st.info(
                "Upload an image to begin identification."
            )

            identify_button = False

    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    with result_col:

        if (
            uploaded_file is None
            or not identify_button
        ):

            st.markdown(
                """
                <div class="card card-result">
                    <div class="card-title">
                        Identification Result
                    </div>
                    <div class="card-description">
                        Your AI result will appear here.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.info(
                "Ready for identification. "
                "Upload a player image and click "
                "Identify Player."
            )

        else:

            with st.spinner(
                "Analyzing image..."
            ):

                try:

                    face, original_image, detection_score = (
                        detect_and_crop_face(image)
                    )

                    if face is None:

                        st.error(
                            "No face detected. "
                            "Please upload a clearer image "
                            "containing a visible face."
                        )

                    else:

                        (
                            predicted_player,
                            score,
                            top_indices,
                            similarities
                        ) = predict_player(face)

                        # Save history
                        history_item = {
                            "player": predicted_player,
                            "score": score,
                            "time": datetime.now().strftime(
                                "%d %b %Y, %I:%M %p"
                            )
                        }

                        st.session_state.history.insert(
                            0,
                            history_item
                        )

                        # Keep only recent 20 results
                        st.session_state.history = (
                            st.session_state.history[:20]
                        )

                        st.markdown(
                            """
                            <div class="match-badge">
                                MATCH FOUND
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        face_col, info_col = st.columns(
                            [0.9, 1.1],
                            gap="medium"
                        )

                        with face_col:

                            st.markdown(
                                '<div class="card-title">'
                                'Detected Face'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            st.image(
                                face,
                                use_container_width=True
                            )

                            st.caption(
                                f"Face detection score: "
                                f"{detection_score:.3f}"
                            )

                        with info_col:

                            st.markdown(
                                '<div class="identified-label">'
                                'IDENTIFIED PLAYER'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="player-name">'
                                f'{predicted_player}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                '<div class="identified-label">'
                                'SIMILARITY SCORE'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="score-number">'
                                f'{score:.4f}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div class="score-label">'
                                f'{score * 100:.2f}% similarity'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            st.progress(
                                max(
                                    0.0,
                                    min(
                                        1.0,
                                        score
                                    )
                                )
                            )

                            st.markdown(
                                '<div class="technical-note">'
                                'Cosine similarity • not probability'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        st.divider()

                        st.markdown(
                            '<div class="card-title">'
                            'Top Matches'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        for rank, index in enumerate(
                            top_indices[:3],
                            start=1
                        ):

                            player = str(
                                player_names[index]
                            )

                            match_score = float(
                                similarities[index]
                            )

                            row_class = (
                                "match-row-first"
                                if rank == 1
                                else "match-row"
                            )

                            st.markdown(
                                f"""
                                <div class="match-row {row_class}">
                                    <span class="match-rank">
                                        {rank:02d}
                                    </span>
                                    &nbsp;&nbsp;
                                    <span class="match-name">
                                        {player}
                                    </span>
                                    <span style="float:right"
                                          class="match-score">
                                        {match_score:.4f}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                except Exception as e:

                    st.error(
                        f"Unable to process the image: {e}"
                    )


# ============================================================
# HISTORY
# ============================================================

elif page == "History":

    st.markdown(
        '<div class="page-heading">'
        'Identification History'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Recent player identification results from this session.'
        '</div>',
        unsafe_allow_html=True
    )

    if len(st.session_state.history) == 0:

        st.info(
            "No identification history yet. "
            "Identify a player to see results here."
        )

    else:

        for item in st.session_state.history:

            col1, col2, col3 = st.columns(
                [2, 2, 1]
            )

            with col1:
                st.markdown(
                    f"**{item['player']}**"
                )

            with col2:
                st.write(
                    f"Similarity: {item['score']:.4f}"
                )

            with col3:
                st.caption(
                    item["time"]
                )

            st.divider()


# ============================================================
# SUPPORTED PLAYERS
# ============================================================

elif page == "Supported Players":

    st.markdown(
        '<div class="page-heading">'
        'Supported Players'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'The current model can identify these 10 selected '
        'Indian cricketers.'
        '</div>',
        unsafe_allow_html=True
    )

    names = [
        str(name)
        for name in player_names
    ]

    for start in range(
        0,
        len(names),
        2
    ):

        cols = st.columns(2)

        for col, name in zip(
            cols,
            names[start:start + 2]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="card"
                         style="margin-bottom:15px;
                                min-height:80px;">
                        <div class="card-title">
                            {name}
                        </div>
                        <div class="card-description"
                             style="margin-bottom:0;">
                            Supported identification class
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.markdown(
        '<div class="page-heading">'
        'About CricketFace AI'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-description">'
        'Computer vision based Indian cricketer identification.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

        <div class="card-title">
            System Overview
        </div>

        <br>

        CricketFace AI is a closed-set facial identification
        system designed to recognize one of 10 selected Indian
        cricketers.

        <br><br>

        <b>Face Detection</b><br>
        YuNet detects and crops the strongest visible face.

        <br><br>

        <b>Face Embeddings</b><br>
        FaceNet converts the detected face into a 512-dimensional
        feature representation.

        <br><br>

        <b>Matching</b><br>
        The generated embedding is compared with stored player
        prototypes using cosine similarity.

        <br><br>

        <b>Prediction</b><br>
        The player with the highest similarity score is returned
        as the predicted identity.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        CRICKETFACE AI &nbsp;•&nbsp;
        YuNet Face Detection &nbsp;•&nbsp;
        FaceNet Embeddings &nbsp;•&nbsp;
        Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True
)
