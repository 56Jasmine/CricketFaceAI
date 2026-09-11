import os
import textwrap

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from keras_facenet import FaceNet


# =========================================================
# CONFIGURATION
# =========================================================

PROTOTYPE_PATH = "outputs/player_prototypes.npy"
NAMES_PATH = "outputs/player_names.npy"
YUNET_MODEL = "models/face_detector/face_detection_yunet_2023mar.onnx"

st.set_page_config(
    page_title="CricketFace AI",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# HTML HELPER
# =========================================================

def render_html(html):
    st.html(textwrap.dedent(html).strip())

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.stApp {
    background:
        radial-gradient(
            circle at 85% 10%,
            rgba(35, 105, 210, 0.16),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 80%,
            rgba(15, 65, 135, 0.12),
            transparent 30%
        ),
        #050f24;
    color: #eef5ff;
}

.main .block-container {
    max-width: 1500px;
    padding: 1.2rem 2.4rem 3rem 2.4rem;
}


/* =====================================================
   SIDEBAR
   ===================================================== */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #071731 0%,
        #061329 55%,
        #040d1e 100%
    );

    border-right: 1px solid rgba(70, 125, 205, 0.24);
    min-width: 250px;
    max-width: 250px;
}

.brand {
    padding: 12px 8px 24px 8px;
    border-bottom: 1px solid rgba(100, 150, 220, 0.12);
    margin-bottom: 22px;
}

.brand-name {
    font-size: 20px;
    font-weight: 800;
    color: #f5f8ff;
}

.brand-name span {
    color: #3284ff;
}

.brand-subtitle {
    color: #719bd1;
    font-size: 9px;
    letter-spacing: 2px;
    margin-top: 4px;
}

.navigation-label {
    color: #5c7da8;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin: 0 8px 10px 8px;
}


/* =====================================================
   TOP BAR
   ===================================================== */

.topbar {
    height: 42px;
    border-bottom: 1px solid rgba(100, 150, 220, 0.10);
    display: flex;
    align-items: center;
    justify-content: flex-end;
    margin-bottom: 22px;
}

.topbar-text {
    color: #8daed8;
    font-size: 12px;
}

.topbar-arrow {
    color: #3985f2;
    padding: 0 7px;
}

.ai-badge {
    margin-left: 16px;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    border: 1px solid rgba(73, 140, 232, 0.45);
    background: rgba(15, 43, 82, 0.8);

    display: flex;
    align-items: center;
    justify-content: center;

    color: #83b6fa;
    font-size: 9px;
    font-weight: 700;
}


/* =====================================================
   HERO
   ===================================================== */

.hero {
    position: relative;
    overflow: hidden;

    min-height: 175px;
    padding: 27px 30px;

    border-radius: 16px;
    margin-bottom: 22px;

    background:
        linear-gradient(
            90deg,
            rgba(5, 15, 36, 0.98) 0%,
            rgba(5, 20, 45, 0.94) 55%,
            rgba(8, 35, 75, 0.76) 100%
        );

    border: 1px solid rgba(57, 119, 205, 0.30);
}

.hero::after {
    content: "";

    position: absolute;
    right: -110px;
    top: -160px;

    width: 430px;
    height: 430px;

    border-radius: 50%;

    border: 1px solid rgba(65, 143, 255, 0.08);

    box-shadow:
        0 0 0 50px rgba(65, 143, 255, 0.025),
        0 0 0 100px rgba(65, 143, 255, 0.018);
}

.hero-label {
    position: relative;
    z-index: 2;

    color: #4d94f5;

    font-size: 9px;
    letter-spacing: 2px;

    margin-bottom: 7px;
}

.hero-title {
    position: relative;
    z-index: 2;

    color: #f4f8ff;

    font-size: 41px;
    line-height: 1.05;

    font-weight: 800;
    letter-spacing: -1px;
}

.hero-title span {
    color: #3183ff;
}

.hero-subtitle {
    position: relative;
    z-index: 2;

    color: #e9f1fd;

    font-size: 19px;
    font-weight: 600;

    margin-top: 7px;
}

.hero-description {
    position: relative;
    z-index: 2;

    max-width: 650px;

    color: #91add3;

    font-size: 12px;
    line-height: 1.6;

    margin-top: 8px;
}


/* =====================================================
   MAIN CARDS
   ===================================================== */

.upload-card,
.result-card {
    min-height: 485px;

    border-radius: 15px;

    padding: 22px;

    background:
        linear-gradient(
            145deg,
            rgba(8, 29, 62, 0.94),
            rgba(4, 17, 38, 0.98)
        );
}

.upload-card {
    border: 1px solid rgba(52, 117, 204, 0.43);
}

.result-card {
    border: 1px solid rgba(41, 220, 177, 0.58);
}

.card-title {
    color: #edf5ff;

    font-size: 19px;
    font-weight: 700;

    margin-bottom: 5px;
}

.card-subtitle {
    color: #718fb9;

    font-size: 11px;

    margin-bottom: 16px;
}


/* =====================================================
   FILE UPLOADER
   ===================================================== */

[data-testid="stFileUploaderDropzone"] {
    background: rgba(5, 20, 44, 0.72);

    border: 1px dashed rgba(82, 143, 222, 0.72);

    border-radius: 12px;

    min-height: 190px;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #3184ff;

    background: rgba(8, 27, 57, 0.88);
}


/* =====================================================
   BUTTON
   ===================================================== */

.stButton > button {
    width: 100%;

    height: 48px;

    border: none;

    border-radius: 9px;

    background:
        linear-gradient(
            90deg,
            #1468e8,
            #287ff6
        );

    color: white;

    font-size: 14px;
    font-weight: 700;

    box-shadow:
        0 8px 25px rgba(17, 103, 225, 0.20);
}

.stButton > button:hover {
    background:
        linear-gradient(
            90deg,
            #1c72f2,
            #3488ff
        );

    color: white;
}


/* =====================================================
   EMPTY RESULT
   ===================================================== */

.empty-result {
    min-height: 395px;

    border: 1px dashed rgba(80, 128, 190, 0.32);

    border-radius: 12px;

    display: flex;

    align-items: center;
    justify-content: center;

    text-align: center;
}

.empty-title {
    color: #c2d4eb;

    font-size: 17px;
    font-weight: 600;

    margin-bottom: 7px;
}

.empty-text {
    color: #6785ad;

    font-size: 11px;

    line-height: 1.7;

    max-width: 300px;
}


/* =====================================================
   MATCH BADGE
   ===================================================== */

.match-badge {
    display: inline-block;

    padding: 6px 13px;

    border-radius: 30px;

    background: rgba(29, 211, 169, 0.13);

    border: 1px solid rgba(44, 224, 181, 0.48);

    color: #3de1b7;

    font-size: 10px;

    font-weight: 700;

    letter-spacing: 1px;

    margin-bottom: 15px;
}


/* =====================================================
   DETECTED FACE
   ===================================================== */

.detected-box {
    background: rgba(5, 21, 45, 0.82);

    border: 1px solid rgba(69, 119, 181, 0.28);

    border-radius: 11px;

    padding: 11px;
}

.detected-title {
    color: #dce9fa;

    font-size: 12px;
    font-weight: 600;

    margin-bottom: 8px;
}


/* =====================================================
   PLAYER RESULT
   ===================================================== */

.player-label {
    color: #86a7d3;

    font-size: 9px;

    letter-spacing: 1px;

    text-transform: uppercase;

    margin-top: 6px;
}

.player-name {
    color: #f5f8ff;

    font-size: 29px;

    font-weight: 800;

    margin-top: 3px;
}

.score-label {
    color: #90a9cc;

    font-size: 12px;
}

.score-value {
    color: #31ddb0;

    font-size: 36px;

    font-weight: 800;

    letter-spacing: -1px;
}

.score-note {
    color: #637f9f;

    font-size: 9px;
}

.section-line {
    height: 1px;

    background: rgba(90, 130, 180, 0.17);

    margin: 13px 0;
}


/* =====================================================
   TOP MATCHES
   ===================================================== */

.matches-title {
    color: #dceafe;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 8px;
}

.match-row {
    background: rgba(8, 28, 58, 0.82);

    border: 1px solid rgba(57, 102, 161, 0.28);

    border-radius: 8px;

    padding: 8px 10px;

    margin-bottom: 6px;
}

.match-row.first {
    border-color: rgba(40, 220, 176, 0.62);

    background: rgba(18, 94, 81, 0.22);
}

.match-rank {
    display: inline-block;

    width: 23px;

    color: #9db8dd;

    font-size: 11px;

    font-weight: 700;
}

.match-name {
    color: #edf5ff;

    font-size: 11px;

    font-weight: 600;
}

.match-score {
    float: right;

    color: #9bbce8;

    font-size: 10px;
}

.first .match-score {
    color: #3ee0b6;
}

.bar-background {
    height: 4px;

    margin-left: 23px;
    margin-top: 5px;

    background: rgba(57, 99, 157, 0.35);

    border-radius: 10px;

    overflow: hidden;
}

.bar-fill {
    height: 100%;

    background:
        linear-gradient(
            90deg,
            #2982ff,
            #6caeff
        );

    border-radius: 10px;
}

.first .bar-fill {
    background:
        linear-gradient(
            90deg,
            #1fd4a9,
            #4de7c1
        );
}


/* =====================================================
   INFO CARDS
   ===================================================== */

.info-card {
    background: rgba(7, 25, 53, 0.72);

    border: 1px solid rgba(55, 107, 176, 0.27);

    border-radius: 13px;

    padding: 19px;

    margin-top: 18px;
}

.info-heading {
    color: #e9f2ff;

    font-size: 14px;

    font-weight: 700;

    margin-bottom: 8px;
}

.info-text {
    color: #7896bd;

    font-size: 11px;

    line-height: 1.7;
}


/* =====================================================
   FOOTER
   ===================================================== */

.footer {
    text-align: center;

    color: #4f6b92;

    font-size: 8px;

    margin-top: 28px;

    padding-top: 15px;

    border-top: 1px solid rgba(80, 120, 175, 0.10);
}


/* =====================================================
   MOBILE
   ===================================================== */

@media (max-width: 900px) {

    .main .block-container {
        padding: 1rem;
    }

    .hero-title {
        font-size: 30px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_facenet():
    return FaceNet()


@st.cache_resource
def load_face_detector():

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        (320, 320),
        0.6,
        0.3,
        5000
    )

    return detector


@st.cache_data
def load_prototypes():

    prototypes = np.load(
        PROTOTYPE_PATH
    )

    player_names = np.load(
        NAMES_PATH
    )

    return prototypes, player_names


embedder = load_facenet()

face_detector = load_face_detector()

player_prototypes, player_names = load_prototypes()


# =========================================================
# SESSION STATE
# =========================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "history" not in st.session_state:
    st.session_state.history = []


# =========================================================
# FACE DETECTION
# =========================================================

def detect_and_crop_face(image):

    image = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    height, width = image.shape[:2]

    face_detector.setInputSize(
        (width, height)
    )

    _, faces = face_detector.detect(
        image
    )

    if faces is None or len(faces) == 0:
        return None, 0

    best_face = max(
        faces,
        key=lambda face: face[-1]
    )

    x, y, w, h = best_face[:4]

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

    face_crop = image[
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

    return face_crop, len(faces)


# =========================================================
# FACE EMBEDDING
# =========================================================

def get_embedding(face_image):

    embedding = embedder.embeddings(
        np.expand_dims(
            face_image,
            axis=0
        )
    )[0]

    embedding = embedding / np.linalg.norm(
        embedding
    )

    return embedding


# =========================================================
# PLAYER PREDICTION
# =========================================================

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

    predicted_player = player_names[
        best_index
    ]

    best_score = float(
        similarities[best_index]
    )

    return (
        predicted_player,
        best_score,
        top_indices,
        similarities
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    render_html(
        """
        <div class="brand">
            <div class="brand-name">
                CRICKETFACE <span>AI</span>
            </div>

            <div class="brand-subtitle">
                AI VISION
            </div>
        </div>
        """
    )

    render_html(
        """
        <div class="navigation-label">
            Navigation
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Identify Player",
            "History",
            "About"
        ],
        label_visibility="collapsed"
    )

    render_html(
        """
        <div style="
            position: fixed;
            bottom: 25px;
            width: 205px;
            text-align: center;
            color: #55749f;
            font-size: 9px;
            line-height: 1.7;
        ">
            INDIAN CRICKETER<br>
            IDENTIFICATION SYSTEM

            <br><br>

            FACE DETECTION &nbsp; • &nbsp;
            FACE EMBEDDINGS &nbsp; • &nbsp;
            AI MATCHING
        </div>
        """
    )


# =========================================================
# TOP BAR
# =========================================================

render_html(
    """
    <div class="topbar">

        <div class="topbar-text">
            Your Face
            <span class="topbar-arrow">→</span>
            Our AI
            <span class="topbar-arrow">→</span>
            Your Cricketer
        </div>

        <div class="ai-badge">
            AI
        </div>

    </div>
    """
)


# =========================================================
# HOME / IDENTIFY
# =========================================================

if page in ["Home", "Identify Player"]:

    render_html(
        """
        <div class="hero">

            <div class="hero-label">
                COMPUTER VISION SYSTEM
            </div>

            <div class="hero-title">
                CRICKETFACE <span>AI</span>
            </div>

            <div class="hero-subtitle">
                Indian Cricketer Identification System
            </div>

            <div class="hero-description">
                Identify one of 10 supported Indian cricketers
                using facial features, deep face embeddings,
                and similarity-based matching.
            </div>

        </div>
        """
    )


    left, right = st.columns(
        [1, 1.08],
        gap="large"
    )


    # =====================================================
    # LEFT — UPLOAD
    # =====================================================

    with left:

        render_html(
            """
            <div class="upload-card">

                <div class="card-title">
                    Upload Player Image
                </div>

                <div class="card-subtitle">
                    Upload a clear image containing a visible face.
                </div>

            </div>
            """
        )

        uploaded_file = st.file_uploader(
            "Upload image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            label_visibility="collapsed"
        )

        render_html(
            """
            <div style="
                text-align: center;
                color: #718fb8;
                font-size: 10px;
                margin-top: 7px;
                margin-bottom: 17px;
            ">
                JPG / PNG &nbsp; • &nbsp;
                Clear frontal face recommended
            </div>
            """
        )

        identify_button = st.button(
            "Identify Player",
            use_container_width=True
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

            if identify_button:

                with st.spinner(
                    "Analyzing facial features..."
                ):

                    face, face_count = (
                        detect_and_crop_face(
                            image
                        )
                    )

                    if face is None:

                        st.session_state.prediction = None

                        st.error(
                            "No face detected. "
                            "Please upload a clearer image "
                            "with a visible face."
                        )

                    else:

                        (
                            predicted_player,
                            score,
                            top_indices,
                            similarities
                        ) = predict_player(face)

                        st.session_state.prediction = {

                            "player":
                                predicted_player,

                            "score":
                                score,

                            "face":
                                face,

                            "top_indices":
                                top_indices,

                            "similarities":
                                similarities,

                            "face_count":
                                face_count
                        }

                        st.session_state.history.insert(
                            0,
                            {
                                "player":
                                    predicted_player,

                                "score":
                                    score
                            }
                        )

                        st.rerun()


    # =====================================================
    # RIGHT — RESULT
    # =====================================================

    with right:

        prediction = (
            st.session_state.prediction
        )

        if prediction is None:

            render_html(
                """
                <div class="result-card">

                    <div class="card-title">
                        Identification Result
                    </div>

                    <div class="empty-result">

                        <div>

                            <div class="empty-title">
                                Ready for identification
                            </div>

                            <div class="empty-text">
                                Upload a player image and click
                                <b>Identify Player</b> to see
                                the AI prediction.
                            </div>

                        </div>

                    </div>

                </div>
                """
            )

        else:

            player = prediction["player"]

            score = prediction["score"]

            face = prediction["face"]

            top_indices = (
                prediction["top_indices"]
            )

            similarities = (
                prediction["similarities"]
            )

            face_count = (
                prediction["face_count"]
            )


            render_html(
                """
                <div class="result-card">

                    <div class="match-badge">
                        MATCH FOUND
                    </div>

                </div>
                """
            )


            result_left, result_right = st.columns(
                [0.75, 1.25],
                gap="large"
            )


            # =============================================
            # FACE
            # =============================================

            with result_left:

                render_html(
                    """
                    <div class="detected-box">

                        <div class="detected-title">
                            Detected Face
                        </div>

                    </div>
                    """
                )

                st.image(
                    face,
                    use_container_width=True
                )

                if face_count > 1:

                    st.caption(
                        f"{face_count} faces detected. "
                        "Highest-confidence face used."
                    )


            # =============================================
            # PLAYER DETAILS
            # =============================================

            with result_right:

                render_html(
                    """
                    <div class="player-label">
                        Identified Player
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="player-name">
                        {player}
                    </div>
                    """
                )

                render_html(
                    """
                    <div class="section-line"></div>

                    <div class="score-label">
                        Similarity Score
                    </div>
                    """
                )

                render_html(
                    f"""
                    <div class="score-value">
                        {score:.4f}
                    </div>

                    <div class="score-note">
                        Cosine similarity • not probability
                    </div>
                    """
                )

                render_html(
                    """
                    <div class="section-line"></div>

                    <div class="matches-title">
                        Top Matches
                    </div>
                    """
                )


                # =========================================
                # TOP 3 MATCHES
                # =========================================

                for rank, index in enumerate(
                    top_indices[:3],
                    start=1
                ):

                    match_name = (
                        player_names[index]
                    )

                    match_score = float(
                        similarities[index]
                    )

                    width = max(
                        5,
                        min(
                            100,
                            match_score * 100
                        )
                    )

                    row_class = (
                        "match-row first"
                        if rank == 1
                        else "match-row"
                    )

                    render_html(
                        f"""
                        <div class="{row_class}">

                            <span class="match-rank">
                                {rank}.
                            </span>

                            <span class="match-name">
                                {match_name}
                            </span>

                            <span class="match-score">
                                {match_score:.4f}
                            </span>

                            <div class="bar-background">

                                <div
                                    class="bar-fill"
                                    style="width:{width}%"
                                ></div>

                            </div>

                        </div>
                        """
                    )


# =========================================================
# HISTORY
# =========================================================

elif page == "History":

    render_html(
        """
        <div class="hero">

            <div class="hero-label">
                IDENTIFICATION RECORDS
            </div>

            <div class="hero-title">
                Prediction <span>History</span>
            </div>

            <div class="hero-description">
                Review player identifications made during
                this application session.
            </div>

        </div>
        """
    )


    if not st.session_state.history:

        render_html(
            """
            <div class="empty-result">

                <div>

                    <div class="empty-title">
                        No predictions yet
                    </div>

                    <div class="empty-text">
                        Your identification results will
                        appear here after using the system.
                    </div>

                </div>

            </div>
            """
        )

    else:

        for i, item in enumerate(
            st.session_state.history
        ):

            render_html(
                f"""
                <div class="info-card">

                    <div class="info-heading">
                        {i + 1}. {item["player"]}
                    </div>

                    <div class="info-text">
                        Similarity Score:
                        <b style="color:#36deb2;">
                            {item["score"]:.4f}
                        </b>
                    </div>

                </div>
                """
            )


# =========================================================
# ABOUT
# =========================================================

elif page == "About":

    render_html(
        """
        <div class="hero">

            <div class="hero-label">
                ABOUT THE SYSTEM
            </div>

            <div class="hero-title">
                CRICKETFACE <span>AI</span>
            </div>

            <div class="hero-description">
                A computer vision based identification system
                designed to identify supported Indian cricketers
                from facial features.
            </div>

        </div>
        """
    )


    col1, col2 = st.columns(
        2,
        gap="large"
    )


    with col1:

        render_html(
            """
            <div class="info-card">

                <div class="info-heading">
                    Computer Vision
                </div>

                <div class="info-text">
                    YuNet detects the face from the uploaded image.
                    FaceNet converts the detected face into a
                    512-dimensional embedding, which is then
                    compared with stored player prototypes.
                </div>

            </div>
            """
        )


    with col2:

        render_html(
            """
            <div class="info-card">

                <div class="info-heading">
                    Supported Players
                </div>

                <div class="info-text">
                    The current system supports 10 selected
                    Indian cricketers. Identification is performed
                    within this closed set of supported players.
                </div>

            </div>
            """
        )


    render_html(
        """
        <div class="info-card">

            <div class="info-heading">
                Technologies
            </div>

            <div class="info-text">
                Python &nbsp; • &nbsp;
                OpenCV &nbsp; • &nbsp;
                YuNet &nbsp; • &nbsp;
                FaceNet &nbsp; • &nbsp;
                NumPy &nbsp; • &nbsp;
                TensorFlow &nbsp; • &nbsp;
                Streamlit
            </div>

        </div>
        """
    )


# =========================================================
# FOOTER
# =========================================================

render_html(
    """
    <div class="footer">
        CRICKETFACE AI &nbsp; • &nbsp;
        YuNet Face Detection &nbsp; • &nbsp;
        FaceNet Embeddings &nbsp; • &nbsp;
        Cosine Similarity
    </div>
    """
)
