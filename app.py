from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Pnuemonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS & PATH CONFIGURATION (UNCHANGED)
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "pneumonia_model_finetuned.pth"
)

THRESHOLD = 0.55

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PREMIUM STYLING SYSTEM
# ============================================================

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
    /* Global Resets & Typography */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp {
        background-color: #080c14;
        background-image: 
            radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.18) 0%, transparent 35%),
            radial-gradient(circle at 85% 5%, rgba(14, 165, 233, 0.14) 0%, transparent 32%),
            radial-gradient(circle at 50% 100%, rgba(15, 23, 42, 0.8) 0%, transparent 50%);
        background-attachment: fixed;
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1140px;
        padding: 1.5rem 1.5rem 4rem 1.5rem;
    }

    #MainMenu, footer, header {
        visibility: hidden !important;
        height: 0 !important;
    }

    /* Navigation Bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 22px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.35);
        margin-bottom: 2rem;
    }

    .nav-brand {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(14, 165, 233, 0.25));
        border: 1px solid rgba(129, 140, 248, 0.35);
        font-size: 20px;
    }

    .brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #ffffff;
    }

    .brand-tag {
        font-size: 0.72rem;
        color: #94a3b8;
        font-weight: 500;
    }

    .nav-badges {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 11px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .pill-success {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.28);
        color: #4ade80;
    }

    .pill-info {
        background: rgba(56, 189, 248, 0.10);
        border: 1px solid rgba(56, 189, 248, 0.25);
        color: #7dd3fc;
    }

    .pulse-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #4ade80;
        box-shadow: 0 0 8px #4ade80;
        animation: pulse 2s infinite ease-in-out;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.85); }
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 1.5rem 0 2.5rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .hero-heading {
        font-size: clamp(2.2rem, 5vw, 3.4rem);
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.1;
        margin: 0 auto 0.85rem;
        background: linear-gradient(180deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-sub {
        font-size: clamp(0.95rem, 1.8vw, 1.1rem);
        color: #94a3b8;
        max-width: 580px;
        margin: 0 auto;
        line-height: 1.6;
        font-weight: 400;
    }

    /* Modern Metric Cards Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 2rem;
    }

    .stat-card {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 14px 16px;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .stat-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-2px);
    }

    .stat-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748b;
        margin-bottom: 4px;
    }

    .stat-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }

    /* Section Subheaders */
    .ui-panel-header {
        font-size: 0.98rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Streamlit File Uploader Customization */
    [data-testid="stFileUploader"] {
        padding: 0;
        background: transparent;
        border: none;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1.5px dashed rgba(99, 102, 241, 0.35) !important;
        border-radius: 16px !important;
        padding: 2.2rem 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(129, 140, 248, 0.8) !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 48px;
        border-radius: 12px;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: #ffffff;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.35);
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        box-shadow: 0 14px 30px -4px rgba(79, 70, 229, 0.5);
        transform: translateY(-1px);
        color: #ffffff;
    }

    .stButton > button:active {
        transform: translateY(1px);
    }

    /* Diagnostics Output Cards */
    .diagnostic-card {
        border-radius: 18px;
        padding: 24px;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(14px);
        position: relative;
        overflow: hidden;
    }

    .card-normal {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 78, 59, 0.15) 100%);
        border: 1px solid rgba(52, 211, 153, 0.28);
        box-shadow: 0 12px 36px -8px rgba(16, 185, 129, 0.15);
    }

    .card-pneumonia {
        background: linear-gradient(145deg, rgba(244, 63, 94, 0.08) 0%, rgba(136, 19, 55, 0.18) 100%);
        border: 1px solid rgba(251, 113, 133, 0.32);
        box-shadow: 0 12px 36px -8px rgba(244, 63, 94, 0.18);
    }

    .diag-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }

    .diag-title {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 8px;
    }

    .color-normal { color: #34d399; }
    .color-pneumonia { color: #fb7185; }

    .diag-confidence {
        font-size: 0.92rem;
        color: #cbd5e1;
    }

    /* Probability Visual Bars */
    .bar-wrapper {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px;
        margin-top: 1rem;
    }

    .bar-row {
        margin-bottom: 12px;
    }

    .bar-row:last-child {
        margin-bottom: 0;
    }

    .bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .bar-track {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 9999px;
        overflow: hidden;
    }

    .bar-fill-normal {
        height: 100%;
        background: linear-gradient(90deg, #10b981, #34d399);
        border-radius: 9999px;
    }

    .bar-fill-pneumonia {
        height: 100%;
        background: linear-gradient(90deg, #f43f5e, #fb7185);
        border-radius: 9999px;
    }

    /* Image Preview Container */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    }

    /* Clean Expander */
    .streamlit-expanderHeader {
        background: rgba(15, 23, 42, 0.45) !important;
        border-radius: 12px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #cbd5e1 !important;
    }

    /* Clinical Disclaimer Box */
    .disclaimer-card {
        display: flex;
        gap: 12px;
        padding: 14px 18px;
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.06);
        margin-top: 2.5rem;
    }

    .disclaimer-icon {
        font-size: 1.1rem;
        line-height: 1.4;
    }

    .disclaimer-text {
        font-size: 0.78rem;
        color: #64748b;
        line-height: 1.55;
    }

    .disclaimer-text strong {
        color: #94a3b8;
    }

    /* Footer */
    .footer-bar {
        text-align: center;
        padding-top: 2rem;
        margin-top: 2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.76rem;
        color: #475569;
    }

    /* Mobile Adaptations */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.8rem 3rem 0.8rem;
        }
        .stats-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .nav-bar {
            flex-direction: column;
            gap: 10px;
            align-items: flex-start;
        }
        .nav-badges {
            width: 100%;
            justify-content: flex-start;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PYTORCH MODEL LOADER (OPTIMIZED & CACHED)
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n{MODEL_PATH}")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# INFERENCE ROUTINE
# ============================================================

def predict(image: Image.Image, model: torch.nn.Module):
    image_rgb = image.convert("RGB")
    tensor = transform(image_rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    normal_prob = probabilities[0].item()
    pneumonia_prob = probabilities[1].item()

    if pneumonia_prob >= THRESHOLD:
        prediction = "PNEUMONIA"
        confidence = pneumonia_prob
    else:
        prediction = "NORMAL"
        confidence = normal_prob

    return prediction, confidence, normal_prob, pneumonia_prob


# ============================================================
# APPLICATION LIFECYCLE & STATE
# ============================================================

if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = None

try:
    model = load_model()
    model_ready = True
except Exception as err:
    model = None
    model_ready = False
    model_error_msg = str(err)


# ============================================================
# NAVIGATION BAR
# ============================================================

device_label = "CUDA ACCELERATED" if DEVICE.type == "cuda" else "CPU ENGINE"

st.markdown(
    f"""
    <div class="nav-bar">
        <div class="nav-brand">
            <div class="brand-icon">🫁</div>
            <div>
                <div class="brand-title">AI Pneumonia Detection</div>
                <div class="brand-tag">CXR Screening System</div>
            </div>
        </div>
        <div class="nav-badges">
            <span class="pill-badge pill-info">{device_label}</span>
            <span class="pill-badge pill-success">
                <span class="pulse-dot"></span>
                {"System Operational" if model_ready else "Check System"}
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">ResNet-18 Neural Classifier</div>
        <h1 class="hero-heading">
            Next-Gen <span style="background: linear-gradient(135deg, #818cf8, #38bdf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Pneumonia Screening</span>
        </h1>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# RUNTIME & ARCHITECTURE METRICS
# ============================================================

st.markdown(
    f"""
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Model Backbone</div>
            <div class="stat-value">ResNet-18</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Validation Acc.</div>
            <div class="stat-value">97.89%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Decision Threshold</div>
            <div class="stat-value">{THRESHOLD:.2f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Execution Target</div>
            <div class="stat-value">{"CUDA" if DEVICE.type == "cuda" else "CPU"}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not model_ready:
    st.error("Failed to load pre-trained weights. Please verify model storage integrity.")
    st.code(model_error_msg)
    st.stop()


# ============================================================
# RADIOGRAPH WORKSPACE
# ============================================================

col_upload, col_result = st.columns([1.05, 0.95], gap="large")

with col_upload:
    st.markdown(
        '<div class="ui-panel-header"><span>📁</span> Upload Radiograph</div>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload chest X-ray",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file).convert("RGB")
            st.image(pil_image, use_container_width=True)
            st.caption(f"Asset: {uploaded_file.name} · Resolution: {pil_image.width} × {pil_image.height}px")
        except UnidentifiedImageError:
            st.error("Uploaded file format is invalid or unreadable.")
            pil_image = None
    else:
        pil_image = None
        st.session_state.analysis_cache = None

with col_result:
    st.markdown(
        '<div class="ui-panel-header"><span>🔬</span> Diagnostic Findings</div>',
        unsafe_allow_html=True,
    )

    if uploaded_file is not None and pil_image is not None:
        run_btn = st.button("Run Diagnostic Inference", use_container_width=True)

        if run_btn:
            with st.spinner("Processing tensor passing through 18 convolutional layers..."):
                pred, conf, p_norm, p_pneu = predict(pil_image, model)
                st.session_state.analysis_cache = {
                    "file_id": uploaded_file.file_id if hasattr(uploaded_file, "file_id") else uploaded_file.name,
                    "pred": pred,
                    "conf": conf,
                    "normal_prob": p_norm,
                    "pneumonia_prob": p_pneu,
                }

        # Persistent render from session cache
        if st.session_state.analysis_cache is not None:
            data = st.session_state.analysis_cache
            is_pneumonia = data["pred"] == "PNEUMONIA"

            card_class = "card-pneumonia" if is_pneumonia else "card-normal"
            text_class = "color-pneumonia" if is_pneumonia else "color-normal"
            status_icon = "⚠️" if is_pneumonia else "✓"

            st.markdown(
                f"""
                <div class="diagnostic-card {card_class}">
                    <div class="diag-eyebrow">Classification Output</div>
                    <div class="diag-title {text_class}">
                        {status_icon} {data["pred"]}
                    </div>
                    <div class="diag-confidence">
                        Prediction confidence: <strong>{data["conf"] * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Custom HTML probability meters
            norm_pct = data["normal_prob"] * 100
            pneu_pct = data["pneumonia_prob"] * 100

            st.markdown(
                f"""
                <div class="bar-wrapper">
                    <div class="bar-row">
                        <div class="bar-header">
                            <span style="color: #94a3b8;">Normal Clarity</span>
                            <span style="color: #34d399;">{norm_pct:.2f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill-normal" style="width: {norm_pct:.2f}%;"></div>
                        </div>
                    </div>
                    <div class="bar-row" style="margin-top: 14px;">
                        <div class="bar-header">
                            <span style="color: #94a3b8;">Pneumonia Infiltration</span>
                            <span style="color: #fb7185;">{pneu_pct:.2f}%</span>
                        </div>
                        <div class="bar-track">
                            <div class="bar-fill-pneumonia" style="width: {pneu_pct:.2f}%;"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Technical Telemetry & Parameters", expanded=False):
                st.markdown(
                    f"""
                    - **Feature Extractor:** ResNet-18 Residual Encoder
                    - **Checkpoint Weight:** `pneumonia_model_finetuned.pth`
                    - **Spatial Input Matrix:** 224 × 224 × 3
                    - **Decision Margin Cutoff:** {THRESHOLD}
                    - **Hardware Accelerator:** {DEVICE}
                    - **Softmax Raw Distribution:** Normal ({data["normal_prob"]:.4f}) | Pneumonia ({data["pneumonia_prob"]:.4f})
                    """
                )
    else:
        st.info("Upload an anterior-posterior (AP) or lateral chest X-ray image from the left panel to initialize diagnostic analysis.")


# ============================================================
# CLINICAL COMPLIANCE & FOOTER
# ============================================================
st.markdown(
    """
    <div class="disclaimer-card">
        <div class="disclaimer-icon">⚠️</div>
        <div class="disclaimer-text">
            <strong>Project Disclaimer:</strong> This is an educational machine learning project trained on a Kaggle chest X-ray dataset. It is built purely as a portfolio demo and is <strong>not</strong> for actual medical diagnosis, treatment, or clinical use. Do not use this tool for real medical decisions.
        </div>
    </div>
    <div class="footer-bar">
        · PneumoVision · Personal ML Project · ResNet-18 ·
    </div>
    """,
    unsafe_allow_html=True,
)