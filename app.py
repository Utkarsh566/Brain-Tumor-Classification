
import streamlit as st
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt

from skimage.feature import hog, graycomatrix, graycoprops
from scipy.special import softmax

# -----------------------------
# 1. Load trained model
# -----------------------------


MODEL_PATH = hf_hub_download(
    repo_id="Utkarsh566/brain-tumor-classification-model",
    filename="brain_tumor_model.pkl"
)

artifacts = joblib.load(MODEL_PATH)

svm_model = artifacts["svm_model"]
scaler = artifacts["scaler"]
class_names = artifacts["class_names"]


# -----------------------------
# 2. Feature extraction
# -----------------------------

def extract_features_from_gray(img):

    hog_features = hog(
        img,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    glcm = graycomatrix(
        img,
        distances=[1, 2],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    properties = [
        'contrast',
        'dissimilarity',
        'homogeneity',
        'energy',
        'correlation',
        'ASM'
    ]

    glcm_features = []

    for prop in properties:
        values = graycoprops(glcm, prop)
        glcm_features.append(values.mean())
        glcm_features.append(values.std())

    return np.concatenate([
        hog_features,
        glcm_features
    ])

# -----------------------------
# 3. Prediction function
# -----------------------------

def get_prediction_scores(gray_img):

    features = extract_features_from_gray(gray_img)

    features = features.reshape(1, -1)

    features_scaled = scaler.transform(features)

    decision_scores = svm_model.decision_function(
        features_scaled
    )

    return softmax(decision_scores, axis=1)[0]


def predict_image(gray_img):

    features = extract_features_from_gray(gray_img)

    features_scaled = scaler.transform(
        features.reshape(1, -1)
    )

    prediction = int(svm_model.predict(features_scaled)[0])

    scores = get_prediction_scores(gray_img)

    return prediction, scores

# -----------------------------
# 4. Occlusion heatmap
# -----------------------------

def generate_occlusion_heatmap(gray_img, predicted_label):

    original_scores = get_prediction_scores(gray_img)

    patch_size = 32
    stride = 16

    heatmap = np.zeros((224, 224), dtype=np.float32)
    count_map = np.zeros((224, 224), dtype=np.float32)

    for y in range(0, 224 - patch_size + 1, stride):

        for x in range(0, 224 - patch_size + 1, stride):

            occluded_img = gray_img.copy()

            occluded_img[
                y:y + patch_size,
                x:x + patch_size
            ] = 0

            new_scores = get_prediction_scores(occluded_img)

            score_change = (
                original_scores[predicted_label]
                - new_scores[predicted_label]
            )

            heatmap[
                y:y + patch_size,
                x:x + patch_size
            ] += score_change

            count_map[
                y:y + patch_size,
                x:x + patch_size
            ] += 1

    heatmap = heatmap / np.maximum(count_map, 1)

    return heatmap

# -----------------------------
# 5. Streamlit interface
# -----------------------------

st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification System")

st.write(
    "MRI classification using traditional machine learning "
    "with HOG and GLCM feature extraction."
)

st.info(
    "Research prototype only. This application is not "
    "clinically validated and must not be used for diagnosis."
)

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    original_img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    if original_img is None:

        st.error("Unable to read this image.")

    else:

        gray_img = cv2.resize(
            original_img,
            (224, 224)
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded MRI")
            st.image(
                gray_img,
                clamp=True,
                use_container_width=True
            )

        with st.spinner("Analyzing MRI image..."):

            predicted_label, scores = predict_image(gray_img)

        predicted_class = class_names[predicted_label]

        with col2:

            st.subheader("Prediction Result")

            st.success(
                f"Predicted class: {predicted_class}"
            )

            st.write("Normalized SVM decision scores:")

            for i, class_name in enumerate(class_names):

                st.write(
                    f"{class_name}: {scores[i] * 100:.2f}%"
                )

                st.progress(float(scores[i]))

        st.divider()

        # Occlusion explanation

        st.subheader("Occlusion Sensitivity Heatmap")

        if st.button("Generate Explanation"):

            with st.spinner(
                "Generating heatmap. Please wait..."
            ):

                heatmap = generate_occlusion_heatmap(
                    gray_img,
                    predicted_label
                )

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].imshow(gray_img, cmap="gray")
            axes[0].set_title("Original MRI")
            axes[0].axis("off")

            axes[1].imshow(gray_img, cmap="gray")
            axes[1].imshow(
                heatmap,
                cmap="jet",
                alpha=0.45
            )
            axes[1].set_title("Occlusion Heatmap")
            axes[1].axis("off")

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "The heatmap shows model sensitivity to image "
                "patches, not tumor boundaries."
            )
