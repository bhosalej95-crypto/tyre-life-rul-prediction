
import os
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# ----------------------------------
# 1. App configuration
# ----------------------------------
st.set_page_config(
    page_title="Tyre Life Detection and RUL Prediction",
    page_icon="🚗",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_MODEL_PATH = hf_hub_download(
    repo_id="jeetendra-bhosale-2026/tyre-resnet50-model",
    filename="tyre_resnet50_finetuned.keras"
)
)
RUL_MODEL_PATH = os.path.join(
    BASE_DIR, "tyre_rul_model.pkl"
)
FEATURES_PATH = os.path.join(
    BASE_DIR, "tyre_rul_features.pkl"
)

IMAGE_SIZE = (224, 224)
MAX_DISTANCE_KM = 113000.0
MAX_RUL_KM = 96752.0

REQUIRED_FEATURES = [
    "current_tread_depth(mm)",
    "tyre_age(years)",
    "kilometers_driven(km)",
    "average_inflation_pressure(psi)"
]

# Remember predictions and form values between reruns.
if "predicted_distance_km" not in st.session_state:
    st.session_state["predicted_distance_km"] = None

if "kilometers_driven_input" not in st.session_state:
    st.session_state["kilometers_driven_input"] = 25000.0


# ----------------------------------
# 2. Load saved models
# ----------------------------------
@st.cache_resource
def load_image_model():
    return tf.keras.models.load_model(IMAGE_MODEL_PATH)


@st.cache_resource
def load_rul_model():
    return joblib.load(RUL_MODEL_PATH)


@st.cache_resource
def load_feature_names():
    return joblib.load(FEATURES_PATH)


st.title("🚗 Tyre Life Detection and RUL Prediction")

st.write(
    "Estimate the distance travelled from a tyre image "
    "and estimate remaining useful life from tyre measurements."
)

st.info(
    "These predictions are experimental estimates. "
    "They do not replace a professional tyre inspection."
)


# ----------------------------------
# 3. Load the image model
# ----------------------------------
image_model = None

try:
    if not os.path.isfile(IMAGE_MODEL_PATH):
        raise FileNotFoundError(IMAGE_MODEL_PATH)

    image_model = load_image_model()
    st.success("MobileNetV2 model loaded successfully!")

except Exception as e:
    st.error(f"Could not load image model: {e}")


# ----------------------------------
# 4. Image-based distance prediction
# ----------------------------------
st.header("1. Tyre Distance Prediction")

st.write(
    "Upload a tyre image to estimate the distance "
    "already travelled."
)

uploaded_file = st.file_uploader(
    "Choose a tyre image",
    type=["jpg", "jpeg", "png"],
    key="tyre_image"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded tyre image",
            width=350
        )

        if st.button(
            "Predict Distance Travelled",
            type="primary"
        ):
            if image_model is None:
                st.error("The image model is unavailable.")

            else:
                with st.spinner("Analysing tyre image..."):
                    resized_image = image.resize(IMAGE_SIZE)

                    image_array = np.asarray(
                        resized_image,
                        dtype=np.float32
                    )

                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    # This model expects raw pixel values.
                    # Its trained architecture handles rescaling.
                    prediction = image_model.predict(
                        image_array, verbose=0
                    )

                    predicted_scaled = float(
                        np.asarray(prediction).reshape(-1)[0]
                    )

                    predicted_distance = float(
                        np.clip(
                            predicted_scaled * MAX_DISTANCE_KM,
                            0,
                            MAX_DISTANCE_KM
                        )
                    )

                    st.session_state[
                        "predicted_distance_km"
                    ] = predicted_distance

                st.success("Image prediction completed!")

    except Exception as e:
        st.error(f"Could not process image: {e}")


# Display the stored image prediction after the button reruns the app.
predicted_distance = st.session_state[
    "predicted_distance_km"
]

if predicted_distance is not None:
    st.metric(
        "Estimated Distance Already Travelled",
        f"{predicted_distance:,.0f} km"
    )

    st.caption(
        "This is an image-model estimate, not a verified "
        "odometer reading."
    )


st.divider()


# ----------------------------------
# 5. Numerical RUL prediction
# ----------------------------------
st.header("2. Remaining Useful Life (RUL) Prediction")

st.write(
    "Enter the tyre measurements to estimate remaining "
    "useful life in kilometres."
)

try:
    if not os.path.isfile(RUL_MODEL_PATH):
        raise FileNotFoundError(RUL_MODEL_PATH)

    if not os.path.isfile(FEATURES_PATH):
        raise FileNotFoundError(FEATURES_PATH)

    rul_model = load_rul_model()
    feature_names = load_feature_names()

    if isinstance(feature_names, dict):
        feature_names = feature_names.get(
            "feature_names",
            feature_names.get("features")
        )

    if feature_names is None:
        raise ValueError(
            "Feature names were not found in the saved file."
        )

    feature_names = list(feature_names)

    if (
        len(feature_names) != len(REQUIRED_FEATURES)
        or set(feature_names) != set(REQUIRED_FEATURES)
    ):
        raise ValueError(
            "The saved feature list does not match the "
            "four features expected by this application. "
            f"Saved features: {feature_names}"
        )

    st.success("Numerical RUL model loaded successfully!")

    # Optional: copy the image prediction into the
    # kilometres-driven field. You can still edit it.
    if predicted_distance is not None:
        if st.button("Use image estimate for kilometres driven"):
            st.session_state[
                "kilometers_driven_input"
            ] = float(predicted_distance)

            st.rerun()

    with st.form("rul_prediction_form"):

        tread_depth = st.number_input(
            "Current tread depth (mm)",
            min_value=0.0,
            max_value=30.0,
            value=5.5,
            step=0.1
        )

        tyre_age = st.number_input(
            "Tyre age (years)",
            min_value=0.0,
            max_value=50.0,
            value=2.0,
            step=0.5
        )

        kilometres_driven = st.number_input(
            "Kilometres already driven (km)",
            min_value=0.0,
            max_value=MAX_DISTANCE_KM,
            step=1000.0,
            key="kilometers_driven_input"
        )

        inflation_pressure = st.number_input(
            "Average inflation pressure (psi)",
            min_value=0.0,
            max_value=100.0,
            value=32.0,
            step=0.5
        )

        predict_rul = st.form_submit_button(
            "Predict Remaining Useful Life",
            type="primary"
        )

    if predict_rul:
        input_values = {
            "current_tread_depth(mm)": tread_depth,
            "tyre_age(years)": tyre_age,
            "kilometers_driven(km)": kilometres_driven,
            "average_inflation_pressure(psi)": inflation_pressure
        }

        # Preserve the feature order used during training.
        input_df = pd.DataFrame(
            [[input_values[name] for name in feature_names]],
            columns=feature_names
        )

        with st.spinner("Predicting remaining useful life..."):
            rul_prediction = rul_model.predict(input_df)

            predicted_rul = float(
                np.asarray(rul_prediction).reshape(-1)[0]
            )

            predicted_rul = float(
                np.clip(predicted_rul, 0, MAX_RUL_KM)
            )

        st.subheader("Remaining Useful Life Result")

        st.metric(
            "Estimated Remaining Useful Life",
            f"{predicted_rul:,.0f} km"
        )

        st.caption(
            "This estimate depends on the training data and "
            "the accuracy of the measurements. It is not a "
            "guarantee of tyre life or safety."
        )

except Exception as e:
    st.error(f"Could not load or use the RUL model: {e}")
