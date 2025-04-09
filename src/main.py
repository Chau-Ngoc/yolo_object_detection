from pathlib import Path

import cv2 as cv
import onnxruntime as rt
import streamlit as st
import yaml

from src.utils import annotate_bounding_boxes, draw_bounding_boxes, get_preds_for_classes, uploaded_img_to_ndarray

v5_model = Path(__file__).resolve().parents[1] / "model" / "v5_runs" / "weights" / "best.onnx"
v10_model = Path(__file__).resolve().parents[1] / "model" / "v10_runs" / "weights" / "best.onnx"
models = {"v5": v5_model, "v10": v10_model}

if "error" not in st.session_state:
    st.session_state["error"] = False

if "preds" not in st.session_state:
    st.session_state["preds"] = []

if "blob" not in st.session_state:
    st.session_state["blob"] = None

if "orig_img" not in st.session_state:
    st.session_state["orig_img"] = None

if "finished_inferencing" not in st.session_state:
    st.session_state["finished_inferencing"] = False
with open("datasets.yaml", "r") as stream:
    yml = yaml.safe_load(stream)

classes_dict = {v: i for i, v in enumerate(yml["names"].values())}


def start_detection(model, orig_img, preprocessed_img, class_names):
    with st.spinner("Detecting...", show_time=True):
        class_idx = tuple([classes_dict[name] for name in class_names])
        session = rt.InferenceSession(model)
        input_name = session.get_inputs()[0].name

        try:
            preds = session.run(None, {input_name: preprocessed_img})[0]
            preds = get_preds_for_classes(preds[0], class_idx)
        except ValueError:
            st.session_state.error = True

        if not st.session_state.error:
            conf = preds[:, 4]
            bbox = preds[:, :4]
            indices = cv.dnn.NMSBoxes(bbox, conf, 0.25, 0.7)
            preds = preds[indices]

            bb_img = draw_bounding_boxes(orig_img, preds)
            annotated = annotate_bounding_boxes(bb_img, preds, list(classes_dict.keys()))
            st.session_state.preds = annotated
            st.session_state.finished_inferencing = True


st.set_page_config(page_title="Object detection")
st.title("Object detection")

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "png", "jpeg"], help="Upload an image to start detection"
    )
    if uploaded_file is not None:
        img = uploaded_img_to_ndarray(uploaded_file)
        img_blob = cv.dnn.blobFromImage(img, 1 / 255, size=(640, 640), swapRB=False, crop=False)
        st.session_state.blob = img_blob
        st.session_state.orig_img = img

    selected_model = st.radio(
        "Select your preferred AI model",
        ["v5", "v10"],
        format_func=lambda x: "YOLOv5 model" if x == "v5" else "YOLOv10 model",
    )
    classes = st.multiselect("Select which classes to detect", classes_dict, default=["car", "person"])
    st.button(
        "Start detection",
        help="Click to start detection",
        type="primary",
        on_click=start_detection,
        args=[models[selected_model], st.session_state.orig_img, st.session_state.blob, classes],
    )

if uploaded_file is not None:
    with st.empty():
        st.image(img, use_container_width=True, caption="Image")
        if st.session_state.finished_inferencing:
            st.image(st.session_state.preds, use_container_width=True, caption="Predictions")

if st.session_state.error:
    st.error(
        "An error has occurred. Please check that you have provide enough input (i.e. the image, the classes, the model)"
    )
    st.session_state.error = False
