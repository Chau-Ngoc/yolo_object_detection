import cv2 as cv
import numpy as np


def uploaded_img_to_ndarray(uploaded_image) -> np.ndarray:
    """According to Streamlit, the uploaded image is an object of type
    UploadedFile.

    We need to convert it into a numpy array by getting the bytes from
    the UploadedFile. The converted image will be in RGB color format.
    """
    bytes_data = uploaded_image.getvalue()
    image = cv.imdecode(np.frombuffer(bytes_data, np.uint8), cv.IMREAD_COLOR)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image_rgb


def annotate_bounding_boxes(
    img: np.ndarray,
    preds: np.ndarray,
    classes: list[str],
    font_face=cv.FONT_HERSHEY_PLAIN,
    font_scale=1,
    thickness=1,
):
    """Annotate the bounding boxes for each element in preds.

    This function does not annotate directly onto the passed image.
    Instead, it will copy the passed image and annotate onto the copied
    version.
    """
    new_img = img.copy()
    h, w, _ = new_img.shape
    fx = w / 640
    fy = h / 640
    for pred in preds:
        x1, y1, x2, y2 = pred[:4].astype(int)
        x1, x2 = int(x1 * fx), int(x2 * fx)
        y1, y2 = int(y1 * fy), int(y2 * fy)
        conf = pred[4] * 100
        class_id = pred[5].astype(int)
        class_label = classes[class_id]

        text = f"{class_label}: {conf:.2f}%"
        (text_w, text_h), _ = cv.getTextSize(text, font_face, font_scale, thickness)
        pad = text_h // 4
        cv.rectangle(new_img, (x1, y1), (x1 + text_w, y1 - text_h - pad), (250, 250, 250), cv.FILLED)
        cv.putText(new_img, text, (x1, y1 - pad), font_face, font_scale, (1, 1, 1), thickness=thickness)

    return new_img


def draw_bounding_boxes(img: np.ndarray, preds: np.ndarray):
    """Draw the bounding boxes for each element in preds.

    This function does not draw directly onto the passed image. Instead
    it will copy the passed image and draw onto the copied version.
    """
    new_img = img.copy()
    h, w, _ = new_img.shape
    fx = w / 640
    fy = h / 640

    for pred in preds:
        x1, y1, x2, y2 = pred[:4].astype(int)
        x1, x2 = int(x1 * fx), int(x2 * fx)
        y1, y2 = int(y1 * fy), int(y2 * fy)
        class_id = pred[5].astype(int)
        cv.rectangle(new_img, (x1, y1), (x2, y2), random_assign_class_to_color(class_id), 2)
    return new_img


def random_assign_class_to_color(class_id: int, seed: int = 42) -> tuple[int, int, int]:
    """Randomly assign a color to a class given the class_id."""
    rng = np.random.default_rng(seed)
    r = int(np.exp(class_id + rng.integers(class_id, class_id + 255)) % 255)
    g = int(np.exp(class_id + rng.integers(class_id, class_id + 255)) % 255)
    b = int(np.exp(class_id + rng.integers(class_id, class_id + 255)) % 255)
    return r, g, b


def get_preds_for_classes(preds: np.ndarray, classes: tuple) -> np.ndarray:
    """Return the new array that contains only the predictions of the specified
    classes."""
    indices = np.isin(preds[:, 5], classes)
    return preds[indices]
