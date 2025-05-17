# =====================================
# IMPORT MODULES
# =====================================
import streamlit as st
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import time
import pandas as pd
import zipfile
import itertools
import matplotlib.pyplot as plt
import base64

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


st.set_page_config(
    page_title= "App",
    page_icon= "🧠",
    initial_sidebar_state= "collapsed"
)
st.sidebar.header("Main app")
with st.sidebar:
    with st.echo():
        st.write("""
                 This is the main app, functionality wise, users can predict using unlabeled CXR images in singles 
                 or in batches (multiple files, or ZIP). Users also can evaluate the models using labeled CXR images 
                 as instructed.
                 """)
# =====================================
# FUNCTIONS
# =====================================

@st.cache_resource
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def clean_single_image(image):
    image = np.array(image)
    image = np.array(Image.fromarray(image).resize((224, 224)))
    return image

def predict_with_model(model, img_input, class_mapping):
    start_time = time.time()
    pred = model.predict(img_input) # returns an array pred[confidence of[cl1, cl2, cl3, cl4]]
    inference_time = time.time() - start_time
    
    predicted_class_index = np.argmax(pred, axis=1)[0]
    
    confidence = pred[0][predicted_class_index]
    # Map the predicted index to the actual class label
    predicted_class = class_mapping[predicted_class_index]
    return predicted_class, confidence, inference_time

def compute_batch_params(ts_length):
    # Find a divisor of ts_length such that the number of steps (ts_length / batch_size) is ≤ 80.
    divisors = [ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and (ts_length/n) <= 80]
    test_batch_size = max(divisors) if divisors else ts_length
    test_steps = ts_length // test_batch_size
    return test_batch_size, test_steps

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    # Create a new figure and axes using subplots.
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    # Normalize the confusion matrix IF required.
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized Confusion Matrix')
    else:
        print('Confusion Matrix, Without Normalization')

    print(cm)

    thresh = cm.max() / 2.
    # Annotate each cell with its corresponding value.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], '.2f') if normalize else f'{cm[i, j]:d}',
                horizontalalignment='center',
                color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    return fig

# =====================================
# APP
# =====================================

class_mapping = {0: 'COVID', 1: 'Lung_Opacity', 2: 'Normal', 3: 'Viral Pneumonia'}

# Load your trained models (PATH SHOULD BE RE-CHECKED)
mobilenet = load_model(r'.\MODELS\mobilenetv3small-COVID-19-94.95.keras')
efficientnet = load_model(r'.\MODELS\efficientnetv2b0-COVID-19-95.81.keras')
resnet = load_model(r'.\MODELS\resnet50v2-COVID-19-94.19.keras')

# Remove Streamlit's default menu and footer for a cleaner interface
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# TITLE AND DESCRIPTION
st.title('COVID-19 Model Comparison')
st.write("Upload either a single or multiple unlabeled chest X-ray image (PNG/JPG/ZIP for multiple) to predict")
st.write("or you can upload multiple labeled chest X-ray images to evaluate each model as instructed below")

# Instruction
file_ = open(r".\asset\instruction.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

# Update uploader to accept both image files and .zip files
uploaded_file = st.file_uploader("Choose an image file or a ZIP archive", type=["png", "jpg", "jpeg", "zip"])

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith('.zip'):
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            # List all file names in the ZIP archive.
            all_files = zip_ref.namelist()
            # Check if the ZIP is organized by class folders.
            top_level_dirs = set(f.split('/')[0] for f in all_files if "/" in f)
            expected_dirs = set(["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"])
            ordered_dataset = expected_dirs.issubset(top_level_dirs)

            if ordered_dataset:
                st.write("**Ordered Batch Detected:** Processing images using folder names as true labels.")
                images = []
                true_labels = []
                for file in all_files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")) and "/" in file:
                        folder = file.split("/")[0]
                        if folder in expected_dirs:
                            file_bytes = zip_ref.read(file)
                            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                            image = clean_single_image(image)
                            images.append(image)
                            true_labels.append(folder)
                if len(images) == 0:
                    st.error("No valid images found in the ZIP file.")
                else:
                    images_array = np.array(images)
                    ts_length = len(images)
                    test_batch_size, test_steps = compute_batch_params(ts_length)
                    st.write(f"Processing {ts_length} images with batch size = {test_batch_size} and {test_steps} steps.")

                    # Process predictions for each model in the ordered dataset
                    for model_name, model in zip(
                        ["MobileNetV3Small", "EfficientNetV2B0", "ResNet50V2"],
                        [mobilenet, efficientnet, resnet]
                    ):
                        st.subheader(model_name)
                        start_time = time.time()
                        preds = model.predict(images_array, batch_size=test_batch_size)
                        inference_time = time.time() - start_time

                        pred_labels = []
                        for pred in preds:
                            idx = np.argmax(pred)
                            pred_labels.append(class_mapping[idx])

                        # Compute confusion matrix and classification metrics
                        class_order = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
                        cm = confusion_matrix(true_labels, pred_labels, labels=class_order)
                        fig = plot_confusion_matrix(cm, classes=class_order, normalize=False, title=f"{model_name} Confusion Matrix")
                        st.pyplot(fig)

                        # Generate classification report for macro average metrics
                        report = classification_report(true_labels, pred_labels, labels=class_order, output_dict=True)
                        accuracy = accuracy_score(true_labels, pred_labels)
                        precision = report['macro avg']['precision']
                        recall = report['macro avg']['recall']
                        f1 = report['macro avg']['f1-score']

                        st.write(f"**Total Inference Time:** {round(inference_time, 4)} seconds")
                        st.write(f"**Accuracy:** {round(accuracy, 4)}")
                        st.write(f"**Precision (Macro Avg):** {round(precision, 4)}")
                        st.write(f"**Recall (Macro Avg):** {round(recall, 4)}")
                        st.write(f"**F1 Score (Macro Avg):** {round(f1, 4)}")
            else: # if not ordered_dataset
                st.write("**Batch of Images Detected:** Processing images without predefined true labels.")
                images = []
                file_names = []
                for file in all_files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        file_bytes = zip_ref.read(file)
                        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
                        image = clean_single_image(image)
                        images.append(image)
                        file_names.append(file.split("/")[-1])
                if len(images) == 0:
                    st.error("No valid images found in the ZIP file.")
                else:
                    images_array = np.array(images)
                    ts_length = len(images)
                    test_batch_size, test_steps = compute_batch_params(ts_length)
                    st.write(f"Processing {ts_length} images with batch size = {test_batch_size} and {test_steps} steps.")

                    # For each model, predict over the batch and show per-image results
                    for model_name, model in zip(["MobileNetV3Small", "EfficientNetV2B0", "ResNet50V2"],
                                                 [mobilenet, efficientnet, resnet]):
                        st.subheader(model_name)
                        start_time = time.time()
                        preds = model.predict(images_array, batch_size=test_batch_size)
                        inference_time = time.time() - start_time

                        pred_labels = []
                        confidences = []
                        for pred in preds:
                            idx = np.argmax(pred)
                            pred_labels.append(class_mapping[idx])
                            confidences.append(round(float(pred[idx]) * 100, 2))

                        df_results = pd.DataFrame({
                            "File": file_names,
                            "Predicted Class": pred_labels,
                            "Confidence (%)": confidences
                        })

                        st.dataframe(df_results)
                        st.write(f"**Total Inference Time for batch:** {round(inference_time, 5)} seconds")
                        
    else:
        # Process as a single image 
        progress_text = st.text("Crunching Image...")
        progress_bar = st.progress(0)

        # Read and display the uploaded image (resized for visualization)
        image = Image.open(io.BytesIO(uploaded_file.read()))
        image = image.convert("RGB")
        display_image = np.array(Image.fromarray(np.array(image)).resize((700, 700)))
        left_co, cent_co, last_co = st.columns(3)
        with cent_co:
            st.image(display_image, caption="Uploaded Chest X-ray")
        progress_bar.progress(40)

        # Pre-process the image for model input
        image = clean_single_image(image)
        img_input = np.expand_dims(image, axis=0)

        # Dictionary to store predictions and inference times
        results = {"Model": [], "Predicted Class": [], "Confidence (%)": [], "Inference Time (s)": []}

        # Prediction and timing for MobileNetV3Small
        pred_class, confidence, time_taken = predict_with_model(mobilenet, img_input, class_mapping)
        results["Model"].append("MobileNetV3Small")
        results["Predicted Class"].append(pred_class)
        results["Confidence (%)"].append(round(confidence * 100, 2))
        results["Inference Time (s)"].append(round(time_taken, 4))
        progress_bar.progress(60)

        # Prediction and timing for EfficientNetV2B0
        pred_class, confidence, time_taken = predict_with_model(efficientnet, img_input, class_mapping)
        results["Model"].append("EfficientNetV2B0")
        results["Predicted Class"].append(pred_class)
        results["Confidence (%)"].append(round(confidence * 100, 2))
        results["Inference Time (s)"].append(round(time_taken, 4))
        progress_bar.progress(80)

        # Prediction and timing for ResNet50V2
        pred_class, confidence, time_taken = predict_with_model(resnet, img_input, class_mapping)
        results["Model"].append("ResNet50V2")
        results["Predicted Class"].append(pred_class)
        results["Confidence (%)"].append(round(confidence * 100, 2))
        results["Inference Time (s)"].append(round(time_taken, 4))
        progress_bar.progress(100)

        progress_text.text("Compiling results, please hold on...")

        # Display results using a table
        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

        progress_text.text("All done!")
        progress_text.empty()
        progress_bar.empty()