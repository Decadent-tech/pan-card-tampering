import gradio as gr
from skimage.metrics import structural_similarity
import cv2
import numpy as np
from PIL import Image
import imutils
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt


def log_detection(ssim_score, verdict, tampered_regions):
    log_file = "detection_log.csv"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "ssim_score": round(ssim_score, 3),
        "verdict": verdict,
        "tampered_regions": tampered_regions
    }])

    if not os.path.exists(log_file):
        log_entry.to_csv(log_file, index=False)
    else:
        log_entry.to_csv(log_file, mode='a', header=False, index=False)


def download_log():
    return "detection_log.csv"

def generate_ssim_plot(log_file="detection_log.csv"):
    try:
        df = pd.read_csv(log_file)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        plt.figure(figsize=(10, 5))
        plt.plot(df["timestamp"], df["ssim_score"], marker='o', linestyle='-')
        plt.title("SSIM Score Trend Over Time")
        plt.xlabel("Time")
        plt.ylabel("SSIM Score")
        plt.xticks(rotation=45)
        plt.grid(True)

        plot_path = "ssim_trend.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    except Exception as e:
        print(f"Error in plotting: {e}")
        return None
    
def show_plot():
    path = generate_ssim_plot()
    return path

def filter_logs(start_date=None, end_date=None, gender=None):
    df = pd.read_csv("detection_log.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert empty strings to None for compatibility with Gradio Textbox
    if not start_date or str(start_date).strip() == '':
        start_date = None
    if not end_date or str(end_date).strip() == '':
        end_date = None
    if not gender or str(gender).strip() == '':
        gender = None

    # Only filter if the value is not None
    if start_date:
        # If user enters only date, match all times on that date
        start_date = pd.to_datetime(str(start_date)[:10])
        df = df[df["timestamp"] >= start_date]
    if end_date:
        # Add 1 day to include the whole end date
        end_date = pd.to_datetime(str(end_date)[:10]) + pd.Timedelta(days=1)
        df = df[df["timestamp"] < end_date]
    if gender:
        if "gender" in df.columns:
            df = df[df["gender"] == gender]

    return df.reset_index(drop=True)

def check_tampering(original_img, tampered_img):
    # Convert to OpenCV format
    original = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    tampered = cv2.cvtColor(np.array(tampered_img), cv2.COLOR_RGB2BGR)

    # Resize both images to the same size
    original = cv2.resize(original, (250, 160))
    tampered = cv2.resize(tampered, (250, 160))

    # Convert to grayscale
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

    # Compute SSIM
    score, diff = structural_similarity(original_gray, tampered_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold and contours
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Set minimum area to ignore noise
    min_area = 30  # pixels
    tampered_regions = 0

    for c in cnts:
        if cv2.contourArea(c) > min_area:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
            tampered_regions += 1

    # Verdict logic
    if score < 0.90 and tampered_regions > 0:
        verdict = f"Tampering Detected in {tampered_regions} region(s)!"
    else:
        verdict = "No Tampering Detected."
    log_detection(score, verdict, tampered_regions)
    output_path = "tampered_output.png"
    cv2.imwrite(output_path, tampered)
    result_img = cv2.cvtColor(tampered, cv2.COLOR_BGR2RGB)
    return f"{score:.3f}", verdict, output_path

# Gradio UI
interface = gr.Interface(
    fn=check_tampering,
    inputs=[
        gr.Image(type="pil", label="Original PAN"),
        gr.Image(type="pil", label="Suspected PAN")
    ],
    outputs=[
        gr.Textbox(label="SSIM Score"),
        gr.Textbox(label="Tampering Verdict"),
        gr.File(label="Download Highlighted Image")  # this adds download link!
    ],
    title="PAN Card Tampering Detection"
)

download_log_interface = gr.Interface(
    fn=download_log,
    inputs=[],
    outputs=gr.File(label="Download Detection Log"),
    title="Download Detection Log"
)

plot_interface = gr.Interface(
    fn=show_plot,
    inputs=[],
    outputs=gr.Image(type="filepath", label="SSIM Trend"),
    title="SSIM Score Trend"
)

filter_logs_interface = gr.Interface(
    fn=filter_logs,
    inputs=[
        gr.Textbox(label="Start Date (YYYY-MM-DD)"),
        gr.Textbox(label="End Date (YYYY-MM-DD)"),
        gr.Textbox(label="Gender (optional, if available)")
    ],
    outputs=gr.Dataframe(label="Filtered Logs"),
    title="Filter Detection Logs"
)

app = gr.TabbedInterface(
    [interface, download_log_interface, plot_interface, filter_logs_interface],
    ["Detection", "Download Log", "SSIM Trend", "Filter Logs"]
)

if __name__ == "__main__":
    app.launch(share=True)
    # If you want a standalone download log interface, uncomment the next line:
    # gr.Interface(fn=download_log, inputs=[], outputs=gr.File(label="Download Detection Logs")).launch()
