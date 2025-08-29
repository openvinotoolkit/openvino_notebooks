import math
import os
import cv2
import numpy as np
import requests
from pathlib import Path

import tensorflow as tf


# Utilities to open video files using CV2
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def download_with_headers(url, path):
    """Download file with proper headers to avoid 403 errors."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }

    path = Path(path)
    if not path.exists():
        print(f"Downloading {url} to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(path, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded to {path}")
        else:
            print(f"Failed to download: HTTP {response.status_code}")

    return path


def load_video(video_url, max_frames=32, resize=(224, 224)):
    if video_url.startswith("http"):
        path = tf.keras.utils.get_file(os.path.basename(video_url)[-128:], video_url, cache_dir=".", cache_subdir="data")
    else:
        path = video_url
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    frames = np.array(frames)
    if len(frames) < max_frames:
        n_repeat = int(math.ceil(max_frames / float(len(frames))))
        frames = frames.repeat(n_repeat, axis=0)
    frames = frames[:max_frames]
    return frames / 255.0


def display_query_and_results_video(query, urls, scores):
    """Display a text query and the top result videos and scores."""
    sorted_ix = np.argsort(-scores)
    html = ""
    html += "<h2>Input query: <i>{}</i> </h2><div>".format(query)
    html += "Results: <div>"
    html += "<table>"
    html += "<tr><th>Rank #1, Score:{:.2f}</th>".format(scores[sorted_ix[0]])
    html += "<th>Rank #2, Score:{:.2f}</th>".format(scores[sorted_ix[1]])
    html += "<th>Rank #3, Score:{:.2f}</th></tr><tr>".format(scores[sorted_ix[2]])
    for i, idx in enumerate(sorted_ix):
        url = urls[sorted_ix[i]]
        html += "<td>"
        html += '<img src="{}" height="224">'.format(url)
        html += "</td>"
    html += "</tr></table>"

    return html
