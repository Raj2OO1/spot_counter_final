from flask import Flask, request, jsonify, render_template, send_from_directory, session, flash, redirect, url_for, Response
from werkzeug.utils import secure_filename
import uuid, csv
import os
import io
import base64
import json
import sys
# Pillow
from PIL import Image, ImageEnhance, ImageOps, ImageDraw, ImageFont

from datetime import datetime
# SciPy/NumPy
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Rolling-ball / morphological operations
from skimage.morphology import disk
import cv2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'replace-this-with-a-random-secret'


@app.before_request
def ensure_log():
    session.setdefault('final_settings', {})


UPLOAD_FOLDER = os.path.abspath('./uploads')
PROCESSED_FOLDER = os.path.abspath('./processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# ----------------------------------------------------------------------
# file_states[filename] structure:
# {
#   'original_image': PIL.Image,
#   'current_image': PIL.Image,
#   'contrast': float,
#   'gamma': float,
#   'rotate': float,
#   'sharpen': float,
#   'grayscale': bool,
#   'negative': bool,
#   'lane_boundaries': [ { 'left': int, 'right': int, 'label': str }, ... ],
#   'band_boundaries': { lane_index (int): [ { 'top': int, 'bottom': int, 'label': str, ... }, ... ] },
#   # Calibration (if computed):
#   # 'baseline_row': int,
#   # 'ladder_lane_index': int,
#   # 'calibration': { 'slope': float, 'intercept': float, 'r2': float, 'plot_b64': str }
# }
file_states = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


def pil_to_base64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def apply_basic_adjustments(image, state):
    """Apply basic image adjustments (contrast, gamma, sharpen, grayscale, negative)"""
    if state['grayscale']:
        image = image.convert('L')

    if state['negative']:
        image = ImageOps.invert(image)

    if state['contrast'] != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(state['contrast'])

    if state['gamma'] != 1.0:
        g = state['gamma']
        image = image.point(lambda p: 255 * ((p / 255) ** (1 / g)))

    if state['sharpen'] != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(state['sharpen'])

    return image


def apply_circle_detection(image, state):
    """Apply Circle Detection"""
    # Convert PIL to numpy array
    img_array = np.array(image)
    if img_array.shape[-1] == 4: # Handle RGBA
        img_array = np.array(image.convert('RGB'))

    x = state['x']
    y = state['y']
    r = state['r']

    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if r > 0:
        cv2.circle(mask, (x, y), r, 255, -1)

    # Create background based on user selection
    if state['circleBackground'] == 'white':
        background = np.ones_like(img_array) * 255
    elif state['circleBackground'] == 'black':
        background = np.zeros_like(img_array)
    else:  # original
        background = img_array.copy()

    # Apply mask
    if len(img_array.shape) == 3:
        mask_3d = np.stack([mask] * img_array.shape[2], axis=-1)
        result = np.where(mask_3d == 255, img_array, background)
    else:
        result = np.where(mask == 255, img_array, background)

    return Image.fromarray(result.astype(np.uint8))


def apply_binary_conversion(image, state):
    """Apply binary conversion using specified method"""
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')

    # Convert to numpy array
    img_array = np.array(image)

    method = state['binaryMethod']

    if method == 'threshold':
        _, binary = cv2.threshold(img_array, int(state['thresholdValue']), 255, cv2.THRESH_BINARY)
    elif method == 'adaptive':
        # Ensure block size is odd
        block_size = int(state['blockSize'])
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            img_array, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, int(state['cConstant'])
        )
    elif method == 'otsu':
        _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = img_array

    return Image.fromarray(binary)


def apply_all_tools(state):
    """Apply all image processing tools in the correct order"""
    image = state['original_image'].copy()

    # 1. Apply basic adjustments first
    image = apply_basic_adjustments(image, state)

    # 2. Apply circle detection if enabled
    if state.get('enableCircleDetection', False):
        image = apply_circle_detection(image, state)

    # 3. Apply binary conversion if enabled (should be last)
    if state.get('enableBinaryConversion', False):
        image = apply_binary_conversion(image, state)

    state['current_image'] = image
    return image


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pil_img = Image.open(file_path).convert('RGB')
        file_states[filename] = {
            'original_image': pil_img.copy(),
            'current_image': pil_img.copy(),
            # Basic adjustments
            'contrast': 1.0,
            'gamma': 1.0,
            'sharpen': 1.0,
            'grayscale': False,
            'negative': False,
            # Circle detection
            'enableCircleDetection': False,
            'circleBackground': 'white',
            'x': 100,
            'y': 100,
            'r': 100,
            # Binary conversion
            'enableBinaryConversion': False,
            'binaryMethod': 'threshold',
            'thresholdValue': 127,
            'blockSize': 11,
            'cConstant': 2,
            # Other features
            'lane_boundaries': [],
            'band_boundaries': {}
        }
        session['orig_image'] = filename
        return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200
    else:
        return jsonify({'error': 'Invalid file format'}), 400


# ========== Image Preparation ==========
@app.route('/process_prep', methods=['POST'])
def process_prep():
    filename = request.form.get('filename')
    if not filename or filename not in file_states:
        return jsonify({'error': 'File state not found'}), 404
    state = file_states[filename]
    try:
        # Basic adjustments
        state['contrast'] = float(request.form.get('contrast', 1.0))
        state['gamma'] = float(request.form.get('gamma', 1.0))
        state['sharpen'] = float(request.form.get('sharpen', 1.0))
        state['grayscale'] = (request.form.get('grayscale') == 'true')
        state['negative'] = (request.form.get('negative') == 'true')

        # Circle detection parameters
        state['enableCircleDetection'] = (request.form.get('enableCircleDetection') == 'true')
        state['circleBackground'] = request.form.get('circleBackground', 'white')
        state['x'] = int(float(request.form.get('x', 100)))
        state['y'] = int(float(request.form.get('y', 100)))
        state['r'] = int(float(request.form.get('r', 100)))

        # Binary conversion parameters
        state['enableBinaryConversion'] = (request.form.get('enableBinaryConversion') == 'true')
        state['binaryMethod'] = request.form.get('binaryMethod', 'threshold')
        state['thresholdValue'] = int(request.form.get('thresholdValue', 127))
        state['blockSize'] = int(request.form.get('blockSize', 11))
        state['cConstant'] = int(request.form.get('cConstant', 2))
    except Exception as e:
        return jsonify({'error': f'Invalid param: {e}'}), 400
    try:
        # Apply all transformations
        processed_image = apply_all_tools(state)

        # Save processed image
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
        processed_image.save(processed_path)

        # Update session settings
        session['processed_image'] = filename
        session['final_settings']['prep'] = {
            'contrast': state['contrast'],
            'gamma': state['gamma'],
            'sharpen': state['sharpen'],
            'grayscale': state['grayscale'],
            'negative': state['negative'],
            'enableCircleDetection': state['enableCircleDetection'],
            'circleBackground': state['circleBackground'],
            'x': state['x'],
            'y': state['y'],
            'r': state['r'],
            'enableBinaryConversion': state['enableBinaryConversion'],
            'binaryMethod': state['binaryMethod'],
            'thresholdValue': state['thresholdValue'],
            'blockSize': state['blockSize'],
            'cConstant': state['cConstant']
        }

        return jsonify({
            'message': 'Image processing completed successfully',
            'filename': filename
        }), 200

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/processed/<filename>')
def serve_processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)


@app.route('/api/image/export')
def export_image():
    t = request.args.get('type')
    filename = session.get('orig_image')
    state = file_states.get(filename)
    if not state:
        return jsonify({'error':'No state'}), 404

    if t == 'prep-overlay':
        img = state['current_image']
        outname = 'prepared_image.tiff'
    buf = io.BytesIO()
    img.save(buf, format='TIFF')
    buf.seek(0)
    return Response(buf.read(),
                    mimetype='image/tiff',
                    headers={'Content-Disposition': f'attachment;filename={outname}'})

# ========== Spot Detection Utilities ==========


directions = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]


def flood_fill(x, y, label, image, labeled_image):
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if 0 <= cx < image.shape[0] and 0 <= cy < image.shape[1] and image[cx, cy] == 255 and labeled_image[cx, cy] == 0:
            labeled_image[cx, cy] = label
            for dx, dy in directions:
                stack.append((cx+dx, cy+dy))


def is_within_tolerance(dim, ref_dim, tolerance=0.5):
    lower = ref_dim * (1 - tolerance)
    upper = ref_dim * (1 + tolerance)
    return lower <= dim <= upper


def apply_spot_counting(image, ref_height=20, ref_width=20, remove_background=True):
    """Count spots in a binary (0/255) grayscale image."""
    if image is None or image.size == 0:
        return 0

    rows, cols = image.shape
    labeled_image = np.zeros_like(image, dtype=int)

    if remove_background:
        outer_x, outer_y = -1, -1
        for i in range(rows):
            for j in range(cols):
                if i in (0, rows - 1) or j in (0, cols - 1):
                    if image[i, j] == 255:
                        outer_x, outer_y = i, j
                        break
            if outer_x != -1:
                break

        if outer_x != -1:
            flood_fill(outer_x, outer_y, -1, image, labeled_image)

    label_counter = 1
    spot_counts = 0

    for i in range(rows):
        for j in range(cols):
            if image[i, j] == 255 and labeled_image[i, j] != -1:
                flood_fill(i, j, label_counter, image, labeled_image)
                x, y, w, h = cv2.boundingRect(np.uint8(labeled_image == label_counter))
                if is_within_tolerance(h, ref_height) and is_within_tolerance(w, ref_width):
                    spot_counts += 1
                label_counter += 1

    return spot_counts


@app.route('/spot_detection')
def spot_detection_page():
    filename = session.get('processed_image')
    if not filename:
        flash("No processed image found. Please prepare an image first.")
        return redirect(url_for('home'))
    return render_template('spot_detection.html', filename=filename)


@app.route('/api/spot_count', methods=['POST'])
def api_spot_count():
    filename = session.get('processed_image')
    if not filename or filename not in file_states:
        return jsonify({'error':'No processed image'}), 404

    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error':'Failed to load image'}), 500

    return jsonify({'spots': apply_spot_counting(image)})


@app.route('/api/spot_filter', methods=['POST'])
def api_spot_filter():
    data = request.json
    x1, y1 = int(data['x1']), int(data['y1'])
    x2, y2 = int(data['x2']), int(data['y2'])

    filename = session.get('processed_image')
    if not filename:
        return jsonify({'error':'No processed image'}), 404

    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error':'Failed to load image'}), 500

    # crop selected region
    roi = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    if roi.size == 0:
        return jsonify({'spots': 0})

    return jsonify({'spots': apply_spot_counting(roi, remove_background=False)})

@app.route("/count_spots", methods=["POST"])
def count_spots():
    rect = request.get_json() or {}

    filename = session.get('processed_image')
    if not filename:
        return jsonify({'error': 'No processed image'}), 404

    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Failed to load image'}), 500

    try:
        x = int(rect.get('x', 0))
        y = int(rect.get('y', 0))
        w = int(rect.get('w', image.shape[1]))
        h = int(rect.get('h', image.shape[0]))
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid region values'}), 400

    x1 = max(0, min(x, x + w))
    x2 = min(image.shape[1], max(x, x + w))
    y1 = max(0, min(y, y + h))
    y2 = min(image.shape[0], max(y, y + h))

    if x1 >= x2 or y1 >= y2:
        return jsonify({'count': 0})

    roi = image[y1:y2, x1:x2]
    remove_background = (x1 == 0 and y1 == 0 and x2 == image.shape[1] and y2 == image.shape[0])
    count = apply_spot_counting(roi, remove_background=remove_background)

    return jsonify({'count': int(count)})


if __name__ == '__main__':
    app.run(debug=True)
