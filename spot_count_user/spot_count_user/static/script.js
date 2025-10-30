// static/script.js

let uploadedFilename = null;

// 1) Upload
const uploadForm = document.getElementById('upload-form');
const uploadPreview = document.getElementById('upload-preview');
if (uploadForm) {
  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) {
      uploadPreview.src = URL.createObjectURL(fileInput.files[0]);
      uploadPreview.style.display = 'block';
    } else {
      uploadPreview.src = "";
      uploadPreview.style.display = 'none';
    }
  });

  uploadForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!fileInput.files[0]) {
      alert('Please select a file.');
      return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
      const response = await fetch('/upload', { method:'POST', body:formData });
      const result = await response.json();
      if (response.ok) {
        alert(result.message);
        uploadedFilename = result.filename;
      } else {
        alert(result.error);
      }
    } catch (err) {
      console.error(err);
      alert('Error uploading file');
    }
  });
}

// 2) Navigation
const uploadMenu = document.getElementById('upload-menu');
const prepMenu = document.getElementById('prep-menu');
const laneLink = document.getElementById('spot-detect');

if (uploadMenu) {
  uploadMenu.addEventListener('click', () => {
    showSection('upload-section');
  });
}
if (prepMenu) {
  prepMenu.addEventListener('click', () => {
    if (!uploadedFilename) {
      alert('No file uploaded');
      return;
    }
    showSection('prep-tools-section');
    loadPrepState();
  });
}
if (laneLink) {
  laneLink.addEventListener('click', (event) => {
    event.preventDefault();
    if (!uploadedFilename) {
      alert('No file uploaded');
      return;
    }
    localStorage.setItem('uploadedFilename', uploadedFilename);
    window.location.href = laneLink.href;
  });
}



function showSection(secId) {
  document.getElementById('upload-section').style.display='none';
  document.getElementById('prep-tools-section').style.display='none';
  const sec = document.getElementById(secId);
  if (sec) sec.style.display='block';
}

// 3) Enhanced Prep State with Circle Detection and Binary Conversion
let currentPrepState = {
    contrast: 1.0,
    gamma: 1.0,
    sharpen: 1.0,
    grayscale: false,
    negative: false,
    // Circle detection
    enableCircleDetection: false,
    circleBackground: 'white',
    x: 100,
    y: 100,
    r: 100,
    // Binary conversion
    enableBinaryConversion: false,
    binaryMethod: 'threshold',
    thresholdValue: 127,
    blockSize: 11,
    cConstant: 2
};

let prepHistory = [];
let prepRedoStack = [];

const originalPreview = document.getElementById('original-preview');
const processedPreview = document.getElementById('processed-preview');
const processingStatus = document.getElementById('processing-status');

// Basic adjustment controls
const contrastSlider = document.getElementById('contrast-slider');
const contrastInput = document.getElementById('contrast-input');
const gammaSlider = document.getElementById('gamma-slider');
const gammaInput = document.getElementById('gamma-input');
const sharpenSlider = document.getElementById('sharpen-slider');
const sharpenInput = document.getElementById('sharpen-input');
const grayscaleBtn = document.getElementById('grayscale-button');
const negativeBtn = document.getElementById('negative-button');
const prepUndoBtn = document.getElementById('prep-undo');
const prepRedoBtn = document.getElementById('prep-redo');

// Circle detection controls
const enableCircleDetection = document.getElementById('enable-circle-detection');
const circleParams = document.getElementById('circle-params');
const centerXSlider = document.getElementById('center-x-slider');
const centerXInput = document.getElementById('center-x-input');
const centerYSlider = document.getElementById('center-y-slider');
const centerYInput = document.getElementById('center-y-input');
const radiusSlider = document.getElementById('circle-radius');
const radiusInput = document.getElementById('radius-input');

// Background selection
const bgOriginal = document.getElementById('bg-original');
const bgWhite = document.getElementById('bg-white');
const bgBlack = document.getElementById('bg-black');

// Binary conversion controls
const enableBinaryConversion = document.getElementById('enable-binary-conversion');
const binaryParams = document.getElementById('binary-params');
const binaryMethod = document.getElementById('binary-method');
const thresholdParams = document.getElementById('threshold-params');
const adaptiveParams = document.getElementById('adaptive-params');
const thresholdValueSlider = document.getElementById('threshold-value-slider');
const thresholdValueInput = document.getElementById('threshold-value-input');
const blockSizeSlider = document.getElementById('block-size-slider');
const blockSizeInput = document.getElementById('block-size-input');
const cConstantSlider = document.getElementById('c-constant-slider');
const cConstantInput = document.getElementById('c-constant-input');

function loadPrepState() {
    if (uploadedFilename) {
        originalPreview.src = `/uploads/${uploadedFilename}?t=${Date.now()}`;
        if (allDefaults(currentPrepState)) {
            processedPreview.src = "";
            processedPreview.alt = "No transformations yet";
        } else {
            processedPreview.src = `/processed/${uploadedFilename}?t=${Date.now()}`;
        }
        updatePrepUI();
    }
}

function allDefaults(st) {
    return (st.contrast===1.0 && st.gamma===1.0 && st.sharpen===1.0 &&
            !st.grayscale && !st.negative && !st.enableCircleDetection &&
            !st.enableBinaryConversion);
}

function updatePrepUI() {
    // Basic adjustments
    contrastSlider.value = currentPrepState.contrast;
    contrastInput.value = currentPrepState.contrast;
    gammaSlider.value = currentPrepState.gamma;
    gammaInput.value = currentPrepState.gamma;
    sharpenSlider.value = currentPrepState.sharpen;
    sharpenInput.value = currentPrepState.sharpen;

    // Circle detection
    enableCircleDetection.checked = currentPrepState.enableCircleDetection;
    circleParams.style.display = currentPrepState.enableCircleDetection ? 'block' : 'none';

    centerXSlider.value = currentPrepState.x;
    centerXInput.value = currentPrepState.x;
    centerYSlider.value = currentPrepState.y;
    centerYInput.value = currentPrepState.y;
    radiusSlider.value = currentPrepState.r;
    radiusInput.value = currentPrepState.r;

    // Background selection
    const bgRadio = document.querySelector(`input[name="circle-background"][value="${currentPrepState.circleBackground}"]`);
    if(bgRadio) bgRadio.checked = true;

    // Binary conversion
    enableBinaryConversion.checked = currentPrepState.enableBinaryConversion;
    binaryParams.style.display = currentPrepState.enableBinaryConversion ? 'block' : 'none';
    binaryMethod.value = currentPrepState.binaryMethod;

    updateBinaryMethodUI();

    thresholdValueSlider.value = currentPrepState.thresholdValue;
    thresholdValueInput.value = currentPrepState.thresholdValue;
    blockSizeSlider.value = currentPrepState.blockSize;
    blockSizeInput.value = currentPrepState.blockSize;
    cConstantSlider.value = currentPrepState.cConstant;
    cConstantInput.value = currentPrepState.cConstant;
}

function updateBinaryMethodUI() {
    const method = currentPrepState.binaryMethod;
    thresholdParams.style.display = (method === 'threshold') ? 'block' : 'none';
    adaptiveParams.style.display = (method === 'adaptive') ? 'block' : 'none';
}

function pushPrepHistory() {
    const snap = JSON.parse(JSON.stringify(currentPrepState));
    prepHistory.push(snap);
}

function showProcessingStatus(show) {
    if (processingStatus) {
        processingStatus.style.display = show ? 'block' : 'none';
    }
}

function applyPrepState() {
    if (!uploadedFilename) return;

    showProcessingStatus(true);

    const formData = new FormData();
    formData.append('filename', uploadedFilename);

    // Basic adjustments
    formData.append('contrast', currentPrepState.contrast);
    formData.append('gamma', currentPrepState.gamma);
    formData.append('sharpen', currentPrepState.sharpen);
    formData.append('grayscale', currentPrepState.grayscale);
    formData.append('negative', currentPrepState.negative);

    // Circle detection
    formData.append('enableCircleDetection', currentPrepState.enableCircleDetection);
    formData.append('circleBackground', currentPrepState.circleBackground);
    formData.append('x', currentPrepState.x);
    formData.append('y', currentPrepState.y);
    formData.append('r', currentPrepState.r);

    // Binary conversion
    formData.append('enableBinaryConversion', currentPrepState.enableBinaryConversion);
    formData.append('binaryMethod', currentPrepState.binaryMethod);
    formData.append('thresholdValue', currentPrepState.thresholdValue);
    formData.append('blockSize', currentPrepState.blockSize);
    formData.append('cConstant', currentPrepState.cConstant);

    fetch('/process_prep', { method:'POST', body: formData })
    .then(res => res.json())
    .then(data => {
        showProcessingStatus(false);
        if (data.error) {
            console.error(data.error);
            alert('Processing error: ' + data.error);
        } else {
            if (allDefaults(currentPrepState)) {
                processedPreview.src = "";
                processedPreview.alt = "No transformations yet";
            } else {
                processedPreview.src = `/processed/${data.filename}?t=${Date.now()}`;
            }
        }
    })
    .catch(err => {
        showProcessingStatus(false);
        console.error(err);
        alert('Network error during processing');
    });
}

function onPrepChange() {
    pushPrepHistory();
    prepRedoStack = [];
    applyPrepState();
}

function setupPrepParam(slider, input, key) {
    if (slider && input) {
        slider.addEventListener('input', () => {
            currentPrepState[key] = parseFloat(slider.value);
            input.value = slider.value;
            onPrepChange();
        });
        input.addEventListener('change', () => {
            currentPrepState[key] = parseFloat(input.value);
            slider.value = input.value;
            onPrepChange();
        });
    }
}

// Setup basic parameter controls
setupPrepParam(contrastSlider, contrastInput, 'contrast');
setupPrepParam(gammaSlider, gammaInput, 'gamma');
setupPrepParam(sharpenSlider, sharpenInput, 'sharpen');

// Setup Circle parameter controls
setupPrepParam(centerXSlider, centerXInput, 'x');
setupPrepParam(centerYSlider, centerYInput, 'y');
setupPrepParam(radiusSlider, radiusInput, 'r');

// Setup binary conversion parameter controls
setupPrepParam(thresholdValueSlider, thresholdValueInput, 'thresholdValue');
setupPrepParam(blockSizeSlider, blockSizeInput, 'blockSize');
setupPrepParam(cConstantSlider, cConstantInput, 'cConstant');

// Toggle controls
if (grayscaleBtn) {
    grayscaleBtn.addEventListener('click', () => {
        pushPrepHistory();
        prepRedoStack=[];
        currentPrepState.grayscale = !currentPrepState.grayscale;
        onPrepChange();
    });
}

if (negativeBtn) {
    negativeBtn.addEventListener('click', () => {
        pushPrepHistory();
        prepRedoStack=[];
        currentPrepState.negative = !currentPrepState.negative;
        onPrepChange();
    });
}

// Circle detection toggle
if (enableCircleDetection) {
    enableCircleDetection.addEventListener('change', () => {
        pushPrepHistory();
        prepRedoStack=[];
        currentPrepState.enableCircleDetection = enableCircleDetection.checked;
        circleParams.style.display = enableCircleDetection.checked ? 'block' : 'none';
        onPrepChange();
    });
}

// Background selection
[bgOriginal, bgWhite, bgBlack].forEach(radio => {
    if (radio) {
        radio.addEventListener('change', () => {
            if (radio.checked) {
                pushPrepHistory();
                prepRedoStack=[];
                currentPrepState.circleBackground = radio.value;
                onPrepChange();
            }
        });
    }
});

// Binary conversion toggle
if (enableBinaryConversion) {
    enableBinaryConversion.addEventListener('change', () => {
        pushPrepHistory();
        prepRedoStack=[];
        currentPrepState.enableBinaryConversion = enableBinaryConversion.checked;
        binaryParams.style.display = enableBinaryConversion.checked ? 'block' : 'none';
        onPrepChange();
    });
}

// Binary method selection
if (binaryMethod) {
    binaryMethod.addEventListener('change', () => {
        pushPrepHistory();
        prepRedoStack=[];
        currentPrepState.binaryMethod = binaryMethod.value;
        updateBinaryMethodUI();
        onPrepChange();
    });
}

function prepUndo() {
  if (prepHistory.length>0) {
    let prev = prepHistory.pop();
    let curr = JSON.parse(JSON.stringify(currentPrepState));
    prepRedoStack.push(curr);
    currentPrepState = prev;
    updatePrepUI();
    applyPrepState();
  }
}
function prepRedo() {
  if (prepRedoStack.length>0) {
    prepHistory.push(JSON.parse(JSON.stringify(currentPrepState)));
    currentPrepState = prepRedoStack.pop();
    updatePrepUI();
    applyPrepState();
  }
}

if (prepUndoBtn) prepUndoBtn.addEventListener('click', prepUndo);
if (prepRedoBtn) prepRedoBtn.addEventListener('click', prepRedo);

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key==='z') {
    e.preventDefault();
    prepUndo();
  } else if (e.ctrlKey && e.key==='y') {
    e.preventDefault();
    prepRedo();
  }
});

// ========== Spot Detection Logic ==========

const spotCanvas = document.getElementById('spotCanvas');
const ctx = spotCanvas ? spotCanvas.getContext('2d') : null;
let imgObj = null;
let startX, startY, endX, endY;
let isDrawing = false;

function getMousePos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

if (spotCanvas && typeof processedImgUrl !== "undefined") {
    imgObj = new Image();
    imgObj.src = processedImgUrl;
    imgObj.onload = () => {
    console.log("Image loaded:", imgObj.width, imgObj.height);
    // Set a max display size
    const maxWidth = 800;
    const scale = imgObj.width > maxWidth ? maxWidth / imgObj.width : 1;

    // Adjust canvas size based on scale
    spotCanvas.width = imgObj.width * scale;
    spotCanvas.height = imgObj.height * scale;

    // Draw the image scaled
    ctx.drawImage(imgObj, 0, 0, spotCanvas.width, spotCanvas.height);
    console.log("Canvas size:", spotCanvas.width, spotCanvas.height);
};


  spotCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const pos = getMousePos(e, spotCanvas);
    startX = pos.x;
    startY = pos.y;
  });

  spotCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const pos = getMousePos(e, spotCanvas);
    endX = pos.x;
    endY = pos.y;
    ctx.drawImage(imgObj, 0, 0);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
  });

  spotCanvas.addEventListener('mouseup', (e) => {
    isDrawing = false;
    const pos = getMousePos(e, spotCanvas);
    endX = pos.x;
    endY = pos.y;
    ctx.drawImage(imgObj, 0, 0);
    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(startX, startY, endX - startX, endY - startY);
  });
}


const countSpotsBtn = document.getElementById('countSpotsBtn');
const spotCountResult = document.getElementById('spotCountResult');
if (countSpotsBtn) {
  countSpotsBtn.addEventListener('click', async () => {
    const res = await fetch('/api/spot_count', { method:'POST' });
    const data = await res.json();
    spotCountResult.innerText = "Total Spots: " + (data.spots ?? 0);
  });
}

const filterSpotsBtn = document.getElementById('filterSpotsBtn');
const spotFilterResult = document.getElementById('spotFilterResult');
if (filterSpotsBtn) {
  filterSpotsBtn.addEventListener('click', async () => {
    if (startX==null || endX==null) {
      alert("Please draw a selection first.");
      return;
    }
    const res = await fetch('/api/spot_filter', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ x1:startX, y1:startY, x2:endX, y2:endY })
    });
    const data = await res.json();
    spotFilterResult.innerText = "Filtered Spots: " + (data.spots ?? 0);
  });
}

