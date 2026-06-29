'use strict';

const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const predictionEl = document.getElementById('prediction');
const probabilityEl = document.getElementById('probability');
const sentenceEl = document.getElementById('sentence');
const barFill = document.getElementById('bar-fill');
const holdRing = document.getElementById('hold-ring');
const holdInner = document.getElementById('hold-inner');
const statusText = document.getElementById('status-text');
const palmToast = document.getElementById('palm-toast');

// FIX: use plain hyphen to avoid "tofu block" on any device
const IDLE_GLYPH = '-';

document.getElementById('clear-btn').addEventListener('click', () => { sentence = ''; updateSentence(); });

document.getElementById('space-btn').addEventListener('click', () => {
    if (sentence.length > 0 && sentence.slice(-1) !== ' ') {
        sentence += ' ';
        updateSentence();
    }
});

document.getElementById('delete-btn').addEventListener('click', () => { sentence = sentence.slice(0, -1); updateSentence(); });

// 26 NSL letters only
const LABELS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');

const CONFIDENCE_THRESHOLD = 0.65;
const HOLD_TIME = 2000;
const PALM_HOLD_TIME = 1500;
const MAX_SENTENCE_LENGTH = 200;

let model = null;
let lastLetter = null;
let holdStart = null;
let sentence = '';
let palmHoldStart = null;
let toastTimeout = null;
let currentFacingMode = 'user';
let cameraInstance = null;

function updateSentence() {
    sentenceEl.textContent = sentence || '_';
}

function resetPredictionUI() {
    predictionEl.textContent = IDLE_GLYPH;
    probabilityEl.textContent = '0%';
    barFill.style.width = '0%';
    barFill.style.opacity = '0';
    lastLetter = null;
    holdStart = null;
}

function updateHoldRing(progress) {
    const pct = Math.min(100, Math.max(0, progress * 100));
    holdRing.style.background =
        `conic-gradient(#2563eb ${pct}%, #e2e8f0 ${pct}%)`;
    holdInner.textContent = progress > 0 ? `${Math.round(pct)}%` : 'HOLD';
}

function showPalmToast() {
    palmToast.classList.add('show');
    clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => palmToast.classList.remove('show'), 1500);
}

function isOpenPalm(landmarks) {
    const tips = [8, 12, 16, 20];
    const pips = [6, 10, 14, 18];
    let extendedFingers = 0;
    for (let i = 0; i < tips.length; i++) {
        const tipDist = Math.hypot(
            landmarks[tips[i]].x - landmarks[0].x,
            landmarks[tips[i]].y - landmarks[0].y
        );
        const pipDist = Math.hypot(
            landmarks[pips[i]].x - landmarks[0].x,
            landmarks[pips[i]].y - landmarks[0].y
        );
        if (tipDist > pipDist) extendedFingers++;
    }
    const thumbTipDist = Math.hypot(
        landmarks[4].x - landmarks[0].x,
        landmarks[4].y - landmarks[0].y
    );
    const thumbMcpDist = Math.hypot(
        landmarks[2].x - landmarks[0].x,
        landmarks[2].y - landmarks[0].y
    );
    const thumbExtended = thumbTipDist > thumbMcpDist;
    return extendedFingers >= 3 && thumbExtended;
}

async function loadModel() {
    try {
        statusText.textContent = 'Loading model...';
        // Updated path to the new NSL-only model
        model = await tf.loadGraphModel('model_nsl/model.json');
        statusText.textContent = 'Model ready · Camera active';
    } catch (err) {
        statusText.textContent = 'Model load failed — check console';
        console.error('[NSL] Model error:', err);
    }
}

async function predict(imageEl) {
    if (!model) return null;
    const tensor = tf.tidy(() =>
        tf.browser.fromPixels(imageEl)
            .resizeBilinear([224, 224])
            .toFloat()
            .div(255.0)
            .expandDims(0)
    );
    const preds = await model.predict(tensor).data();
    tensor.dispose();
    const maxIdx = preds.indexOf(Math.max(...preds));
    const letter = LABELS[maxIdx];
    return { letter, confidence: preds[maxIdx] };
}

// MediaPipe setup
const hands = new Hands({
    locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.5
});

hands.onResults(async (results) => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results.multiHandLandmarks?.length) {
        resetPredictionUI();
        palmHoldStart = null;
        updateHoldRing(0);
        return;
    }

    const landmarks = results.multiHandLandmarks[0];

    drawConnectors(ctx, landmarks, HAND_CONNECTIONS,
        { color: '#2563eb', lineWidth: 2 });
    drawLandmarks(ctx, landmarks,
        { color: '#ffffff', lineWidth: 1, radius: 3 });

    // Open palm → space
    if (isOpenPalm(landmarks)) {
        if (!palmHoldStart) {
            palmHoldStart = Date.now();
        } else {
            const elapsed = Date.now() - palmHoldStart;
            updateHoldRing(elapsed / PALM_HOLD_TIME);
            if (elapsed >= PALM_HOLD_TIME) {
                if (sentence.length > 0 &&
                    sentence.length < MAX_SENTENCE_LENGTH &&
                    sentence.slice(-1) !== ' ') {
                    sentence += ' ';
                    updateSentence();
                    showPalmToast();
                }
                palmHoldStart = null;
                updateHoldRing(0);
            }
        }
        predictionEl.textContent = '✋';
        probabilityEl.textContent = '—';
        barFill.style.width = '0%';
        barFill.style.opacity = '0';
        lastLetter = null;
        holdStart = null;
        return;
    }

    palmHoldStart = null;

    // Crop hand region
    const xs = landmarks.map(l => l.x * canvas.width);
    const ys = landmarks.map(l => l.y * canvas.height);
    const pad = 30;
    const x1 = Math.max(0, Math.min(...xs) - pad);
    const y1 = Math.max(0, Math.min(...ys) - pad);
    const x2 = Math.min(canvas.width, Math.max(...xs) + pad);
    const y2 = Math.min(canvas.height, Math.max(...ys) + pad);
    const w = x2 - x1;
    const h = y2 - y1;

    if (w < 10 || h < 10) {
        resetPredictionUI();
        updateHoldRing(0);
        return;
    }

    const offscreen = document.createElement('canvas');
    offscreen.width = w;
    offscreen.height = h;
    offscreen.getContext('2d')
        .drawImage(video, x1, y1, w, h, 0, 0, w, h);

    const result = await predict(offscreen);
    if (!result) return;

    if (result.confidence >= CONFIDENCE_THRESHOLD) {
        predictionEl.textContent = result.letter;
        const pct = (result.confidence * 100).toFixed(1);
        probabilityEl.textContent = `${pct}%`;
        barFill.style.width = `${pct}%`;
        barFill.style.opacity = '1';

        if (result.letter === lastLetter) {
            const elapsed = Date.now() - holdStart;
            updateHoldRing(elapsed / HOLD_TIME);
            if (elapsed >= HOLD_TIME) {
                if (sentence.length < MAX_SENTENCE_LENGTH) {
                    sentence += result.letter;
                    updateSentence();
                }
                lastLetter = null;
                holdStart = null;
                updateHoldRing(0);
            }
        } else {
            lastLetter = result.letter;
            holdStart = Date.now();
            updateHoldRing(0);
        }
    } else {
        resetPredictionUI();
        updateHoldRing(0);
    }
});

// Camera functions (with toggle support)
async function startCamera() {
    try {
        if (cameraInstance) {
            cameraInstance.stop();
            cameraInstance = null;
        }
        if (video.srcObject) {
            video.srcObject.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        await new Promise(resolve => setTimeout(resolve, 300));

        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: currentFacingMode },
                audio: false
            });
        } catch (constraintErr) {
            console.warn('[NSL] Exact constraints failed, retrying with relaxed constraints:', constraintErr);
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode },
                audio: false
            });
        }

        video.srcObject = stream;
        try {
            await video.play();
        } catch (playErr) {
            console.warn('[NSL] video.play() did not resolve immediately:', playErr);
        }

        cameraInstance = new Camera(video, {
            onFrame: async () => { await hands.send({ image: video }); },
            width: 640,
            height: 480
        });
        cameraInstance.start();
        statusText.textContent = 'Model ready · Camera active';
    } catch (err) {
        statusText.textContent = 'Camera access denied';
        console.error('[NSL] Camera error:', err);
    }
}

// Switch camera button
document.getElementById('switch-camera-btn').addEventListener('click', async () => {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    statusText.textContent = `Switching to ${currentFacingMode === 'user' ? 'front' : 'back'} camera...`;
    await startCamera();
});

// Init
loadModel();
startCamera();
