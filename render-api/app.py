from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError


MODEL_SIZE = 256
MODEL_PIXELS = MODEL_SIZE * MODEL_SIZE
THRESHOLD = 0.625
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MODEL_DIR = Path(__file__).with_name("model")


def create_session() -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(
        str(MODEL_DIR / "model.onnx"),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


SESSION = create_session()
SESSION_LOCK = threading.Lock()

app = FastAPI(
    title="Echo LV Segmentation API",
    version="1.0.0",
    description="Research-only left-ventricle segmentation for 2-D echocardiography.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Inference-Ms", "X-Model-Provider", "X-Threshold"],
)


def bilinear_resize(gray: np.ndarray) -> np.ndarray:
    """Match torch.interpolate(..., align_corners=False) without shipping PyTorch."""
    source_height, source_width = gray.shape
    target = np.arange(MODEL_SIZE, dtype=np.float32)
    source_y = (target + 0.5) * (source_height / MODEL_SIZE) - 0.5
    source_x = (target + 0.5) * (source_width / MODEL_SIZE) - 0.5
    y0_raw = np.floor(source_y).astype(np.int32)
    x0_raw = np.floor(source_x).astype(np.int32)
    y1_raw = y0_raw + 1
    x1_raw = x0_raw + 1
    y_weight = (source_y - y0_raw).reshape(-1, 1)
    x_weight = (source_x - x0_raw).reshape(1, -1)
    y0 = np.clip(y0_raw, 0, source_height - 1)
    y1 = np.clip(y1_raw, 0, source_height - 1)
    x0 = np.clip(x0_raw, 0, source_width - 1)
    x1 = np.clip(x1_raw, 0, source_width - 1)

    top = gray[y0[:, None], x0[None, :]] * (1 - x_weight) + gray[y0[:, None], x1[None, :]] * x_weight
    bottom = gray[y1[:, None], x0[None, :]] * (1 - x_weight) + gray[y1[:, None], x1[None, :]] * x_weight
    return top * (1 - y_weight) + bottom * y_weight


def preprocess(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    resized = bilinear_resize(gray)
    low, high = np.percentile(resized, [1, 99])
    clipped = np.clip(resized, low, high)
    normalized = (clipped - clipped.min()) / max(float(clipped.max() - clipped.min()), 1e-6)
    return normalized.astype(np.float32)[None, None]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=bool)
    largest: list[tuple[int, int]] = []

    for row in range(MODEL_SIZE):
        for column in range(MODEL_SIZE):
            if not mask[row, column] or visited[row, column]:
                continue
            queue = [(row, column)]
            visited[row, column] = True
            head = 0
            while head < len(queue):
                current_row, current_column = queue[head]
                head += 1
                for row_offset in (-1, 0, 1):
                    for column_offset in (-1, 0, 1):
                        if row_offset == 0 and column_offset == 0:
                            continue
                        next_row = current_row + row_offset
                        next_column = current_column + column_offset
                        if not (0 <= next_row < MODEL_SIZE and 0 <= next_column < MODEL_SIZE):
                            continue
                        if mask[next_row, next_column] and not visited[next_row, next_column]:
                            visited[next_row, next_column] = True
                            queue.append((next_row, next_column))
            if len(queue) > len(largest):
                largest = queue

    result = np.zeros_like(mask, dtype=bool)
    if largest:
        rows, columns = zip(*largest)
        result[rows, columns] = True
    return result


def fill_holes(mask: np.ndarray) -> np.ndarray:
    outside = np.zeros_like(mask, dtype=bool)
    queue: list[tuple[int, int]] = []

    def enqueue(row: int, column: int) -> None:
        if not mask[row, column] and not outside[row, column]:
            outside[row, column] = True
            queue.append((row, column))

    for coordinate in range(MODEL_SIZE):
        enqueue(0, coordinate)
        enqueue(MODEL_SIZE - 1, coordinate)
        enqueue(coordinate, 0)
        enqueue(coordinate, MODEL_SIZE - 1)

    head = 0
    while head < len(queue):
        row, column = queue[head]
        head += 1
        for next_row, next_column in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if 0 <= next_row < MODEL_SIZE and 0 <= next_column < MODEL_SIZE:
                enqueue(next_row, next_column)
    return mask | ~outside


def postprocess(logits: np.ndarray) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    mask = probabilities >= THRESHOLD
    if not mask.any():
        return np.asarray(mask, dtype=np.uint8)
    return np.asarray(fill_holes(keep_largest_component(mask)), dtype=np.uint8)


def make_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    full_mask = np.asarray(
        Image.fromarray(mask * 255).resize(image.size, Image.Resampling.NEAREST)
    ) > 0
    output = rgb.astype(np.float32)
    output[full_mask] = output[full_mask] * 0.5 + np.array([0, 255, 0], dtype=np.float32) * 0.5
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


@app.get("/")
def index() -> JSONResponse:
    return JSONResponse(
        {
            "name": "Echo LV Segmentation API",
            "status": "ready",
            "docs": "/docs",
            "notice": "Research use only - not a medical device.",
        }
    )


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": "Pix2Pix-style 8-level U-Net",
            "input": [1, 1, MODEL_SIZE, MODEL_SIZE],
            "threshold": THRESHOLD,
            "provider": SESSION.get_providers()[0],
        }
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> StreamingResponse:
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Upload a PNG or JPG image.")

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
        if image.width * image.height > 16_000_000:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from error

    started = time.perf_counter()
    input_tensor = preprocess(image)
    with SESSION_LOCK:
        logits = SESSION.run(["logits"], {"input": input_tensor})[0][0, 0]
    mask = postprocess(logits)
    overlay = make_overlay(image, mask)
    output = io.BytesIO()
    overlay.save(output, format="PNG", optimize=True)
    output.seek(0)
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    return StreamingResponse(
        output,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Inference-Ms": str(elapsed_ms),
            "X-Model-Provider": "ONNX Runtime CPU",
            "X-Threshold": str(THRESHOLD),
        },
    )
