import os
import re

import torch
import numpy as np

# OpenCV for video decoding
try:
    import cv2

    _has_cv2 = True
except Exception:
    _has_cv2 = False


def extract_first_number(s):
    match = re.search(r"\d+", s)
    return int(match.group()) if match else float("inf")


sort_methods = [
    "None",
    "Alphabetical (ASC)",
    "Alphabetical (DESC)",
    "Numerical (ASC)",
    "Numerical (DESC)",
    "Datetime (ASC)",
    "Datetime (DESC)",
]


def sort_by(items, base_path=".", method=None):
    def fullpath(x):
        return os.path.join(base_path, x)

    def get_timestamp(path):
        try:
            return os.path.getmtime(path)
        except FileNotFoundError:
            return float("-inf")

    if method == "Alphabetical (ASC)":
        return sorted(items)
    elif method == "Alphabetical (DESC)":
        return sorted(items, reverse=True)
    elif method == "Numerical (ASC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]))
    elif method == "Numerical (DESC)":
        return sorted(items, key=lambda x: extract_first_number(os.path.splitext(x)[0]), reverse=True)
    elif method == "Datetime (ASC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)))
    elif method == "Datetime (DESC)":
        return sorted(items, key=lambda x: get_timestamp(fullpath(x)), reverse=True)
    else:
        return items


def target_size(width, height, custom_width, custom_height, downscale_ratio=8):
    if downscale_ratio is None:
        downscale_ratio = 8

    if custom_width == 0 and custom_height == 0:
        new_w, new_h = width, height
    elif custom_height == 0:
        new_h = int(height * (custom_width / width))
        new_w = int(custom_width)
    elif custom_width == 0:
        new_w = int(width * (custom_height / height))
        new_h = int(custom_height)
    else:
        new_w, new_h = int(custom_width), int(custom_height)

    # round to multiple of downscale_ratio (VHS behavior)
    new_w = int(new_w / downscale_ratio + 0.5) * downscale_ratio
    new_h = int(new_h / downscale_ratio + 0.5) * downscale_ratio
    return new_w, new_h


def _read_frames_vhs_like(
    video_path: str,
    force_rate: float = 0,
    custom_width: int = 0,
    custom_height: int = 0,
    downscale_ratio: int = 8,
    frame_load_cap: int = 0,  # <-- ВАЖНО: максимум кадров на видео. 0 = без лимита (вся длина)
) -> torch.Tensor:
    """
    Returns: IMAGE batch [F,H,W,3] float32 0..1

    Логика:
      - force_rate = 0 => native fps
      - else => time-based sampling to target fps
      - optional resize with rounding to multiple of downscale_ratio
      - frame_load_cap:
          0 => без лимита (все кадры по выбранной частоте)
          N>0 => максимум N кадров (если N=1, вернет ровно 1 кадр)
    """
    if not _has_cv2:
        raise RuntimeError("OpenCV (cv2) not available. Install opencv-python.")

    cap = cv2.VideoCapture(video_path)
    # grab() нужен, чтобы retrieve() отдал первый кадр сразу
    if not cap.isOpened() or not cap.grab():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0  # fallback

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sometimes CAP_PROP_* gives 0; retrieve first frame to get real shape
    # (мы его все равно читаем как первый кадр)
    ok0, frame0 = cap.retrieve()
    if not ok0 or frame0 is None:
        cap.release()
        raise RuntimeError(f"Cannot retrieve first frame from: {video_path}")

    if width <= 0 or height <= 0:
        height, width = frame0.shape[:2]

    base_dt = 1.0 / float(fps)
    target_dt = base_dt if force_rate == 0 else (1.0 / float(force_rate))

    # compute target resize
    new_w, new_h = target_size(width, height, custom_width, custom_height, downscale_ratio)
    do_resize = (new_w != width) or (new_h != height)

    frames = []

    # time-accumulator: на старте считаем, что пора отдать первый кадр
    time_offset = target_dt

    def _process_frame(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if do_resize:
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return rgb

    # Первый кадр (уже считан retrieve выше)
    frames.append(_process_frame(frame0))
    if frame_load_cap > 0 and len(frames) >= frame_load_cap:
        cap.release()
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(arr)

    # после отдачи первого кадра — сбрасываем offset
    time_offset -= target_dt

    # Далее читаем поток: grab() двигает на следующий кадр, retrieve() забирает текущий
    while cap.isOpened():
        # накапливаем время, пока не пора выдавать кадр
        if time_offset < target_dt:
            ok = cap.grab()
            if not ok:
                break
            time_offset += base_dt
            continue

        # пора выдать кадр
        ok, frame_bgr = cap.retrieve()
        if not ok or frame_bgr is None:
            break

        frames.append(_process_frame(frame_bgr))

        if frame_load_cap > 0 and len(frames) >= frame_load_cap:
            break

        time_offset -= target_dt

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames could be read from: {video_path}")

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # [F,H,W,3]
    return torch.from_numpy(arr)


class LoadVideoBatchListFromDir:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "force_rate": ("FLOAT", {"default": 30, "min": 1, "max": 120, "step": 1}),
                "width": ("INT", {"default": 720, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1280, "min": 1, "max": 8192, "step": 1}),
            },
            "optional": {
                # сколько ВИДЕО грузить из папки (как было)
                "video_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                # сколько КАДРОВ на КАЖДОЕ видео (НОВОЕ)
                # 0 = вся длина видео, 1 = один кадр, N = максимум N кадров
                "frame_load_cap": ("INT", {"default": 0, "min": 0, "step": 1}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1}),
                "load_always": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                "sort_method": (sort_methods,),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("IMAGE", "MASK", "COUNT")
    OUTPUT_IS_LIST = (True, True, False)

    FUNCTION = "load_videos"
    CATEGORY = "InspirePack/video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("load_always"):
            return float("NaN")
        return hash(frozenset(kwargs.items()))

    def load_videos(
        self,
        directory: str,
        force_rate: float = 0,
        width: int = 0,
        height: int = 0,
        video_load_cap: int = 0,
        frame_load_cap: int = 0,  # <-- новое
        start_index: int = 0,
        load_always: bool = False,
        sort_method=None,
    ):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")

        files = os.listdir(directory)
        if len(files) == 0:
            raise FileNotFoundError(f"No files in directory '{directory}'.")

        valid_ext = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        files = [
            f
            for f in files
            if os.path.isfile(os.path.join(directory, f)) and os.path.splitext(f)[1].lower() in valid_ext
        ]
        if len(files) == 0:
            raise FileNotFoundError(f"No video files in directory '{directory}' (expected: {sorted(valid_ext)}).")

        files = sort_by(files, directory, sort_method)
        files = files[start_index:]
        if video_load_cap > 0:
            files = files[:video_load_cap]

        images_list = []
        masks_list = []

        for fname in files:
            path = os.path.join(directory, fname)

            vid = _read_frames_vhs_like(
                path,
                force_rate=force_rate,
                custom_width=width,
                custom_height=height,
                downscale_ratio=8,  # кратность 8 как в VHS
                frame_load_cap=frame_load_cap,  # <-- применяем лимит кадров
            )  # [F,H,W,3] float32 0..1

            f, h, w, _ = vid.shape
            mask = torch.zeros((f, h, w), dtype=torch.float32, device="cpu")

            images_list.append(vid)
            masks_list.append(mask)

        return (images_list, masks_list, len(images_list))
