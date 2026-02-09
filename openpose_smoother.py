from __future__ import annotations

import copy
import math
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
import torch


# ============================================================
# ComfyUI Node (pose_data + PKL)
# ============================================================

_GLOBAL_LOCK = threading.Lock()


class KPSSmoothPoseDataAndRender:
    """
    Сглаживание + рендер позы.
    Вход: POSEDATA (как объект/dict; обычно приходит из TSLoadPoseDataPickle).
    Выход: IMAGE (torch [T,H,W,3] float 0..1), POSEDATA (в том же формате, но сглаженный).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_data": ("POSEDATA",),  # <-- ВАЖНО: именно POSEDATA
                "filter_extra_people": ("BOOLEAN", {"default": True}),
                # общий набор параметров сглаживания (вместо body + face_hands)
                "smooth_alpha": ("FLOAT", {"default": 0.7, "min": 0.01, "max": 0.99, "step": 0.01}),
                "gap_frames": ("INT", {"default": 12, "min": 0, "max": 100, "step": 1}),
                "min_run_frames": ("INT", {"default": 2, "min": 1, "max": 60, "step": 1}),
                # пороги отрисовки (в инпут добавляем body/hands, face НЕ добавляем)
                "conf_thresh_body": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.01}),
                "conf_thresh_hands": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA")  # <-- ВАЖНО: именно POSEDATA
    RETURN_NAMES = ("IMAGE", "pose_data")
    FUNCTION = "run"
    CATEGORY = "KPS"

    def run(self, pose_data, **kwargs):
        filter_extra_people = bool(kwargs.get("filter_extra_people", True))

        # общий набор
        smooth_alpha = float(kwargs.get("smooth_alpha", 0.7))
        gap_frames = int(kwargs.get("gap_frames", 12))
        min_run_frames = int(kwargs.get("min_run_frames", 2))

        # пороги рендера
        conf_thresh_body = float(kwargs.get("conf_thresh_body", 0.20))
        conf_thresh_hands = float(kwargs.get("conf_thresh_hands", 0.50))
        conf_thresh_face = 0.20  # <- НЕ добавляем в INPUT, но фиксируем как ты просил

        force_body_18 = bool(kwargs.get("force_body_18", False))

        pose_data = _coerce_pose_data_to_obj(pose_data)

        # pose_data -> frames_json_like
        frames_json_like, meta_ref = _pose_data_to_kps_frames(pose_data, force_body_18=force_body_18)

        with _GLOBAL_LOCK:
            old = _snapshot_tunable_globals()
            try:
                # BODY
                globals()["ALPHA_BODY"] = smooth_alpha
                globals()["SUPER_SMOOTH_ALPHA"] = smooth_alpha
                globals()["MAX_GAP_FRAMES"] = gap_frames
                globals()["MIN_RUN_FRAMES"] = min_run_frames

                # FACE+HANDS (dense) тоже от общего набора
                globals()["DENSE_SUPER_SMOOTH_ALPHA"] = smooth_alpha
                globals()["DENSE_MAX_GAP_FRAMES"] = gap_frames
                globals()["DENSE_MIN_RUN_FRAMES"] = min_run_frames

                globals()["FILTER_EXTRA_PEOPLE"] = filter_extra_people

                smoothed_frames = smooth_KPS_json_obj(
                    frames_json_like,
                    keep_face_untouched=False,
                    keep_hands_untouched=False,
                    filter_extra_people=filter_extra_people,
                )
            finally:
                _restore_tunable_globals(old)

        # frames_json_like -> pose_data (обратно в pose_metas)
        out_pose_data = _kps_frames_to_pose_data(pose_data, smoothed_frames, meta_ref, force_body_18=force_body_18)

        # render
        w, h = _extract_canvas_wh(smoothed_frames, default_w=720, default_h=1280)
        frames_np = []
        for fr in smoothed_frames:
            if isinstance(fr, dict) and fr.get("people"):
                img = _draw_pose_frame_full(
                    w,
                    h,
                    fr["people"][0],
                    conf_thresh_body=conf_thresh_body,
                    conf_thresh_hands=conf_thresh_hands,
                    conf_thresh_face=conf_thresh_face,
                )
            else:
                img = np.zeros((h, w, 3), dtype=np.uint8)
            frames_np.append(img)

        frames_t = torch.from_numpy(np.stack(frames_np, axis=0)).float() / 255.0
        return (frames_t, out_pose_data)


# ============================================================
# PKL / pose_data IO
# ============================================================


class _PoseDummyObj:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        # поддержка dict и (dict, slotstate)
        if isinstance(state, dict):
            self.__dict__.update(state)
        elif isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], dict):
            self.__dict__.update(state[0])
            if isinstance(state[1], dict):
                self.__dict__.update(state[1])
            else:
                self.__dict__["_slotstate"] = state[1]
        else:
            self.__dict__["_state"] = state


class _SafeUnpickler(pickle.Unpickler):
    """
    Безопасно грузим PKL из ComfyUI окружения:
    - ремап numpy._core -> numpy.core
    - неизвестные классы (WanAnimatePreprocess.*) превращаем в простые объекты с __dict__
    """

    def find_class(self, module, name):
        # ремап внутренних путей numpy (частая проблема между версиями)
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        if module.startswith("numpy._globals"):
            module = module.replace("numpy._globals", "numpy", 1)

        # конкретные классы метаданных (если встречаются)
        if name in {"AAPoseMeta"}:
            return _PoseDummyObj

        try:
            return super().find_class(module, name)
        except Exception:
            return _PoseDummyObj


def _load_pose_data_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return _SafeUnpickler(f).load()


def _coerce_pose_data_to_obj(pd: Any) -> Any:
    """
    Accepts:
      - dict pose_data
      - object with attributes like .pose_metas (AAPoseMeta-like)
      - str path to .pkl
      - dict wrapper with 'pose_data'
    """
    if isinstance(pd, str):
        obj = _load_pose_data_pkl(pd)
        return obj

    if isinstance(pd, dict) and "pose_data" in pd:
        return pd["pose_data"]

    return pd


# ============================================================
# pose_data <-> JSON-like KPS frames
# ============================================================


def _as_attr(x: Any, key: str, default=None):
    if isinstance(x, dict):
        return x.get(key, default)
    return getattr(x, key, default)


def _set_attr(x: Any, key: str, value: Any):
    if isinstance(x, dict):
        x[key] = value
    else:
        setattr(x, key, value)


def _xy_p_to_flat(xy: Optional[np.ndarray], p: Optional[np.ndarray]) -> Optional[List[float]]:
    if xy is None:
        return None
    arr = np.asarray(xy)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    N = arr.shape[0]
    if p is None:
        pp = np.ones((N,), dtype=np.float32)
    else:
        pp = np.asarray(p).reshape(-1)
        if pp.shape[0] != N:
            # если вдруг не совпали — подстрахуемся
            pp = np.ones((N,), dtype=np.float32)

    out: List[float] = []
    for i in range(N):
        out.extend([float(arr[i, 0]), float(arr[i, 1]), float(pp[i])])
    return out


def _flat_to_xy_p(flat: Optional[List[float]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not isinstance(flat, list) or len(flat) % 3 != 0:
        return None, None
    N = len(flat) // 3
    xy = np.zeros((N, 2), dtype=np.float32)
    p = np.zeros((N,), dtype=np.float32)
    for i in range(N):
        xy[i, 0] = float(flat[3 * i + 0])
        xy[i, 1] = float(flat[3 * i + 1])
        p[i] = float(flat[3 * i + 2])
    return xy, p


def _pose_data_to_kps_frames(pose_data: Any, *, force_body_18: bool) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Делает "как JSON" список кадров:
      frame = {"people":[{pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, hand_right_keypoints_2d}],
               "canvas_width": W, "canvas_height": H}
    meta_ref: ссылки на pose_metas + тип/доступ, чтобы правильно записать обратно.
    """
    pose_metas = _as_attr(pose_data, "pose_metas", None)
    if pose_metas is None:
        # иногда называют иначе
        pose_metas = _as_attr(pose_data, "frames", None)

    if pose_metas is None or not isinstance(pose_metas, list):
        raise ValueError("pose_data does not contain 'pose_metas' list.")

    frames: List[Dict[str, Any]] = []
    for meta in pose_metas:
        h = _as_attr(meta, "height", 1280)
        w = _as_attr(meta, "width", 720)

        kps_body = _as_attr(meta, "kps_body", None)
        kps_body_p = _as_attr(meta, "kps_body_p", None)

        kps_face = _as_attr(meta, "kps_face", None)
        kps_face_p = _as_attr(meta, "kps_face_p", None)

        kps_lhand = _as_attr(meta, "kps_lhand", None)
        kps_lhand_p = _as_attr(meta, "kps_lhand_p", None)

        kps_rhand = _as_attr(meta, "kps_rhand", None)
        kps_rhand_p = _as_attr(meta, "kps_rhand_p", None)

        # to flat
        pose_flat = _xy_p_to_flat(kps_body, kps_body_p)
        face_flat = _xy_p_to_flat(kps_face, kps_face_p)
        lh_flat = _xy_p_to_flat(kps_lhand, kps_lhand_p)
        rh_flat = _xy_p_to_flat(kps_rhand, kps_rhand_p)

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        person = {
            "pose_keypoints_2d": pose_flat if pose_flat is not None else [],
            "face_keypoints_2d": face_flat if face_flat is not None else [],
            "hand_left_keypoints_2d": lh_flat,
            "hand_right_keypoints_2d": rh_flat,
        }

        frame = {"people": [person], "canvas_height": int(h), "canvas_width": int(w)}
        frames.append(frame)

    meta_ref = {
        "pose_metas": pose_metas,
        "len": len(pose_metas),
    }
    return frames, meta_ref


def _kps_frames_to_pose_data(
    pose_data_in: Any,
    frames_kps: List[Dict[str, Any]],
    meta_ref: Dict[str, Any],
    *,
    force_body_18: bool,
) -> Any:
    """
    Записывает обратно сглаженные keypoints в pose_metas[*].kps_* / kps_*_p.
    Остальные поля pose_data сохраняем.
    """
    out_pd = copy.deepcopy(pose_data_in)
    pose_metas_out = _as_attr(out_pd, "pose_metas", None)
    if pose_metas_out is None:
        # fallback: вдруг другой ключ
        pose_metas_out = meta_ref.get("pose_metas")

    if pose_metas_out is None or not isinstance(pose_metas_out, list):
        raise ValueError("Failed to locate pose_metas in output pose_data.")

    T = min(len(pose_metas_out), len(frames_kps))
    for t in range(T):
        meta = pose_metas_out[t]
        fr = frames_kps[t]
        people = fr.get("people", []) if isinstance(fr, dict) else []
        p0 = people[0] if people else None
        if not isinstance(p0, dict):
            continue

        pose_flat = p0.get("pose_keypoints_2d")
        face_flat = p0.get("face_keypoints_2d")
        lh_flat = p0.get("hand_left_keypoints_2d")
        rh_flat = p0.get("hand_right_keypoints_2d")

        if force_body_18 and isinstance(pose_flat, list) and len(pose_flat) >= 18 * 3:
            pose_flat = pose_flat[: 18 * 3]

        body_xy, body_p = _flat_to_xy_p(pose_flat if isinstance(pose_flat, list) else None)
        face_xy, face_p = _flat_to_xy_p(face_flat if isinstance(face_flat, list) else None)
        lh_xy, lh_p = _flat_to_xy_p(lh_flat if isinstance(lh_flat, list) else None)
        rh_xy, rh_p = _flat_to_xy_p(rh_flat if isinstance(rh_flat, list) else None)

        if body_xy is not None and body_p is not None:
            _set_attr(meta, "kps_body", body_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_body_p", body_p.astype(np.float32, copy=False))

        if face_xy is not None and face_p is not None:
            _set_attr(meta, "kps_face", face_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_face_p", face_p.astype(np.float32, copy=False))

        if lh_xy is not None and lh_p is not None:
            _set_attr(meta, "kps_lhand", lh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_lhand_p", lh_p.astype(np.float32, copy=False))

        if rh_xy is not None and rh_p is not None:
            _set_attr(meta, "kps_rhand", rh_xy.astype(np.float32, copy=False))
            _set_attr(meta, "kps_rhand_p", rh_p.astype(np.float32, copy=False))

        # обновим width/height если нужно
        if isinstance(fr, dict):
            if "canvas_width" in fr:
                _set_attr(meta, "width", int(fr["canvas_width"]))
            if "canvas_height" in fr:
                _set_attr(meta, "height", int(fr["canvas_height"]))

    # обязательно положим pose_metas обратно
    _set_attr(out_pd, "pose_metas", pose_metas_out)
    return out_pd


def _extract_canvas_wh(data: Any, default_w: int, default_h: int) -> Tuple[int, int]:
    w, h = int(default_w), int(default_h)
    if isinstance(data, list):
        for fr in data:
            if isinstance(fr, dict) and "canvas_width" in fr and "canvas_height" in fr:
                try:
                    w = int(fr["canvas_width"])
                    h = int(fr["canvas_height"])
                    break
                except Exception:
                    pass
    return w, h


# ============================================================
# === START: smooth_KPS_json.py logic (ported as-is)
# ============================================================

# --- Root+Scale carry (when torso disappears on close-up) ---
ROOTSCALE_CARRY_ENABLED = True
CARRY_MAX_FRAMES = 48
CARRY_MIN_ANCHORS = 2
CARRY_ANCHOR_JOINTS = [0, 1, 2, 5, 3, 6, 4, 7]
CARRY_CONF_GATE = 0.20

# --- Main person selection / multi-person filtering ---
FILTER_EXTRA_PEOPLE = True
MAIN_PERSON_MODE = "longest_track"
TRACK_MATCH_MIN_PX = 80.0
TRACK_MATCH_FACTOR = 3.0
TRACK_MAX_FRAME_GAP = 32

# --- Spatial outlier suppression ---
SPATIAL_OUTLIER_FIX = True
BONE_MAX_FACTOR = 2.3
TORSO_RADIUS_FACTOR = 4.0

# EMA smoothing for BODY only (online)
ALPHA_BODY = 0.70
MAX_STEP_BODY = 60.0
VEL_ALPHA = 0.45
EPS = 0.3
CONF_GATE_BODY = 0.20
CONF_FLOOR_BODY = 0.00

TRACK_DIST_PENALTY = 1.5
FACE_WEIGHT_IN_SCORE = 0.15
HAND_WEIGHT_IN_SCORE = 0.35

ALLOW_DISAPPEAR_JOINTS = {3, 4, 6, 7}

GAP_FILL_ENABLED = True
MAX_GAP_FRAMES = 12
MIN_RUN_FRAMES = 2

TORSO_SYNC_ENABLED = True
TORSO_JOINTS = {1, 2, 5, 8, 11}
TORSO_LOOKAHEAD_FRAMES = 32

SUPER_SMOOTH_ENABLED = True
SUPER_SMOOTH_ALPHA = 0.7
SUPER_SMOOTH_MIN_CONF = 0.20

MEDIAN3_ENABLED = True

FACE_SMOOTH_ENABLED = True
HANDS_SMOOTH_ENABLED = False

CONF_GATE_FACE = 0.20
CONF_GATE_HAND = 0.50

HAND_MIN_POINTS_PRESENT = 7
MIN_HAND_RUN_FRAMES = 6

DENSE_GAP_FILL_ENABLED = False
DENSE_MAX_GAP_FRAMES = 8
DENSE_MIN_RUN_FRAMES = 2

DENSE_MEDIAN3_ENABLED = False
DENSE_SUPER_SMOOTH_ENABLED = False
DENSE_SUPER_SMOOTH_ALPHA = 0.7


def _snapshot_tunable_globals() -> Dict[str, Any]:
    keys = [
        "FILTER_EXTRA_PEOPLE",
        "SUPER_SMOOTH_ALPHA",
        "MAX_GAP_FRAMES",
        "MIN_RUN_FRAMES",
        "DENSE_SUPER_SMOOTH_ALPHA",
        "DENSE_MAX_GAP_FRAMES",
        "DENSE_MIN_RUN_FRAMES",
    ]
    return {k: globals().get(k) for k in keys}


def _restore_tunable_globals(old: Dict[str, Any]) -> None:
    for k, v in old.items():
        globals()[k] = v


def _is_valid_xyc(x: float, y: float, c: float) -> bool:
    if c is None:
        return False
    if c <= 0:
        return False
    if x == 0 and y == 0:
        return False
    if math.isnan(x) or math.isnan(y) or math.isnan(c):
        return False
    return True


def _reshape_keypoints_2d(arr: List[float]) -> List[Tuple[float, float, float]]:
    if arr is None:
        return []
    if len(arr) % 3 != 0:
        raise ValueError(f"keypoints length not multiple of 3: {len(arr)}")
    out = []
    for i in range(0, len(arr), 3):
        out.append((float(arr[i]), float(arr[i + 1]), float(arr[i + 2])))
    return out


def _flatten_keypoints_2d(kps: List[Tuple[float, float, float]]) -> List[float]:
    out: List[float] = []
    for x, y, c in kps:
        out.extend([float(x), float(y), float(c)])
    return out


def _sum_conf(arr: Optional[List[float]], sample_step: int = 1) -> float:
    if not arr:
        return 0.0
    s = 0.0
    for i in range(2, len(arr), 3 * sample_step):
        try:
            c = float(arr[i])
        except Exception:
            c = 0.0
        if c > 0:
            s += c
    return s


def _body_center_from_pose(pose_arr: Optional[List[float]]) -> Optional[Tuple[float, float]]:
    if not pose_arr:
        return None
    kps = _reshape_keypoints_2d(pose_arr)
    idxs = [2, 5, 8, 11, 1]
    pts = []
    for idx in idxs:
        if idx < len(kps):
            x, y, c = kps[idx]
            if _is_valid_xyc(x, y, c):
                pts.append((x, y))
    if not pts:
        for x, y, c in kps:
            if _is_valid_xyc(x, y, c):
                pts.append((x, y))
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return (cx, cy)


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _choose_single_person(
    people: List[Dict[str, Any]], prev_center: Optional[Tuple[float, float]]
) -> Optional[Dict[str, Any]]:
    if not people:
        return None
    best = None
    best_score = -1e18

    for p in people:
        pose = p.get("pose_keypoints_2d")
        face = p.get("face_keypoints_2d")
        lh = p.get("hand_left_keypoints_2d")
        rh = p.get("hand_right_keypoints_2d")

        score = _sum_conf(pose)
        score += FACE_WEIGHT_IN_SCORE * _sum_conf(face, sample_step=4)
        score += HAND_WEIGHT_IN_SCORE * (_sum_conf(lh, sample_step=2) + _sum_conf(rh, sample_step=2))

        center = _body_center_from_pose(pose)
        if prev_center is not None and center is not None:
            score -= TRACK_DIST_PENALTY * _dist(prev_center, center)

        if score > best_score:
            best_score = score
            best = p

    return best


@dataclass
class _Track:
    frames: Dict[int, Dict[str, Any]]
    centers: Dict[int, Tuple[float, float]]
    last_t: int
    last_center: Tuple[float, float]


def _estimate_torso_scale(pose: List[Tuple[float, float, float]]) -> Optional[float]:
    def dist(i, k) -> Optional[float]:
        if i >= len(pose) or k >= len(pose):
            return None
        xi, yi, ci = pose[i]
        xk, yk, ck = pose[k]
        if not _is_valid_xyc(xi, yi, ci) or not _is_valid_xyc(xk, yk, ck):
            return None
        return math.hypot(xi - xk, yi - yk)

    cand = [dist(2, 5), dist(8, 11), dist(1, 8), dist(1, 11)]
    cand = [c for c in cand if c is not None and c > 1e-3]
    if not cand:
        return None
    return float(sum(cand) / len(cand))


def _track_match_threshold_from_pose(pose_arr: Optional[List[float]]) -> float:
    if isinstance(pose_arr, list):
        pose = _reshape_keypoints_2d(pose_arr)
        s = _estimate_torso_scale(pose)
        if s is not None:
            return max(float(TRACK_MATCH_MIN_PX), float(TRACK_MATCH_FACTOR) * float(s))
    return float(max(TRACK_MATCH_MIN_PX, 120.0))


def _build_tracks_over_video(frames_data: List[Any]) -> List[_Track]:
    tracks: List[_Track] = []

    for t, frame in enumerate(frames_data):
        if not isinstance(frame, dict):
            continue
        people = frame.get("people", [])
        if not isinstance(people, list) or not people:
            continue

        cand: List[Tuple[int, Dict[str, Any], Tuple[float, float]]] = []
        for i, p in enumerate(people):
            if not isinstance(p, dict):
                continue
            pose = p.get("pose_keypoints_2d")
            c = _body_center_from_pose(pose)
            if c is None:
                continue
            cand.append((i, p, c))

        if not cand:
            continue

        used = set()
        track_order = sorted(range(len(tracks)), key=lambda k: tracks[k].last_t, reverse=True)

        for k in track_order:
            tr = tracks[k]
            age = t - tr.last_t
            if age > int(TRACK_MAX_FRAME_GAP):
                continue

            best_idx = None
            best_d = 1e18

            for i, p, cc in cand:
                if i in used:
                    continue

                thr = _track_match_threshold_from_pose(p.get("pose_keypoints_2d"))
                d = _dist(tr.last_center, cc)
                if d <= thr and d < best_d:
                    best_d = d
                    best_idx = i

            if best_idx is not None:
                i, p, cc = next(x for x in cand if x[0] == best_idx)
                used.add(i)
                tr.frames[t] = p
                tr.centers[t] = cc
                tr.last_t = t
                tr.last_center = cc

        for i, p, cc in cand:
            if i in used:
                continue
            tracks.append(_Track(frames={t: p}, centers={t: cc}, last_t=t, last_center=cc))

    return tracks


def _track_presence_score(tr: _Track) -> Tuple[int, float, float]:
    frames_count = len(tr.frames)
    face_sum = 0.0
    body_sum = 0.0
    for p in tr.frames.values():
        face_sum += _sum_conf(p.get("face_keypoints_2d"), sample_step=4)
        body_sum += _sum_conf(p.get("pose_keypoints_2d"), sample_step=1)
    return (frames_count, face_sum, body_sum)


def _pick_main_track(tracks: List[_Track]) -> Optional[_Track]:
    if not tracks:
        return None
    best = None
    best_key = (-1, -1e18, -1e18)
    for tr in tracks:
        key = _track_presence_score(tr)
        if key > best_key:
            best_key = key
            best = tr
    return best


@dataclass
class BodyState:
    last_xy: List[Optional[Tuple[float, float]]]
    last_v: List[Tuple[float, float]]

    def __init__(self, joints: int):
        self.last_xy = [None] * joints
        self.last_v = [(0.0, 0.0)] * joints


def _smooth_body_pose(pose_arr: Optional[List[float]], state: BodyState) -> Optional[List[float]]:
    if pose_arr is None:
        return None

    kps = _reshape_keypoints_2d(pose_arr)
    J = len(kps)
    if len(state.last_xy) != J:
        state.last_xy = [None] * J
        state.last_v = [(0.0, 0.0)] * J

    out: List[Tuple[float, float, float]] = []

    for j in range(J):
        x, y, c = kps[j]
        last = state.last_xy[j]
        vx_last, vy_last = state.last_v[j]

        valid_in = _is_valid_xyc(x, y, c) and (c >= CONF_GATE_BODY)

        if valid_in:
            if last is None:
                nx, ny = x, y
                state.last_xy[j] = (nx, ny)
                state.last_v[j] = (0.0, 0.0)
                out.append((nx, ny, float(c)))
                continue

            dx_raw = x - last[0]
            dy_raw = y - last[1]
            if abs(dx_raw) < EPS:
                dx_raw = 0.0
            if abs(dy_raw) < EPS:
                dy_raw = 0.0

            vx = VEL_ALPHA * dx_raw + (1.0 - VEL_ALPHA) * vx_last
            vy = VEL_ALPHA * dy_raw + (1.0 - VEL_ALPHA) * vy_last

            px = last[0] + vx
            py = last[1] + vy

            nx = ALPHA_BODY * x + (1.0 - ALPHA_BODY) * px
            ny = ALPHA_BODY * y + (1.0 - ALPHA_BODY) * py

            ddx = nx - last[0]
            ddy = ny - last[1]
            d = math.hypot(ddx, ddy)
            if d > MAX_STEP_BODY and d > 1e-6:
                scale = MAX_STEP_BODY / d
                nx = last[0] + ddx * scale
                ny = last[1] + ddy * scale
                vx = nx - last[0]
                vy = ny - last[1]

            state.last_xy[j] = (nx, ny)
            state.last_v[j] = (vx, vy)

            out.append((nx, ny, float(c)))
        else:
            out.append((float(x), float(y), float(c)))

    return _flatten_keypoints_2d(out)


COCO18_EDGES = [
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (8, 11),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

HAND21_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

_NEIGHBORS = None


def _build_neighbors():
    global _NEIGHBORS
    if _NEIGHBORS is not None:
        return
    neigh = {}
    for a, b in COCO18_EDGES:
        neigh.setdefault(a, set()).add(b)
        neigh.setdefault(b, set()).add(a)
    _NEIGHBORS = neigh


def _suppress_spatial_outliers_in_pose_arr(
    pose_arr: Optional[List[float]], *, conf_gate: float
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr

    pose = _reshape_keypoints_2d(pose_arr)
    J = len(pose)

    center = _body_center_from_pose(pose_arr)
    scale = _estimate_torso_scale(pose)
    if center is None or scale is None:
        return pose_arr

    cx, cy = center
    max_r = TORSO_RADIUS_FACTOR * scale
    max_bone = BONE_MAX_FACTOR * scale

    out = [list(p) for p in pose]

    def visible(j: int) -> bool:
        if j >= J:
            return False
        x, y, c = out[j]
        return (c >= conf_gate) and not (x == 0 and y == 0)

    for j in range(J):
        x, y, c = out[j]
        if c >= conf_gate and not (x == 0 and y == 0):
            if math.hypot(x - cx, y - cy) > max_r:
                out[j] = [0.0, 0.0, 0.0]

    for a, b in COCO18_EDGES:
        if a >= J or b >= J:
            continue
        if not visible(a) or not visible(b):
            continue
        ax, ay, ac = out[a]
        bx, by, bc = out[b]
        d = math.hypot(ax - bx, ay - by)
        if d > max_bone:
            if ac <= bc:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]

    flat: List[float] = []
    for x, y, c in out:
        flat.extend([float(x), float(y), float(c)])
    return flat


def _suppress_isolated_joints_in_pose_arr(
    pose_arr: Optional[List[float]], *, conf_gate: float, keep: set[int] = None
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr

    _build_neighbors()
    pose = _reshape_keypoints_2d(pose_arr)
    J = len(pose)
    out = [list(p) for p in pose]

    if keep is None:
        keep = set()

    def vis(j: int) -> bool:
        if j >= J:
            return False
        x, y, c = out[j]
        return (c >= conf_gate) and not (x == 0 and y == 0)

    for j in range(J):
        if j in keep:
            continue
        if not vis(j):
            continue
        neighs = _NEIGHBORS.get(j, set())
        if not any((n < J and vis(n)) for n in neighs):
            out[j] = [0.0, 0.0, 0.0]

    flat = []
    for x, y, c in out:
        flat.extend([float(x), float(y), float(c)])
    return flat


def _denoise_and_fill_gaps_pose_seq(
    pose_arr_seq: List[Optional[List[float]]],
    *,
    conf_gate: float,
    min_run: int,
    max_gap: int,
) -> List[Optional[List[float]]]:
    if not pose_arr_seq:
        return pose_arr_seq

    J = None
    for arr in pose_arr_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_arr_seq

    T = len(pose_arr_seq)
    out_seq: List[Optional[List[float]]] = []
    for arr in pose_arr_seq:
        if isinstance(arr, list) and len(arr) == J * 3:
            out_seq.append(list(arr))
        else:
            out_seq.append(arr)

    def is_vis(arr: List[float], j: int) -> bool:
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    # 1) remove short flashes
    for j in range(J):
        start = None
        for t in range(T + 1):
            cur = False
            if t < T and isinstance(out_seq[t], list):
                cur = is_vis(out_seq[t], j)
            if cur and start is None:
                start = t
            if (not cur) and start is not None:
                run_len = t - start
                if run_len < min_run:
                    for k in range(start, t):
                        if not isinstance(out_seq[k], list):
                            continue
                        out_seq[k][3 * j + 0] = 0.0
                        out_seq[k][3 * j + 1] = 0.0
                        out_seq[k][3 * j + 2] = 0.0
                start = None

    # 2) gap fill only if returns
    for j in range(J):
        last_vis_t = None
        t = 0
        while t < T:
            arr = out_seq[t]
            if not isinstance(arr, list):
                t += 1
                continue

            cur_vis = is_vis(arr, j)
            if cur_vis:
                last_vis_t = t
                t += 1
                continue

            if last_vis_t is None:
                t += 1
                continue

            gap_start = t
            t2 = t
            while t2 < T:
                arr2 = out_seq[t2]
                if isinstance(arr2, list) and is_vis(arr2, j):
                    break
                t2 += 1

            if t2 >= T:
                break

            gap_len = t2 - gap_start
            if gap_len <= 0:
                t = t2
                continue

            if gap_len <= max_gap:
                a = out_seq[last_vis_t]
                b = out_seq[t2]
                if isinstance(a, list) and isinstance(b, list):
                    ax, ay, ac = float(a[3 * j + 0]), float(a[3 * j + 1]), float(a[3 * j + 2])
                    bx, by, bc = float(b[3 * j + 0]), float(b[3 * j + 1]), float(b[3 * j + 2])
                    if not (ax == 0 and ay == 0) and not (bx == 0 and by == 0):
                        conf_fill = min(ac, bc)
                        for k in range(gap_len):
                            tt = gap_start + k
                            if not isinstance(out_seq[tt], list):
                                continue
                            r = (k + 1) / (gap_len + 1)
                            x = ax + (bx - ax) * r
                            y = ay + (by - ay) * r
                            out_seq[tt][3 * j + 0] = float(x)
                            out_seq[tt][3 * j + 1] = float(y)
                            out_seq[tt][3 * j + 2] = float(conf_fill)

            t = t2

    return out_seq


def _zero_lag_ema_pose_seq(
    pose_seq: List[Optional[List[float]]], *, alpha: float, conf_gate: float
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq

    J = None
    for arr in pose_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_seq

    T = len(pose_seq)

    def is_vis(arr: List[float], j: int) -> bool:
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    fwd = [None] * T
    last = [None] * J
    for t in range(T):
        arr = pose_seq[t]
        if not isinstance(arr, list) or len(arr) != J * 3:
            fwd[t] = arr
            continue
        out = list(arr)
        for j in range(J):
            if is_vis(arr, j):
                x = float(arr[3 * j + 0])
                y = float(arr[3 * j + 1])
                if last[j] is None:
                    sx, sy = x, y
                else:
                    sx = alpha * x + (1 - alpha) * last[j][0]
                    sy = alpha * y + (1 - alpha) * last[j][1]
                last[j] = (sx, sy)
                out[3 * j + 0] = float(sx)
                out[3 * j + 1] = float(sy)
        fwd[t] = out

    bwd = [None] * T
    last = [None] * J
    for t in range(T - 1, -1, -1):
        arr = fwd[t]
        if not isinstance(arr, list) or len(arr) != J * 3:
            bwd[t] = arr
            continue
        out = list(arr)
        for j in range(J):
            if is_vis(arr, j):
                x = float(arr[3 * j + 0])
                y = float(arr[3 * j + 1])
                if last[j] is None:
                    sx, sy = x, y
                else:
                    sx = alpha * x + (1 - alpha) * last[j][0]
                    sy = alpha * y + (1 - alpha) * last[j][1]
                last[j] = (sx, sy)
                out[3 * j + 0] = float(sx)
                out[3 * j + 1] = float(sy)
        bwd[t] = out

    return bwd


def _apply_root_scale(
    pose_arr: Optional[List[float]],
    *,
    src_root: Tuple[float, float],
    src_scale: float,
    dst_root: Tuple[float, float],
    dst_scale: float,
) -> Optional[List[float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return pose_arr
    if src_scale <= 1e-6 or dst_scale <= 1e-6:
        return pose_arr

    kps = _reshape_keypoints_2d(pose_arr)
    out = []
    s = dst_scale / src_scale

    for x, y, c in kps:
        if c <= 0 or (x == 0 and y == 0):
            out.append((x, y, c))
            continue
        nx = dst_root[0] + (x - src_root[0]) * s
        ny = dst_root[1] + (y - src_root[1]) * s
        out.append((nx, ny, c))

    return _flatten_keypoints_2d(out)


def _carry_pose_when_torso_missing(
    pose_seq: List[Optional[List[float]]],
    *,
    conf_gate: float,
    max_carry: int,
    anchor_joints: List[int],
    min_anchors: int,
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq

    J = None
    for arr in pose_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_seq

    out = [a if a is None else list(a) for a in pose_seq]

    FILL_JOINTS = {1, 8, 9, 10, 11, 12, 13}
    FILL_JOINTS -= set(ALLOW_DISAPPEAR_JOINTS)

    def is_vis_flat(arr: List[float], j: int) -> bool:
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    def count_visible(arr: List[float], joints: List[int]) -> int:
        c = 0
        for j in joints:
            if j < J and is_vis_flat(arr, j):
                c += 1
        return c

    def root_scale_from_anchors(arr: List[float]) -> Optional[Tuple[Tuple[float, float], float]]:
        pts = []
        for j in anchor_joints:
            if j >= J:
                continue
            if is_vis_flat(arr, j):
                x = float(arr[3 * j + 0])
                y = float(arr[3 * j + 1])
                pts.append((x, y))
        if len(pts) < min_anchors:
            return None

        rx = sum(p[0] for p in pts) / len(pts)
        ry = sum(p[1] for p in pts) / len(pts)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = max(max(xs) - min(xs), max(ys) - min(ys))
        if scale <= 1e-3:
            return None

        return (rx, ry), float(scale)

    last_good: Optional[List[float]] = None
    last_good_rs: Optional[Tuple[Tuple[float, float], float]] = None
    carry_left = 0

    for t in range(len(out)):
        arr = out[t]
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue

        anchors_ok = count_visible(arr, anchor_joints) >= min_anchors
        fill_vis = sum(1 for j in FILL_JOINTS if j < J and is_vis_flat(arr, j))
        rs = root_scale_from_anchors(arr)

        if anchors_ok and rs is not None and fill_vis >= 2:
            last_good = list(arr)
            last_good_rs = rs
            carry_left = max_carry
            continue

        if anchors_ok and rs is not None and last_good is not None and last_good_rs is not None and carry_left > 0:
            dst_root, dst_scale = rs
            src_root, src_scale = last_good_rs

            carried_full = _apply_root_scale(
                last_good,
                src_root=src_root,
                src_scale=src_scale,
                dst_root=dst_root,
                dst_scale=dst_scale,
            )
            if isinstance(carried_full, list) and len(carried_full) == J * 3:
                for j in FILL_JOINTS:
                    if j >= J:
                        continue
                    if is_vis_flat(arr, j):
                        continue

                    cx = float(carried_full[3 * j + 0])
                    cy = float(carried_full[3 * j + 1])
                    cc = float(carried_full[3 * j + 2])

                    if (cx == 0 and cy == 0) or cc <= 0:
                        continue

                    arr[3 * j + 0] = cx
                    arr[3 * j + 1] = cy
                    arr[3 * j + 2] = max(min(cc, 0.60), conf_gate)

                out[t] = arr
                carry_left -= 1
                continue

        carry_left = max(carry_left - 1, 0)

    return out


def _force_full_torso_pair(
    pose_seq: List[Optional[List[float]]],
    *,
    conf_gate: float,
    anchor_joints: List[int],
    min_anchors: int,
    max_lookback: int = 240,
    fill_legs_with_hip: bool = True,
    always_fill_if_one_hip: bool = True,
) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq

    J = None
    for arr in pose_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_seq

    out = [a if a is None else list(a) for a in pose_seq]

    R_HIP, R_KNEE, R_ANK = 8, 9, 10
    L_HIP, L_KNEE, L_ANK = 11, 12, 13

    def is_vis(arr: List[float], j: int) -> bool:
        if j >= J:
            return False
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    def count_visible(arr: List[float], joints: List[int]) -> int:
        c = 0
        for j in joints:
            if is_vis(arr, j):
                c += 1
        return c

    def root_scale_from_anchors(arr: List[float]) -> Optional[Tuple[Tuple[float, float], float]]:
        pts = []
        for j in anchor_joints:
            if j >= J:
                continue
            if is_vis(arr, j):
                pts.append((float(arr[3 * j + 0]), float(arr[3 * j + 1])))
        if len(pts) < min_anchors:
            return None

        rx = sum(p[0] for p in pts) / len(pts)
        ry = sum(p[1] for p in pts) / len(pts)

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scale = max(max(xs) - min(xs), max(ys) - min(ys))
        if scale <= 1e-3:
            return None
        return (rx, ry), float(scale)

    last_full_idx = None
    last_full = None
    last_full_rs = None

    for t in range(len(out)):
        arr = out[t]
        if not isinstance(arr, list) or len(arr) != J * 3:
            continue

        rs = root_scale_from_anchors(arr)

        r_ok = is_vis(arr, R_HIP)
        l_ok = is_vis(arr, L_HIP)

        anchors_ok = count_visible(arr, anchor_joints) >= min_anchors

        if anchors_ok and rs is not None and r_ok and l_ok:
            last_full_idx = t
            last_full = list(arr)
            last_full_rs = rs
            continue

        if last_full is None or last_full_rs is None or last_full_idx is None:
            continue
        if (t - last_full_idx) > max_lookback:
            continue
        if not (r_ok or l_ok):
            continue
        if r_ok and l_ok:
            continue
        if not always_fill_if_one_hip:
            continue
        if rs is None:
            continue

        dst_root, dst_scale = rs
        src_root, src_scale = last_full_rs

        carried = _apply_root_scale(
            last_full,
            src_root=src_root,
            src_scale=src_scale,
            dst_root=dst_root,
            dst_scale=dst_scale,
        )
        if not (isinstance(carried, list) and len(carried) == J * 3):
            continue

        def copy_joint(j: int):
            if j >= J:
                return
            if is_vis(arr, j):
                return
            cx = float(carried[3 * j + 0])
            cy = float(carried[3 * j + 1])
            cc = float(carried[3 * j + 2])
            if (cx == 0 and cy == 0) or cc <= 0:
                return
            arr[3 * j + 0] = cx
            arr[3 * j + 1] = cy
            arr[3 * j + 2] = max(min(cc, 0.60), conf_gate)

        if not r_ok:
            copy_joint(R_HIP)
            if fill_legs_with_hip:
                copy_joint(R_KNEE)
                copy_joint(R_ANK)

        if not l_ok:
            copy_joint(L_HIP)
            if fill_legs_with_hip:
                copy_joint(L_KNEE)
                copy_joint(L_ANK)

        out[t] = arr

    return out


def _median3_pose_seq(pose_seq: List[Optional[List[float]]], *, conf_gate: float) -> List[Optional[List[float]]]:
    if not pose_seq:
        return pose_seq

    J = None
    for arr in pose_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_seq

    T = len(pose_seq)

    def is_vis(arr: List[float], j: int) -> bool:
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    out_seq: List[Optional[List[float]]] = []
    for t in range(T):
        arr = pose_seq[t]
        if not isinstance(arr, list) or len(arr) != J * 3:
            out_seq.append(arr)
            continue

        out = list(arr)
        t0 = max(0, t - 1)
        t1 = t
        t2 = min(T - 1, t + 1)

        a0 = pose_seq[t0]
        a1 = pose_seq[t1]
        a2 = pose_seq[t2]

        for j in range(J):
            if not is_vis(arr, j):
                continue

            xs, ys = [], []
            for aa in (a0, a1, a2):
                if isinstance(aa, list) and len(aa) == J * 3 and is_vis(aa, j):
                    xs.append(float(aa[3 * j + 0]))
                    ys.append(float(aa[3 * j + 1]))

            if len(xs) >= 2:
                xs.sort()
                ys.sort()
                out[3 * j + 0] = float(xs[len(xs) // 2])
                out[3 * j + 1] = float(ys[len(ys) // 2])

        out_seq.append(out)

    return out_seq


def _sync_group_appearances(
    pose_arr_seq: List[Optional[List[float]]],
    *,
    group: set[int],
    conf_gate: float,
    lookahead: int,
) -> List[Optional[List[float]]]:
    if not pose_arr_seq:
        return pose_arr_seq

    J = None
    for arr in pose_arr_seq:
        if isinstance(arr, list) and len(arr) % 3 == 0 and len(arr) > 0:
            J = len(arr) // 3
            break
    if J is None:
        return pose_arr_seq

    T = len(pose_arr_seq)
    out_seq: List[Optional[List[float]]] = []
    for arr in pose_arr_seq:
        if isinstance(arr, list) and len(arr) == J * 3:
            out_seq.append(list(arr))
        else:
            out_seq.append(arr)

    def is_vis(arr: List[float], j: int) -> bool:
        x = float(arr[3 * j + 0])
        y = float(arr[3 * j + 1])
        c = float(arr[3 * j + 2])
        return (c >= conf_gate) and not (x == 0 and y == 0)

    for t in range(T):
        arr = out_seq[t]
        if not isinstance(arr, list):
            continue

        vis = {j for j in group if j < J and is_vis(arr, j)}
        if not vis:
            continue

        missing = {j for j in group if j < J and j not in vis}
        if not missing:
            continue

        appear_t: dict[int, int] = {}
        for j in list(missing):
            t2 = t + 1
            while t2 < T and t2 <= t + lookahead:
                arr2 = out_seq[t2]
                if isinstance(arr2, list) and is_vis(arr2, j):
                    appear_t[j] = t2
                    break
                t2 += 1

        if not appear_t:
            continue

        for j, t2 in appear_t.items():
            last_t = None
            for tb in range(t - 1, -1, -1):
                arrb = out_seq[tb]
                if isinstance(arrb, list) and is_vis(arrb, j):
                    last_t = tb
                    break

            if last_t is None:
                b = out_seq[t2]
                if not isinstance(b, list):
                    continue
                bx, by, bc = float(b[3 * j + 0]), float(b[3 * j + 1]), float(b[3 * j + 2])
                for k in range(t, t2):
                    a = out_seq[k]
                    if not isinstance(a, list):
                        continue
                    a[3 * j + 0] = bx
                    a[3 * j + 1] = by
                    a[3 * j + 2] = bc
                continue

            a0 = out_seq[last_t]
            b0 = out_seq[t2]
            if not (isinstance(a0, list) and isinstance(b0, list)):
                continue

            ax, ay, ac = float(a0[3 * j + 0]), float(a0[3 * j + 1]), float(a0[3 * j + 2])
            bx, by, bc = float(b0[3 * j + 0]), float(b0[3 * j + 1]), float(b0[3 * j + 2])

            if (ax == 0 and ay == 0) or (bx == 0 and by == 0):
                continue

            conf_fill = min(ac, bc)
            total = t2 - last_t
            if total <= 0:
                continue

            for tt in range(t, t2):
                a = out_seq[tt]
                if not isinstance(a, list):
                    continue
                r = (tt - last_t) / total
                x = ax + (bx - ax) * r
                y = ay + (by - ay) * r
                a[3 * j + 0] = float(x)
                a[3 * j + 1] = float(y)
                a[3 * j + 2] = float(conf_fill)

    return out_seq


def _count_valid_points(arr: Optional[List[float]], *, conf_gate: float) -> int:
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return 0
    cnt = 0
    for i in range(0, len(arr), 3):
        x, y, c = float(arr[i]), float(arr[i + 1]), float(arr[i + 2])
        if c >= conf_gate and not (x == 0 and y == 0):
            cnt += 1
    return cnt


def _zero_out_kps(arr: Optional[List[float]]) -> Optional[List[float]]:
    if not isinstance(arr, list) or len(arr) % 3 != 0:
        return arr
    out = list(arr)
    for i in range(0, len(out), 3):
        out[i + 0] = 0.0
        out[i + 1] = 0.0
        out[i + 2] = 0.0
    return out


def _pin_body_wrist_to_hand(
    p_out: Dict[str, Any],
    *,
    side: str,
    conf_gate_body: float = 0.2,
    conf_gate_hand: float = 0.2,
    blend: float = 1.0,
) -> None:
    if side == "right":
        bw = 4
        hk = "hand_right_keypoints_2d"
    else:
        bw = 7
        hk = "hand_left_keypoints_2d"

    pose = p_out.get("pose_keypoints_2d")
    hand = p_out.get(hk)

    if not (isinstance(pose, list) and isinstance(hand, list)):
        return
    if len(pose) < (bw * 3 + 3):
        return
    if len(hand) < 3:
        return

    hx, hy, hc = float(hand[0]), float(hand[1]), float(hand[2])
    if hc < conf_gate_hand or (hx == 0.0 and hy == 0.0):
        return

    bx, by, bc = float(pose[bw * 3 + 0]), float(pose[bw * 3 + 1]), float(pose[bw * 3 + 2])

    if bc < conf_gate_body or (bx == 0.0 and by == 0.0):
        pose[bw * 3 + 0] = hx
        pose[bw * 3 + 1] = hy
        pose[bw * 3 + 2] = float(max(bc, min(hc, 0.9)))
    else:
        nx = bx * (1.0 - blend) + hx * blend
        ny = by * (1.0 - blend) + hy * blend
        pose[bw * 3 + 0] = nx
        pose[bw * 3 + 1] = ny
        pose[bw * 3 + 2] = float(min(bc, hc))

    p_out["pose_keypoints_2d"] = pose


def _fix_elbow_using_wrist(p_out: Dict[str, Any], *, side: str, conf_gate: float = 0.2) -> None:
    pose = p_out.get("pose_keypoints_2d")
    if not isinstance(pose, list) or len(pose) % 3 != 0:
        return

    if side == "right":
        sh, el, wr = 2, 3, 4
    else:
        sh, el, wr = 5, 6, 7

    def get(j):
        return float(pose[3 * j + 0]), float(pose[3 * j + 1]), float(pose[3 * j + 2])

    def vis(x, y, c):
        return c >= conf_gate and not (x == 0.0 and y == 0.0)

    sx, sy, sc = get(sh)
    ex, ey, ec = get(el)
    wx, wy, wc = get(wr)

    if not (vis(sx, sy, sc) and vis(wx, wy, wc)):
        return

    if vis(ex, ey, ec):
        Lse = math.hypot(ex - sx, ey - sy)
        Lew = math.hypot(wx - ex, wy - ey)
    else:
        dsw = math.hypot(wx - sx, wy - sy)
        if dsw < 1e-3:
            return
        Lse = 0.55 * dsw
        Lew = 0.45 * dsw

    dx = wx - sx
    dy = wy - sy
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return

    d2 = max(min(d, (Lse + Lew) - 1e-3), abs(Lse - Lew) + 1e-3)

    a = (Lse * Lse - Lew * Lew + d2 * d2) / (2.0 * d2)
    h2 = max(Lse * Lse - a * a, 0.0)
    h = math.sqrt(h2)

    ux = dx / d
    uy = dy / d
    px = sx + a * ux
    py = sy + a * uy

    rx = -uy
    ry = ux

    e1x, e1y = px + h * rx, py + h * ry
    e2x, e2y = px - h * rx, py - h * ry

    if vis(ex, ey, ec):
        if math.hypot(e1x - ex, e1y - ey) <= math.hypot(e2x - ex, e2y - ey):
            nx, ny = e1x, e1y
        else:
            nx, ny = e2x, e2y
    else:
        nx, ny = e1x, e1y

    pose[3 * el + 0] = float(nx)
    pose[3 * el + 1] = float(ny)
    pose[3 * el + 2] = float(max(min(ec, 0.8), conf_gate))

    p_out["pose_keypoints_2d"] = pose


def _remove_short_presence_runs_kps_seq(
    seq: List[Optional[List[float]]],
    *,
    conf_gate: float,
    min_points_present: int,
    min_run: int,
) -> List[Optional[List[float]]]:
    if not seq:
        return seq

    present = [(_count_valid_points(a, conf_gate=conf_gate) >= min_points_present) for a in seq]
    out = [None if a is None else list(a) for a in seq]

    start = None
    for t in range(len(seq) + 1):
        cur = present[t] if t < len(seq) else False
        if cur and start is None:
            start = t
        if (not cur) and start is not None:
            run_len = t - start
            if run_len < min_run:
                for k in range(start, t):
                    out[k] = _zero_out_kps(out[k])
            start = None

    return out


def _zero_sparse_frames_kps_seq(
    seq: List[Optional[List[float]]], *, conf_gate: float, min_points_present: int
) -> List[Optional[List[float]]]:
    if not seq:
        return seq

    out: List[Optional[List[float]]] = []
    for a in seq:
        if not isinstance(a, list):
            out.append(a)
            continue
        if _count_valid_points(a, conf_gate=conf_gate) < min_points_present:
            out.append(_zero_out_kps(a))
        else:
            out.append(a)
    return out


def _suppress_spatial_outliers_in_hand_arr(
    hand_arr: Optional[List[float]], *, conf_gate: float, max_bone_factor: float = 3.0
) -> Optional[List[float]]:
    if not isinstance(hand_arr, list) or len(hand_arr) % 3 != 0:
        return hand_arr
    pts = _reshape_keypoints_2d(hand_arr)
    J = len(pts)
    if J < 21:
        return hand_arr

    out = [list(p) for p in pts]

    def vis(j: int) -> bool:
        x, y, c = out[j]
        return c >= conf_gate and not (x == 0 and y == 0)

    vv = [(x, y) for (x, y, c) in out if c >= conf_gate and not (x == 0 and y == 0)]
    if len(vv) < 6:
        return hand_arr
    xs = [p[0] for p in vv]
    ys = [p[1] for p in vv]
    scale = max(max(xs) - min(xs), max(ys) - min(ys))
    if scale <= 1e-3:
        return hand_arr
    max_bone = max_bone_factor * scale

    for a, b in HAND21_EDGES:
        if a >= J or b >= J:
            continue
        if not vis(a) or not vis(b):
            continue
        ax, ay, ac = out[a]
        bx, by, bc = out[b]
        d = math.hypot(ax - bx, ay - by)
        if d > max_bone:
            if ac <= bc:
                out[a] = [0.0, 0.0, 0.0]
            else:
                out[b] = [0.0, 0.0, 0.0]

    return _flatten_keypoints_2d([(x, y, c) for x, y, c in out])


def _body_head_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, conf_gate: float
) -> Optional[Tuple[Tuple[float, float], float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)

    def vis(j: int) -> Optional[Tuple[float, float]]:
        if j >= len(kps):
            return None
        x, y, c = kps[j]
        if c >= conf_gate and not (x == 0 and y == 0):
            return (float(x), float(y))
        return None

    pts = []
    for j in [0, 1, 14, 15, 16, 17]:
        p = vis(j)
        if p is not None:
            pts.append(p)

    if not pts:
        return None

    rx = sum(p[0] for p in pts) / len(pts)
    ry = sum(p[1] for p in pts) / len(pts)
    root = (rx, ry)

    def dist(a: int, b: int) -> Optional[float]:
        pa, pb = vis(a), vis(b)
        if pa is None or pb is None:
            return None
        d = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        return d if d > 1e-3 else None

    cands = [dist(14, 15), dist(16, 17), dist(2, 5)]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None

    scale = float(sum(cands) / len(cands))
    return root, scale


def _body_wrist_root_scale_from_pose(
    pose_arr: Optional[List[float]], *, side: str, conf_gate: float
) -> Optional[Tuple[Tuple[float, float], float]]:
    if not isinstance(pose_arr, list) or len(pose_arr) % 3 != 0:
        return None
    kps = _reshape_keypoints_2d(pose_arr)

    if side == "right":
        w, e = 4, 3
    else:
        w, e = 7, 6

    def vis(j: int) -> Optional[Tuple[float, float]]:
        if j >= len(kps):
            return None
        x, y, c = kps[j]
        if c >= conf_gate and not (x == 0 and y == 0):
            return (float(x), float(y))
        return None

    pw = vis(w)
    if pw is None:
        return None
    root = pw

    pe = vis(e)
    scale = None
    if pe is not None:
        d = math.hypot(pw[0] - pe[0], pw[1] - pe[1])
        if d > 1e-3:
            scale = d

    if scale is None:
        p2 = vis(2)
        p5 = vis(5)
        if p2 is not None and p5 is not None:
            d = math.hypot(p2[0] - p5[0], p2[1] - p5[1])
            if d > 1e-3:
                scale = d

    if scale is None:
        return None

    return root, float(scale)


def _smooth_dense_seq_anchored_to_body(
    dense_seq: List[Optional[List[float]]],
    body_pose_seq: List[Optional[List[float]]],
    *,
    kind: str,
    conf_gate_dense: float,
    conf_gate_body: float,
    median3: bool,
    zero_lag_alpha: float,
) -> List[Optional[List[float]]]:
    if not dense_seq:
        return dense_seq

    Jd = None
    for a in dense_seq:
        if isinstance(a, list) and len(a) % 3 == 0 and len(a) > 0:
            Jd = len(a) // 3
            break
    if Jd is None:
        return dense_seq

    T = len(dense_seq)
    out = [None if a is None else list(a) for a in dense_seq]

    norm_seq: List[Optional[List[float]]] = [None] * T

    for t in range(T):
        arr = out[t]
        body = body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arr, list) or len(arr) != Jd * 3 or not isinstance(body, list):
            norm_seq[t] = arr
            continue

        if kind == "face":
            rs = _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
        elif kind == "hand_left":
            rs = _body_wrist_root_scale_from_pose(body, side="left", conf_gate=conf_gate_body)
        else:
            rs = _body_wrist_root_scale_from_pose(body, side="right", conf_gate=conf_gate_body)

        if rs is None:
            norm_seq[t] = arr
            continue

        (rx, ry), s = rs
        if s <= 1e-6:
            norm_seq[t] = arr
            continue

        nn = list(arr)
        for j in range(Jd):
            x = float(arr[3 * j + 0])
            y = float(arr[3 * j + 1])
            c = float(arr[3 * j + 2])
            if c >= conf_gate_dense and not (x == 0 and y == 0):
                nn[3 * j + 0] = (x - rx) / s
                nn[3 * j + 1] = (y - ry) / s
        norm_seq[t] = nn

    if median3:
        norm_seq = _median3_pose_seq(norm_seq, conf_gate=conf_gate_dense)

    norm_seq = _zero_lag_ema_pose_seq(norm_seq, alpha=zero_lag_alpha, conf_gate=conf_gate_dense)

    for t in range(T):
        arrn = norm_seq[t]
        body = body_pose_seq[t] if t < len(body_pose_seq) else None
        if not isinstance(arrn, list) or len(arrn) != Jd * 3 or not isinstance(body, list):
            continue

        if kind == "face":
            rs = _body_head_root_scale_from_pose(body, conf_gate=conf_gate_body)
        elif kind == "hand_left":
            rs = _body_wrist_root_scale_from_pose(body, side="left", conf_gate=conf_gate_body)
        else:
            rs = _body_wrist_root_scale_from_pose(body, side="right", conf_gate=conf_gate_body)

        if rs is None:
            continue

        (rx, ry), s = rs
        if s <= 1e-6:
            continue

        orig = out[t]
        for j in range(Jd):
            x = float(arrn[3 * j + 0])
            y = float(arrn[3 * j + 1])
            c = float(arrn[3 * j + 2])

            ox = float(orig[3 * j + 0])
            oy = float(orig[3 * j + 1])
            oc = float(orig[3 * j + 2])

            if oc >= conf_gate_dense and not (ox == 0 and oy == 0) and c >= conf_gate_dense:
                orig[3 * j + 0] = rx + x * s
                orig[3 * j + 1] = ry + y * s

        out[t] = orig

    return out


def smooth_KPS_json_obj(
    data: Any,
    *,
    keep_face_untouched: bool = True,
    keep_hands_untouched: bool = True,
    filter_extra_people: Optional[bool] = None,
) -> Any:
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON to be a list of frames.")

    if filter_extra_people is None:
        filter_extra_people = bool(FILTER_EXTRA_PEOPLE)

    chosen_people: List[Optional[Dict[str, Any]]] = [None] * len(data)

    if MAIN_PERSON_MODE == "longest_track":
        tracks = _build_tracks_over_video(data)
        main_tr = _pick_main_track(tracks)

        if main_tr is not None:
            for t in range(len(data)):
                if t in main_tr.frames:
                    chosen_people[t] = main_tr.frames[t]
        else:
            prev_center: Optional[Tuple[float, float]] = None
            for i, frame in enumerate(data):
                if not isinstance(frame, dict):
                    continue
                people = frame.get("people", [])
                if not isinstance(people, list) or len(people) == 0:
                    continue
                chosen = _choose_single_person(people, prev_center)
                chosen_people[i] = chosen
                if chosen is not None:
                    c = _body_center_from_pose(chosen.get("pose_keypoints_2d"))
                    if c is not None:
                        prev_center = c
    else:
        prev_center: Optional[Tuple[float, float]] = None
        for i, frame in enumerate(data):
            if not isinstance(frame, dict):
                continue
            people = frame.get("people", [])
            if not isinstance(people, list) or len(people) == 0:
                continue
            chosen = _choose_single_person(people, prev_center)
            chosen_people[i] = chosen
            if chosen is not None:
                c = _body_center_from_pose(chosen.get("pose_keypoints_2d"))
                if c is not None:
                    prev_center = c

    pose_seq: List[Optional[List[float]]] = []
    for p in chosen_people:
        pose_seq.append(p.get("pose_keypoints_2d") if isinstance(p, dict) else None)

    if SPATIAL_OUTLIER_FIX:
        pose_seq = [
            _suppress_spatial_outliers_in_pose_arr(arr, conf_gate=CONF_GATE_BODY) if arr is not None else None
            for arr in pose_seq
        ]

    if GAP_FILL_ENABLED:
        pose_seq = _denoise_and_fill_gaps_pose_seq(
            pose_seq,
            conf_gate=CONF_GATE_BODY,
            min_run=MIN_RUN_FRAMES,
            max_gap=MAX_GAP_FRAMES,
        )

    if TORSO_SYNC_ENABLED:
        pose_seq = _sync_group_appearances(
            pose_seq,
            group=TORSO_JOINTS,
            conf_gate=CONF_GATE_BODY,
            lookahead=TORSO_LOOKAHEAD_FRAMES,
        )

    pose_seq = [
        (
            _suppress_isolated_joints_in_pose_arr(arr, conf_gate=CONF_GATE_BODY, keep=TORSO_JOINTS)
            if arr is not None
            else None
        )
        for arr in pose_seq
    ]

    if MEDIAN3_ENABLED:
        pose_seq = _median3_pose_seq(pose_seq, conf_gate=CONF_GATE_BODY)

    if SUPER_SMOOTH_ENABLED:
        pose_seq = _zero_lag_ema_pose_seq(pose_seq, alpha=SUPER_SMOOTH_ALPHA, conf_gate=SUPER_SMOOTH_MIN_CONF)

    if ROOTSCALE_CARRY_ENABLED:
        pose_seq = _carry_pose_when_torso_missing(
            pose_seq,
            conf_gate=CARRY_CONF_GATE,
            max_carry=CARRY_MAX_FRAMES,
            anchor_joints=CARRY_ANCHOR_JOINTS,
            min_anchors=CARRY_MIN_ANCHORS,
        )

    pose_seq = _force_full_torso_pair(
        pose_seq,
        conf_gate=CARRY_CONF_GATE,
        anchor_joints=CARRY_ANCHOR_JOINTS,
        min_anchors=CARRY_MIN_ANCHORS,
        max_lookback=240,
        fill_legs_with_hip=True,
        always_fill_if_one_hip=True,
    )

    face_seq: List[Optional[List[float]]] = []
    lh_seq: List[Optional[List[float]]] = []
    rh_seq: List[Optional[List[float]]] = []

    for p in chosen_people:
        if isinstance(p, dict):
            face_seq.append(p.get("face_keypoints_2d"))
            lh_seq.append(p.get("hand_left_keypoints_2d"))
            rh_seq.append(p.get("hand_right_keypoints_2d"))
        else:
            face_seq.append(None)
            lh_seq.append(None)
            rh_seq.append(None)

    if HANDS_SMOOTH_ENABLED and (not keep_hands_untouched):
        lh_seq = [
            _suppress_spatial_outliers_in_hand_arr(a, conf_gate=CONF_GATE_HAND) if a is not None else None
            for a in lh_seq
        ]
        rh_seq = [
            _suppress_spatial_outliers_in_hand_arr(a, conf_gate=CONF_GATE_HAND) if a is not None else None
            for a in rh_seq
        ]

        lh_seq = _remove_short_presence_runs_kps_seq(
            lh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT, min_run=MIN_HAND_RUN_FRAMES
        )
        rh_seq = _remove_short_presence_runs_kps_seq(
            rh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT, min_run=MIN_HAND_RUN_FRAMES
        )

        lh_seq = _zero_sparse_frames_kps_seq(
            lh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT
        )
        rh_seq = _zero_sparse_frames_kps_seq(
            rh_seq, conf_gate=CONF_GATE_HAND, min_points_present=HAND_MIN_POINTS_PRESENT
        )

        if DENSE_GAP_FILL_ENABLED:
            lh_seq = _denoise_and_fill_gaps_pose_seq(
                lh_seq, conf_gate=CONF_GATE_HAND, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )
            rh_seq = _denoise_and_fill_gaps_pose_seq(
                rh_seq, conf_gate=CONF_GATE_HAND, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )

    if FACE_SMOOTH_ENABLED and (not keep_face_untouched):
        if DENSE_GAP_FILL_ENABLED:
            face_seq = _denoise_and_fill_gaps_pose_seq(
                face_seq, conf_gate=CONF_GATE_FACE, min_run=DENSE_MIN_RUN_FRAMES, max_gap=DENSE_MAX_GAP_FRAMES
            )

    if FACE_SMOOTH_ENABLED and (not keep_face_untouched):
        face_seq = _smooth_dense_seq_anchored_to_body(
            face_seq,
            pose_seq,
            kind="face",
            conf_gate_dense=CONF_GATE_FACE,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )

    if HANDS_SMOOTH_ENABLED and (not keep_hands_untouched):
        lh_seq = _smooth_dense_seq_anchored_to_body(
            lh_seq,
            pose_seq,
            kind="hand_left",
            conf_gate_dense=CONF_GATE_HAND,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )
        rh_seq = _smooth_dense_seq_anchored_to_body(
            rh_seq,
            pose_seq,
            kind="hand_right",
            conf_gate_dense=CONF_GATE_HAND,
            conf_gate_body=CONF_GATE_BODY,
            median3=DENSE_MEDIAN3_ENABLED,
            zero_lag_alpha=DENSE_SUPER_SMOOTH_ALPHA,
        )

    out_frames = []
    body_state: Optional[BodyState] = None

    for i, frame in enumerate(data):
        if not isinstance(frame, dict):
            out_frames.append(frame)
            continue

        frame_out = copy.deepcopy(frame)
        chosen = chosen_people[i]

        if chosen is None:
            if filter_extra_people:
                frame_out["people"] = []
            out_frames.append(frame_out)
            continue

        p_out = copy.deepcopy(chosen)
        p_out["pose_keypoints_2d"] = pose_seq[i]

        pose_arr = p_out.get("pose_keypoints_2d")
        joints = (len(pose_arr) // 3) if isinstance(pose_arr, list) else 0
        if body_state is None:
            body_state = BodyState(joints if joints > 0 else 18)

        p_out["pose_keypoints_2d"] = _smooth_body_pose(p_out.get("pose_keypoints_2d"), body_state)

        if FACE_SMOOTH_ENABLED and (not keep_face_untouched):
            p_out["face_keypoints_2d"] = face_seq[i]
        else:
            p_out["face_keypoints_2d"] = chosen.get("face_keypoints_2d", p_out.get("face_keypoints_2d"))

        if HANDS_SMOOTH_ENABLED and (not keep_hands_untouched):
            p_out["hand_left_keypoints_2d"] = lh_seq[i]
            p_out["hand_right_keypoints_2d"] = rh_seq[i]
        else:
            p_out["hand_left_keypoints_2d"] = chosen.get("hand_left_keypoints_2d", p_out.get("hand_left_keypoints_2d"))
            p_out["hand_right_keypoints_2d"] = chosen.get(
                "hand_right_keypoints_2d", p_out.get("hand_right_keypoints_2d")
            )

        _pin_body_wrist_to_hand(
            p_out, side="left", conf_gate_body=CONF_GATE_BODY, conf_gate_hand=CONF_GATE_HAND, blend=1.0
        )
        _pin_body_wrist_to_hand(
            p_out, side="right", conf_gate_body=CONF_GATE_BODY, conf_gate_hand=CONF_GATE_HAND, blend=1.0
        )

        _fix_elbow_using_wrist(p_out, side="left", conf_gate=CONF_GATE_BODY)
        _fix_elbow_using_wrist(p_out, side="right", conf_gate=CONF_GATE_BODY)

        if filter_extra_people:
            frame_out["people"] = [p_out]
        else:
            orig_people = frame.get("people", [])
            if not isinstance(orig_people, list):
                frame_out["people"] = [p_out]
            else:
                replaced = False
                new_people = []
                for op in orig_people:
                    if (not replaced) and (op is chosen):
                        new_people.append(p_out)
                        replaced = True
                    else:
                        new_people.append(copy.deepcopy(op))
                if not replaced:
                    new_people = [p_out] + [copy.deepcopy(op) for op in orig_people]
                frame_out["people"] = new_people

        out_frames.append(frame_out)

    return out_frames


# ============================================================
# === END: smooth_KPS_json.py logic
# ============================================================


# ============================================================
# === START: render_pose_video.py logic (ported to frame render)
# ============================================================

OP_COLORS: List[Tuple[int, int, int]] = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
]

BODY_EDGES: List[Tuple[int, int]] = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]

BODY_EDGE_COLORS = OP_COLORS[: len(BODY_EDGES)]
BODY_JOINT_COLORS = OP_COLORS

HAND_EDGES: List[Tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _valid_pt(x: float, y: float, c: float, conf_thresh: float) -> bool:
    return (c is not None) and (c >= conf_thresh) and not (x == 0 and y == 0)


def _hsv_to_bgr(h: float, s: float, v: float) -> Tuple[int, int, int]:
    H = int(np.clip(h, 0.0, 1.0) * 179.0)
    S = int(np.clip(s, 0.0, 1.0) * 255.0)
    V = int(np.clip(v, 0.0, 1.0) * 255.0)
    hsv = np.uint8([[[H, S, V]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _looks_normalized(points: List[Tuple[float, float, float]], conf_thresh: float) -> bool:
    valid = [(x, y, c) for (x, y, c) in points if _valid_pt(x, y, c, conf_thresh)]
    if not valid:
        return False
    in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
    return (in01 / float(len(valid))) >= 0.7


def _draw_body(
    canvas: np.ndarray, pose: List[Tuple[float, float, float]], conf_thresh: float, xinsr_stick_scaling: bool = False
) -> None:
    CH, CW = canvas.shape[:2]
    stickwidth = 2

    valid = [(x, y, c) for (x, y, c) in pose if _valid_pt(x, y, c, conf_thresh)]
    norm = False
    if valid:
        in01 = sum(1 for (x, y, _) in valid if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
        norm = (in01 / float(len(valid))) >= 0.7

    def to_px(x: float, y: float) -> Tuple[float, float]:
        if norm:
            return x * CW, y * CH
        return x, y

    max_side = max(CW, CH)
    if xinsr_stick_scaling:
        stick_scale = 1 if max_side < 500 else min(2 + (max_side // 1000), 7)
    else:
        stick_scale = 1

    for idx, (a, b) in enumerate(BODY_EDGES):
        if a >= len(pose) or b >= len(pose):
            continue

        ax, ay, ac = pose[a]
        bx, by, bc = pose[b]
        if not (_valid_pt(ax, ay, ac, conf_thresh) and _valid_pt(bx, by, bc, conf_thresh)):
            continue

        ax, ay = to_px(ax, ay)
        bx, by = to_px(bx, by)

        base = BODY_EDGE_COLORS[idx] if idx < len(BODY_EDGE_COLORS) else (255, 255, 255)

        X = np.array([ay, by], dtype=np.float32)
        Y = np.array([ax, bx], dtype=np.float32)

        mX = float(np.mean(X))
        mY = float(np.mean(Y))
        length = float(np.hypot(X[0] - X[1], Y[0] - Y[1]))
        if length < 1.0:
            continue

        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))

        polygon = cv2.ellipse2Poly(
            (int(mY), int(mX)),
            (int(length / 2), int(stickwidth * stick_scale)),
            int(angle),
            0,
            360,
            1,
        )

        cv2.fillConvexPoly(
            canvas,
            polygon,
            (int(base[0] * 0.6), int(base[1] * 0.6), int(base[2] * 0.6)),
        )

    for j, (x, y, c) in enumerate(pose):
        if not _valid_pt(x, y, c, conf_thresh):
            continue
        x, y = to_px(x, y)
        col = BODY_JOINT_COLORS[j] if j < len(BODY_JOINT_COLORS) else (255, 255, 255)
        cv2.circle(canvas, (int(x), int(y)), 2, col, thickness=-1)


def _draw_hand(canvas: np.ndarray, hand: List[Tuple[float, float, float]], conf_thresh: float) -> None:
    if not hand or len(hand) < 21:
        return

    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(hand, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    n_edges = len(HAND_EDGES)
    for i, (a, b) in enumerate(HAND_EDGES):
        x1, y1, c1 = hand[a]
        x2, y2, c2 = hand[b]
        if _valid_pt(x1, y1, c1, conf_thresh) and _valid_pt(x2, y2, c2, conf_thresh):
            x1, y1 = to_px(x1, y1)
            x2, y2 = to_px(x2, y2)
            bgr = _hsv_to_bgr(i / float(n_edges), 1.0, 1.0)
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 1, cv2.LINE_AA)

    for x, y, c in hand:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 1, (0, 0, 255), -1, cv2.LINE_AA)


def _draw_face(canvas: np.ndarray, face: List[Tuple[float, float, float]], conf_thresh: float) -> None:
    if not face:
        return

    CH, CW = canvas.shape[:2]
    norm = _looks_normalized(face, conf_thresh)

    def to_px(x: float, y: float) -> Tuple[float, float]:
        return (x * CW, y * CH) if norm else (x, y)

    for x, y, c in face:
        if _valid_pt(x, y, c, conf_thresh):
            x, y = to_px(x, y)
            cv2.circle(canvas, (int(x), int(y)), 0, (255, 255, 255), -1, cv2.LINE_AA)


def _draw_pose_frame_full(
    w: int,
    h: int,
    person: Dict[str, Any],
    conf_thresh_body: float = 0.10,
    conf_thresh_hands: float = 0.10,
    conf_thresh_face: float = 0.10,
) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)

    pose = _reshape_keypoints_2d(person.get("pose_keypoints_2d") or [])
    face = _reshape_keypoints_2d(person.get("face_keypoints_2d") or [])
    hand_l = _reshape_keypoints_2d(person.get("hand_left_keypoints_2d") or [])
    hand_r = _reshape_keypoints_2d(person.get("hand_right_keypoints_2d") or [])

    if pose:
        _draw_body(img, pose, conf_thresh_body)
    if hand_l:
        _draw_hand(img, hand_l, conf_thresh_hands)
    if hand_r:
        _draw_hand(img, hand_r, conf_thresh_hands)
    if face:
        _draw_face(img, face, conf_thresh_face)

    return img


# ============================================================
# === END: render_pose_video.py logic
# ============================================================


# ============================================================
# ComfyUI mappings
# ============================================================

NODE_CLASS_MAPPINGS = {
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSPoseDataSmoother": "KPS: Smooth + Render (pose_data/PKL)",
}
