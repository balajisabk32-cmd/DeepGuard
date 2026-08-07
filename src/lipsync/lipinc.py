"""LIPINC-V2 lip-sync deepfake detector.

    lipinc.analyze_video(path) -> LipincResult      # P(manipulated)

WHY THIS MODEL
LIPINC-V2 (Datta, Jia, Lyu — arXiv 2504.01470) targets lip-syncing deepfakes
specifically: a vision temporal transformer over mouth-region crops looking for
spatiotemporal inconsistency across adjacent frames. Trained against Wav2Lip,
Wav2Lip_GAN, Diff2Lip, Video_Retalking and IP_LAP — exactly the attack class the
classical MAR-vs-envelope branch cannot separate.

USES UPSTREAM'S OWN PREPROCESSING, DELIBERATELY
An earlier version of this module reimplemented dlib's 68-point alignment on
MediaPipe's mesh, because dlib has no cp313 source build here. It loaded and ran
and returned ~0.000 for every clip including known fakes — the crops were far
enough from the training distribution to be meaningless. `dlib-bin` provides a
prebuilt cp313 wheel, so upstream's `utils.get_color_structure_frames` is now
called directly. Reimplementing a trained model's preprocessing is a silent
accuracy leak; use the original whenever it can be made to run.

POLARITY — INVERTED RELATIVE TO EVERY OTHER MODEL HERE
demo.py reads `result[0][1]` under the comment "#real probability". For LIPINC
index 1 is REAL, index 0 is FAKE — the opposite of xception/capsule/effb7. This
module returns P(manipulated) = out[0][0] so the repo-wide
`*_manipulation_score` convention holds at the boundary.

KERAS 2 vs 3
The checkpoint is a Keras 2 .hdf5. Under Keras 3 the rebuilt layer expects 66
weights while the file holds 67 (and the layer is renamed), so it will not load.
TF_USE_LEGACY_KERAS=1 routes tf.keras to the tf-keras package and it loads
exactly. The variable must be set BEFORE the first tensorflow import.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# Must precede any tensorflow import (see docstring).
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LIPINC_REPO = Path(os.environ.get(
    "LIPINC_REPO", REPO.parent / "Repos" / "LIPINC-V2"))
WEIGHTS = REPO / "models" / "lipinc_v2.hdf5"
SHAPE_PREDICTOR = REPO / "models" / "shape_predictor_68_face_landmarks.dat"

N_LOCAL = 5          # -> 8 combined frames (5 local + 3 global), 7 residue


@dataclass
class LipincResult:
    lipinc_manipulation_score: float = 0.5
    lipinc_quality: float = 0.0
    local_frame_ids: list = field(default_factory=list)
    global_frame_ids: list = field(default_factory=list)
    processing_time_ms: int = 0
    degraded_reason: str | None = None


def _deps_present() -> bool:
    return WEIGHTS.exists() and SHAPE_PREDICTOR.exists() and \
        (LIPINC_REPO / "utils.py").exists()


def _patch_imutils_for_opencv5() -> None:
    """imutils 0.5.4 is incompatible with OpenCV 5; coerce the argument types.

    `imutils.face_utils.FaceAligner.align` builds its rotation centre with
    integer division on numpy values and hands the resulting numpy-int64 tuple to
    cv2.getRotationMatrix2D. OpenCV 5 rejects it:

        TypeError: Can't parse 'center'. Sequence item with index 0 has a
                   wrong type

    Upstream swallows that inside a bare `except: continue`, so every frame was
    silently discarded and the only symptom was "Number of frames with faces 0" —
    a face-detection message for what is actually a type error. imutils is
    unmaintained, so the fix is a narrow shim that coerces to Python floats and
    changes nothing else.
    """
    import cv2

    if getattr(cv2.getRotationMatrix2D, "_deepguard_shim", False):
        return
    original = cv2.getRotationMatrix2D

    def shimmed(center, angle, scale):
        return original((float(center[0]), float(center[1])),
                        float(angle), float(scale))

    shimmed._deepguard_shim = True
    cv2.getRotationMatrix2D = shimmed


FACE_CAP = 220           # aligned faces to collect before attempting a match


def _build_face_array(utils, video_path: str, cap: int | None = FACE_CAP):
    """Upstream's create_face_array at FULL 800px detection width, optionally
    stopping after `cap` aligned faces.

    NEVER CHANGE THE DETECTION WIDTH. An earlier attempt combined a 400px width
    with a frame cap and inverted the result on the same clip:

        FAKE2.mp4   800px, all frames  -> P(fake) 0.974
                    400px + 220 cap    -> P(fake) 0.029

    Isolating the two variables showed the WIDTH was the culprit, not the cap.
    Detecting at lower resolution shifts which frames pass the mouth-openness
    filter, which changes the selected 5-frame window, which changes the input
    entirely.

    The cap on its own is safe, because find_LGframes walks frames in order and
    breaks at the first valid window. Measured on FAKE2.mp4 at full width:

        cap  faces  build   local/global      P(fake)
        none   385   91.0s  [19,34,..]/[17,28,32]  0.9737
        300    300   58.9s  identical              0.9737
        200    200   29.8s  identical              0.9737   <- 3x faster, bit-identical
        120    120   21.2s  no global match found

    So the cap is a genuine 3x win with no fidelity cost, provided the caller
    falls back to uncapped when the match fails (see analyze_video).
    """
    import cv2
    import imutils

    fa = utils.FaceAligner(utils.predictor, desiredFaceWidth=256)
    capture = cv2.VideoCapture(video_path)
    faces = []
    try:
        while cap is None or len(faces) < cap:
            ok, frame = capture.read()
            if not ok:
                break
            try:
                image = imutils.resize(frame, width=800)   # 800 is load-bearing
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces.append(fa.align(image, gray, utils.detector_pre(gray)[0]))
            except Exception:
                continue
    finally:
        capture.release()
    return np.asarray(faces)


def _select_frames(utils, video_path: str):
    """(combined, residue, local_ids, global_ids) or all-None.

    Tries the capped pass first (3x faster, bit-identical when it succeeds) and
    falls back to the full clip when no geometry-matching global frames are found
    within the cap.
    """
    for cap in (FACE_CAP, None):
        face_array = _build_face_array(utils, video_path, cap=cap)
        if len(face_array) < 31:
            continue
        try:
            local, glob, l_id, g_id = utils.find_LGframes(
                N_LOCAL, face_array, utils.predictor)
        except UnboundLocalError:
            # Upstream bug: `g_id` is only bound inside the match loop, so a clip
            # with no geometry-matching global frames raises instead of returning
            # empty. Treat it as "no match" and let the fallback try more frames.
            continue
        local = np.asarray(local)
        glob = np.asarray(glob)
        if local.size == 0 or glob.size == 0:
            continue
        combined = np.concatenate((local, glob), axis=0)
        return combined, utils.create_residue(combined), l_id, g_id
    return None, None, None, None


@lru_cache(maxsize=1)
def _load():
    """(model, utils_module) or (None, None).

    utils.py resolves `shape_predictor_68_face_landmarks.dat` relative to the
    CURRENT WORKING DIRECTORY at import time, so the import happens with cwd set
    to models/. Without that it raises on import, not on use.
    """
    if not _deps_present():
        return None, None
    try:
        import sys

        _patch_imutils_for_opencv5()

        if str(LIPINC_REPO) not in sys.path:
            sys.path.insert(0, str(LIPINC_REPO))
        with contextlib.chdir(SHAPE_PREDICTOR.parent):
            import utils as lipinc_utils
        import tensorflow as tf  # noqa: F401  (triggers legacy-keras routing)

        from model import LIPINC_V2

        net = LIPINC_V2()
        net.load_weights(str(WEIGHTS))
        return net, lipinc_utils
    except Exception:
        return None, None


def weights_available() -> bool:
    return _load()[0] is not None


def analyze_video(video_path: str, **_ignored) -> LipincResult:
    """LIPINC-V2 score for a clip. Never raises.

    `**_ignored` absorbs pipeline kwargs (clip=, max_sec=) that do not apply:
    upstream reads the file itself and needs its own frame selection, so a
    pre-decoded clip cannot be reused here.
    """
    t0 = time.time()
    ms = lambda: int((time.time() - t0) * 1000)  # noqa: E731
    try:
        net, utils = _load()
        if net is None:
            return LipincResult(degraded_reason="weights_or_deps_unavailable",
                                processing_time_ms=ms())

        # Resolve BEFORE the chdir. Inside it, a relative path like
        # "TEST_VIDEOS/FAKE.mp4" resolves against models/ instead of the repo
        # root, VideoCapture silently opens nothing, and upstream reports
        # "Number of frames with faces 0" — which reads as a face-detection
        # failure rather than a missing file.
        resolved = str(Path(video_path).resolve())

        combined, residue, l_id, g_id = _select_frames(utils, resolved)
        if combined is None:
            return LipincResult(degraded_reason="no_matching_global_frames",
                                processing_time_ms=ms())

        out = net.predict([combined[None, ...], residue[None, ...]], verbose=0)
        # index 1 is REAL upstream; this repo reports P(manipulated).
        return LipincResult(
            lipinc_manipulation_score=round(float(out[0][0]), 4),
            lipinc_quality=1.0,
            local_frame_ids=list(l_id) if l_id is not None else [],
            global_frame_ids=list(g_id) if g_id is not None else [],
            processing_time_ms=ms(),
        )
    except Exception as exc:  # noqa: BLE001
        return LipincResult(degraded_reason=f"unhandled:{type(exc).__name__}",
                            processing_time_ms=ms())
