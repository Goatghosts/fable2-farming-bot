import os
import json
import time
import numpy as np
import cv2
import mss
import vgamepad as vg

A_PRE_PRESS_DELAY_MS = 30

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
    "axis_points_screen": None,
    "roi_screen": None,
    "axis_points_roi": None,
    "axis_samples_roi": None,
    "roi_padding": 80,
    "axis_resample_points": 260,
    "axis_sample_radius": 0,
    "green_hsv_low": [40, 80, 80],
    "green_hsv_high": [90, 255, 255],
    "green_open": 1,
    "green_close": 2,
    "green_min_area": 60,
    "green_max_area": 12000,
    "green_hold_frames": 4,
    "green_smooth_alpha": 0.35,
    "green_min_width_s": 0.015,
    "green_max_width_s": 0.22,
    "green_min_run": 4,
    "white_hsv_low": [0, 0, 185],
    "white_hsv_high": [180, 140, 255],
    "white_v_min": 185,
    "white_s_max": 160,
    "white_open": 1,
    "white_min_area": 18,
    "white_max_area": 2600,
    "white_min_circularity": 0.50,
    "white_min_run": 2,
    "white_max_run": 24,
    "axis_dist_white": 26,
    "axis_dist_green": 34,
    "track_max_jump_s": 0.10,
    "lost_reset_frames": 10,
    "arming_frames": 3,
    "confirm_in_green_frames": 1,
    "lead_ms": 55,
    "press_ms": 45,
    "cooldown_ms": 90,
    "show_debug": True,
}


def load_config():
    if not os.path.exists(CONFIG_PATH):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        if isinstance(data, dict):
            cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def grab_frame(sct, monitor):
    img = np.array(sct.grab(monitor), dtype=np.uint8)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def press_a(gamepad, press_ms):
    time.sleep(max(0.0, float(A_PRE_PRESS_DELAY_MS) / 1000.0))
    gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()
    time.sleep(max(0.0, press_ms / 1000.0))
    gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
    gamepad.update()


def resample_polyline(pts, n):
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] < 2:
        return p
    d = p[1:] - p[:-1]
    seg = np.sqrt((d[:, 0] ** 2) + (d[:, 1] ** 2))
    total = float(np.sum(seg))
    if total <= 1e-6:
        return np.repeat(p[:1], n, axis=0)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    t = np.linspace(0.0, total, n).astype(np.float32)

    out = np.zeros((n, 2), dtype=np.float32)
    j = 0
    for i in range(n):
        ti = float(t[i])
        while j < len(cum) - 2 and ti > float(cum[j + 1]):
            j += 1
        t0 = float(cum[j])
        t1 = float(cum[j + 1])
        if t1 - t0 <= 1e-6:
            out[i] = p[j]
        else:
            a = (ti - t0) / (t1 - t0)
            out[i] = (1.0 - a) * p[j] + a * p[j + 1]
    return out


def compute_roi_from_points(pts, pad, screen_w, screen_h):
    p = np.asarray(pts, dtype=np.int32)
    x0 = int(np.min(p[:, 0])) - int(pad)
    y0 = int(np.min(p[:, 1])) - int(pad)
    x1 = int(np.max(p[:, 0])) + int(pad)
    y1 = int(np.max(p[:, 1])) + int(pad)
    x0 = max(0, min(screen_w - 2, x0))
    y0 = max(0, min(screen_h - 2, y0))
    x1 = max(x0 + 1, min(screen_w - 1, x1))
    y1 = max(y0 + 1, min(screen_h - 1, y1))
    return [x0, y0, x1 - x0, y1 - y0]


def axis_selection_ui(frame_bgr):
    h, w = frame_bgr.shape[:2]
    max_w = 1400
    scale = 1.0
    view = frame_bgr.copy()
    if w > max_w:
        scale = float(max_w) / float(w)
        view = cv2.resize(frame_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    pts = []
    win = "Axis select: LMB add | RMB undo | Enter finish | Esc cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    state = {"done": False, "cancel": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ox = int(round(float(x) / scale))
            oy = int(round(float(y) / scale))
            pts.append((ox, oy))
        if event == cv2.EVENT_RBUTTONDOWN:
            if pts:
                pts.pop()

    cv2.setMouseCallback(win, on_mouse)

    while True:
        dbg = view.copy()
        if pts:
            pts_view = [(int(round(p[0] * scale)), int(round(p[1] * scale))) for p in pts]
            for i in range(1, len(pts_view)):
                cv2.line(dbg, pts_view[i - 1], pts_view[i], (0, 255, 255), 2)
            for p in pts_view:
                cv2.circle(dbg, p, 4, (0, 0, 0), 2)
                cv2.circle(dbg, p, 4, (0, 255, 255), -1)

        txt = f"points={len(pts)}"
        cv2.putText(dbg, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
        cv2.imshow(win, dbg)

        key = cv2.waitKey(16) & 0xFF
        if key == 27:
            state["cancel"] = True
            break
        if key in (10, 13):
            state["done"] = True
            break

    cv2.destroyWindow(win)
    if state["cancel"] or len(pts) < 2:
        return None
    return pts


def _h_in_range(h, lo, hi):
    if lo <= hi:
        return (h >= lo) & (h <= hi)
    return (h >= lo) | (h <= hi)


def sample_hsv_along_axis(hsv, axis_samples, radius):
    h, w = hsv.shape[:2]
    pts = np.rint(axis_samples).astype(np.int32)
    xs = np.clip(pts[:, 0], 0, w - 1)
    ys = np.clip(pts[:, 1], 0, h - 1)

    r = int(radius)
    if r <= 0:
        return hsv[ys, xs].copy()

    out = np.empty((pts.shape[0], 3), dtype=np.uint8)
    for i in range(pts.shape[0]):
        x = int(xs[i])
        y = int(ys[i])
        x0 = max(0, x - r)
        y0 = max(0, y - r)
        x1 = min(w - 1, x + r)
        y1 = min(h - 1, y + r)
        patch = hsv[y0 : y1 + 1, x0 : x1 + 1].reshape(-1, 3).astype(np.float32)
        med = np.median(patch, axis=0)
        out[i, 0] = np.uint8(max(0, min(180, int(round(float(med[0]))))))
        out[i, 1] = np.uint8(max(0, min(255, int(round(float(med[1]))))))
        out[i, 2] = np.uint8(max(0, min(255, int(round(float(med[2]))))))
    return out


def bits_green_from_samples(samples_hsv, cfg):
    low = np.array(cfg["green_hsv_low"], dtype=np.int32)
    high = np.array(cfg["green_hsv_high"], dtype=np.int32)
    h = samples_hsv[:, 0].astype(np.int32)
    s = samples_hsv[:, 1].astype(np.int32)
    v = samples_hsv[:, 2].astype(np.int32)
    in_h = _h_in_range(h, int(low[0]), int(high[0]))
    in_s = (s >= int(low[1])) & (s <= int(high[1]))
    in_v = (v >= int(low[2])) & (v <= int(high[2]))
    return in_h & in_s & in_v


def bits_white_from_samples(samples_hsv, cfg):
    low = np.array(cfg["white_hsv_low"], dtype=np.int32)
    high = np.array(cfg["white_hsv_high"], dtype=np.int32)
    h = samples_hsv[:, 0].astype(np.int32)
    s = samples_hsv[:, 1].astype(np.int32)
    v = samples_hsv[:, 2].astype(np.int32)
    mask_hsv = (
        _h_in_range(h, int(low[0]), int(high[0]))
        & (s >= int(low[1]))
        & (s <= int(high[1]))
        & (v >= int(low[2]))
        & (v <= int(high[2]))
    )
    mask_sv = (v >= int(cfg["white_v_min"])) & (s <= int(cfg["white_s_max"]))
    return mask_hsv | mask_sv


def runs_from_bits(bits):
    n = int(bits.shape[0])
    runs = []
    start = None
    for i in range(n):
        if bits[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, n - 1))
    return runs


def detect_green_interval_1d(bits_green, cfg):
    n = int(bits_green.shape[0])
    runs = runs_from_bits(bits_green)
    if not runs:
        return None

    min_w = float(cfg["green_min_width_s"])
    max_w = float(cfg["green_max_width_s"])
    min_run = int(cfg.get("green_min_run", 1))

    best = None
    best_len = -1

    denom = float(max(1, n - 1))
    for i0, i1 in runs:
        run_len = int(i1 - i0 + 1)
        if run_len < min_run:
            continue
        width_s = float(i1 - i0) / denom
        if width_s < min_w or width_s > max_w:
            continue
        if run_len > best_len:
            best_len = run_len
            best = (i0, i1)

    if best is None:
        return None

    i0, i1 = best
    lo = float(i0) / float(n - 1) if n > 1 else 0.0
    hi = float(i1) / float(n - 1) if n > 1 else 0.0
    return {"lo": lo, "hi": hi, "len": int(i1 - i0 + 1)}


def detect_white_marker_1d(bits_white, axis_samples, cfg, prev_s):
    n = int(bits_white.shape[0])
    runs = runs_from_bits(bits_white)
    if not runs:
        return None

    min_run = int(cfg.get("white_min_run", 1))
    max_run = int(cfg.get("white_max_run", n))
    max_jump_s = float(cfg["track_max_jump_s"])

    best = None
    best_score = -1e18

    denom = float(max(1, n - 1))

    for i0, i1 in runs:
        run_len = int(i1 - i0 + 1)
        if run_len < min_run or run_len > max_run:
            continue

        ic = int((i0 + i1) // 2)
        s = float(ic) / denom

        if prev_s is not None and abs(s - float(prev_s)) > max_jump_s:
            continue

        score = 0.0
        score += float(run_len) * 50.0
        if prev_s is not None:
            score -= abs(s - float(prev_s)) * 600.0

        if score > best_score:
            best_score = score
            best = ic

    if best is None:
        ic = int((runs[0][0] + runs[0][1]) // 2)
        s = float(ic) / denom
        if prev_s is not None and abs(s - float(prev_s)) > max_jump_s:
            return None
        best = ic

    cx = int(round(float(axis_samples[best, 0])))
    cy = int(round(float(axis_samples[best, 1])))
    s = float(best) / float(n - 1) if n > 1 else 0.0
    return {"cx": cx, "cy": cy, "s": s, "idx": int(best)}


def draw_axis(dbg, axis_samples, color=(200, 200, 200), thickness=1):
    pts = axis_samples.astype(np.int32)
    for i in range(1, pts.shape[0]):
        cv2.line(dbg, (int(pts[i - 1, 0]), int(pts[i - 1, 1])), (int(pts[i, 0]), int(pts[i, 1])), color, thickness)


def draw_interval(dbg, axis_samples, lo, hi, color=(0, 255, 0), thickness=3):
    n = axis_samples.shape[0]
    i0 = int(round(lo * float(n - 1)))
    i1 = int(round(hi * float(n - 1)))
    i0 = max(0, min(n - 1, i0))
    i1 = max(0, min(n - 1, i1))
    if i1 < i0:
        i0, i1 = i1, i0
    pts = axis_samples.astype(np.int32)
    for i in range(i0 + 1, i1 + 1):
        cv2.line(dbg, (int(pts[i - 1, 0]), int(pts[i - 1, 1])), (int(pts[i, 0]), int(pts[i, 1])), color, thickness)


class Runtime:
    def __init__(self):
        self.prev_s = None
        self.prev_t = None
        self.lost_white = 0

        self.green_lo = None
        self.green_hi = None
        self.green_hold = 0

        self.armed_count = 0
        self.armed = False

        self.in_run = 0
        self.was_in = False
        self.last_press = 0.0


def main():
    cfg = load_config()

    gamepad = vg.VX360Gamepad()
    gamepad.update()

    sct = mss.mss()
    monitor = sct.monitors[1]
    screen_w = int(monitor["width"])
    screen_h = int(monitor["height"])

    if cfg["axis_points_screen"] is None or cfg["roi_screen"] is None or cfg["axis_samples_roi"] is None:
        frame = grab_frame(sct, monitor)
        pts = axis_selection_ui(frame)
        if pts is None:
            return

        roi = compute_roi_from_points(pts, int(cfg["roi_padding"]), screen_w, screen_h)
        rx, ry, rw, rh = roi

        pts_roi = [(int(p[0] - rx), int(p[1] - ry)) for p in pts]
        axis_resampled = resample_polyline(pts_roi, int(cfg["axis_resample_points"]))
        cfg["axis_points_screen"] = pts
        cfg["roi_screen"] = roi
        cfg["axis_points_roi"] = pts_roi
        cfg["axis_samples_roi"] = axis_resampled.tolist()
        save_config(cfg)

    rx, ry, rw, rh = [int(v) for v in cfg["roi_screen"]]
    region = {"left": rx, "top": ry, "width": rw, "height": rh}

    axis_samples = np.asarray(cfg["axis_samples_roi"], dtype=np.float32)

    show_debug = bool(cfg["show_debug"])
    press_ms = int(cfg["press_ms"])
    cooldown_ms = float(cfg["cooldown_ms"])
    lead_ms = float(cfg["lead_ms"])
    arming_frames = int(cfg["arming_frames"])
    confirm_frames = int(cfg["confirm_in_green_frames"])
    lost_reset = int(cfg["lost_reset_frames"])
    hold_frames = int(cfg["green_hold_frames"])
    alpha = float(cfg["green_smooth_alpha"])
    radius = int(cfg.get("axis_sample_radius", 0))

    st = Runtime()

    cv2.namedWindow("fable2-farming-bot", cv2.WINDOW_NORMAL)

    while True:
        frame = grab_frame(sct, region)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        samples = sample_hsv_along_axis(hsv, axis_samples, radius)

        bits_green = bits_green_from_samples(samples, cfg)
        g = detect_green_interval_1d(bits_green, cfg)

        if g is not None:
            lo = float(g["lo"])
            hi = float(g["hi"])
            if st.green_lo is None or st.green_hi is None:
                st.green_lo, st.green_hi = lo, hi
            else:
                st.green_lo = (1.0 - alpha) * float(st.green_lo) + alpha * lo
                st.green_hi = (1.0 - alpha) * float(st.green_hi) + alpha * hi
            st.green_hold = hold_frames
        else:
            if st.green_hold > 0:
                st.green_hold -= 1
            else:
                st.green_lo, st.green_hi = None, None

        bits_white = bits_white_from_samples(samples, cfg)
        w = detect_white_marker_1d(bits_white, axis_samples, cfg, st.prev_s)

        wx = wy = None
        ws = None
        widx = None
        if w is not None:
            wx, wy = int(w["cx"]), int(w["cy"])
            ws = float(w["s"])
            widx = int(w["idx"])
            st.lost_white = 0
        else:
            st.lost_white += 1
            if st.lost_white >= lost_reset:
                st.prev_s = None
                st.prev_t = None
                ws = None

        now = time.perf_counter()

        if ws is not None:
            if st.prev_s is not None and st.prev_t is not None:
                dt = float(now - float(st.prev_t))
                if dt > 1e-4:
                    v = (float(ws) - float(st.prev_s)) / dt
                else:
                    v = 0.0
            else:
                v = 0.0
            st.prev_s = float(ws)
            st.prev_t = float(now)
        else:
            v = 0.0

        have_green = st.green_lo is not None and st.green_hi is not None
        have_white = ws is not None

        if have_green and have_white:
            st.armed_count += 1
        else:
            st.armed_count = 0
            st.armed = False
            st.in_run = 0
            st.was_in = False

        if st.armed_count >= arming_frames:
            st.armed = True

        in_green = False
        if st.armed and have_green and have_white:
            s_pred = float(ws) + float(v) * (lead_ms / 1000.0)
            s_pred = max(0.0, min(1.0, s_pred))
            in_green = float(st.green_lo) <= s_pred <= float(st.green_hi)

        if in_green:
            st.in_run += 1
        else:
            st.in_run = 0
            st.was_in = False

        if st.armed and st.in_run >= confirm_frames and not st.was_in:
            if (now - float(st.last_press)) * 1000.0 >= cooldown_ms:
                press_a(gamepad, press_ms)
                st.last_press = float(now)
            st.was_in = True

        if show_debug:
            dbg = frame.copy()
            draw_axis(dbg, axis_samples, color=(170, 170, 170), thickness=1)

            if have_green:
                draw_interval(
                    dbg, axis_samples, float(st.green_lo), float(st.green_hi), color=(0, 255, 0), thickness=4
                )

            if wx is not None:
                cv2.circle(dbg, (wx, wy), 6, (255, 255, 255), -1)
                cv2.circle(dbg, (wx, wy), 10, (0, 0, 0), 2)

            state_txt = "IN GREEN" if in_green else ("ARMED" if st.armed else "WAIT")
            col = (0, 255, 0) if in_green else ((255, 255, 0) if st.armed else (200, 200, 200))
            cv2.putText(dbg, state_txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)

            gtxt = (
                "green=none" if not have_green else f"green=[{st.green_lo:.3f},{st.green_hi:.3f}] hold={st.green_hold}"
            )
            wtxt = "white=none" if not have_white else f"white_s={ws:.3f} v={v:+.2f}"
            cv2.putText(dbg, gtxt, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)
            cv2.putText(dbg, wtxt, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2, cv2.LINE_AA)

            if widx is not None:
                n = int(axis_samples.shape[0])
                bar_y = 105
                x0 = 10
                x1 = min(dbg.shape[1] - 10, x0 + 320)
                cv2.rectangle(dbg, (x0, bar_y), (x1, bar_y + 14), (40, 40, 40), -1)
                pos = int(round((float(widx) / float(max(1, n - 1))) * float(x1 - x0)))
                cv2.rectangle(dbg, (x0, bar_y), (x0 + pos, bar_y + 14), (255, 255, 255), -1)

            cv2.putText(
                dbg,
                "T=Test A | R=Redefine axis | D=Debug | Q=Quit",
                (10, dbg.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("fable2-farming-bot", dbg)
        else:
            cv2.imshow("fable2-farming-bot", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in (ord("d"), ord("D")):
            show_debug = not show_debug
        if key in (ord("t"), ord("T")):
            press_a(gamepad, press_ms)
            st.last_press = float(time.perf_counter())
        if key in (ord("r"), ord("R")):
            frame_full = grab_frame(sct, monitor)
            pts = axis_selection_ui(frame_full)
            if pts is not None:
                roi = compute_roi_from_points(pts, int(cfg["roi_padding"]), screen_w, screen_h)
                rx, ry, rw, rh = roi
                region = {"left": rx, "top": ry, "width": rw, "height": rh}
                pts_roi = [(int(p[0] - rx), int(p[1] - ry)) for p in pts]
                axis_resampled = resample_polyline(pts_roi, int(cfg["axis_resample_points"]))
                cfg["axis_points_screen"] = pts
                cfg["roi_screen"] = roi
                cfg["axis_points_roi"] = pts_roi
                cfg["axis_samples_roi"] = axis_resampled.tolist()
                save_config(cfg)
                axis_samples = np.asarray(cfg["axis_samples_roi"], dtype=np.float32)
                st = Runtime()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
