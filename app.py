import os
import cv2
import numpy as np
import torch
import threading
import time
import datetime
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# BGR colors for up to 3 simultaneous prompts: red, green, blue
PROMPT_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
MODES = ["heatmap", "contours", "both"]


def build_overlay(frame, probs, mode, threshold, blend):
    """
    Build a visualization overlay on top of frame.

    Args:
        frame:     np.uint8 BGR image, shape (H, W, 3)
        probs:     np.float32 array, shape (N, H, W), values in [0, 1]
                   already resized to match frame dimensions
        mode:      one of "heatmap", "contours", "both"
        threshold: float — pixels at or above this prob are "detected"
        blend:     float in [0.1, 0.9] — overlay opacity

    Returns:
        np.uint8 BGR image, same shape as frame
    """
    n = probs.shape[0]

    if mode in ("heatmap", "both"):
        heat = probs.max(axis=0)                                      # (H, W)
        heat_uint8 = (np.clip(heat / max(threshold, 1e-6), 0, 1) * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
        result = cv2.addWeighted(frame, 1.0 - blend, heat_color, blend, 0)
    else:
        result = frame.copy()

    if mode in ("contours", "both"):
        for i in range(n):
            color = PROMPT_COLORS[min(i, len(PROMPT_COLORS) - 1)]
            mask = (probs[i] >= threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 2)

    return result


def render_hud(frame, prompts, threshold, blend, mode, fps, save_msg=""):
    """
    Draw an informational overlay (HUD) onto frame in-place.

    Args:
        frame:     np.uint8 BGR image — modified in place
        prompts:   list of str
        threshold: float
        blend:     float
        mode:      str
        fps:       float
        save_msg:  optional str — filename of last saved snapshot

    Returns:
        frame (same object, modified in place)
    """
    lines = [
        f"Prompts: {', '.join(prompts)}",
        f"Threshold: {threshold:.2f}   Blend: {blend:.2f}",
        f"Mode: {mode}   FPS: {fps:.1f}",
        "e=edit  c=mode  s=save  +/-=threshold  [/]=blend  q=quit",
    ]
    if save_msg:
        lines.append(f"Saved: {os.path.basename(save_msg)}")
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    return frame


def capture_loop(cap, state, stop_event):
    """Daemon thread: continuously writes the latest frame into state."""
    while not stop_event.is_set():
        ok, frame = cap.read()
        if ok:
            with state["lock"]:
                state["latest_frame"] = frame


def run_inference(frame, prompts, proc, model, device):
    """Run CLIPSeg on a single BGR frame and return per-prompt probability maps."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = proc(text=prompts, images=[rgb] * len(prompts), return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits          # (N, 352, 352)
    if logits.ndim == 2:                         # single prompt: (352, 352) -> (1, 352, 352)
        logits = logits.unsqueeze(0)
    return torch.sigmoid(logits).float().cpu().numpy()   # (N, 352, 352)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIPSeg on {device}...")
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
    if device == "cuda":
        model = model.half()
    model.eval()
    print("Model ready. Press 'q' to quit.")

    state = {
        "prompts": ["head"],
        "threshold": 0.4,
        "blend": 0.45,
        "mode": "heatmap",
        "latest_frame": None,
        "lock": threading.Lock(),
        "save_msg": "",
    }

    stop_event = threading.Event()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam at index 0. Check that a camera is connected and not in use.")
    thread = threading.Thread(target=capture_loop, args=(cap, state, stop_event), daemon=True)
    thread.start()

    fps_times = []

    while True:
        with state["lock"]:
            frame = state["latest_frame"]
        if frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        prompts = list(state["prompts"])
        if not prompts:
            continue

        raw_probs = run_inference(frame, prompts, proc, model, device)  # (N, 352, 352)

        probs = np.stack([
            cv2.resize(raw_probs[i], (w, h), interpolation=cv2.INTER_LINEAR)
            for i in range(raw_probs.shape[0])
        ])

        result = build_overlay(frame, probs, state["mode"], state["threshold"], state["blend"])

        fps_times.append(time.time())
        fps_times = fps_times[-10:]
        fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0] + 1e-9) if len(fps_times) > 1 else 0.0

        render_hud(result, prompts, state["threshold"], state["blend"], state["mode"], fps, state.get("save_msg", ""))
        state["save_msg"] = ""
        cv2.imshow("CLIPSeg Demo — press q to quit", result)

        key = cv2.waitKey(1)
        char = key & 0xFF

        if char == ord('q'):
            break
        elif char == ord('e'):
            # NOTE: display freezes while user types — this is expected
            raw = input("\nEnter prompts (comma-separated): ").strip()
            new_prompts = [p.strip() for p in raw.split(",") if p.strip()]
            if new_prompts:
                state["prompts"] = new_prompts
        elif char == ord('c'):
            idx = MODES.index(state["mode"])
            state["mode"] = MODES[(idx + 1) % len(MODES)]
        elif char == ord('s'):
            fname = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            cv2.imwrite(fname, result)
            print(f"Saved {fname}")
            state["save_msg"] = fname
        elif char == ord('+') or char == ord('='):
            state["threshold"] = min(0.95, round(state["threshold"] + 0.05, 2))
        elif char == ord('-'):
            state["threshold"] = max(0.05, round(state["threshold"] - 0.05, 2))
        elif char == ord(']'):
            state["blend"] = min(0.9, round(state["blend"] + 0.05, 2))
        elif char == ord('['):
            state["blend"] = max(0.1, round(state["blend"] - 0.05, 2))

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
