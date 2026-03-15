"""
Prosthetic Simulator Interface
Provides a real-time gesture prediction loop using a trained model and
streaming synthetic EMG windows that simulate a live electrode feed.
"""

import time
import threading
import queue
import numpy as np
import torch
import torch.nn as nn

# Gesture label map (NinaPro DB1: exercises A+B+C combined, 52 gestures)
GESTURE_NAMES = {
    # Exercise A – finger movements (0-11)
    0:  "Index finger flexion",
    1:  "Index finger extension",
    2:  "Middle finger flexion",
    3:  "Middle finger extension",
    4:  "Ring finger flexion",
    5:  "Ring finger extension",
    6:  "Little finger flexion",
    7:  "Little finger extension",
    8:  "Thumb flexion",
    9:  "Thumb extension",
    10: "Thumb adduction",
    11: "Thumb abduction",
    # Exercise B – wrist/hand isometric (12-28)
    12: "Wrist flexion",
    13: "Wrist extension",
    14: "Wrist radial deviation",
    15: "Wrist ulnar deviation",
    16: "Wrist pronation",
    17: "Wrist supination",
    18: "Hand open",
    19: "Hand close",
    20: "Pointer",
    21: "Fist",
    22: "OK sign",
    23: "Pinch",
    24: "Lateral prehension",
    25: "Tripod grasp",
    26: "Prismatic grip",
    27: "Spherical grip",
    28: "Power grip",
    # Exercise C – grasping (29-51)
    **{i: f"Grasp {i-29+1}" for i in range(29, 52)},
}


class EMGStreamSimulator:
    """
    Simulates a real-time streaming EMG acquisition device.
    Generates synthetic windows at a given update rate.
    """

    def __init__(self, n_channels: int = 10, window_size: int = 200,
                 update_hz: float = 10.0, n_classes: int = 52):
        self.n_channels  = n_channels
        self.window_size = window_size
        self.interval    = 1.0 / update_hz
        self.n_classes   = n_classes
        self._running    = False
        self._queue: queue.Queue = queue.Queue(maxsize=5)
        self._current_gesture    = 0

    def set_gesture(self, gesture_id: int):
        """Simulate user performing a specific gesture."""
        self._current_gesture = max(0, min(gesture_id, self.n_classes - 1))

    def _synthesize_window(self) -> np.ndarray:
        """Generate one window for the current gesture."""
        from data.synthetic_emg import generate_emg_window
        snr = np.random.uniform(15, 25)
        return generate_emg_window(self._current_gesture,
                                   self.window_size, self.n_channels,
                                   snr_db=snr)

    def _stream_loop(self):
        while self._running:
            t0  = time.time()
            win = self._synthesize_window()
            try:
                self._queue.put_nowait(win)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                self._queue.put_nowait(win)
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.interval - elapsed))

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def get_window(self, timeout: float = 1.0):
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


class ProstheticController:
    """
    Real-time gesture decoder.
    Wraps a trained classification model and provides a smoothed prediction
    output suitable for prosthetic actuation.
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_classes: int = 52, smoothing_window: int = 5,
                 confidence_threshold: float = 0.5):
        self.model       = model.eval().to(device)
        self.device      = device
        self.n_classes   = n_classes
        self.smoothing   = smoothing_window
        self.threshold   = confidence_threshold
        self._prob_buf   = []      # rolling buffer of probability vectors

    @torch.no_grad()
    def predict(self, window: np.ndarray):
        """
        Parameters
        ----------
        window : (W, C) float32

        Returns
        -------
        gesture_id   : int
        gesture_name : str
        confidence   : float
        probabilities: np.ndarray (n_classes,)
        """
        x      = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
        logits = self.model(x)                           # (1, n_classes)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Temporal smoothing
        self._prob_buf.append(probs)
        if len(self._prob_buf) > self.smoothing:
            self._prob_buf.pop(0)
        smoothed      = np.mean(self._prob_buf, axis=0)

        gesture_id    = int(smoothed.argmax())
        confidence    = float(smoothed[gesture_id])
        gesture_name  = GESTURE_NAMES.get(gesture_id, f"Gesture {gesture_id+1}")

        if confidence < self.threshold:
            return -1, "Rest / uncertain", confidence, smoothed

        return gesture_id, gesture_name, confidence, smoothed

    def reset(self):
        self._prob_buf.clear()


class ProstheticSimulator:
    """
    End-to-end simulator combining streaming EMG + gesture decoder.

    Usage
    -----
    sim = ProstheticSimulator(model, device="cpu")
    sim.start()
    sim.set_target_gesture(12)    # wrist flexion
    result = sim.step()
    sim.stop()
    """

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_classes: int = 52, update_hz: float = 10.0):
        self.stream     = EMGStreamSimulator(update_hz=update_hz,
                                              n_classes=n_classes)
        self.controller = ProstheticController(model, device, n_classes)
        self._log       = []

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def set_target_gesture(self, gesture_id: int):
        self.stream.set_gesture(gesture_id)

    def step(self):
        """Get next prediction."""
        window = self.stream.get_window()
        if window is None:
            return None
        g_id, g_name, conf, probs = self.controller.predict(window)
        result = {
            "gesture_id":   g_id,
            "gesture_name": g_name,
            "confidence":   conf,
            "probabilities": probs,
            "timestamp":    time.time(),
        }
        self._log.append(result)
        return result

    def run_demo(self, n_steps: int = 20, target_gesture: int = 12):
        """Quick CLI demo."""
        print(f"\n{'─'*60}")
        print(f"  Prosthetic Simulator Demo")
        print(f"  Target gesture: {GESTURE_NAMES.get(target_gesture, target_gesture)}")
        print(f"{'─'*60}")

        self.start()
        self.set_target_gesture(target_gesture)

        correct = 0
        for step in range(1, n_steps + 1):
            result = self.step()
            if result is None:
                continue
            mark  = "✓" if result["gesture_id"] == target_gesture else "✗"
            print(f"  Step {step:>2} | {mark} {result['gesture_name']:<35} "
                  f"conf={result['confidence']:.3f}")
            if result["gesture_id"] == target_gesture:
                correct += 1

        self.stop()
        print(f"{'─'*60}")
        print(f"  Accuracy over {n_steps} steps: "
              f"{correct/n_steps*100:.1f}%")
        print(f"{'─'*60}\n")
