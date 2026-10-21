import cv2
import numpy as np


class DrawingEngine:
    def __init__(self, width=900, height=600):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.previous_point = None
        self.undo_stack = []
        self.redo_stack = []

    def save_state(self):
        self.undo_stack.append(self.canvas.copy())
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def draw(self, point, color=(255, 180, 80), brush_size=8):
        x, y = point
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        if self.previous_point is not None:
            cv2.line(
                self.canvas,
                self.previous_point,
                (x, y),
                color,
                brush_size,
                cv2.LINE_AA
            )
        else:
            cv2.circle(
                self.canvas,
                (x, y),
                max(1, brush_size // 2),
                color,
                -1,
                cv2.LINE_AA
            )

        self.previous_point = (x, y)

    def stop_drawing(self):
        self.previous_point = None

    def erase(self, point, brush_size=25):
        x, y = point
        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        cv2.circle(
            self.canvas,
            (x, y),
            brush_size,
            (0, 0, 0),
            -1,
            cv2.LINE_AA
        )

        self.previous_point = None

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
        self.stop_drawing()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
        self.stop_drawing()

    def clear(self):
        self.save_state()
        self.canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8
        )
        self.stop_drawing()

    def get_canvas(self):
        return self.canvas.copy()

    def export_png(self):
        success, encoded = cv2.imencode(".png", self.canvas)
        if not success:
            raise RuntimeError("PNG export failed")
        return encoded.tobytes()