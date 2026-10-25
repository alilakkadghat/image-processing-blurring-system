# Line-by-Line Code Explanation of `v1.py`

This document provides a line-by-line explanation of every statement, variable, function, and callback in **`v1.py`** (Lines 1 to 169).

---

## Lines 1–6: Module Imports & External Dependencies

```python
1: import cv2
2: import dlib
3: import numpy as np
4: import matplotlib.pyplot as plt
5: from matplotlib.widgets import RectangleSelector, CheckButtons, Button
6: import os
```

* **Line 1 (`import cv2`):** Imports the OpenCV (Open Source Computer Vision) library, used for reading images, color-space conversion (`BGR2GRAY`, `BGR2RGB`), resizing images during pixelation, and writing the final output image to disk.
* **Line 2 (`import dlib`):** Imports the `dlib` machine learning toolkit, used for frontal face detection (via HOG + Linear SVM) and facial landmark shape prediction (via Kazemi & Sullivan's Ensemble of Regression Trees).
* **Line 3 (`import numpy as np`):** Imports NumPy for multidimensional array manipulation, coordinate slicing, min/max calculations across landmark arrays, and memory operations.
* **Line 4 (`import matplotlib.pyplot as plt`):** Imports Matplotlib's plotting interface to build the desktop GUI, render canvases, and manage display windows without requiring heavyweight GUI frameworks like Qt or Tkinter.
* **Line 5 (`from matplotlib.widgets import ...`):** Imports interactive UI widgets from Matplotlib:
  * `RectangleSelector`: Enables interactive click-and-drag bounding box selection directly on the image canvas.
  * `CheckButtons`: Provides multi-option checkboxes for toggling automated feature anonymization (`Face`, `Left Eye`, etc.).
  * `Button`: Provides clickable action buttons (`Pixelate / Save` and `Reset`).
* **Line 6 (`import os`):** Imports the operating system module to check for the presence of the landmark weights file (`shape_predictor_68_face_landmarks.dat`) on disk before attempting to load it.

---

## Lines 8–14: Facial Landmark Mapping Configuration

```python
8: # Define facial landmark indices for dlib
9: LANDMARKS = {
10:     "Left Eye": list(range(42, 48)),
11:     "Right Eye": list(range(36, 42)),
12:     "Nose": list(range(27, 36)),
13:     "Mouth": list(range(48, 68))
14: }
```

* **Line 8:** A comment indicating the dictionary definition for facial landmark indices.
* **Line 9 (`LANDMARKS = {`):** Declares a dictionary mapping human-readable anatomical feature names to specific point index ranges in the standard 68-point iBUG facial landmark schema.
* **Line 10 (`"Left Eye": list(range(42, 48))`):** Generates `[42, 43, 44, 45, 46, 47]`, representing the 6 points outlining the person's left eye (from an observer's perspective on the right).
* **Line 11 (`"Right Eye": list(range(36, 42))`):** Generates `[36, 37, 38, 39, 40, 41]`, representing the 6 points outlining the person's right eye.
* **Line 12 (`"Nose": list(range(27, 36))`):** Generates `[27, 28, ..., 35]`, covering the 4 points of the nasal bridge and the 5 points of the nasal tip and nostrils.
* **Line 13 (`"Mouth": list(range(48, 68))`):** Generates `[48, 49, ..., 67]`, spanning the 12 outer lip contour points and 8 inner lip contour points.
* **Line 14 (`}`):** Closes the landmark dictionary definition.

---

## Lines 16–34: The Core Mosaic Pixelation Function

```python
16: def pixelate(img, roi, size=15):
17:     x1, y1, x2, y2 = roi
18:     
19:     # Ensure coordinates are within image bounds
20:     x1, y1 = max(0, x1), max(0, y1)
21:     x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
22:     
23:     region = img[y1:y2, x1:x2]
24:     h, w = region.shape[:2]
25: 
26:     if h <= 0 or w <= 0:
27:         return img
28: 
29:     temp = cv2.resize(region, (max(1, w // size), max(1, h // size)))
30:     pixel = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
31: 
32:     out = img.copy()
33:     out[y1:y2, x1:x2] = pixel
34:     return out
```

* **Line 16 (`def pixelate(img, roi, size=15):`):** Defines the function responsible for applying the mosaic pixelation effect to a specified bounding box.
  * `img`: The source image (NumPy array).
  * `roi`: A tuple of coordinates `(x1, y1, x2, y2)`.
  * `size`: The block scaling factor (defaults to 15; higher values create larger, coarser pixel blocks).
* **Line 17 (`x1, y1, x2, y2 = roi`):** Unpacks the bounding box coordinate tuple.
* **Line 19:** A comment regarding boundary sanitization.
* **Line 20 (`x1, y1 = max(0, x1), max(0, y1)`):** Clamps the top-left coordinate to non-negative values, preventing negative indexing from wrapping around array edges.
* **Line 21 (`x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)`):** Clamps the bottom-right coordinate to the image's maximum width (`img.shape[1]`) and maximum height (`img.shape[0]`), avoiding index overflow.
* **Line 23 (`region = img[y1:y2, x1:x2]`):** Crops the target Region of Interest (ROI) from the image array using 2D slicing.
* **Line 24 (`h, w = region.shape[:2]`):** Extracts the height (`h`) and width (`w`) of the cropped ROI slice.
* **Line 26 (`if h <= 0 or w <= 0:`):** Defensive check: verifies whether the ROI is empty or degenerate (e.g., zero-width or zero-height selection).
* **Line 27 (`return img`):** If invalid, returns the unaltered image immediately.
* **Line 29 (`temp = cv2.resize(...)`):** **Stage 1 (Downsampling):** Decimates the resolution of `region` by dividing width and height by `size`. `max(1, ...)` guarantees that dimensions never drop below 1 pixel, even on tiny regions.
* **Line 30 (`pixel = cv2.resize(...)`):** **Stage 2 (Upsampling):** Re-expands the low-resolution thumbnail back to the original size `(w, h)` using `cv2.INTER_NEAREST` (Nearest-Neighbor interpolation). Instead of creating a smooth blur, nearest-neighbor repeats pixel values into solid, crisp square blocks.
* **Line 32 (`out = img.copy()`):** Creates an independent clone of the original image array to preserve immutability and avoid unwanted in-place side effects.
* **Line 33 (`out[y1:y2, x1:x2] = pixel`):** Replaces the original ROI pixels with the newly generated mosaic pixel block.
* **Line 34 (`return out`):** Returns the pixelated image.

---

## Lines 36–47: Landmark Bounding Box Calculation

```python
36: def get_landmark_bbox(shape, indices, img_shape, padding=10):
37:     """Calculates a padded bounding box for specific facial landmarks."""
38:     pts = np.array([[shape.part(i).x, shape.part(i).y] for i in indices])
39:     x1, y1 = np.min(pts, axis=0)
40:     x2, y2 = np.max(pts, axis=0)
41:     
42:     return (
43:         max(0, x1 - padding),
44:         max(0, y1 - padding),
45:         min(img_shape[1], x2 + padding),
46:         min(img_shape[0], y2 + padding)
47:     )
```

* **Line 36 (`def get_landmark_bbox(...)`):** Function to calculate the rectangular boundary encompassing a given set of landmark points, with added padding and image-boundary clamping.
* **Line 37:** Docstring describing the function's intent.
* **Line 38 (`pts = np.array(...)`):** Iterates over each landmark index in `indices`, extracts its $(x, y)$ coordinate via `shape.part(i).x` and `shape.part(i).y`, and constructs an $N \times 2$ NumPy array of points.
* **Line 39 (`x1, y1 = np.min(pts, axis=0)`):** Finds the minimum $x$ and $y$ coordinate among all points in this feature.
* **Line 40 (`x2, y2 = np.max(pts, axis=0)`):** Finds the maximum $x$ and $y$ coordinate among all points in this feature.
* **Lines 42–47 (`return (...)`):** Returns a 4-tuple representing the padded bounding box:
  * `max(0, x1 - padding)`: Left edge expanded outward by `padding` (10 pixels), clamped to 0.
  * `max(0, y1 - padding)`: Top edge expanded upward by `padding`, clamped to 0.
  * `min(img_shape[1], x2 + padding)`: Right edge expanded outward, clamped to image width.
  * `min(img_shape[0], y2 + padding)`: Bottom edge expanded downward, clamped to image height.

---

## Lines 49–55: Interactive ROI Region Selection

```python
49: def select_region(ax, state):
50:     def onselect(e1, e2):
51:         state["roi"]["x1"], state["roi"]["y1"] = int(e1.xdata), int(e1.ydata)
52:         state["roi"]["x2"], state["roi"]["y2"] = int(e2.xdata), int(e2.ydata)
53: 
54:     selector = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
55:     return selector
```

* **Line 49 (`def select_region(ax, state):`):** Configures interactive rectangular dragging on the Matplotlib image axis `ax`.
* **Line 50 (`def onselect(e1, e2):`):** Inner callback triggered automatically whenever the user clicks, drags, and releases a rectangle. `e1` contains mouse press coordinates; `e2` contains mouse release coordinates.
* **Line 51 (`state["roi"]["x1"], ...`):** Converts the float click coordinates `e1.xdata` and `e1.ydata` to integers and stores them in the shared `state["roi"]` dictionary.
* **Line 52 (`state["roi"]["x2"], ...`):** Converts and stores the release coordinates `e2.xdata` and `e2.ydata`.
* **Line 54 (`selector = RectangleSelector(...)`):** Instantiates the Matplotlib widget:
  * `ax`: Target image plot axis.
  * `onselect`: The callback defined in line 50.
  * `useblit=True`: Uses blitting for responsive, flicker-free rendering of the selection rectangle.
  * `button=[1]`: Binds only the primary (left) mouse button.
  * `interactive=True`: Enables draggable grab handles to resize the rectangle after drawing.
* **Line 55 (`return selector`):** Returns the selector instance (persisted in memory to prevent Python's garbage collector from destroying it).

---

## Lines 57–90: Full Image Processing & Pixelation Pipeline

```python
57: def apply_pixelation(state):
58:     """Applies pixelation based on selected checkboxes AND manual ROI."""
59:     img_copy = state["orig_img"].copy()
60:     gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
61:     
62:     # 1. Apply automated facial landmark pixelation
63:     selected_features = [label for label, state_val in zip(state["labels"], state["check"].get_status()) if state_val]
64:     
65:     if selected_features and state["detector"] and state["predictor"]:
66:         faces = state["detector"](gray)
67:         for face in faces:
68:             # Pixelate entire face bounding box
69:             if "Face" in selected_features:
70:                 bbox = (face.left(), face.top(), face.right(), face.bottom())
71:                 img_copy = pixelate(img_copy, bbox, size=20)
72:             
73:             # Pixelate specific features
74:             shape = state["predictor"](gray, face)
75:             for feature in selected_features:
76:                 if feature in LANDMARKS:
77:                     bbox = get_landmark_bbox(shape, LANDMARKS[feature], img_copy.shape)
78:                     img_copy = pixelate(img_copy, bbox, size=15)
79: 
80:     # 2. Apply manual selection pixelation
81:     if "x1" in state["roi"]:
82:         x1, x2 = sorted([state["roi"]["x1"], state["roi"]["x2"]])
83:         y1, y2 = sorted([state["roi"]["y1"], state["roi"]["y2"]])
84:         img_copy = pixelate(img_copy, (x1, y1, x2, y2), size=20)
85: 
86:     # Update display
87:     state["curr_img"] = img_copy
88:     state["ax_img"].set_data(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
89:     state["fig"].canvas.draw_idle()
```

* **Line 57 (`def apply_pixelation(state):`):** Core orchestration function that gathers UI choices, applies automated and manual pixelation, and refreshes the canvas.
* **Line 58:** Docstring.
* **Line 59 (`img_copy = state["orig_img"].copy()`):** Always begins with a clean clone of the original input image so filters do not accumulate destructively upon multiple clicks.
* **Line 60 (`gray = cv2.cvtColor(...)`):** Converts the image to grayscale for dlib's face detector.
* **Line 63 (`selected_features = [...]`):** Uses list comprehension with `zip()` over `state["labels"]` and `state["check"].get_status()` (which returns a list of booleans) to extract only the labels whose checkboxes are currently checked.
* **Line 65 (`if selected_features and state["detector"] and state["predictor"]:`):** Checks whether any checkbox is active and verifies that the detector and model predictor are loaded.
* **Line 66 (`faces = state["detector"](gray)`):** Runs dlib's HOG + Linear SVM detector across the grayscale image to locate all face rectangles.
* **Line 67 (`for face in faces:`):** Loops through every detected face.
* **Line 69 (`if "Face" in selected_features:`):** Checks if the user checked the "Face" box.
* **Line 70 (`bbox = (face.left(), face.top(), face.right(), face.bottom())`):** Extracts coordinates from dlib's `rectangle` object.
* **Line 71 (`img_copy = pixelate(img_copy, bbox, size=20)`):** Applies coarse pixelation (`size=20`) over the full face rectangle.
* **Line 74 (`shape = state["predictor"](gray, face)`):** Executes Kazemi & Sullivan's ERT regressor to identify the 68 landmark coordinates for this face.
* **Line 75 (`for feature in selected_features:`):** Iterates over the user's checked options (`Left Eye`, `Right Eye`, `Nose`, `Mouth`).
* **Line 76 (`if feature in LANDMARKS:`):** Checks if the feature is defined in our landmark mapping.
* **Line 77 (`bbox = get_landmark_bbox(...)`):** Calculates the padded bounding box enclosing that specific feature.
* **Line 78 (`img_copy = pixelate(img_copy, bbox, size=15)`):** Applies pixelation (`size=15`) strictly over that feature.
* **Line 81 (`if "x1" in state["roi"]:`):** Checks if the user manually dragged a custom rectangular ROI.
* **Line 82 (`x1, x2 = sorted(...)`):** Sorts coordinates so $x_1 < x_2$, regardless of whether the user dragged left-to-right or right-to-left.
* **Line 83 (`y1, y2 = sorted(...)`):** Sorts coordinates so $y_1 < y_2$, regardless of whether the user dragged top-to-bottom or bottom-to-top.
* **Line 84 (`img_copy = pixelate(...)`):** Applies pixelation (`size=20`) over the manually selected area.
* **Line 87 (`state["curr_img"] = img_copy`):** Updates the currently active image in the state dictionary.
* **Line 88 (`state["ax_img"].set_data(...)`):** Converts the image from BGR to RGB (required by Matplotlib) and updates the on-screen image buffer.
* **Line 89 (`state["fig"].canvas.draw_idle()`):** Requests a redraw of the Matplotlib GUI window efficiently during the next idle cycle.

---

## Lines 91–112: Button & Keyboard Handlers

```python
91: def btn_pixelate_clicked(event, state):
92:     apply_pixelation(state)
93:     cv2.imwrite("pixelized_output.jpg", state["curr_img"])
94:     print("Saved as pixelized_output.jpg")
95: 
96: def btn_reset_clicked(event, state):
97:     state["roi"] = {}  # Clear manual selection
98:     state["curr_img"] = state["orig_img"].copy()
99:     
100:     # Uncheck all boxes
101:     for i, is_checked in enumerate(state["check"].get_status()):
102:         if is_checked:
103:             state["check"].set_active(i)
104:             
105:     state["ax_img"].set_data(cv2.cvtColor(state["curr_img"], cv2.COLOR_BGR2RGB))
106:     state["fig"].canvas.draw_idle()
107:     print("Reset applied.")
108: 
109: def key_handler(event, state):
110:     if event.key == "enter":
111:         btn_pixelate_clicked(event, state)
```

* **Line 91 (`def btn_pixelate_clicked(event, state):`):** Callback triggered when clicking the "Pixelate / Save" button.
* **Line 92 (`apply_pixelation(state)`):** Executes the full pixelation algorithm.
* **Line 93 (`cv2.imwrite(...)`):** Saves the processed image array to disk as `pixelized_output.jpg`.
* **Line 94 (`print(...)`):** Emits a console log confirming the file has been saved.
* **Line 96 (`def btn_reset_clicked(event, state):`):** Callback triggered when clicking the "Reset" button.
* **Line 97 (`state["roi"] = {}`):** Clears any stored manual ROI coordinates.
* **Line 98 (`state["curr_img"] = state["orig_img"].copy()`):** Replaces the display image with the pristine original.
* **Lines 101–103:** Loops through all checkboxes; if a box is currently checked (`is_checked is True`), calls `set_active(i)` to toggle it back to unchecked.
* **Line 105 (`state["ax_img"].set_data(...)`):** Refreshes the display canvas with the original RGB image.
* **Line 106 (`state["fig"].canvas.draw_idle()`):** Redraws the figure.
* **Line 107:** Prints confirmation to the terminal.
* **Line 109 (`def key_handler(event, state):`):** Keyboard event listener.
* **Line 110 (`if event.key == "enter":`):** Detects if the pressed key is the **Enter** key.
* **Line 111 (`btn_pixelate_clicked(...)`):** Triggers pixelation and saving directly from the keyboard for convenience.

---

## Lines 113–166: Application Initialization & GUI Construction (`main()`)

```python
113: def main():
114:     image_path = "image.jpg"
115:     model_path = "shape_predictor_68_face_landmarks.dat"
116: 
117:     img = cv2.imread(image_path)
118:     if img is None:
119:         print(f"Error: '{image_path}' not found.")
120:         return
121: 
122:     # Initialize dlib
123:     detector = dlib.get_frontal_face_detector()
124:     predictor = None
125:     if os.path.exists(model_path):
126:         predictor = dlib.shape_predictor(model_path)
127:     else:
128:         print(f"Warning: '{model_path}' not found. Facial mapping disabled.")
129: 
130:     # Application state to share data across callbacks easily
131:     state = {
132:         "orig_img": img,
133:         "curr_img": img.copy(),
134:         "roi": {},
135:         "detector": detector,
136:         "predictor": predictor,
137:         "labels": ["Face", "Left Eye", "Right Eye", "Nose", "Mouth"]
138:     }
139: 
140:     # Setup Main Figure and Subplots
141:     state["fig"] = plt.figure(figsize=(10, 7))
142:     
143:     # Image axes (leaving room on the left for UI)
144:     ax_main = state["fig"].add_axes([0.3, 0.05, 0.65, 0.9])
145:     state["ax_img"] = ax_main.imshow(cv2.cvtColor(state["curr_img"], cv2.COLOR_BGR2RGB))
146:     ax_main.axis("off")
147: 
148:     # UI Controls Axes
149:     ax_chk = state["fig"].add_axes([0.05, 0.5, 0.2, 0.35])
150:     ax_btn_pix = state["fig"].add_axes([0.05, 0.35, 0.2, 0.08])
151:     ax_btn_res = state["fig"].add_axes([0.05, 0.25, 0.2, 0.08])
152: 
153:     # Add Checkboxes
154:     state["check"] = CheckButtons(ax_chk, state["labels"], [False] * len(state["labels"]))
155:     
156:     # Add Buttons
157:     btn_pix = Button(ax_btn_pix, 'Pixelate / Save')
158:     btn_res = Button(ax_btn_res, 'Reset')
159: 
160:     # Connect Callbacks
161:     selector = select_region(ax_main, state)
162:     btn_pix.on_clicked(lambda event: btn_pixelate_clicked(event, state))
163:     btn_res.on_clicked(lambda event: btn_reset_clicked(event, state))
164:     state["fig"].canvas.mpl_connect("key_press_event", lambda event: key_handler(event, state))
165: 
166:     plt.show()
```

* **Line 113 (`def main():`):** The primary entry point function that orchestrates setup and starts the application.
* **Line 114 (`image_path = "image.jpg"`):** Specifies the path to the input target image.
* **Line 115 (`model_path = "shape_predictor_68_face_landmarks.dat"`):** Specifies the path to the pre-trained landmark weights file.
* **Line 117 (`img = cv2.imread(image_path)`):** Loads the image from disk into a BGR NumPy matrix.
* **Lines 118–120:** Verifies the image was loaded successfully; if `img is None`, prints an error and exits cleanly.
* **Line 123 (`detector = dlib.get_frontal_face_detector()`):** Instantiates dlib's HOG-based face detector.
* **Line 124 (`predictor = None`):** Initializes the predictor variable to `None` for safe fallback.
* **Lines 125–128:** Checks if the `.dat` weights file exists on disk. If found, initializes `dlib.shape_predictor(model_path)`; otherwise, prints a warning and runs in a degraded mode (manual ROI and full-face mode remain functional).
* **Lines 131–138 (`state = { ... }`):** Constructs the state dictionary holding references to image arrays, models, UI labels, and bounding boxes.
* **Line 141 (`state["fig"] = plt.figure(figsize=(10, 7))`):** Creates a Matplotlib window of 10 inches by 7 inches.
* **Line 144 (`ax_main = state["fig"].add_axes([0.3, 0.05, 0.65, 0.9])`):** Adds the primary viewport axis for the image, positioned on the right side of the window `[left=30%, bottom=5%, width=65%, height=90%]`.
* **Line 145 (`state["ax_img"] = ax_main.imshow(...)`):** Displays the image in RGB color format.
* **Line 146 (`ax_main.axis("off")`):** Hides axis numbers and tick marks for a clean image viewing experience.
* **Line 149 (`ax_chk = state["fig"].add_axes([0.05, 0.5, 0.2, 0.35])`):** Allocates a rectangular area on the upper-left for checkboxes.
* **Line 150 (`ax_btn_pix = state["fig"].add_axes([0.05, 0.35, 0.2, 0.08])`):** Allocates space for the "Pixelate / Save" button.
* **Line 151 (`ax_btn_res = state["fig"].add_axes([0.05, 0.25, 0.2, 0.08])`):** Allocates space for the "Reset" button.
* **Line 154 (`state["check"] = CheckButtons(...)`):** Creates the checkbox widget with all labels initialized to `False` (unchecked).
* **Line 157 (`btn_pix = Button(ax_btn_pix, 'Pixelate / Save')`):** Instantiates the pixelate action button.
* **Line 158 (`btn_res = Button(ax_btn_res, 'Reset')`):** Instantiates the reset action button.
* **Line 161 (`selector = select_region(ax_main, state)`):** Attaches the interactive rectangle selector to the image canvas.
* **Line 162 (`btn_pix.on_clicked(...)`):** Connects the pixelate button's click event to `btn_pixelate_clicked`.
* **Line 163 (`btn_res.on_clicked(...)`):** Connects the reset button's click event to `btn_reset_clicked`.
* **Line 164 (`state["fig"].canvas.mpl_connect(...)`):** Attaches the key press event listener to `key_handler`.
* **Line 166 (`plt.show()`):** Starts Matplotlib's blocking event loop, rendering the window and listening for user interactions.

---

## Lines 168–169: Python Execution Entry Point

```python
168: if __name__ == "__main__":
169:     main()
```

* **Line 168 (`if __name__ == "__main__":`):** Standard Python idiom that checks if the script is being executed directly (rather than imported as a module into another script).
* **Line 169 (`main()`):** Invokes the `main()` function to launch the application.
