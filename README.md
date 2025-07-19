# 🖐️ Hand Gesture Mouse Controller

Control your mouse using hand gestures in real time using just your webcam!  
Move, click, drag, and scroll naturally — no hardware needed beyond your hands.

---

## ✨ Features

- 👆 Move the cursor with your **index finger**
- ✊ **Fist** gesture triggers a **left click**
- ✌️ **Index + middle fingers** trigger a **right click**
- 🤏 **Pinch (thumb + index)** triggers **drag & drop**
- 🖖 **Exactly 3 fingers up** triggers **scrolling**
- 🖼️ Live webcam GUI showing gestures and feedback
- 🧠 Smart gesture detection with smoothing & cooldown
- 💻 Fully works on real desktop environments

---

## 🛠️ Requirements

- Python 3.7+
- Webcam (internal or external)

### 📦 Install dependencies:

```bash
pip install opencv-python cvzone pyautogui numpy
```

---

## 🚀 How to Run

1. Clone this repository or download the script.

2. Run the main file:

```bash
python hand_mouse_controller.py
```

---

## 🖥️ Controls

| Gesture                 | Action         |
|-------------------------|----------------|
| ☝️ Index Finger Up       | Move cursor    |
| ✊ Fist                  | Left Click     |
| ✌️ Index + Middle Up     | Right Click    |
| 🤏 Thumb + Index Pinch   | Drag & Drop    |
| 🖖 Exactly 3 Fingers Up  | Scroll         |

> Press **`Q`** to quit the application.

---

## ⚙️ Customization

You can tweak behavior directly in the code:

| Parameter              | Purpose                         |
|------------------------|---------------------------------|
| `self.margin`          | Camera frame margin             |
| `self.pinch_threshold` | Pinch detection distance        |
| `self.smoothing_factor`| Cursor movement smoothing       |
| `self.click_cooldown`  | Delay between click events      |

---

## 📂 Project Structure

```
hand-gesture-mouse/
├── hand_mouse_controller.py  # Main script
├── README.md                 # You're here!
└── requirements.txt          # (optional)
```

---

## 🧠 How It Works

- Uses **cvzone** and **OpenCV** to detect hands and finger landmarks
- Recognizes gestures based on finger positions and movement
- Maps index finger position to screen coordinates
- Triggers **PyAutoGUI** to control the actual system mouse

---

## 🙌 Credits

- [cvzone](https://github.com/cvzone/cvzone)
- [OpenCV](https://opencv.org/)
- [PyAutoGUI](https://github.com/asweigart/pyautogui)

---

## ❓ Questions?

Feel free to open an issue or fork and experiment. Happy coding! 🚀
