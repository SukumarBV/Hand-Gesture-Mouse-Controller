import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import pyautogui
import numpy as np
import time
import math
from collections import deque
from enum import Enum

class GestureState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"

class HandGestureMouseController:
    def __init__(self):
        # Initialize CVZone Hand Detector
        self.detector = HandDetector(
            staticMode=False,
            maxHands=1,
            modelComplexity=1,
            detectionCon=0.7, 
            minTrackCon=0.5
        )

        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()

        # Camera dimensions
        self.cam_width = 640
        self.cam_height = 480

        # Define camera to screen mapping area (ignore edges for better control)
        self.cam_margin_x = 150 
        self.cam_margin_y = 100 
        self.cam_usable_width = self.cam_width - 2 * self.cam_margin_x
        self.cam_usable_height = self.cam_height - 2 * self.cam_margin_y


        # Gesture state management for debouncing and edge-triggering
        self.raw_current_gesture = GestureState.IDLE 
        self.stable_gesture_state = GestureState.IDLE 
        self.prev_stable_gesture_state = GestureState.IDLE 
        self.gesture_start_time = 0 
        self.gesture_threshold = 0.08  # Time (seconds) a gesture must be stable to be recognized

        # Mouse control parameters
        self.smoothing_factor = 0.3 
        self.prev_mouse_x = self.screen_width / 2 
        self.prev_mouse_y = self.screen_height / 2

        # Movement history for smoothing
        self.movement_history = deque(maxlen=7) 

        # Click detection (last_click_time is not for cooldown now, but could be repurposed)
        self.last_click_time = 0 

        # Drag state
        self.is_dragging = False
        self.drag_start_pos = None

        # Scroll parameters
        self.scroll_sensitivity = 1     # How many camera pixels translate to 1 scroll unit (smaller value = more scroll)
        self.scroll_sensitivity_threshold = 10 # Minimum Y movement in pixels to trigger any scroll (increase if too jumpy, decrease if unresponsive)
        self.last_scroll_y = 0 # This will be initialized only when SCROLLING gesture becomes stable

        # Finger distance thresholds
        self.finger_fold_threshold = 40 
        self.pinch_threshold = 50       

        # Safety settings for pyautogui
        pyautogui.FAILSAFE = True 
        pyautogui.PAUSE = 0.005 

        print("Hand Gesture Mouse Controller Initialized!")
        print("Screen Resolution:", self.screen_width, "x", self.screen_height)
        print("Camera Resolution:", self.cam_width, "x", self.cam_height)
        print("\nControls:")
        print("- Index finger pointing: Mouse movement")
        print("- Fist (all fingers down) or Thumb only: Left click")
        print("- Index + Middle up: Right click")
        print("- Thumb + Index pinch: Drag and drop")
        print("- Three fingers (index, middle, ring) up: Scrolling")
        print("- Press 'q' to quit")
        print("\n--- Troubleshooting Tips ---")
        print("1. Ensure 'pyautogui' has Accessibility permissions (macOS) or xdotool (Linux).")
        print("2. The blue rectangle on the camera feed is your 'active zone'. Keep your index finger TIP within it for movement.")
        print("3. If mouse movement is sluggish, try decreasing `self.smoothing_factor` (e.g., 0.5, 0.4).")
        print("4. **SCROLLING DEBUG:** Look for 'DEBUG: Entering SCROLLING state' and '--- Scroll Debug ---' messages in the console.")
        print("   Adjust `self.scroll_sensitivity` (smaller for more scroll) and `self.scroll_sensitivity_threshold` (smaller for more sensitive scroll trigger).")


    def get_finger_distances(self, hand):
        """Calculate distances between fingertips and hand center (not used for current logic, but kept)"""
        lm_list = hand["lmList"]
        hand_center = lm_list[0] 

        fingertip_ids = [4, 8, 12, 16, 20] 
        distances = []
        for tip_id in fingertip_ids:
            tip = lm_list[tip_id]
            distance = math.sqrt((tip[0] - hand_center[0])**2 + (tip[1] - hand_center[1])**2)
            distances.append(distance)
        return distances

    def is_finger_up(self, hand, finger_id):
        """Check if a specific finger is up using cvzone method"""
        fingers = self.detector.fingersUp(hand)
        return fingers[finger_id] == 1

    def get_fingers_up(self, hand):
        """Get list of fingers that are up (0-thumb, 1-index, 2-middle, 3-ring, 4-pinky)"""
        fingers = self.detector.fingersUp(hand)
        fingers_up = []
        for i, finger_status in enumerate(fingers):
            if finger_status == 1:
                fingers_up.append(i)
        return fingers_up

    def calculate_pinch_distance(self, hand):
        """Calculate distance between thumb and index finger tips"""
        lm_list = hand["lmList"]
        if len(lm_list) > 8 and len(lm_list[4]) > 1 and len(lm_list[8]) > 1:
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            distance = math.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
            return distance
        return float('inf') 

    def detect_gesture(self, hand):
        """Detect current gesture based on hand landmarks"""
        fingers_up = self.get_fingers_up(hand)
        num_fingers = len(fingers_up)
        
        pinch_distance = self.calculate_pinch_distance(hand)

        # Index finger only - Mouse movement
        if fingers_up == [1]:
            return GestureState.MOVING

        # Pinch gesture (thumb + index close) - Drag and drop
        elif set(fingers_up) == {0, 1} and pinch_distance < self.pinch_threshold:
            return GestureState.DRAGGING

        # Index + Middle finger - Right click
        elif set(fingers_up) == {1, 2}:
            return GestureState.RIGHT_CLICK
        
        # Three fingers (index, middle, ring) up - Scrolling
        elif set(fingers_up) == {1, 2, 3}:
            return GestureState.SCROLLING

        # Fist (all fingers down) or just thumb - Left click
        elif num_fingers == 0 or fingers_up == [0]:
            return GestureState.LEFT_CLICK

        return GestureState.IDLE

    def smooth_movement(self, x, y):
        """Apply smoothing to mouse movement using a deque for recent history"""
        self.movement_history.append((x, y))

        if len(self.movement_history) < 2:
            return x, y

        avg_x = sum(p[0] for p in self.movement_history) / len(self.movement_history)
        avg_y = sum(p[1] for p in self.movement_history) / len(self.movement_history)

        smooth_x = self.prev_mouse_x + (avg_x - self.prev_mouse_x) * self.smoothing_factor
        smooth_y = self.prev_mouse_y + (avg_y - self.prev_mouse_y) * self.smoothing_factor

        self.prev_mouse_x = smooth_x
        self.prev_mouse_y = smooth_y

        return smooth_x, smooth_y

    def convert_to_screen_coords(self, x, y):
        """Convert camera coordinates to screen coordinates within the usable area"""
        screen_x = np.interp(x, [self.cam_margin_x, self.cam_width - self.cam_margin_x], [0, self.screen_width])
        screen_y = np.interp(y, [self.cam_margin_y, self.cam_height - self.cam_margin_y], [0, self.screen_height])

        screen_x = max(0, min(self.screen_width - 1, screen_x))
        screen_y = max(0, min(self.screen_height - 1, screen_y))

        return screen_x, screen_y

    def handle_mouse_movement(self, hand):
        """Handle mouse movement based on index finger position"""
        lm_list = hand["lmList"]
        index_tip = lm_list[8]

        screen_x, screen_y = self.convert_to_screen_coords(index_tip[0], index_tip[1])
        smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y)
        
        try:
            pyautogui.moveTo(smooth_x, smooth_y)
        except pyautogui.FailSafeException:
            print("ERROR: PyAutoGUI FailSafe triggered during mouse movement. (Cursor likely in a corner.)")
        except Exception as e:
            print(f"ERROR: Failed to move mouse: {e}")

    def handle_left_click(self):
        """Handle left mouse click (edge-triggered)"""
        try:
            pyautogui.click()
            print("ACTION: Left click")
        except pyautogui.FailSafeException:
            print("ERROR: PyAutoGUI FailSafe triggered during left click.")
        except Exception as e:
            print(f"ERROR: Failed to perform left click: {e}")

    def handle_right_click(self):
        """Handle right mouse click (edge-triggered)"""
        try:
            pyautogui.rightClick()
            print("ACTION: Right click")
        except pyautogui.FailSafeException:
            print("ERROR: PyAutoGUI FailSafe triggered during right click.")
        except Exception as e:
            print(f"ERROR: Failed to perform right click: {e}")

    def handle_drag_drop(self, hand):
        """Handle drag and drop operations"""
        lm_list = hand["lmList"]
        index_tip = lm_list[8]
        
        screen_x, screen_y = self.convert_to_screen_coords(index_tip[0], index_tip[1])
        # FIX: Corrected the argument from smooth_y to screen_y
        smooth_x, smooth_y = self.smooth_movement(screen_x, screen_y) 
        
        if not self.is_dragging:
            try:
                pyautogui.mouseDown()
                self.is_dragging = True
                self.drag_start_pos = (smooth_x, smooth_y)
                print("ACTION: Drag started")
            except pyautogui.FailSafeException:
                print("ERROR: PyAutoGUI FailSafe triggered during drag start.")
            except Exception as e:
                print(f"ERROR: Failed to start drag: {e}")
        else:
            try:
                pyautogui.moveTo(smooth_x, smooth_y)
            except pyautogui.FailSafeException:
                print("ERROR: PyAutoGUI FailSafe triggered during dragging movement.")
            except Exception as e:
                print(f"ERROR: Failed to continue drag: {e}")


    def handle_scrolling(self, hand):
        """Handle scrolling based on hand movement"""
        lm_list = hand["lmList"]
        middle_tip = lm_list[12] 
        current_y = middle_tip[1]
        
        y_diff = current_y - self.last_scroll_y # Positive means hand moved down
        
        # --- SCROLL DEBUG PRINTS START ---
        # Uncomment these lines to see scroll values in console
        print(f"Middle Tip Y (cam): {current_y}")
        print(f"Last Scroll Y: {self.last_scroll_y}")
        print(f"Y Diff: {y_diff}")
        print(f"Abs Y Diff: {abs(y_diff)}")
        # --- SCROLL DEBUG PRINTS END ---

        if abs(y_diff) > self.scroll_sensitivity_threshold:
            scroll_amount = int(y_diff / self.scroll_sensitivity)
            # print(f"Calculated scroll_amount: {scroll_amount}") # Uncomment for scroll_amount debug
            if scroll_amount != 0:
                try:
                    pyautogui.scroll(-scroll_amount) # Scroll up for positive, down for negative
                    print(f"ACTION: Scrolled by {-scroll_amount} units")
                except pyautogui.FailSafeException:
                    print("ERROR: PyAutoGUI FailSafe triggered during scrolling.")
                except Exception as e:
                    print(f"ERROR: Failed to scroll: {e}")
            self.last_scroll_y = current_y 
        self.last_scroll_y = current_y


    def process_gesture_state(self, hand, raw_current_gesture):
        """
        Processes the current gesture, applying debouncing and triggering actions.
        This function manages state transitions for edge-triggered actions like clicks.
        """
        current_time = time.time()

        # Step 1: Detect raw gesture changes and reset debounce timer
        if raw_current_gesture != self.raw_current_gesture:
            self.raw_current_gesture = raw_current_gesture 
            self.gesture_start_time = current_time 
            
            # If a new raw gesture starts, we might need to release a previous action (e.g., drag)
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                    print("ACTION: Drag ended (raw gesture changed)")
                except pyautogui.FailSafeException:
                    print("ERROR: PyAutoGUI FailSafe triggered during drag end (raw gesture change).")
                except Exception as e:
                    print(f"ERROR: Failed to release drag on raw gesture change: {e}")
            
            # --- IMPORTANT FOR SCROLLING INITIALIZATION ---
            # Reset scroll reference only when moving OUT of SCROLLING or into a new non-SCROLLING state
            if self.stable_gesture_state == GestureState.SCROLLING and raw_current_gesture != GestureState.SCROLLING:
                 self.last_scroll_y = 0 
            
            return # A new raw gesture just started, it's not yet stable. Exit.

        # Step 2: Check if the current raw gesture has been stable for the debounce threshold
        if (current_time - self.gesture_start_time < self.gesture_threshold):
            return # Not stable yet, continue waiting for stability

        # --- If we reach here, the raw_current_gesture has been stable for `gesture_threshold` time. ---

        # Step 3: Update stable gesture states for action triggering
        self.prev_stable_gesture_state = self.stable_gesture_state 
        self.stable_gesture_state = raw_current_gesture # The raw gesture is now stable

        # Step 4: Perform actions based on the NEW stable gesture state and its transition

        if self.stable_gesture_state == GestureState.MOVING:
            self.handle_mouse_movement(hand)
        
        elif self.stable_gesture_state == GestureState.LEFT_CLICK:
            if self.prev_stable_gesture_state != GestureState.LEFT_CLICK: # Only trigger on transition TO left click
                self.handle_left_click()
        
        elif self.stable_gesture_state == GestureState.RIGHT_CLICK:
            if self.prev_stable_gesture_state != GestureState.RIGHT_CLICK: # Only trigger on transition TO right click
                self.handle_right_click()
        
        elif self.stable_gesture_state == GestureState.DRAGGING:
            self.handle_drag_drop(hand)
        
        elif self.stable_gesture_state == GestureState.SCROLLING:
            # Initialize last_scroll_y ONLY when ENTERING the SCROLLING state
            if self.prev_stable_gesture_state != GestureState.SCROLLING:
                lm_list = hand["lmList"]
                middle_tip = lm_list[12]
                self.last_scroll_y = middle_tip[1] # Set initial Y when gesture starts
                print(f"DEBUG: Entering SCROLLING state. Initial last_scroll_y set to {self.last_scroll_y}")
            self.handle_scrolling(hand)
        
        elif self.stable_gesture_state == GestureState.IDLE:
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                    self.is_dragging = False
                    print("ACTION: Drag ended (IDLE stable state)")
                except pyautogui.FailSafeException:
                    print("ERROR: PyAutoGUI FailSafe triggered during drag end (IDLE).")
                except Exception as e:
                    print(f"ERROR: Failed to release drag on IDLE: {e}")
            # If idle, and was previously scrolling, ensure scroll_y is reset
            if self.prev_stable_gesture_state == GestureState.SCROLLING:
                 self.last_scroll_y = 0 


    def draw_info_overlay(self, img, hand, raw_current_gesture):
        """Draw gesture information and visual feedback on the camera feed"""
        
        fingers_up = self.get_fingers_up(hand)
        pinch_distance = self.calculate_pinch_distance(hand)
        
        cv2.putText(img, f"Raw Gesture: {raw_current_gesture.value}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Stable Gesture: {self.stable_gesture_state.value}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(img, f"Fingers Up: {fingers_up}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.putText(img, f"Pinch Dist: {int(pinch_distance)}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        mouse_x, mouse_y = pyautogui.position()
        cv2.putText(img, f"Mouse @: ({mouse_x}, {mouse_y})", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if self.is_dragging:
            cv2.putText(img, "DRAGGING ACTIVE", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.rectangle(img, (self.cam_margin_x, self.cam_margin_y),
                      (self.cam_width - self.cam_margin_x, self.cam_height - self.cam_margin_y),
                      (255, 0, 0), 2)
        
        if raw_current_gesture == GestureState.MOVING and hand:
            lm_list = hand["lmList"]
            index_tip = lm_list[8]
            cv2.circle(img, (index_tip[0], index_tip[1]), 10, (0, 255, 0), cv2.FILLED)

    def run(self):
        """Main loop for the hand gesture mouse controller"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream. Check camera connection or if another app is using it.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        
        print("\nStarting hand gesture mouse controller...")
        print("Show your hand to the camera and make gestures!")
        print("Blue rectangle shows the active control area. Keep your index finger TIP within it for mouse movement.")
        
        try:
            while True:
                ret, img = cap.read()
                if not ret:
                    print("Failed to grab frame, exiting...")
                    break
                
                img = cv2.flip(img, 1)
                
                hands, img = self.detector.findHands(img, draw=True)
                
                if hands:
                    hand = hands[0]
                    raw_current_gesture = self.detect_gesture(hand)
                    self.process_gesture_state(hand, raw_current_gesture)
                    self.draw_info_overlay(img, hand, raw_current_gesture)
                    
                else:
                    # No hand detected: Reset states
                    if self.is_dragging:
                        try:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                            print("ACTION: Drag ended (no hand detected)")
                        except pyautogui.FailSafeException:
                            print("ERROR: PyAutoGUI FailSafe triggered during drag end (no hand).")
                        except Exception as e:
                            print(f"ERROR: Failed to release drag on no hand detection: {e}")
                    
                    # Reset scroll and gesture states when no hand is detected
                    self.last_scroll_y = 0 
                    self.raw_current_gesture = GestureState.IDLE
                    self.prev_stable_gesture_state = self.stable_gesture_state # Capture current stable before resetting
                    self.stable_gesture_state = GestureState.IDLE
                    self.gesture_start_time = time.time() # Reset timer for potential new gesture
                    
                    cv2.putText(img, "No hand detected. Show your hand!", (10, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    cv2.rectangle(img, (self.cam_margin_x, self.cam_margin_y),
                                  (self.cam_width - self.cam_margin_x, self.cam_height - self.cam_margin_y),
                                  (255, 0, 0), 2)
                
                cv2.imshow('Hand Gesture Mouse Controller (Q to Quit)', img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("'q' pressed. Exiting...")
                    break
                    
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping...")
        except Exception as e:
            print(f"\n!!! AN UNEXPECTED ERROR OCCURRED IN MAIN LOOP !!!: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if self.is_dragging:
                try:
                    pyautogui.mouseUp()
                    print("Final cleanup: Released mouse drag.")
                except pyautogui.FailSafeException:
                    print("Final cleanup: PyAutoGUI FailSafe triggered during mouseUp.")
                except Exception as e:
                    print(f"Final cleanup: Error releasing mouseUp: {e}")
            cap.release()
            cv2.destroyAllWindows()
            print("Hand gesture mouse controller stopped.")

if __name__ == "__main__":
    try:
        controller = HandGestureMouseController()
        controller.run()
    except Exception as e:
        print(f"\nERROR: Failed to initialize or start controller: {e}")
        import traceback
        traceback.print_exc()