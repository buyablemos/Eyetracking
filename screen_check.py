import pyautogui
import os

def capture_and_verify_screenshot(file_path):
    try:
        # Capture the screenshot
        screenshot = pyautogui.screenshot()

        # Save the screenshot to the specified file path
        screenshot.save(file_path)

        # Check if the file exists
        if os.path.exists(file_path):
            print(f"Screenshot successfully saved to {file_path}")
            return True
        else:
            print(f"Failed to save the screenshot to {file_path}")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

# Specify the file path where the screenshot will be saved
screenshot_file_path = "screenshot_test.png"

# Capture and verify the screenshot
capture_and_verify_screenshot(screenshot_file_path)
