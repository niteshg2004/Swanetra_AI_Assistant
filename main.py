import cv2
import torch
import pyttsx3
import speech_recognition as sr
import threading
import requests
import json
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import os
import warnings
from config import API_URL, bot_token, chat_id, known_width, focal_length

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=FutureWarning)

# Global variables
latest_frame = None
lock = threading.Lock()
running = True

# ====================== TTS Setup ======================
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# ====================== LM Studio (Llama 3) ======================
def get_ai_response(prompt):
    try:
        payload = {
            "model": "hugging-quants/Llama-3.2-1B-Instruct-Q8_0-GGUF",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return "I'm sorry, I couldn't process that."

# ====================== YOLOv5 Setup ======================
yolo_model = YOLO("yolov8n.pt")
yolo_model.verbose = False

# ====================== BLIP Setup ======================
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
blip_model.to(device) # type: ignore

# ====================== Webcam ======================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# ====================== Speech Recognition ======================
def list_microphones():
    try:
        names = sr.Microphone.list_microphone_names() or []
        for index, name in enumerate(names):
            print(f"Mic {index}: {name}")
    except Exception as e:
        print(f"Microphone enumeration error: {e}")


def recognize_speech():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6
    device_index_env = os.getenv("MIC_DEVICE_INDEX")
    mic_kwargs = {}
    if device_index_env is not None:
        try:
            mic_kwargs["device_index"] = int(device_index_env)
        except ValueError:
            print("Invalid MIC_DEVICE_INDEX; ignoring.")
    try:
        with sr.Microphone(**mic_kwargs) as source:
            print("Calibrating mic for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=6)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except sr.UnknownValueError:
            print("Speech not understood.")
            return None
        except sr.RequestError as e:
            print(f"Speech API request error: {e}")
            return None
    except sr.WaitTimeoutError:
        print("No speech detected (timeout).")
        return None
    except AttributeError as e:
        # Often indicates missing PyAudio or permissions
        print(f"Audio input error: {e}. If on macOS, grant Microphone permission to Terminal/VS Code.")
        list_microphones()
        return None
    except OSError as e:
        # No default input device or permission denied
        print(f"OS audio error: {e}. Try selecting a different mic via MIC_DEVICE_INDEX env.")
        list_microphones()
        return None
# ====================== Object and Scene Description ======================
def describe_objects_and_scene(frame):
    results = yolo_model(frame)[0]
    descriptions = []

    for box in results.boxes:
        conf = box.conf.item()
        if conf < 0.4:
            continue
        cls_id = int(box.cls.item())
        label = yolo_model.names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        pixel_width = x2 - x1
        distance = (known_width * focal_length) / pixel_width
        x_center = (x1 + x2) / 2
        frame_width = frame.shape[1]
        if x_center < frame_width / 3:
            direction = "on the left"
        elif x_center > 2 * frame_width / 3:
            direction = "on the right"
        else:
            direction = "in the center"
        descriptions.append(f"a {label} about {int(distance)} centimeters away {direction}")

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs) # pyright: ignore[reportArgumentType]
    caption = processor.decode(out[0], skip_special_tokens=True)

    if descriptions:
        return "I see " + ", ".join(descriptions) + f". It looks like {caption}."
    else:
        return f"It looks like {caption}."

# ====================== Telegram Alert ======================
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        print("Telegram:", response.text)
    except Exception as e:
        print(f"Telegram Error: {e}")

# ====================== Location from IP ======================
def get_ip_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        loc = data["loc"].split(",")
        return float(loc[0]), float(loc[1])
    except:
        return None, None

# ====================== Object Detection Loop ======================
def object_detection_loop():
    global latest_frame, running
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_frame = frame.copy()
        results = yolo_model(frame, verbose=False)[0]
        annotated = results.plot()
        cv2.imshow("Swanetra View", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break
    cap.release()
    cv2.destroyAllWindows()

# ====================== Voice Loop ======================
def voice_loop():
    global running
    while running:
        command = recognize_speech()
        if not command:
            continue
        print(f"You said: {command}")
        
        if any(phrase in command for phrase in [
            "what is in front", "what do you see", "what do i see",
            "mere saamne kya hai", "mere samne kya hai",
            "kya hai mere saamne", "kya hai mere samne",
            "aage kya hai", "kya dikh rha hai",
            "kya dikh rahe hai", "kya dikh rahe", "yeh kya hai"
        ]):
            with lock:
                if latest_frame is not None:
                    response = describe_objects_and_scene(latest_frame)
                else:
                    print("Warning: latest_frame is None. Possible causes: webcam not initialized, detection thread not running, or no frames captured yet.")
                    response = "I can't see anything right now."

        elif "i need help" in command:
            lat, lon = get_ip_location()
            if lat and lon:
                location_link = f"https://www.google.com/maps/q={lat},{lon}"
                message = f"# Alert from Swanetra: The user needs help!\n📍 Location: {location_link}"
                send_telegram_alert(message)
                response = "Emergency message sent with your location."
            else:
                message = "# Alert from Swanetra: The user needs help!\n⚠ Location could not be determined."
                send_telegram_alert(message)
                response = "Emergency message sent, but location could not be found."
        
        elif "exit" in command or "quit" in command:
            response = "Goodbye!"
            running = False
            break
        
        else:
            response = get_ai_response(command)
        
        print(f"AI: {response}")
        speak(response)

# ====================== Start Threads ======================
if __name__ == '__main__':
    try:
        detection_thread = threading.Thread(target=object_detection_loop, daemon=True)
        detection_thread.start()
        voice_loop()
    except KeyboardInterrupt:
        print("Program interrupted. Shutting down...")
        running = False
    finally:
        cap.release()
        cv2.destroyAllWindows()