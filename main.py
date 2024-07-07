import cv2
import threading
import speech_recognition as sr
import google.generativeai as genai
import pyttsx3
from deepface import DeepFace
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# Configure your Gemini API key
API_KEY = 'AIzaSyDSKpYnzLuBySNUSQLzNJvECIXA2Txkcfc'
genai.configure(api_key=API_KEY)

# Initialize text-to-speech engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to recognize speech from the microphone
def recognize_speech_from_mic(recognizer, microphone):
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)
    
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }
    
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"
    
    return response

# Function to interact with the Gemini chatbot
def chat_with_gemini(input_text):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    response = chat.send_message(input_text)
    return response.text

# Function to perform emotion detection using DeepFace and handle chatbot interaction
def detect_emotions_and_chat(gui):
    cap = cv2.VideoCapture(0)
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1


        if frame_count % 5 == 0:

            try:
                small_frame = cv2.resize(frame, (320, 240))
                analysis = DeepFace.analyze(small_frame, actions=['emotion'])

                for face in analysis:
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    emotion = face['dominant_emotion']
                    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    gui.update_chat(f"Detected emotion: {emotion}")
                
                # Only proceed with chatbot interaction if a face is detected
                if emotion:
                    response = recognize_speech_from_mic(recognizer, microphone)
                    if response["transcription"]:
                        gui.update_chat(f"You said: {response['transcription']}")
                        chatbot_response = chat_with_gemini(response["transcription"])
                        gui.update_chat(f"Gemini Chatbot says: {chatbot_response}")
                        speak_text(chatbot_response)
                    
                    if not response["success"]:
                        gui.update_chat(f"Error: {response['error']}")
            except Exception as e:
                print(f"Error analyzing frame: {e}")
            
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        gui.update_video(imgtk)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# GUI class
class ChatbotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot with Emotion Detection")
        self.root.geometry("800x680")

        self.video_frame = tk.Label(root, width=800, height=400)
        self.video_frame.grid(column=0, row=0, padx=10, pady=10)

        self.chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=90, height=10)
        self.chat_display.grid(column=0, row=1, padx=10, pady=10)
        self.chat_display.config(state=tk.DISABLED)

        self.quit_button = tk.Button(root, text="Quit", command=root.quit)
        self.quit_button.grid(column=0, row=2, padx=10, pady=10)

    def update_chat(self, message):
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.insert(tk.END, message + "\n")
        self.chat_display.config(state=tk.DISABLED)
        self.chat_display.yview(tk.END)

    def update_video(self, imgtk):
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

def main():
    root = tk.Tk()
    gui = ChatbotGUI(root)

    # Start emotion detection and chatbot interaction in a separate thread
    emotion_and_chat_thread = threading.Thread(target=detect_emotions_and_chat, args=(gui,))
    emotion_and_chat_thread.daemon = True
    emotion_and_chat_thread.start()

    root.mainloop()

if __name__ == "__main__":
    main()
