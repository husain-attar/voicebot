import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
import tempfile
import time
import requests
import json
from openai import OpenAI
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav



# Set page configuration
st.set_page_config(page_title="Voice Conversational Bot", page_icon="🎙️")

# def recognize_speech():
#     recognizer = sr.Recognizer()
    
#     with sr.Microphone() as source:
#         st.write("Listening... Speak now.")
#         recognizer.adjust_for_ambient_noise(source)
#         try:
#             audio = recognizer.listen(source, timeout=5)
#             st.write("Processing your speech...")
            
#             try:
#                 text = recognizer.recognize_google(audio)
#                 return text
#             except sr.UnknownValueError:
#                 st.error("Sorry, I couldn't understand what you said.")
#                 return None
#             except sr.RequestError:
#                 st.error("Sorry, speech recognition service is unavailable.")
#                 return None
                
#         except sr.WaitTimeoutError:
#             st.error("No speech detected. Please try again.")
#             return None

def recognize_speech():
    # Record audio
    st.write("Listening... Speak now.")
    
    # Set recording parameters
    duration = 5  # seconds
    sample_rate = 44100
    
    # Record audio
    recording = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, 
                       channels=1,
                       dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save to temporary wav file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        wav.write(temp_file.name, sample_rate, recording)
        temp_filename = temp_file.name
    
    # Use SpeechRecognition to transcribe the saved audio file
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(temp_filename) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        st.error("Sorry, I couldn't understand the audio.")
        return None
    except sr.RequestError:
        st.error("Could not request results from speech recognition service.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None



# Function to get response from OpenAI
def get_openai_response(prompt, conversation_history, system_prompt="You are a helpful Human assistant. Respond in a casual, conversational manner as a human would."):
    try:
        # Prepare messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add previous conversation context
        for msg in conversation_history:
            messages.append({
                "role": msg["role"], 
                "content": msg["message"]
            })
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        # Use API key from session state
        client = OpenAI(api_key=st.session_state.get('openai_api_key', ''))
        
        response = client.chat.completions.create(
            model="gpt-4o",  # Or use another OpenAI model
            messages=messages,
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return None

# Function to get response from Ollama
def get_ollama_response(prompt, conversation_history, system_prompt="You are a helpful Human assistant Imagine yourself as a software developer. Respond in a casual, conversational manner as a human would.", model="llama3"):
    try:
        ollama_url = st.session_state.ollama_url
        
        # Prepare messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add previous conversation context
        for msg in conversation_history:
            messages.append({
                "role": msg["role"], 
                "content": msg["message"]
            })
        
        # Add current user message
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the payload
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        # Send request to Ollama API
        response = requests.post(f"{ollama_url}/api/chat", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            return result['message']['content'].strip()
        else:
            st.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error with Ollama API: {e}")
        return None

# Function to convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
            
        tts.save(temp_filename)
        
        # Play the audio
        st.audio(temp_filename)
        
        # Clean up the file after a delay
        time.sleep(1)
        os.unlink(temp_filename)
        
    except Exception as e:
        st.error(f"Error generating speech: {e}")

# Function to setup OpenAI
def setup_openai():
    # OpenAI API Key input
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''
    
    openai_api_key = st.sidebar.text_input(
        "Enter OpenAI API Key:", 
        value=st.session_state.openai_api_key, 
        type="password"
    )
    
    # Validate and store API key
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.sidebar.success("OpenAI API Key set successfully!")
    
    return bool(openai_api_key)

# Function to setup Ollama
def setup_ollama():
    # Initialize ollama settings in session state
    if 'ollama_url' not in st.session_state:
        st.session_state.ollama_url = "http://localhost:11434"
    
    ollama_url = st.sidebar.text_input("Ollama API URL:", value=st.session_state.ollama_url)
    
    # Get available models from Ollama
    available_models = []
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        if response.status_code == 200:
            models_data = response.json()
            available_models = [model['name'] for model in models_data['models']]
        else:
            st.sidebar.warning("Could not fetch models from Ollama. Using default models.")
            available_models = ["llama3", "llama3:instruct", "mistral", "gemma"]
    except:
        st.sidebar.warning("Could not connect to Ollama. Using default models.")
        available_models = ["llama3", "llama3:instruct", "mistral", "gemma"]
    
    # Store models in session state
    st.session_state.ollama_models = available_models
    st.session_state.ollama_url = ollama_url

# Main app
def main():
    st.title("🎙️ Voice Conversational Bot")
    st.write("A voice bot that responds as a human would")
    
    # Setup sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    st.sidebar.subheader("Choose Model Provider")
    model_provider = st.sidebar.radio("Model Provider:", ["OpenAI", "Ollama"])
    
    # API Key or URL setup
    if model_provider == "OpenAI":
        if not setup_openai():
            st.sidebar.warning("Please enter a valid OpenAI API Key to continue.")
            return
    else:  # Ollama
        setup_ollama()
        st.sidebar.subheader("Ollama Model")
        ollama_model = st.sidebar.selectbox(
            "Choose Ollama model:",
            options=st.session_state.get('ollama_models', ["llama3", "llama3:instruct", "mistral", "gemma"])
        )
    
    # Personality settings
    st.sidebar.subheader("Bot Personality")
    personality = st.sidebar.selectbox(
        "Choose a personality style:",
        ["Casual & Friendly", "Professional", "Humorous", "Empathetic"]
    )
    
    # System prompts for different personalities
    personality_prompts = {
        "Casual & Friendly": "You are a helpful humnan assistant. Respond in a casual, friendly tone using contractions, simple language, and occasional slang. Keep responses short and conversational, as if chatting with a friend.",
        "Professional": "You are a professional human assistant. Provide clear, concise, and informative responses using proper language. Be polite but direct, focusing on delivering accurate information efficiently.",
        "Humorous": "You are a witty human assistant with a good sense of humor. Use casual language, inject jokes or puns when appropriate, and maintain a light-hearted tone. Keep responses engaging and fun.",
        "Empathetic": "You are an empathetic human assistant. Acknowledge feelings, use warm language, and respond with compassion. Focus on understanding and validating the person's perspective while providing helpful information."
    }
    
    system_prompt = personality_prompts[personality]
    
    # Initialize conversation history if not exists
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Display conversation history
    for message in st.session_state.conversation:
        if message["role"] == "user":
            st.write(f"You: {message['message']}")
        else:
            st.write(f"Bot: {message['message']}")
    
    # Voice input button
    if st.button("🎙️ Speak to the Bot"):
        user_input = recognize_speech()
        
        if user_input:
            # Display what was recognized
            st.write(f"You said: {user_input}")
            
            # Update conversation history
            st.session_state.conversation.append({"role": "user", "message": user_input})
            
            # Get response based on selected model provider
            response = None
            if model_provider == "OpenAI":
                response = get_openai_response(user_input, st.session_state.conversation, system_prompt)
            else:  # Ollama
                response = get_ollama_response(user_input, st.session_state.conversation, system_prompt, ollama_model)
            
            if response:
                # Display and speak the response
                st.write(f"Bot: {response}")
                
                # Update conversation history
                st.session_state.conversation.append({"role": "assistant", "message": response})
                
                text_to_speech(response)
    
    # Text input as alternative
    text_input = st.text_input("Or type your message here:")
    if text_input:
        # Update conversation history
        st.session_state.conversation.append({"role": "user", "message": text_input})
        
        # Get response based on selected model provider
        response = None
        if model_provider == "OpenAI":
            response = get_openai_response(text_input, st.session_state.conversation, system_prompt)
        else:  # Ollama
            response = get_ollama_response(text_input, st.session_state.conversation, system_prompt, ollama_model)
        
        if response:
            # Display and speak the response
            st.write(f"Bot: {response}")
            
            # Update conversation history
            st.session_state.conversation.append({"role": "assistant", "message": response})
            
            text_to_speech(response)
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.conversation = []
        st.rerun()  # Use st.rerun() instead of experimental_rerun()

if __name__ == "__main__":
    main()
