﻿# Voice Conversational Bot

## Overview

This is a **Voice Conversational Bot** built using **Streamlit** that allows users to interact with an AI-powered assistant using voice or text input. The assistant responds in a conversational manner and supports different personalities. It uses two different AI models for generating responses: **OpenAI GPT-4** and **Ollama**.

The bot listens for your speech, processes it, and gives a spoken or written response. It also supports text input as an alternative for communication.

## Features

- **Voice Input**: Speak directly to the bot, and it will recognize your speech using the `speech_recognition` library.
- **Text Input**: Alternatively, users can type their input in a text box.
- **Multiple AI Models**: Choose between **OpenAI GPT-4** or **Ollama** for response generation.
- **Personality Selection**: Customize the bot’s personality with options such as "Casual & Friendly", "Professional", "Humorous", or "Empathetic".
- **Text-to-Speech**: The bot responds with voice output using the `gTTS` library.
- **Conversation History**: The conversation is saved in the session and displayed for easy reference.
- **Reset Conversation**: Users can clear the conversation history anytime.

## Requirements

To run the application, you'll need the following dependencies:

- `Python`
- `streamlit` - for the web app interface
- `speech_recognition` - for recognizing speech input
- `gtts` - for converting text to speech
- `requests` - for making HTTP requests to Ollama
- `openai` - for using the OpenAI GPT-4 API

### Install Dependencies

Use the following command to install the required libraries:

```bash
pip install -r requirements.txt
## Setup

### 1. **OpenAI Setup**
- Get your OpenAI API key from [OpenAI](https://beta.openai.com/signup/).
- Enter the OpenAI API key in the sidebar on the app when prompted.

### 2. **Ollama Setup**
- Install Ollama and run it locally by following the instructions at [Ollama](https://ollama.com/).
- Enter the Ollama API URL in the sidebar on the app when prompted (default URL is `http://localhost:11434`).

### 3. **Model Setup**
- Once you select a model provider (OpenAI or Ollama) in the sidebar, choose the appropriate model you want to use.

## How to Run

After setting up the dependencies and configurations:

1. Run the app using Streamlit:

```bash
streamlit run main.py

