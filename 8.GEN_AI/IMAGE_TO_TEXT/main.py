import gradio as gr
from transformers import pipeline
from gtts import gTTS
import os

# Load the BLIP image captioning model
captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

# Function to generate captions for an input image
def generate_caption(image):
    result = captioner(image)
    caption = result[0]['generated_text']
    return caption

# Function to convert text to speech and return the audio file path
def text_to_speech(text, file_path="output.mp3"):
    """
    Converts text to speech using gTTS and saves the audio.
    """
    tts = gTTS(text=text, lang="en", slow=False)
    tts.save(file_path)
    return file_path

# Combined function for caption generation and TTS
def generate_caption_and_speech(image):
    # Generate the image caption
    caption = generate_caption(image)
    
    # Convert the caption to speech
    audio_file = text_to_speech(caption, file_path="caption_audio.mp3")
    
    # Return both caption text and audio file
    return caption, audio_file

# Gradio Interface with a Logo
with gr.Blocks() as demo:
    # Title Section with Logo
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <img src="https://raw.githubusercontent.com/Nithish-github/DeepLearning-Projects/master/assets/letsgodeep.png"  alt="Logo" width="500"/>
        <h1 style="font-family: Arial, sans-serif;">Image Captioning App</h1>
        <p style="color: gray; font-size: 16px;">Upload an image and generate a descriptive caption using the BLIP model.</p>
    </div>
    """)

    # Input and Output Section
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload an Image")
        text_output = gr.Textbox(label="Generated Caption")
        audio_output = gr.Audio(label="Caption Audio", type="filepath")

    # Submit Button
    submit_btn = gr.Button("Generate Caption and Speech")
    submit_btn.click(generate_caption_and_speech, inputs=image_input, outputs=[text_output, audio_output])

# Launch the Gradio Interface
demo.launch()
