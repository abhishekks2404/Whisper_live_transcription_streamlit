import streamlit as st
import argparse
import io
import os
import speech_recognition as sr
import whisper
import torch
from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
import time





def main():
    
    st.title("Real-Time Whisper Transcription")
    model_options = ["tiny", "base", "small", "medium", "large"]
    selected_model = st.selectbox("Select a model:", model_options)
    if selected_model:
        st.success("Model Loaded Successfully....")

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        nonlocal last_sample
        nonlocal is_recording
        if not is_recording:
            return
        # Grab the raw bytes and push it into the thread-safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    

    with st.spinner("Loading model"):

        
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--model", default=selected_model, help="Model to use",
                            choices=["tiny", "base", "small", "medium", "large"])
        parser.add_argument("--non_english", action='store_true',
                            help="Don't use the English model.")
        parser.add_argument("--energy_threshold", default=1000,
                            help="Energy level for the microphone to detect.", type=int)
        parser.add_argument("--record_timeout", default=7,
                            help="How real-time the recording is in seconds.", type=float)
        parser.add_argument("--phrase_timeout", default=2,
                            help="How much empty space between recordings before we "
                                "consider it a new line in the transcription.", type=float)
        if 'linux' in platform:
            parser.add_argument("--default_microphone", default='pulse',
                                help="Default microphone name for SpeechRecognition. "
                                    "Run this with 'list' to view available Microphones.", type=str)
        args = parser.parse_args()

        # The last time a recording was retrieved from the queue.
        phrase_time = None
        # Current raw audio bytes.
        last_sample = bytes()
        # Thread safe Queue for passing data from the threaded recording callback.
        data_queue = Queue()
        # We use SpeechRecognizer to record our audio because it has a nice feature where it can detect when speech ends.
        recorder = sr.Recognizer()
        recorder.energy_threshold = args.energy_threshold
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramatically to a point where the SpeechRecognizer never stops recording.
        recorder.dynamic_energy_threshold = False

        start_time = time.time()
        timeout = 5

        
        if 'linux' in platform:
            mic_name = args.default_microphone
            if not mic_name or mic_name == 'list':
                st.write("Available microphone devices are: ")
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    st.write(f"Microphone with name \"{name}\" found")
                return
            else:
                for index, name in enumerate(sr.Microphone.list_microphone_names()):
                    if mic_name in name:
                        source = sr.Microphone(sample_rate=16000, device_index=index)
                        break
        else:
            source = sr.Microphone(sample_rate=16000)

        # Load / Download model
        model = args.model

        if args.model != "large" and not args.non_english:
            model = model + ".en"
        audio_model = whisper.load_model(model)

        record_timeout = args.record_timeout
        phrase_timeout = args.phrase_timeout

        temp_file = NamedTemporaryFile().name
        transcription = ['']

        with source:
            recorder.adjust_for_ambient_noise(source)

        is_recording = False
        st.session_state.eye = True



    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        # Record button
        

        if st.button("Record"):
            is_recording = not is_recording


            # Display transcribed text in real-time
            st.header("Transcription:")
            for line in transcription:
                last_line = line

            last_voice_time = None
            last_print_time = None
            timeout = 5
            

            while True:

                
                now = datetime.utcnow()
                # Pull raw recorded audio from the queue.
                if not data_queue.empty():
                    last_voice_time = now

                    phrase_complete = False
                    # If enough time has passed between recordings, consider the phrase complete.
                    # Clear the current working audio buffer to start over with the new data.
                    if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                        last_sample = bytes()
                        phrase_complete = True
                    # This is the last time we received new audio data from the queue.
                    phrase_time = now

                    # Concatenate our current audio data with the latest audio data.
                    while not data_queue.empty():
                        data = data_queue.get()
                        last_sample += data

                    # Use AudioData to convert the raw data to wav data.
                    audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
                    wav_data = io.BytesIO(audio_data.get_wav_data())

                    # Write wav data to the temporary file as bytes.
                    with open(temp_file, 'w+b') as f:
                        f.write(wav_data.read())

                    # Read the transcription.
                    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    # If we detected a pause between recordings, add a new item to our transcription.
                    # Otherwise, edit the existing one.
                    if phrase_complete:
                        transcription.append(text)
                    else:
                        transcription[-1] = text

                    # Update Streamlit display
                    for line in transcription:
                        last_line = line
                    if last_line is not None:
                        st.write(last_line)

                
                    
                        last_print_time = now

                if ((last_voice_time and now - last_voice_time > timedelta(seconds=timeout)) and (last_print_time and now - last_print_time > timedelta(seconds=timeout))):
                    print("No voice or text detected for 5 seconds, exiting loop")
                    break

                    # Sleep to avoid high CPU usage
                sleep(0.25)

                

                    
                

        

if __name__ == "__main__":
    with open("style.css") as source_des:
        st.markdown(f"<style>{source_des.read()}</style>", unsafe_allow_html=True)

    def decode_name(encoded_name):
        return base64.b64decode(encoded_name).decode()
    encoded_name = "Q29kZSBpcyBmdWxseSBnZW5lcmF0ZWQgYnkgQWJoaXNoZWsgS3VtYXIgU2luZ2g="
    decode_name(encoded_name)

    main()
