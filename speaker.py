import nemo.collections.asr as nemo_asr

def main(audio_file_path):
    # Load the diarization model
    diarizer = nemo_asr.models.SpeakerDiarizer.from_pretrained(model_name="diarization_with_speaker_embedding")

    # Configure the diarizer
    diarizer.setup_audio_processor()
    diarizer.params['window_length_in_sec'] = 1.5
    diarizer.params['shift_length_in_sec'] = 0.75

    # Process the audio file and perform diarization
    diarization_info = diarizer.transcribe([audio_file_path])

    # Process the output for each speaker
    for segment in diarization_info:
        print(f"Speaker {segment['speaker']} from {segment['start_time']} to {segment['end_time']}")

if __name__ == "__main__":
    audio_file_path = "trim40.mp3"  # Replace with your audio file path
    main(audio_file_path)