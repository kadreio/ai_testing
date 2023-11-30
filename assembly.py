import assemblyai as aai

aai.settings.api_key = "13c62e08e4b74adda0ea1a283b8f1ba2"
transcriber = aai.Transcriber()

# # transcript = transcriber.transcribe("https://storage.googleapis.com/aai-web-samples/news.mp4")
# transcript = transcriber.transcribe()

# print(transcript.text)
# 4

config = aai.TranscriptionConfig(speaker_labels=True)

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(
  "./data/trim40.wav",
  config=config
)

for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")