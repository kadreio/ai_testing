from pyannote.audio import Pipeline
import torch

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0",
                                    use_auth_token="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")

# pipeline = pipeline.to(torch.device("cuda:0"))
pipeline.to(torch.device("cuda"))
# pipeline=pipeline.to(0)
print(torch.cuda.is_available())

diarization = pipeline("./data/trim40.wav")
# apply the pipeline to an audio file
# diarization = pipeline("data/trim40.mp3")

# dump the diarization output to disk using RTTM format
with open("data/trim40.rttm", "w") as rttm:
    diarization.write_rttm(rttm)