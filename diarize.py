import matplotlib.pyplot as plt
import numpy as np
import librosa
import os
import wget
import json
from omegaconf import OmegaConf
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels, labels_to_pyannote_object
from nemo.collections.asr.models import ClusteringDiarizer

ROOT = os.getcwd()
data_dir = os.path.join(ROOT,'data')
os.makedirs(data_dir, exist_ok=True)
an4_audio = os.path.join(data_dir,'an4_diarize_test.wav')
an4_rttm = os.path.join(data_dir,'an4_diarize_test.rttm')
if not os.path.exists(an4_audio):
    an4_audio_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.wav"
    an4_audio = wget.download(an4_audio_url, data_dir)
if not os.path.exists(an4_rttm):
    an4_rttm_url = "https://nemo-public.s3.us-east-2.amazonaws.com/an4_diarize_test.rttm"
    an4_rttm = wget.download(an4_rttm_url, data_dir)

sr = 16000
signal, sr = librosa.load(an4_audio,sr=sr) 

fig,ax = plt.subplots(1,1)
fig.set_figwidth(20)
fig.set_figheight(2)
plt.plot(np.arange(len(signal)),signal,'gray')
fig.suptitle('Reference merged an4 audio', fontsize=16)
plt.xlabel('time (secs)', fontsize=18)
ax.margins(x=0)
plt.ylabel('signal strength', fontsize=16);
a,_ = plt.xticks();plt.xticks(a,a/sr);

plt.savefig('books_read.png')
labels = rttm_to_labels(an4_rttm)
reference = labels_to_pyannote_object(labels)
print(labels)
reference

meta = {
    'audio_filepath': an4_audio, 
    'offset': 0, 
    'duration':None, 
    'label': 'infer', 
    'text': '-', 
    'num_speakers': 2, 
    'rttm_filepath': an4_rttm, 
    'uem_filepath' : None
}
with open('data/input_manifest.json','w') as fp:
    json.dump(meta,fp)
    fp.write('\n')

output_dir = os.path.join(ROOT, 'oracle_vad')
os.makedirs(output_dir,exist_ok=True)

MODEL_CONFIG = os.path.join(data_dir,'diar_infer_telephonic.yaml')
if not os.path.exists(MODEL_CONFIG):
    config_url = "https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/diar_infer_telephonic.yaml"
    MODEL_CONFIG = wget.download(config_url,data_dir)

config = OmegaConf.load(MODEL_CONFIG)
print(OmegaConf.to_yaml(config))

config.diarizer.manifest_filepath = 'data/input_manifest.json'
config.diarizer.out_dir = output_dir # Directory to store intermediate files and prediction outputs
pretrained_speaker_model = 'titanet_large'
config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
config.diarizer.speaker_embeddings.parameters.window_length_in_sec = [1.5,1.25,1.0,0.75,0.5] 
config.diarizer.speaker_embeddings.parameters.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1] 
config.diarizer.speaker_embeddings.parameters.multiscale_weights= [1,1,1,1,1] 
config.diarizer.speaker_embeddings.oracle_vad_manifest = None
config.diarizer.paths2audio_files = [os.path.join(data_dir, 'an4_diarize_test.wav')]
config.diarizer.oracle_vad = False # ----> ORACLE VAD 
# config.diarizer.oracle_num_speakers = True

config.diarizer.oracle_num_speakers = 2
# config.diarizer.vad.window_length_in_sec  = [1.5,1.25,1.0,0.75,0.5] 
# config.diarizer.vad.shift_length_in_sec = [0.75,0.625,0.5,0.375,0.1]
config.diarizer.vad.window_length_in_sec=0.15  # Window length in sec for VAD context input 
config.diarizer.vad.shift_length_in_sec=0.01 # Shift length in sec for generate frame level VAD prediction
config.diarizer.vad.vad_decision_smoothing="median" # False or type of smoothing method (eg=median)
config.diarizer.vad.smoothing="median" # False or type of smoothing method (eg=median)
config.diarizer.vad.overlap=0.5 # Overlap ratio for overlapped mean/median smoothing filter
config.diarizer.vad.onset=0.1 # Onset threshold for detecting the beginning and end of a speech 
config.diarizer.vad.offset=0.1 # Offset threshold for detecting the end of a speech
config.diarizer.vad.pad_onset=0.1 # Adding durations before each speech segment 
config.diarizer.vad.pad_offset=0 # Adding durations after each speech segment 
config.diarizer.vad.min_duration_on=0 # Threshold for small non_speech deletion
config.diarizer.vad.min_duration_off=0.2 # Threshold for short speech segment deletion
config.diarizer.vad.filter_speech_first=True
config.diarizer.vad.smoothing_params = {"method": "median", "overlap": 0.875} # False or type of smoothing_params method (eg=median)
config.diarizer.vad.postprocessing_params= {"onset": .8, "offset": 0.7, "min_duration_on":0.1, "min_duration_off":0.3}
# config.diarizer.vad.postprocessing_params.offset = 0.7
# config.diarizer.vad.postprocessing_params.min_duration_on = 0.1
# config.diarizer.vad.postprocessing_params.min_duration_off = 0.3
# config.diarizer.vad.smoothing_params.overlap=0.875 # False or type of smoothing method (eg=median)
config.diarizer.path2groundtruth_rttm_files = [os.path.join(data_dir, 'an4_diarize_test.rttm')]
oracle_vad_clusdiar_model = ClusteringDiarizer(cfg=config)


oracle_vad_clusdiar_model.diarize()