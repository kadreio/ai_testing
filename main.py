import torch
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp

# print(nemo_asr.models.EncDecCTCModel.list_available_models())
print(nemo_nlp.modules.get_tokenizer_list())

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. Running on CPU.")