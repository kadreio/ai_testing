from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
tokenizer = AutoTokenizer.from_pretrained("OpenLemur/lemur-70b-v1")

device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": "cpu",
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.embed_tokens": "cpu",
    "model.layers":"cpu",
    "model.norm":"cpu"
}
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

model = AutoModelForCausalLM.from_pretrained("OpenLemur/lemur-70b-v1",     
                                             device_map=device_map,
                                             quantization_config=quantization_config,
    load_in_8bit=True)


# Text Generation Example
prompt = "The world is "
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=50, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Code Generation Example
prompt = """
def factorial(n):
    if n == 0:
        return 1
"""
input = tokenizer(prompt, return_tensors="pt")
output = model.generate(**input, max_length=200, num_return_sequences=1)
generated_code = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_code)