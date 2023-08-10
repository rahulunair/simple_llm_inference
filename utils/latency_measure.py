import logging
import warnings

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
import intel_extension_for_pytorch
from transformers import LlamaForCausalLM, LlamaTokenizer
import psutil
import time

# Set seed
seed_value = 88
torch.manual_seed(seed_value)
if torch.xpu.is_available():
    torch.xpu.manual_seed_all(seed_value)

# Device configuration
device = "xpu" if torch.xpu.is_available() else "cpu"

# Default dtype configuration
# dtype_options = [torch.float16, torch.bfloat16, torch.float32]
dtype = torch.float16 if device == "xpu" else torch.float32

# Autocast configuration
if device == "xpu":
    autocast = torch.xpu.amp.autocast(
        enabled=True if dtype != torch.float32 else False, dtype=dtype
    )
else:
    autocast = torch.cpu.amp.autocast(
        enabled=True if dtype != torch.float32 else False, dtype=dtype
    )

# Default values
max_tokens = 100
temperature = 1.0
num_beams = 4
do_sample = False

# Load the model and tokenizer
model_name = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
model = model.to(device)

# Print configurations
print("\n___")
print("Model Configuration:")
print("  Device:", device)
print("  Dtype:", dtype)
print("  Model name:", model_name)
print("  Tokenizer name:", tokenizer.__class__.__name__)
print("\n___")
print("Generation Params:")
print("  max_tokens:", max_tokens)
print("  temperature:", temperature)
print("  num_beams:", num_beams)
print("  do_sample:", do_sample)

# Prepare the input
input_text = "Once upon a time there was"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
input_token_length = len(input_ids[0])

# Warm-up with autocasting
for _ in range(5):
    with autocast:
        _ = model.generate(input_ids, max_length=input_ids.shape[1] + 1)

# Create an XPU stream
stream = torch.xpu.Stream() if device == "xpu" else None

with torch.xpu.stream(stream) if device == "xpu" else torch.no_grad():
    start_time = (
        torch.xpu.Event(enable_timing=True) if device == "xpu" else time.perf_counter()
    )
    end_time = torch.xpu.Event(enable_timing=True) if device == "xpu" else None

    # First token
    with autocast:
        start_time.record() if device == "xpu" else None
        _ = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 1,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
        )
        end_time.record() if device == "xpu" else None
        torch.xpu.synchronize() if device == "xpu" else None
        first_token_latency = (
            start_time.elapsed_time(end_time) / 1000
            if device == "xpu"
            else time.perf_counter() - start_time
        )

    # Second token
    with autocast:
        start_time.record() if device == "xpu" else time.perf_counter()
        _ = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 2,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
        )
        end_time.record() if device == "xpu" else None
        torch.xpu.synchronize() if device == "xpu" else None
        second_token_latency = (
            start_time.elapsed_time(end_time) / 1000
            if device == "xpu"
            else time.perf_counter() - start_time
        )

    if device == "xpu":
        memory_allocated_before = round(torch.xpu.memory_reserved() / 1024**3, 3)
    else:
        memory_allocated_before = round(
            psutil.Process().memory_info().rss / 1024**3, 3
        )

    with autocast:
        start_time.record() if device == "xpu" else time.perf_counter()
        generated_output = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
        )
        end_time.record() if device == "xpu" else None
        torch.xpu.synchronize() if device == "xpu" else None
        average_latency_all_tokens = (
            start_time.elapsed_time(end_time) / 1000
            if device == "xpu"
            else time.perf_counter() - start_time
        )
        generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        total_generated_tokens = len(generated_output[0]) - input_token_length
    if device == "xpu":
        memory_allocated_after = round(torch.xpu.memory_reserved() / 1024**3, 3)
    else:
        memory_allocated_after = round(
            psutil.Process().memory_info().rss / 1024**3, 3
        )

memory_utilized_for_generation = memory_allocated_after - memory_allocated_before

print("\n___")
print("Generate text:")
print(f"  {generated_text}")

print("\n___")
print("Token Latency")
print(f"  Input token length: {input_token_length}")
print(f"  Total generated tokens: {total_generated_tokens}")
print(f"  First token generation latency: {first_token_latency} sec")
print(f"  Second token generation latency: {second_token_latency} sec")
print(f"  Average latency for generating all tokens: {average_latency_all_tokens} sec")

print("\n___")
print("Memory Stats:")
print(f"  Total memory used for the benchmark: {memory_allocated_after} GB")
print(f"  Memory utilized for text generation: {memory_utilized_for_generation} GB")
