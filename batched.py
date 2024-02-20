

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

# Input prompts

batch_size = 5

prompts = \
[
"Write about the execution of a famous pirate. Write from the perspective of the hooded executioner.",
"The Secret in the Attic: Uncover a mysterious object or letter that changes everything.",
"The Map to Yesterday: A character finds an ancient map that doesn't lead to a place, but to a past time, uncovering family secrets.",
"The Painter Who Used the Wind: An artist who can capture the essence of the wind in their paintings, affecting the weather and emotions of the town.",
"A Whisper from the Stars: In a world where people receive guidance from the stars, one starless night leads to unexpected revelations.",
"Echoes of a Dream: A character starts living the life of a person they consistently dream about, blurring the lines between reality and dreams.",
"The Last Library on Earth: In a future where books are banned, the discovery of the last existing library changes a young rebel's life.",
"The Clock in the River: A town where time flows backward, after a mysterious old clock is found in the river.",
"Dance of the Fireflies: On one special night every year, the fireflies' dance grants wishes, but with unexpected consequences.",
"The Last Message from a Sinking Ship: A bottle washes ashore with a message from a long-lost ship, unraveling a historic mystery."
]

# Sort by length to minimize padding

s_prompts = sorted(prompts, key = len)

# Apply prompt format

def format_prompt(sp, p):
    return f"[INST] <<SYS>>\n{sp}\n<</SYS>>\n\n{p} [/INST]"

system_prompt = "Answer the question to the best of your ability."
f_prompts = [format_prompt(system_prompt, p) for p in s_prompts]

# Split into batches

batches = [f_prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]

# Initialize model and cache

model_directory =  "/home/dnhkng/Documents/models/Mistral-7B-Instruct-v0.2"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

config.max_batch_size = batch_size  # Model instance needs to allocate temp buffers to fit the max batch size

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)  # Cache needs to accommodate the batch size
model.load_autosplit(cache)

tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Sampling settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.85
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.05

max_new_tokens = 512

# generator.warmup()  # Only needed to fully initialize CUDA, for correct benchmarking

# Generate for each batch

collected_outputs = []
for b, batch in enumerate(batches):

    print(f"Batch {b + 1} of {len(batches)}...")

    outputs = generator.generate_simple(batch, settings, max_new_tokens, seed = 1234)


    print(outputs)
    trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
    collected_outputs += trimmed_outputs

# Print the results

for q, a in zip(s_prompts, collected_outputs):
    print("---------------------------------------")
    print("Q: " + q)
    print("A: " + a)

# print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")