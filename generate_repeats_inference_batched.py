import os
import pickle
import pprint
import sys
from copy import copy

from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

model = "TinyLlama"
batch_size = 5

match model:
    case "TinyLlama":
        modelName = "TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2"
    case "Mistral":
        modelName = "Mistral-7B-Instruct-v0.2"
    case "Mixtral":
        modelName = "Mixtral-8x7B-instruct-exl2"
    case "Nous-Capybara":
        modelName = "Nous-Capybara-34B-4.0bpw"
    case "Miqu":
        modelName = "miqu-1-70b-sf-4.0bpw-h6-exl2"
    case _:
        print("Invalid generator model")
        quit()


themes = [
    "Write about the execution of a famous pirate. Write from the perspective of the hooded executioner.",
    "The Secret in the Attic: Uncover a mysterious object or letter that changes everything.",
    "The Map to Yesterday: A character finds an ancient map that doesn't lead to a place, but to a past time, uncovering family secrets.",
    "The Painter Who Used the Wind: An artist who can capture the essence of the wind in their paintings, affecting the weather and emotions of the town.",
    "A Whisper from the Stars: In a world where people receive guidance from the stars, one starless night leads to unexpected revelations.",
    "Echoes of a Dream: A character starts living the life of a person they consistently dream about, blurring the lines between reality and dreams.",
    "The Last Library on Earth: In a future where books are banned, the discovery of the last existing library changes a young rebel's life.",
    "The Clock in the River: A town where time flows backward, after a mysterious old clock is found in the river.",
    "Dance of the Fireflies: On one special night every year, the fireflies' dance grants wishes, but with unexpected consequences.",
    "The Last Message from a Sinking Ship: A bottle washes ashore with a message from a long-lost ship, unraveling a historic mystery.",
]

prompts = [f"""<s> [INST] You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win! The theme for the contest is: "{theme}" [/INST]""" for theme in themes]

batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]


print(batches)

# Initialize model and cache
seed = 1234

model_directory = f"/home/dnhkng/Documents/models/{modelName}"


config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.max_seq_len = 4096
config.max_batch_size = batch_size  # Model instance needs to allocate temp buffers to fit the max batch size

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)  # Cache needs to accommodate the batch size


model.load_autosplit(cache)

# Initialize tokenizer
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)


# Print model layers
def printModules(modules):
    for i, m in enumerate(modules):
        if hasattr(m, "layer_idx"):
            print(i, m.key, m.layer_idx)
        else:
            print(i, m.key)
    print()


## Store base modules
baseModules = model.modules
cache_class = type(cache)

print(vars(cache))
def buildModel(model, layers):
    model.modules = baseModules[:1]
    print("building model")
    for i, idx in enumerate(layers):
        model.modules += [copy(baseModules[idx*2 + 1])]
        model.modules[-1].layer_idx = i # for duplicate layers to use a different cache
        model.modules += [baseModules[idx*2 + 2]]

    model.modules += baseModules[-2:]

    model.head_layer_idx = len(model.modules) - 1
    model.config.num_hidden_layers = len(layers)
    model.last_kv_layer_idx = len(model.modules) - 4
    if 'cache' in globals():
        del globals()['cache']
    torch.cuda.empty_cache()
    model.cache_map = {}
    model.set_cache_map()

    cache = ExLlamaV2Cache(model, lazy = True, batch_size = batch_size)  # Cache needs to accommodate the batch size
    # cache = cache_class(model)

    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)


    return model, cache, generator


# Generate some text
max_new_tokens = 128
settings = ExLlamaV2Sampler.Settings()
# settings.temperature = 0.0  We sjip temperature, as we will sett it later.
settings.top_k = 50
settings.top_p = 0.8
settings.top_a = 0.0
settings.token_repetition_penalty = 1.05


def generate(self, prompt, settings, max_new_tokens, seed=seed):
    generator.warmup()
    output = generator.generate_simple(prompt, settings, max_new_tokens, seed=seed)
    return output


def generateLayerDict(size):
    layersDict = {}
    layersDict["0_0"] = list(range(size))  #  0_0 is the base model
    for j in range(size):
        for i in range(j):
            layersList = list(range(0, j)) + list(range(i, size))
            layersDict[f"{i}_{j}"] = layersList
    return layersDict


def saveData(
    modelOutput,
    layersDict,
    prompts,
    settings,
    max_new_tokens,
    seed,
    themes,
    modelName,
):
    experiment = {}
    experiment["modelOutput"] = modelOutput
    experiment["layersDict"] = layersDict
    experiment["prompts"] = prompts
    experiment["config"] = vars(settings)
    experiment["config"]["max_new_tokens"] = max_new_tokens
    experiment["config"]["seed"] = seed
    experiment["themes"] = themes
    experiment["modelName"] = modelName

    pickle.dump(experiment, open(f"{modelName}-stories_all.p", "wb"))




num_layers = int(
    (len(baseModules) - 3) / 2
)  # the is an embedder and 2 final layers (3), with 2 layers per attention block
layersDict = generateLayerDict(num_layers)
modelOutput = {}
for i in range(len(prompts)):
    modelOutput[i] = {}



# generated many samples form the base model, so we can compare them
settings.temperature = 0.8
for i in range(1, 3):
    key = f"{i}_0"
    model, cache, generator = buildModel(model, layersDict["0_0"])

    for b, batch in enumerate(batches):
        print(f"Batch {b + 1} of {len(batches)} for {key}...")

        outputs = generator.generate_simple(batch, settings, max_new_tokens, seed = 1234)
        trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
        for j, output in enumerate(trimmed_outputs):
            modelOutput[j+b*batch_size][key] = output
    

# now generate the rest the repeats
seed = 1234
settings.temperature = 0.0
for i, (key, layers) in tqdm(enumerate(layersDict.items()), total=len(layersDict)):
    if key in modelOutput:
        continue
    # del cache
    

    model, cache, generator = buildModel(model, layers)
    # generatedText = generate(generator, prompt, settings, max_new_tokens)
    # generatedText = generatedText[promptLength:]  # remove prompt
    # modelOutput[key] = generatedText


    for b, batch in enumerate(batches):
        print(f"Batch {b} of {len(batches)} for {key}...")

        outputs = generator.generate_simple(batch, settings, max_new_tokens, seed = 1234)
        trimmed_outputs = [o[len(p):] for p, o in zip(batch, outputs)]
        for j, output in enumerate(trimmed_outputs):
            modelOutput[j+b*batch_size][key] = output

    if i % 20 == 0:
        saveData(
            modelOutput,
            layersDict,
            prompts,
            settings,
            max_new_tokens,
            seed,
            themes,
            modelName,
        )



saveData(
    modelOutput,
    layersDict,
    prompts,
    settings,
    max_new_tokens,
    seed,
    themes,
    modelName,
)
