
import copy
import os
import pickle
import sys

import torch
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import argparse

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

# Create the parser
parser = argparse.ArgumentParser(description='Generate stories from a model.')

# Add arguments
parser.add_argument('theme', type=str, help='The theme of the story')
parser.add_argument('myint', type=int, help='An integer for saving the file')
parser.add_argument('model', type=str, help='Model used to generate stories')

# Parse the arguments
args = parser.parse_args()

# Print or process the arguments
print(f"String argument: {args.theme}")
print(f"Integer argument: {args.myint}")
print(f"Model argument: {args.model}")

theme = args.theme
number = args.myint
model = args.model



match args.model:
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

# Initialize model and cache
reuseCache = False
seed = 1234

model_directory =  f"/home/dnhkng/Documents/models/{modelName}"


config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
config.max_seq_len=4096

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

cache = ExLlamaV2Cache(model, lazy = True)
model.load_autosplit(cache)

# Initialize tokenizer
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize generator
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Print model layers
def printModules(modules):
    for i, m in enumerate(modules):
        if hasattr(m, 'layer_idx'):
            print(i, m.key, m.layer_idx)
        else:
            print(i, m.key)
    print()


class ExLlamaV2AttentionWrapperNoCache(ExLlamaV2Attention):
    def __init__(self, obj, new_idx):
        object.__setattr__(self, '_obj', obj)
        object.__setattr__(self, '_new_idx', new_idx)

    def __getattribute__(self, name):
        if name == 'layer_idx':
            return object.__getattribute__(self, '_new_idx')

        # Delegate all other attributes to the wrapped object
        try:
            return getattr(object.__getattribute__(self, '_obj'), name)
        except AttributeError:
            return object.__getattribute__(self, name)


## Store base modules
baseModules = model.modules
cache_class = type(cache)

def buildModel(model, layers, cache_class):
    model.modules = baseModules[:1]
    print('building model')
    for i, idx in enumerate(layers):
        if reuseCache:
            nextModule = ExLlamaV2AttentionWrapperNoCache(baseModules[idx*2 + 1], i)
        else:
            nextModule = copy.copy(baseModules[idx*2 + 1])
            nextModule.layer_idx = i
        model.modules.append(nextModule)
        nextModule = copy.copy(baseModules[idx*2 + 2])
        nextModule.layer_idx = i
        model.modules.append(nextModule)

    model.modules += baseModules[-2:]
    if 'cache' in globals():
        del globals()['cache']
    torch.cuda.empty_cache()

    num_layers = int((len(model.modules) - 3) / 2)
    model.head_layer_idx = len(model.modules) -1
    model.config.num_hidden_layers = num_layers
    model.last_kv_layer_idx = len(model.modules) -4
    model.cache_map = {}
    model.set_cache_map()
    cache = cache_class(model)
    generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

    return model, cache, generator


# Generate some text
max_new_tokens = 1024
settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.0
settings.top_k = 50
settings.top_p = 0.8
settings.top_a = 0.0
settings.token_repetition_penalty = 1.05

# theme = "A Whisper from the Stars: In a world where people receive guidance from the stars, one starless night leads to unexpected revelations."
# number = 5
# theme = "Write about the execution of a famous pirate. Write from the perspective of the hooded executioner."
# theme = "The Painter Who Used the Wind: An artist who can capture the essence of the wind in their paintings, affecting the weather and emotions of the town."

# TinyLLama
# def tinyLlamaPrompt(theme):
#     return f"""<|system|>
#     You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win!</s>
#     <|user|>
#     {theme}</s>
#     <|assistant|>"""

def tinyLlamaPrompt(theme):
    return f"""<|im_start|>system
    You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win!<|im_end|>
    <|im_start|>user
    The theme is: "{theme}"<|im_end|>
    <|im_start|>assistant"""


# Mistral





def generate(self, prompt, settings, max_new_tokens, seed = seed):
    generator.warmup()
    output = generator.generate_simple(prompt, settings, max_new_tokens, seed = seed)
    return output


def generateLayerDict(size):
    layersDict = {}
    layersDict['0_0'] = list(range(size)) #  0_0 is the base model
    for j in range(size):
        for i in range(j):
            layersList = list(range(0,j)) + list(range(i,size))
            layersDict[f"{i}_{j}"] = layersList
    return layersDict

def saveData(modelOutput, layersDict, prompt, settings, max_new_tokens, seed, reuseCache, theme, modelName, number):
    experiment = {}
    experiment['modelOutput'] = modelOutput
    experiment['layersDict'] = layersDict
    experiment['prompt'] = prompt
    experiment['config'] = vars(settings)
    experiment['config']['max_new_tokens'] = max_new_tokens
    experiment['config']['seed'] = seed
    experiment['theme'] = theme
    experiment['modelName'] = modelName
    
    if reuseCache:
        pickle.dump( experiment, open( f"{modelName}-stories-reuseCache_{number}.p", "wb" ) )
    else:
        pickle.dump( experiment, open( f"{modelName}-stories_{number}.p", "wb" ) )
                    

if modelName == "Nous-Capybara-34B-4.0bpw":
    from transformers import AutoTokenizer

    ratingModelDirectory =  "/home/dnhkng/Documents/models/Nous-Capybara-34B"
    formatter = AutoTokenizer.from_pretrained("NousResearch/Nous-Capybara-34B", trust_remote_code=True)
    formatter.chat_template = """{% for message in messages %}
        {% if message['role'] == 'user' %}
            {{ bos_token + 'USER: ' + message['content'] }}
        {% elif message['role'] == 'assistant' %}
            {{ 'ASSISTANT: '  + message['content'] + '</s>'}}
        {% endif %}
    {% endfor %}"""

    systemPrompt = f"""You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win! The theme for the contest is: {theme}"""

    chat = [
        {"role": "user", "content": systemPrompt },
    ]
    prompt = formatter.apply_chat_template(chat, tokenize=False) + 'ASSISTANT:'

# else:
#     prompt = f"""<s> [INST] You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win! The theme for the contest is: "{theme}" [/INST]"""
prompt = f"""<s> [INST] You are a profession author who writes excellent creative prose. You are writing for a competition, and are aiming to win! The theme for the contest is: "{theme}" [/INST]"""

# prompt = tinyLlamaPrompt(theme)
promptLength = len(prompt) 
print(prompt)

# quit()
num_layers = int((len(baseModules) - 3) / 2) # the is an embedder and 2 final layers (3), with 2 layers per attention block
layersDict = generateLayerDict(num_layers)
modelOutput = {}

# CONTINUE
# previous = pickle.load( open( f"{modelName}-stories_{number}.p", "rb" ) )
# modelOutput = previous['modelOutput']

# generated many samples, so we can compare them
settings.temperature = 0.8
for i in range(1,10):
    key = f"{i}_0"
    model, cache, generator = buildModel(model, layersDict['0_0'], cache_class)
    generatedText = generate(generator, prompt, settings, max_new_tokens, seed = i)
    generatedText = generatedText[promptLength:] # remove prompt
    modelOutput[key] = generatedText
    print(f"{key=}")
    print(generatedText)
    print()


# now generate the rest the repeats
seed = 1234
settings.temperature = 0.0
for i, (key, layers) in tqdm(enumerate(layersDict.items()), total=len(layersDict)):

    if key in modelOutput:
        continue

    model, cache, generator = buildModel(model, layers, cache_class)
    generatedText = generate(generator, prompt, settings, max_new_tokens)
    generatedText = generatedText[promptLength:] # remove prompt
    modelOutput[key] = generatedText
    print(f"{key=}")
    print(generatedText)
    print()
    if i % 20 == 0:
        saveData(modelOutput, layersDict, prompt, settings, max_new_tokens, seed, reuseCache, theme, modelName, number)



saveData(modelOutput, layersDict, prompt, settings, max_new_tokens, seed, reuseCache, theme, modelName, number)

