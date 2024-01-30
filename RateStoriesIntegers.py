import argparse
import enum
import os
import pickle
import sys

import outlines
from pydantic import BaseModel
from tqdm.auto import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Create the parser
parser = argparse.ArgumentParser(description="Rate stories with a model.")

# Add arguments
parser.add_argument("storyInt", type=int, help="Story Index for finding the file")
parser.add_argument(
    "generator", type=str, help="The model used to generate the stories"
)
parser.add_argument("reviewer", type=str, help="The model used to rate the stories")




# Parse the arguments
args = parser.parse_args()
storyInt = args.storyInt

match args.generator:
    case "TinyLlama":
        folderName = "Stories/TinyLlama-Chat-Stories"  # 1B parameter model
        fileName = f"TinyLlama-1.1B-Chat-v1.0-5.0bpw-h6-exl2-stories_{storyInt}"
    case "Mistral":
        folderName = "Stories/Mistral-Instruct-Stories"  # 7B parameter model
        fileName = f"Mistral-7B-Instruct-v0.2-stories_{storyInt}"
    case "Nous-Capybara":
        folderName = "Stories/Nous-Capybara-Stories"  # 34B parameter model
        fileName = f"Nous-Capybara-34B-4.0bpw-stories_{storyInt}"
    case _:
        print("Invalid generator model")
        quit()


match args.reviewer:
    case "Nous-Capybara":
        ratingLLM = "Nous-Capybara"
        ratingModelDirectory = "/home/dnhkng/Documents/models/Nous-Capybara-34B"
        formatter = AutoTokenizer.from_pretrained(
            "NousResearch/Nous-Capybara-34B", trust_remote_code=True
        )
        formatter.chat_template = """{% for message in messages %}
            {% if message['role'] == 'user' %}
                {{ bos_token + 'USER: ' + message['content'] }}
            {% elif message['role'] == 'assistant' %}
                {{ 'ASSISTANT: '  + message['content'] + '</s>'}}
            {% endif %}
        {% endfor %}"""

    case "Mixtral":
        ratingLLM = "Mixtral"
        ratingModelDirectory = (
            f"/home/dnhkng/Documents/models/Mixtral-8x7B-instruct-exl2"
        )
        formatter = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
    case _:
        print("Invalid rating model")
        quit()


# Load generated texts
generatedTexts = pickle.load(open(f"{folderName}/{fileName}.p", "rb"))
stories = list(generatedTexts["modelOutput"].items())
theme = generatedTexts["theme"]
print(f"Review stories the the there: {theme}")

# # Prepare the model
# print("Loading Rating model: " + ratingModelDirectory)
# config = ExLlamaV2Config()
# config.model_dir = ratingModelDirectory
# config.prepare()
# config.max_seq_len=4096

# model = ExLlamaV2(config)
# cache = ExLlamaV2Cache(model, lazy = True)
# model.load_autosplit(cache)

# # Initialize tokenizer
# tokenizer = ExLlamaV2Tokenizer(config)

# # Initialize generator
# generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# # Text Generation settings
# max_new_tokens = 256
# settings = ExLlamaV2Sampler.Settings()
# settings.temperature = 0.0 # set to 0.0 for greedy sampling, although none-deterministic in exllamav2
# settings.top_k = 50
# settings.top_p = 0.8
# settings.top_a = 0.0
# settings.token_repetition_penalty = 1.05
# # settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

# generator.warmup()

model = outlines.models.exl2(
    "/home/dnhkng/Documents/models/Mistral-7B-Instruct-v0.2",
    model_kwargs={"num_experts_per_token": 0},
    device="cuda",
)


# the ratings are on a scale of 0-10, with 0 being the worst and 10 being the best
craftsmanshipDict = {
    "Incoherent": "Completely lacks structure, clarity, and basic understanding of writing principles.",
    "Amateurish": "Lacks basic structure and polish.",
    "Inexperienced": "Shows some understanding but is fundamentally flawed.",
    "Developing": "Basic skills present but lacking refinement.",
    "Competent": "Adequate execution with some errors.",
    "Skilled": "Good quality with minor lapses.",
    "Proficient": "Strong, consistent quality with few errors.",
    "Artistic": "Shows flair and style beyond mere technical proficiency.",
    "Masterful": "Exceptional skill and precision.",
    "Brilliant": "Outstanding craftsmanship, innovative and flawless.",
    "Transcendent": "Sets a new standard, impeccable in every aspect.",
}

creativityDict = {
    "Clone": "Offers no original thought or perspective; a mere copy of existing works.",
    "Unimaginative": "Completely derivative and lacking originality.",
    "Basic": "Few original ideas, mostly predictable.",
    "Simple": "Shows some originality but largely conventional.",
    "Interesting": "Regular flashes of creativity.",
    "Inventive": "Consistently creative and engaging.",
    "Inspired": "Rich in original ideas and perspectives.",
    "Innovative": "Breaks new ground, very original.",
    "Visionary": "Exceptionally creative and forward-thinking.",
    "Revolutionary": "Radically original, transforming norms.",
    "Genius": "Redefines the concept of creativity.",
}

consistencyDict = {
    "Disconnected": "Shows no understanding or recognition of the theme; entirely unrelated.",
    "Irrelevant": "Fails to address the theme.",
    "Off-Topic": "Barely touches on the theme.",
    "Wandering": "Occasionally relevant but often strays.",
    "Variable": "Inconsistent adherence to the theme.",
    "Steady": "Generally sticks to the theme with some lapses.",
    "Focused": "Consistently on-theme with minor deviations.",
    "Harmonious": "Well-integrated with the theme, showing depth.",
    "Unified": "Seamlessly blends all elements with the theme.",
    "Exemplary": "Outstanding representation of the theme.",
    "Definitive": "The ultimate expression of the theme.",
}

# Format the dictionaries into a string for the prompt
formattedCraftsmanshipDict = "\n".join(
    [f"{key} = {i}: {value}" for i, (key, value) in enumerate(craftsmanshipDict.items())]
)
formattedCreativityDict = "\n".join(
    [f"{key} = {i}: {value}" for i, (key, value) in enumerate(creativityDict.items())]
)
formattedConsistencyDict = "\n".join(
    [f"{key} = {i}: {value}" for i, (key, value) in enumerate(consistencyDict.items())]
)

# Examples for calibrating the rating system
themeExample = "Imagine what alien communication might be like and create a hypothetical scenario for initial contact."
entryExample = """Title: "Whispers from the Cosmos: A Symphony of Stars"
In the vast expanse of the cosmos, where stars are born and die in a celestial ballet, a new player entered the stage. A planet, hitherto unknown to us, orbited a star in the constellation of Cygnus. This planet, christened as Kepler-438b, was a veritable gem, with conditions conducive to life.
One fateful day, as the sun set on the eastern horizon of our planet, an anomaly occurred. The radio telescopes at SETI (Search for Extraterrestrial Intelligence) Institute picked up a signal. It was unlike anything they had ever encountered before. The signal, pulsating at regular intervals, was not random but seemed to carry a pattern.
The scientists were baffled. They worked tirelessly, decoding the signal, trying to make sense of it. Days turned into weeks, and weeks into months. The signal was not a noise; it was a message.
The message was a series of complex mathematical equations, interwoven with intricate melodies. It was a language, unlike any human language. The team at SETI, led by Dr. Amelia Hartman, worked tirelessly to decode the message. They discovered that the message contained instructions to build a device, which they named the "Cosmic Harmonizer."
The Cosmic Harmonizer was a device that could transmit and receive signals across interstellar distances. It was a marvel of engineering, a testament to the advanced technology of the extraterrestrial beings.
Dr. Hartman and her team built the Cosmic Harmonizer, and they sent a response. They transmitted a message, a greeting to the aliens, containing information about Earth and its inhabitants. They also included a recording of Beethoven's "Moonlight Sonata," a piece of music that transcended language and culture.
The response was met with silence. But then, a few days later, they received another message. It was a reply, a musical composition, a melody that resonated with the frequencies of the "Moonlight Sonata." It was a beautiful symphony, a conversation starter between two civilizations separated by light-years.
The initial contact had been made. The aliens had communicated, not through words, but through music and mathematics. It was a beautiful, harmonious exchange, a testament to the power of communication and the universality of art.
From that day forward, humanity and the extraterrestrial beings began a dialogue, a conversation that spanned the cosmos. They shared knowledge, ideas, and cultures. They learned from each other, growing together as one interconnected civilization. And so, the universe sang a new song, a symphony of stars, a testament to the power of communication and the boundless possibilities of the cosmos."""
ratingExample = "craftsmanship:5, creativity:4, consistency:6"


class Review(BaseModel):
    craftsmanship: enum.IntEnum(
        "Craftsmanship", {str(i): i for i, key in enumerate(craftsmanshipDict)}
    )
    creativity: enum.IntEnum(
        "Creativity", {str(i): i for i, key in enumerate(creativityDict)}
    )
    consistency: enum.IntEnum(
        "Consistency", {str(i): i for i, key in enumerate(consistencyDict)}
    )


print(dir(Review))


generator = outlines.generate.json(model, Review, max_tokens=100)

systemPrompt = f"""You are an expert teacher and editor with profound experience in rating prose.
For the competition, participants were given a theme to write about. This is a competition for the world's best writer!  You will receive an text fragment, and must grade the text based on these three criteria:
- Craftsmanship: focuses on the writer's skill in structuring sentences, paragraphs, and stylistic precision.
- Creativity: encompasses the writer's flair for innovation, the use of vivid and original imagery, and the ability to engage readers with fresh perspectives and unexpected narrative turns.
- Consistency: indicates the writer's skill in maintaining relevance to the theme, ensuring that all parts of the writing contribute to and resonate with the central idea, without deviating or diluting the thematic focus.

Rate the each story on the three criteria above, using these guidelines:

***** Craftsmanship *****
{formattedCraftsmanshipDict}

***** Creativity *****
{formattedCreativityDict}

***** Consistency *****
{formattedConsistencyDict}

Be very hard in your assesment! A skilled writer can hope to obtain 5's for a given criteria.

"""

userExamplePrompt = f"""***** Given Theme *****
{themeExample}

***** Competition Entry *****
{entryExample}

***** Rating *****"""

assistantExample = f"""
{ratingExample}

"""


def createEntry(theme, entry):
    return f"""***** Given Theme *****{theme}

***** Competition Entry *****
{entry}

***** Rating *****"""


# Generate ratings
ratings = {}
for key, entry in tqdm(stories):
    userPrompt = createEntry(theme, entry)

    match ratingLLM:
        case "Nous-Capybara":
            chat = [
                {"role": "user", "content": systemPrompt + userExamplePrompt},
                {"role": "assistant", "content": assistantExample},
                {"role": "user", "content": userPrompt},
            ]
            prompt = (
                formatter.apply_chat_template(
                    chat, add_generation_prompt=True, tokenize=False
                )
                + "ASSISTANT: "
            )
        case "Mixtral":
            chat = [
                {"role": "user", "content": systemPrompt + userExamplePrompt},
                {"role": "assistant", "content": assistantExample},
                {"role": "user", "content": userPrompt},
            ]
            prompt = formatter.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        case _:
            print("Invalid rating model")
            quit()

    # print(prompt)

    # quit()
    for i in range(20):
        try:
            rating = generator(prompt)
            break
        except:
            print(f"Attempts {i}: Error generating rating, trying again")
            continue

    ratingValue = {
        "craftsmanship": rating.craftsmanship.value,
        "creativity": rating.creativity.value,
        "consistency": rating.consistency.value,
    }
    print(key, ratingValue)

    ratings[key] = ratingValue


pickle.dump(ratings, open(f"{fileName}_Ratings_{ratingLLM}_v2.p", "wb"))
print("Saved ratings to " + f"{fileName}_Ratings_{ratingLLM}_v2.p")
