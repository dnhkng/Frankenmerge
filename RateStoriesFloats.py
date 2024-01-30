import argparse
import pickle

import torch
import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from transformers import AutoTokenizer

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
    case "Mistral":
        ratingLLM = "Mistral"
        ratingModelDirectory = "/home/dnhkng/Documents/models/Mistral-7B-Instruct-v0.2"
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
            "/home/dnhkng/Documents/models/Mixtral-8x7B-instruct-exl2"
        )
        formatter = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
    case _:
        print("Invalid rating model")
        quit()


# the ratings are on a scale of 0-10, with 0 being the worst and 10 being the best, although the this can be continued and is not limited to 10. What is important is the descriprion of the rating!
craftsmanshipDefinition = {
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

creativityDefinition = {
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

consistencyDefinition = {
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

criteria = {
    "craftsmanship": "Craftsmanship - focuses on the writer's skill in structuring sentences, paragraphs, and stylistic precision.\n",
    "creativity": "Creativity - encompasses the writer's flair for innovation, the use of vivid and original imagery, and the ability to engage readers with fresh perspectives and unexpected narrative turns.\n",
    "consistency": "Consistency - indicates the writer's skill in maintaining relevance to the theme, ensuring that all parts of the writing contribute to and resonate with the central idea, without deviating or diluting the thematic focus.\n",
}


def dictToString(ratingsDict: dict) -> str:
    """Convert a dictionary of ratings to a string.

    Each rating is a line in the string, with the key, value, and letter of the rating the LLM should use.
    This is done by converting the key to a letter, starting with 'a' and incrementing by one for each rating.

    LLM's are very bad at understanding numbers, so we use a descriptive word for each rating instead, and have the LLM
    convert it to a letter.

    Args:
        ratingsDict (dict): Dictionary of ratings"""

    return "\n".join(
        [
            f"{key} = {chr(ord('a')+i)}: {value}"
            for i, (key, value) in enumerate(ratingsDict.items())
        ]
    )


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

entry = """In the heart of the Old World, where the sun sets in a blaze of crimson and gold, lies the bustling seaport of Port Royal. Its cobblestone streets echo with the cacophony of merchants hawking their wares, sailors singing shanties, and children laughing. But beneath this veneer of merriment, lurks an inescapable truth: this is a town built on the blood of the damned. I am its grim guardian.

I, the Hooded Reaper, have borne witness to countless lives claimed by the merciless sea and the even more merciless men who ply her waters. Today, I stand at the precipice of another tale of infamy, as the life of a notorious pirate comes to an end.

The sun had barely risen when the shackled figure was led before me. His name was Blackbeard, the terror of the Seven Seas. His legend had grown like a cancer, spreading fear and awe in equal measure. He stood tall and defiant, his eyes burning with the fire of rebellion. But as he looked upon me, he knew his time had come.

As the crowd gathered, I could feel the weight of their anticipation. They came to see justice served, to witness the spectacle of a pirate's end. I, too, had grown weary of Blackbeard's reign of terror. Yet, as I prepared to execute him, I couldn't help but feel a pang of sadness. For beneath the fear and the violence, there was a man - a man who had once been a part of this very community.

Blackbeard's hands were bound, his beard hidden beneath a thick hood. He looked every inch the pirate king, his eyes filled with a mixture of defiance and resignation. As I approached, he spoke, his voice barely above a whisper.

"Reaper," he said, "I know what you are. I've seen the likes of you before. But I've lived a good life, taken what I wanted, and given as good as I got. I've earned my place in the afterlife."

I remained silent, my face hidden behind the mask of my hood. I had heard such words before, from men and women who thought they had lived lives worth living. But the law was the law, and there was no room for mercy in its cold, unyielding grasp.

As the noose was placed around his neck, Blackbeard's demeanor changed. He closed his eyes, took a deep breath, and spoke one final words.

"Farewell, Reaper. May the sea be kind to you."

With that, he jumped from the makeshift gallows, the noose tightening around his neck. The crowd gasped in shock, but I knew what was coming. I watched as the life drained from his eyes, his body twitching and convulsing in its final moments. And then, silence.

As the sun set over Port Royal, I stood there, the Hooded Reaper, watching as the tide carried Blackbeard's lifeless body away. Another pirate's tale had come to an end, another chapter in the endless saga of the sea written in the blood of the damned. But as I turned to leave, I couldn't help but wonder: would there ever be an end to this cycle of violence and retribution? Or would the sea forever be stained with the blood of those who dared to defy the law?

And so, I continue my vigil, the Hooded Reaper, the grim guardian of Port Royal, waiting for the next tale of infamy to unfold. For the sea is a cruel mistress, and her children are a restless, violent lot. But I will be there, ready to mete out justice, no matter the cost."""

# entry = "the kitty cat done sat on mat!"


def generatePrompt(theme, entry):
    return {
        "craftsmanship": "Rate the folling story, which should following the given theme.\n***** Given Theme *****\n"
        + themeExample
        + "\n***** Competition Entry *****\n"
        + entryExample
        + "\n**** Rating *****\nUse the following rating system:\n"
        + criteria["craftsmanship"]
        + dictToString(craftsmanshipDefinition)
        + "\n\nReturn a single character at the rating:\n'f'\n\nRate the folling story:\n****\n"
        + entry
        + "\n****\nUse the following rating system:\n"
        + criteria["craftsmanship"]
        + dictToString(craftsmanshipDefinition)
        + "\n\nReturn a single character at the rating:\n'",
        "creativity": "Rate the folling story, which should following the given theme.\n***** Given Theme *****\n"
        + themeExample
        + "\n***** Competition Entry *****\n"
        + entryExample
        + "\n**** Rating *****\nUse the following rating system:\n"
        + criteria["creativity"]
        + dictToString(creativityDefinition)
        + "\n\nReturn a single character at the rating:\n'f'\n\nRate the folling story:\n****\n"
        + entry
        + "\n****\nUse the following rating system:\n"
        + criteria["creativity"]
        + dictToString(creativityDefinition)
        + "\n\nReturn a single character at the rating:\n'",
        "consistency": "Rate the folling story, which should following the given theme.\n***** Given Theme *****\n"
        + themeExample
        + "\n***** Competition Entry *****\n"
        + entryExample
        + "\n**** Rating *****\nUse the following rating system:\n"
        + criteria["consistency"]
        + dictToString(consistencyDefinition)
        + "\n\nReturn a single character at the rating:\n'f'\n\nRate the folling story:\n****\n"
        + entry
        + "\n****\nUse the following rating system:\n"
        + criteria["consistency"]
        + dictToString(consistencyDefinition)
        + "\n\nReturn a single character at the rating:\n'",
    }


config = ExLlamaV2Config()
config.model_dir = ratingModelDirectory
config.prepare()

config.max_seq_len = 4096

model = ExLlamaV2(config)
print("Loading model: " + ratingModelDirectory)

# model.load()

tokenizer = ExLlamaV2Tokenizer(config)


cache = ExLlamaV2Cache(
    model, lazy=True, max_seq_len=16384
)  # Cache needs to accommodate the batch size
model.load_autosplit(cache)


max_new_tokens = 1

ranking = {}

for i in range(len(creativityDefinition)):
    ranking[chr(i + ord("a"))] = int(tokenizer.encode(chr(i + ord("a"))))

generatedTexts = pickle.load(open(f"{folderName}/{fileName}.p", "rb"))
stories = generatedTexts["modelOutput"]
theme = generatedTexts["theme"]


ratings = {}
for key, entry in tqdm.tqdm(stories.items()):
    prompts = generatePrompt(theme, entry)
    # print(prompts['creativity'])
    ratingValue = {}
    for crit, prompt in prompts.items():
        cache.current_seq_len = 0
        ids = tokenizer.encode(prompt)
        model.forward(ids[:, :-1], cache, preprocess_only=True)
        input = ids[:, -1:]
        logits = model.forward(input, cache, input_mask=None).float().cpu()

        logitsResults = []
        for i, (k, v) in enumerate(ranking.items()):
            logitsResults.append(float(logits[:, :, v]))
        logitsResults = torch.tensor(logitsResults)
        probabilities = torch.nn.functional.softmax(logitsResults, dim=0)
        scores = torch.tensor(range(len(probabilities)), dtype=torch.float)
        print(key, crit, torch.mean(scores * probabilities) * 10)
        ratingValue[crit] = float(torch.mean(scores * probabilities) * 10)
    ratings[key] = ratingValue

print(ratings)


pickle.dump(ratings, open(f"{fileName}_Ratings_{ratingLLM}_v2.p", "wb"))
print("Saved ratings to " + f"{fileName}_Ratings_{ratingLLM}_v2.p")
