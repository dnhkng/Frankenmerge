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

firstRatingChar = "0"

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
    case "Miqu":
        folderName = "Stories/Miqu-Stories"  # 34B parameter model
        fileName = f"miqu-1-70b-sf-4.0bpw-h6-exl2-stories_{storyInt}"
    case _:
        print("Invalid generator model")
        quit()


match args.reviewer:
    case "Mistral":
        outputFolder = "Ratings/Mistral"
        ratingLLM = "Mistral"
        ratingModelDirectory = "/home/dnhkng/Documents/models/Mistral-7B-Instruct-v0.2"
    # case "Nous-Capybara":
    #     outputFolder = "Ratings/Miqu"
    #     ratingLLM = "Nous-Capybara"
    #     ratingModelDirectory = "/home/dnhkng/Documents/models/Nous-Capybara-34B"
    #     formatter = AutoTokenizer.from_pretrained(
    #         "NousResearch/Nous-Capybara-34B", trust_remote_code=True
    #     )
    #     formatter.chat_template = """{% for message in messages %}
    #         {% if message['role'] == 'user' %}
    #             {{ bos_token + 'USER: ' + message['content'] }}
    #         {% elif message['role'] == 'assistant' %}
    #             {{ 'ASSISTANT: '  + message['content'] + '</s>'}}
    #         {% endif %}
    #     {% endfor %}"""

    case "Mixtral":
        outputFolder = "Ratings/Mixtral"
        ratingLLM = "Mixtral"
        ratingModelDirectory = (
            "/home/dnhkng/Documents/models/Mixtral-8x7B-instruct-exl2"
        )
        formatter = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1"
        )
    case "Miqu":
        ratingLLM = "Miqu"
        ratingModelDirectory = (
            "/home/dnhkng/Documents/models/miqu-1-70b-sf-4.0bpw-h6-exl2"
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
    "Developing": "Basic skills present with some errors.",
    "Competent": "Adequate execution.",
    "Skilled": "Good quality with minor lapses.",
    "Proficient": "Strong, consistent quality with few errors.",
    "Artistic": "Shows flair and style beyond mere technical proficiency.",
    "Masterful": "Exceptional skill and precision.",
    "Brilliant": "Outstanding craftsmanship, innovative and flawless.",
    # "Transcendent": "Sets a new standard, impeccable in every aspect.",
}

creativityDefinition = {
    "Clone": "Offers no original thought or perspective; a mere copy of existing works.",
    "Unimaginative": "Completely derivative and lacking originality.",
    "Basic": "Few original ideas, mostly predictable.",
    "Simple": "Shows some originality but largely conventional.",
    "Interesting": "Regular flashes of creativity and is engaging.",
    "Inventive": "Consistently creative and engaging.",
    "Inspired": "Rich in original ideas and perspectives.",
    "Innovative": "Breaks new ground, very original.",
    "Visionary": "Exceptionally creative and forward-thinking.",
    "Revolutionary": "Radically original, transforming norms.",
    # "Genius": "Redefines the concept of creativity.",
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
    # "Definitive": "The ultimate expression of the theme.",
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
            f"{key} = <{chr(ord(firstRatingChar)+i)}>: {value}"
            for i, (key, value) in enumerate(ratingsDict.items())
        ]
    )


themeExample = "Imagine what alien communication might be like and create a hypothetical scenario for initial contact."

entryExample1 = """
So like, there was this super busy park and everyone was doing their thing, when suddenly this weird humming noise started. Then, boom, there's this funky-looking spaceship just chilling in the sky, all glowing with crazy colors that don't even look like they're from here.

The aliens inside were trying to say hi or something, but dude, they really didn't get how humans work or talk. They tried to send these pictures and sounds right into people's heads. They wanted to show they're cool and got some sick tech. But man, it got messed up. Everyone was grabbing their heads, seeing these trippy alien places and weird symbols, and the sounds were like a piano falling down the stairs. Some peeps were into it, but most were just like, "What the heck?" thinking it was some artsy street show or a weird ad stunt.  All aound the park there were people having picnics, playing games, and just hanging out, and tsome were riding on bikes and skateboards. 

The aliens didn't give up though. They opened this door thingy under their ship and one of them came down, all flickery and not keeping its shape. It reached out, trying to be friendly I guess. People crowded around, kinda interested but also kinda scared. The alien tried to talk, said something like, "We come... in... harmony?" but it was super awkward and kinda creepy.

Even though the aliens were really trying and some people were curious, nobody really got each other. Everyone in the park was either amazed, laughing, or just weirded out. Not much actual talking happened. The aliens couldn't really get past how different we all are, so they just bailed and went back to their ship. Everyone left in the park was just confused and started making up wild stories about what happened. It was the biggest thing everyone talked about that day, but no one really understood what it was all about.  

Do you like my story? I think it's pretty cool and different, right? I hope you like it!"""

entryExample2 = """Out in the wide-open spaces of the Mojave Desert, where the sky stretches out like a big, starry blanket, a group of star-gazers experienced something straight out of a sci-fi movie. This sleek, super cool spaceship came down from the heavens, quiet as a whisper, looking like nothing anyone had ever seen before. The folks inside this spaceship had done their homework, poring over everything humans had ever said or done, all to say "hello" in the nicest way possible.

When the spaceship finally landed, it started making these beautiful sounds and flashing lights, kind of like it was talking in math and music at the same time. The scientists were all wide-eyed at first, but then they got it. They were blown away by how smart and thoughtful these space visitors were, sending a hello that was simple on the surface but deep enough to drown in.

Then, out came these alien beings, looking all mystical and changeable but also oddly comforting, like they stepped right out of a painting or something. They showed this 3D message in the air, all about wanting to be friends, sharing what they know, and learning from us too. Their message was fancy and polite, tipping their hats to what humans have done while also offering a peek into their super-advanced world of science, art, and being nice.

This whole meetup was a game-changer, making people think and work together like never before. The way these aliens came in, so carefully and thoughtfully, it built a bridge between worlds. That first "hello" wasn't just talk; it was the start of a friendship that could change the stars."""


entryExample3 = """On the edge of the observable universe, a civilization advanced beyond the wildest dreams of humankind sent forth a vessel, a masterpiece of technology and empathy, destined for the little blue planet Earth. The beings aboard were architects of harmony, weavers of the fabric of consciousness itself, and they bore a message of cosmic significance. Their understanding of communication transcended mere language; it embraced the full spectrum of sentient expression, resonating with the fundamental frequencies of life itself.

As the vessel approached Earth, it did not disturb the sky with a sonic boom or blaze a fiery trail through the atmosphere. Instead, it folded into reality like a whisper, its arrival marked by a gentle symphony of light and color that danced across the night sky, captivating every soul that beheld it. The aliens communicated not in words, but in a cascade of shared experiences and emotions, a telepathic tapestry that connected every living mind on the planet in a moment of awe and understanding. People from every corner of the globe, regardless of their differences, found themselves enveloped in a profound, collective epiphany: a realization of their place in the vast, living cosmos.

The emissaries from the stars shared the story of their civilization, a saga of triumphs and tribulations, not through sterile data or cold facts, but as a living narrative that every human could feel and experience. They offered insights into the nature of reality, art, and love, elevating humanity's consciousness and offering a glimpse of the potential within each soul. The encounter was not just a meeting of two species; it was a fusion of destinies, a joining of paths that had wound their way through the cosmos to this single, perfect point in space and time.

In the days that followed, humanity was transformed. The arts flourished as new perspectives and inspirations took hold. Science leaped forward, propelled by shared knowledge and a newfound unity of purpose. The Earth itself began to heal, as the people of the world, now profoundly connected to each other and to the life-giving planet they shared, moved forward with a collective, harmonious vision. The visit of the alien emissaries was not just a moment in history; it was the dawn of a new era, a testament to the power of empathy, understanding, and the unbreakable bonds that stretch across the stars, binding all life in a shared dance of light and being."""


def generatePrompt(theme, entry, criteriaName, criteriaDefinition):
    prompt = (
        " [INST] \n--------------------\nAs a professional editor and connoisseur of literature, your role is pivotal in assessing the upcoming story. This narrative is a submission for a prestigious competition aimed at nurturing amateur writers, making it imperative that your evaluation is both equitable and comprehensive. Prior to assigning your score, you are encouraged to engage deeply with the content, ensuring that your critique is informed and nuanced. The central theme of the narrative will be disclosed to guide your analysis. Additionally, a structured ranking system will be made available to you, designed to streamline the evaluation process and maintain consistency across various dimensions of literary excellence. Your expert judgement is not just a contribution, but a cornerstone in the advancement and recognition of emerging literary talent. \n\n***** Given Theme *****\n"
        + theme
        + "\n\n***** Competition Entry *****\n"
        + entry
        + "\n\n**** Rating System *****\nUse the following rating system "
        + criteria[criteriaName]
        + "\n"
        + dictToString(criteriaDefinition)
        + "\n\nUpon reviewing the narrative in accordance with the provided rating systems, please concentrate your evaluation into a singular, precise judgement for the entry. Respond with only one character that aligns with your considered evaluation, ranging from the lowest to the highest tier as defined.\n"
        + "\n***** Rating *****\n"
        + criteriaName
        + " Rating: = [/INST] "
    )
    return prompt


def generateAllPrompt(theme, entry):
    return {
        "craftsmanship": 
        "<s>"
        + generatePrompt(
            themeExample, entryExample1, "craftsmanship", craftsmanshipDefinition
        )
        + "2</s>\n"
        + generatePrompt(
            themeExample, entryExample2, "craftsmanship", craftsmanshipDefinition
        )
        + "5</s>\n"
        + generatePrompt(
            themeExample, entryExample3, "craftsmanship", craftsmanshipDefinition
        )
        + "7</s>\n"
        + generatePrompt(theme, entry, "craftsmanship", craftsmanshipDefinition),

        "creativity": generatePrompt(
            themeExample, entryExample1, "creativity", creativityDefinition
        )
        + "3</s>\n"
        + generatePrompt(
            themeExample, entryExample2, "creativity", creativityDefinition
        )
        + "5</s>\n"
        + generatePrompt(
            themeExample, entryExample3, "creativity", creativityDefinition
        )
        + "7</s>\n"
        + generatePrompt(theme, entry, "creativity", creativityDefinition),
        "consistency": generatePrompt(
            themeExample, entryExample1, "consistency", consistencyDefinition
        )
        + "3</s>\n"
        + generatePrompt(
            themeExample, entryExample2, "consistency", consistencyDefinition
        )
        + "5</s>\n"
        + generatePrompt(
            themeExample, entryExample3, "consistency", consistencyDefinition
        )
        + "7</s>\n"
        + generatePrompt(theme, entry, "consistency", consistencyDefinition),
    }


config = ExLlamaV2Config()
config.model_dir = ratingModelDirectory
config.prepare()
config.max_seq_len = 8162


model = ExLlamaV2(config)
tokenizer = ExLlamaV2Tokenizer(config)


cache = ExLlamaV2Cache(
    model, lazy=True, max_seq_len=8162
)  # Cache needs to accommodate the batch size
model.load_autosplit(cache)


max_new_tokens = 1


# Dictionary for ranking the letters, i.e. a=0, b=1, c=2, etc.
ranking = {}
for i in range(len(creativityDefinition)):
    ranking[chr(i + ord(firstRatingChar))] = int(
        tokenizer.encode(chr(i + ord(firstRatingChar)))[0][-1]
    )


print(ranking)

generatedTexts = pickle.load(open(f"{folderName}/{fileName}.p", "rb"))
stories = generatedTexts["stories"]
theme = generatedTexts["theme"]


ratings = {}
for key, entry in tqdm.tqdm(stories.items()):
    prompts = generateAllPrompt(theme, entry)


    ratingValue = {}
    for crit, prompt in prompts.items():
        # print(prompt)
        # print("####################################################")
        
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

        token = tokenizer.decode_unspecial(int(logits.argmax()))
        print(probabilities)
        print(f"token: {token}")
        print(f"current_seq_len: {cache.current_seq_len}")
        ratingValue[crit] = probabilities.numpy()
    ratings[key] = ratingValue



print(ratings)

saveFile = f"{outputFolder}/{fileName}_Ratings_{ratingLLM}_instruct2.p"
pickle.dump(
    ratings, open(saveFile, "wb")
)
print("Saved ratings to " + saveFile)
