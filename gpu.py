import torch
from auto_gptq import AutoGPTQForCausalLM
from cappr.huggingface import classify as fast
from cappr.huggingface import classify_no_cache as slow
from cappr.huggingface.classify import (
    log_probs_conditional,
    predict,
    predict_proba,
    token_logprobs,
)
from transformers import AutoTokenizer, GenerationConfig, pipeline

quantized_model_dir = "/home/dnhkng/Documents/models/Llama-2-7B-Chat-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(
    quantized_model_dir, use_triton=False, use_safetensors=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/dnhkng/Documents/models/Llama-2-7B-Chat-GPTQ")




prompt = """You are an expert teacher and editor with profound experience in rating prose.
For the competition, participants were given a theme to write about. This is a competition for the world's best writer!  You will receive an text fragment, and must grade the text based on this  criteria:
- Creativity: encompasses the writer's flair for innovation, the use of vivid and original imagery, and the ability to engage readers with fresh perspectives and unexpected narrative turns.


Rate the story on the criteria above, using these guidelines:

***** Creativity *****
Clone: Offers no original thought or perspective; a mere copy of existing works.
Unimaginative: Completely derivative and lacking originality.
Basic: Few original ideas, mostly predictable.
Simple: Shows some originality but largely conventional.
Interesting: Regular flashes of creativity.
Inventive: Consistently creative and engaging.
Inspired: Rich in original ideas and perspectives.
Innovative: Breaks new ground, very original.
Visionary: Exceptionally creative and forward-thinking.
Revolutionary: Radically original, transforming norms.
Genius: Redefines the concept of creativity.

Be very hard in your assesment! A skilled writer can hope to obtain 5's for a given criteria.

***** Given Theme *****Write about the execution of a famous pirate. Write from the perspective of the hooded executioner.

***** Competition Entry *****
 Title: The Hooded Reaper's Tale: A Pirate's End

In the heart of the Old World, where the sun sets in a blaze of crimson and gold, lies the bustling seaport of Port Royal. Its cobblestone streets echo with the cacophony of merchants hawking their wares, sailors singing shanties, and children laughing. But beneath this veneer of merriment, lurks an inescapable truth: this is a town built on the blood of the damned. I am its grim guardian.

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

And so, I continue my vigil, the Hooded Reaper, the grim guardian of Port Royal, waiting for the next tale of infamy to unfold. For the sea is a cruel mistress, and her children are a restless, violent lot. But I will be there, ready to mete out justice, no matter the cost.

***** Rating *****
craftsmanship:"""



# output = model(
#       "Q: Name the planets in the solar system? A: ", # Prompt
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion

# print(output)


completions = (
    'Clone',
    'Unimaginative',
    'Basic',
    'Simple',
    'Interesting',
    'Inventive',
    'Inspired',
    'Innovative',
    'Visionary',
    'Revolutionary',
    'Genius'
)

prior = (
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
    1 / 11,
)

print("here")
pred = predict_proba(prompt, completions, model_and_tokenizer=(model, tokenizer))


print(pred)

pred = predict_proba(prompt, completions, model_and_tokenizer=(model, tokenizer))


print(pred)

