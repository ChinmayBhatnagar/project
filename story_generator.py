from transformers import pipeline

# Load the pre-trained GPT-2 model
generator = pipeline('text-generation', model='gpt2')

# User inputs
character = input("Enter a character: ")
setting = input("Enter a setting: ")
theme = input("Enter a theme: ")

# Combine inputs to form a prompt
prompt = f"{character} in {setting} with a theme of {theme}"

# Generate the story
story = generator(prompt, max_length=150, num_return_sequences=1)

# Display the generated story
print("\nGenerated Story:")
print(story[0]['generated_text'])
