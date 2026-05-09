import os
import openai
import time
import json
from openai import OpenAI
client = OpenAI(api_key = "")

filepath = 'notebooks/dataset_generation/dataset_tasks.jsonl'

convo_queries = []

# Open and load the JSON file
with open(filepath, 'r', encoding='utf-8') as f:
    for line in f:
        convo_queries.append(json.loads(line))

i = 0
num_samples = 250
print("loaded convo_queries of length", len(convo_queries))

for line in convo_queries:
    priority = line["priority"]
    topic = line["topic"]
    print(f'In for loop at {topic}')
    output_dir = f'data/dataset/openai_{topic}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Made new output directory {output_dir}')
    while i < num_samples:
        print(f'In while loop at {topic}, {priority}')
        try:
            print(i, priority)
            response = client.responses.create(
                model="gpt-5-nano-2025-08-07",
                input=line["description"],
                reasoning={
                    "effort": "low"
                }
            )
            with open(f'{output_dir}/conversation_{i}_{topic}_{priority}.jsonl', 'w') as f:
                json.dump(response.output_text, f)
            f.close()
            i+= 1
        except Exception as e:
            print(f"Error: {e}") 
            time.sleep(5)
    i=0
