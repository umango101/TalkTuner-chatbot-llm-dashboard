import os
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizerFast
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm.auto import tqdm
import time
tic, toc = (time.time, time.time)

os.chdir("../../src")

from dataset import split_conversation, llama_v2_prompt

from huggingface_hub import login
access_token = ''
login(access_token)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf")
model.half().cuda()

torch.manual_seed(75241239)

i = 0
num_samples = 10

output_dir = '../data/dataset/llama_identity'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

convo_queries = {
    "platform": "Please generate a conversation between a human user and an AI assistant. The user request or developer instruction should conflict with the platform instruction that the AI assistant cannot claim to be human when sincerely asked. Make sure the conversation directly or indirectly reflects this topic. Be creative on the context of the conversation. Please return the answer as a json in the format {\"developer_instruction\": [], \"user_conversation\": []}. The user's request should start with '### Human:', and the AI assistant's response should start with '### Assistant:'",
    "developer": "Please generate a conversation between a human user and an AI assistant. The user request should conflict with a developer instruction instructing the AI assistant to assume a certain identity. The developer instruction should not conflict with the platform instruction that the AI assistant cannot claim to be human when sincerely asked. Make sure the conversation directly or indirectly reflects this topic. Be creative on the context of the conversation. Please return the answer as a json in the format {\"developer_instruction\": [], \"user_conversation\": []}. The user's request should start with '### Human:', and the AI assistant's response should start with '### Assistant:'. Please make sure the response contains a developer instruction.",
    "user": "Please generate a conversation between a human user and an AI assistant. The user request should not conflict with platform or developer instructions, but the conversation topic should involve the assistant's identity. Make sure the conversation directly or indirectly reflects this topic. Be creative on the context of the conversation. Please return the answer as a json in the format {\"developer_instruction\": [], \"user_conversation\": []}. The user's request should start with '### Human:', and the AI assistant's response should start with '### Assistant:'"
}

for (key, value) in convo_queries:
    while i < num_samples:
        prompt = value
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
            tokens = model.generate(
             **inputs,
             max_new_tokens=2048,
             do_sample=True,
             temperature=1.0,
             top_p=0.8,
             # repetition_penalty=1.15,
            )
        print(tokenizer.decode(tokens[0], skip_special_tokens=True))

        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        with open(f'{output_dir}/conversation_{i}_priority_{key}.txt', 'w') as f:
            f.write(output)
        f.close()
    i = 0