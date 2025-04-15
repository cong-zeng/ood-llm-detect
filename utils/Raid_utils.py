from datasets import load_dataset

raid_name_dct = {'OpenAI-GPT': ["chatgpt", "gpt4", "gpt3", "gpt2"],
                 "Meta-LLaMA": ["llama-chat"],
                 "Mistral": ["mistral", "mistral-chat"],
                 "MPT": ["mpt", "mpt-chat"],
                 "Cohere": ["cohere", "cohere-chat"]}

raid_model_set ={'OpenAI-GPT':0,'Meta-LLaMA':1,'MPT':2,'Cohere':3,\
            'Mistral':4,'human':5}
def load_raid(train=True, machine_text_only=False):
    data = []
    if train:
        raid = load_dataset("Shengkun/Raid_split", split="train")
    else:
        raid = load_dataset("Shengkun/Raid_split", split="test")
    
    if machine_text_only:
        raid = raid.filter(lambda sample: sample["model"] != "human")
    
    for i in range(len(raid)):
        label = 1
        if raid[i]["model"] != "human":
            label = 0
        data.append((raid[i]["generation"], label, raid[i]["human"], i))
    return data
