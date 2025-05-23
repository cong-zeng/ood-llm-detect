from datasets import load_dataset

raid_name_dct = {'GPT': ["gpt3", "gpt2"],
                 'ChatGPT': ["chatgpt", "gpt4"],
                 "Meta-LLaMA": ["llama-chat"],
                 "Mistral": ["mistral", "mistral-chat"],
                 "MPT": ["mpt", "mpt-chat"],
                 "Cohere": ["cohere", "cohere-chat"],
                 "human": ["human"]}

raid_model_set ={'GPT':0, 'ChatGPT':1, 'Meta-LLaMA':2,'MPT':3,'Cohere':4,\
            'Mistral':5,'human':6}

def load_raid(machine_text_only=False):
    data_new = {"train": [], "test": []}
    raid_train = load_dataset("Shengkun/Raid_split", split="train")
    raid_test = load_dataset("Shengkun/Raid_split", split="test")
    
    if machine_text_only:
        raid_train = raid_train.filter(lambda sample: sample["model"] != "human")
    
    for i in range(len(raid_train)):
        label = "1"
        if raid_train[i]["model"] != "human":
            label = "0"
        data_new["train"].append((raid_train[i]["generation"], label, raid_train[i]["model"], i))

    for i in range(len(raid_test)):
        label = "1"
        if raid_test[i]["model"] != "human":
            label = "0"
        data_new["test"].append((raid_test[i]["generation"], label, raid_test[i]["model"], i))
    return data_new

def data_process(): 
    raid = load_dataset("liamdugan/raid", split="train")
    raid = raid.train_test_split(test_size=0.1)
    raid_train = raid["train"]
    raid_test = raid["test"].train_test_split(test_size=0.2)["test"]
    raid_train = raid_train.filter(lambda input: input["attack"] == "none")
    raid_train = raid_train.train_test_split(test_size=0.2)["train"]
    
    raid_train.push_to_hub("Shengkun/Raid_split", split="train")
    raid_test.push_to_hub("Shengkun/Raid_split", split="test")

if __name__ == "__main__":
    data_process()
