import warnings
warnings.filterwarnings("ignore")

import torch

from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForMaskedLM, AdamW

class NextWordDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_token, target_token = self.data[idx]
        input_ids = self.tokenizer.convert_tokens_to_ids(input_token)
        target_token_id = self.tokenizer.convert_tokens_to_ids(target_token)
        return torch.tensor(input_ids), torch.tensor(target_token_id)

class fineTune:
    def __init__(self, modelname, file_name):
        print("Fine tuning the model {}".format(modelname))
        self.modelname = modelname
        self.filename = file_name

    def download_bert(self):
        # model_name      = "bert-base-multilingual-cased"
        self.tokenizer  = BertTokenizer.from_pretrained(self.model_name)
        self.model      = BertForMaskedLM.from_pretrained(self.model_name)

        return self.model, self.tokenizer

    def createData(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            tokens = self.tokenizer.tokenize(line)
            for i in range(len(tokens)):
                data.append((tokens[i],tokens[i+1]))

        return data
    
    def optimize(self):
        return AdamW(self.model.parameters(), lr=2e-5)
        
    def loss(self):
        return torch.nn.CrossEntropyLoss()
    
    def dataLoad(self,data):
        train_dataset = NextWordDataset(data, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        return train_dataset, train_dataloader

if __name__ == "__main__":
    finetuning = fineTune("bert-base-multilingual-cased","../tuneCorpus.txt")
    # print(finetuning.modelname)