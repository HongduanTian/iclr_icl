import os
import sys

from torch.utils.data import DataLoader, Dataset
from typing import List, Dict


class JsonDataset(Dataset):
    
    def __init__(self, data, tokenizer) -> None:
        """
        Args:
            data (List): A list of prompt;
        """
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        conversation = self.tokenizer.apply_chat_template(
            #[{"role": "system", "content": "You are a professional model for machine learning classification tasks. The user will provide you with a set of labeled data and several query data. Your tasks include two aspects: (1) predicting labels for the query data based on the given labeled data; (2) evaluating the confidence of your answer. Your responses must follow the following format. You must predict the labels of the query data with the format 'Label: <label>', where <label> denotes your prediction of the label. You must evaluate your confidence in your prediction with the format: 'Confidence: <confidence>', where <confidence> denotes the confidence of your prediction. Moreover, please obey the instruction about whether providing reasoning process or not in your response. If the reasoning process is not required, please do not provide any reasoning process in your response. If the reasoning process is required, please provide the reasoning process in your response. However, you must predict the labels and provide your confidence before the reasoning process."}],
            [{"role": "user", "content": self.data[index]}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        return conversation


def load_data(promptList:List, tokenizer, batch_size:int) -> DataLoader:
    
    # generate index for each data
    dataDict = {}
    
    for idx, item in enumerate(promptList):
        if dataDict.get(str(idx)) is None:
            dataDict[str(idx)] = item
        else:
            raise ValueError("Repeat data index.")
    
    processed_dataset = JsonDataset(data=promptList, tokenizer=tokenizer)
    dataloader = DataLoader(dataset=processed_dataset, batch_size=batch_size)
    
    return dataloader, dataDict
