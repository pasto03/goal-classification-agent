import os
import torch.nn as nn
from torch.optim import Adam

from utils import *
from cmd_classifier import CommandClassifier

import json


class ObjectiveClassifier:
    def __init__(self, ckpt_path=None, load_ckpt=True):
        self.goals_inv = {0: "TOP LEFT", 1: "BOTTOM LEFT", 2: "TOP RIGHT", 3: "BOTTOM RIGHT"}
        # do not change hyperparameters unless load_ckpt=False
        self.model = CommandClassifier(input_len=MAX_LEN_U, output_dim=4, 
                                       hidden_size=512, embedding_dim=300).to(device)
        self.ckpt_path = "classifier_checkpoint/classifier_en.pt" if not ckpt_path else ckpt_path
        # load previous checkpoint if exists
        if load_ckpt:
            if os.path.exists(self.ckpt_path):
                self.model.load_state_dict(torch.load(self.ckpt_path))
                print("---classifier checkpoint loaded---")
               
    def translate_command(self, user_input, debug=False):
        """
        convert user command to goal index
        """
        self.model.eval()
        input_t = encode_input(user_input).to(device)
        out = self.model(input_t)
        goal_prob = torch.softmax(out, dim=-1).max().item()
        goal_idx = out.argmax(dim=1).item()
        goal_name = self.goals_inv[goal_idx]

        if debug:
            log_dict = {"log": {"user_input": user_input, "predicted_index": goal_idx, 
                                "goal_prob" :goal_prob, "goal_name": goal_name}}
            log_text = json.dumps(log_dict, indent=4, ensure_ascii=False)
            print(log_text+'\n')

        return goal_idx, goal_name
    
    def fine_tune(self, train_data: list[str, int], n_epochs=1, device=device, lr=5e-5):
        """fine tune of one specific goal command"""
        user_input, goal_idx = train_data
        input_t = encode_input(user_input).to(device)
        label_t = torch.tensor([goal_idx], dtype=torch.long).to(device)

        self.model.train()

        opt = Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for ep in range(n_epochs):
            out = self.model(input_t)

            loss = criterion(out, label_t)

            self.model.zero_grad()
            loss.backward()
            opt.step()

        print("--fine tune completed--")

if __name__ == "__main__":
    cls = ObjectiveClassifier(load_ckpt=True)
    print(cls.translate_command("Go to the bottom leftmost side", debug=True))
