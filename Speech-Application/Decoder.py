import torch
class Decoder:
    def __init__(self):
        self.char_map = {}
        self.characters = ['<SPACE>', "'"] + [chr(i) for i in range(97, 123)]  # Add <SPACE> and ' along with lowercase alphabets
        for i, char in enumerate(self.characters):
           self.char_map[char] = i
        self.index_map = {i: char for char, i in self.char_map.items()}
        self.index_map[0] = ' '

    def int_to_text(self,labels):
                # Decode number sequence to text
                string = []
                for i in labels:
                    string.append(self.index_map[i])
                return ''.join(string).replace('<SPACE>', ' ')
    
    def decode(self, output, blank_label = 28, collapse_repeated = True):
        arg_maxes = torch.argmax(output, dim=2)
        decodes = []

        for i, args in enumerate(arg_maxes):
            decode = []

            for j, index in enumerate(args):
                if index != blank_label:
                    if collapse_repeated and j != 0 and index == args[j -1]:
                        continue
                    decode.append(index.item())
            decodes.append(self.int_to_text(decode))
        return decodes

    