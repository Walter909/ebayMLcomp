import torch

##DATA ENCODING
class AspectDataset:
    def __init__(self, dataframe, name_tokenizer, value_tokenizer, aspect_names):
        self.data = dataframe
        self.name_tokenizer = name_tokenizer
        self.value_tokenizer = value_tokenizer
        self.aspect_names = aspect_names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        aspect_name = self.data['Aspect Name'].iloc[index]
        aspect_values = self.data['Aspect Value'].iloc[index]

        aspect_name_tokens = self.name_tokenizer.tokenize(aspect_name)
        aspect_value_tokens = self.value_tokenizer.tokenize(aspect_values)

        aspect_name_ids = self.name_tokenizer.convert_tokens_to_ids(aspect_name_tokens)
        aspect_value_ids = self.value_tokenizer.convert_tokens_to_ids(aspect_value_tokens)

        # Create attention mask
        attention_mask = [1] * len(aspect_name_ids) + [1] * len(aspect_value_ids)

        # Convert aspect name labels to their corresponding indices
        aspect_name_labels = [self.aspect_names.index(name) for name in aspect_name_tokens]

        # Print tokenized sequences
        print('Aspect Name Tokens:', aspect_name_tokens)
        print('Aspect Value Tokens:', aspect_value_tokens)

        return {
            'aspect_name_ids': torch.tensor(aspect_name_ids),
            'aspect_value_ids': torch.tensor(aspect_value_ids),
            'aspect_name_labels': torch.tensor(aspect_name_labels),
            'attention_mask': torch.tensor(attention_mask)
        }

