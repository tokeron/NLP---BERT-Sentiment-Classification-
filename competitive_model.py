import torch
from torch.utils.data import Dataset


class UnlabeledDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        a = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return a

    def __len__(self):
        return len(self.encodings.input_ids)

# ------------------------------------------------------------------------
def dataset_2_masked_labeled_inputs(dataset, tokenizer):
    # creating labels for the unsuprvised (Masked)
    inputs = tokenizer(dataset,
                       max_length=256,
                       return_tensors='pt', truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()

    # masking
    # create random array of floats in equal dimension to input_ids
    rand = torch.rand(inputs.input_ids.shape)
    random_thresh = 0.15  # where the random array is less than 0.15, we set true
    mask_arr = (rand < random_thresh) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (
                inputs.input_ids != 0)
    # [cls, sep, pad] tokens
    # create selection from mask_arr
    # selection = torch.flatten((mask_arr[0]).nonzero()).tolist()  # find true values indexes
    selection = []
    for i in range(inputs.input_ids.shape[0]):
        selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())

    # apply selection index to inputs.input_ids, adding MASK tokens
    # inputs.input_ids[0, selection] = 103  # [Mask] token
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
    return inputs