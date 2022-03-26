import os
import torch

from torchvision.datasets import ImageNet

from transformers import ViTFeatureExtractor, ViTForImageClassification

from hfutils.pipe.vit import ViTPyTorchPipeForImageClassification
from hfutils.sanity import test_parameters_consistency
from hfutils.preprocess import split_train_test, vit_collate_fn, ViTFeatureExtractorTransforms

device = "cuda:0"

# home_dir = os.path.expanduser("~")
home_dir = "/mnt/raid0nvme1"
base_dir = os.path.join(home_dir, "HuggingFace")
dataset_path = os.path.join(home_dir, "ImageNet")
model_dir = os.path.join(base_dir, "google", "vit-base-patch16-224")

dataset = ImageNet(
    dataset_path, 
    split="val", 
    transform=ViTFeatureExtractorTransforms(
        model_dir, 
        split="val"
    )
)

if __name__ == "__main__":
    
    tokenizer = ViTFeatureExtractor.from_pretrained(model_dir)

    model_gold = ViTForImageClassification.from_pretrained(model_dir)
    model_gold = model_gold.to(device)
    model_gold.eval()

    model_test = ViTPyTorchPipeForImageClassification(model_gold)
    model_test.convert(device)
    model_test.eval()

    test_parameters_consistency(model_gold, model_test)

    image = dataset[0][0]
    encodings = vit_collate_fn([dataset[0]])
    pixel_values = encodings['pixel_values'].to(device)

    outputs_gold = model_gold(pixel_values, output_hidden_states=True, return_dict=True)
    print(pixel_values.shape, outputs_gold.logits.shape)

    print(outputs_gold.keys())
    print(type(outputs_gold.hidden_states), len(outputs_gold.hidden_states))

    hidden_states_gold = outputs_gold.hidden_states

    outputs_test = model_test(pixel_values, output_hidden_states=True)
    hidden_states_test = outputs_test[-1]
    print(pixel_values.shape, outputs_test[0].shape)

    print(len(hidden_states_gold), len(hidden_states_test))

    for i in range(len(hidden_states_gold)):
        # print(i, hidden_states_gold[i], hidden_states_test[i])
        print(i, hidden_states_gold[i].shape, hidden_states_test[i].shape)
        assert torch.all(torch.isclose(hidden_states_gold[i], hidden_states_test[i]))
        print("=====================================================")
    # print(hidden_states_gold)
    # print("=====================================================")
    # print(hidden_states_test)
    assert len(hidden_states_gold) == len(hidden_states_test)