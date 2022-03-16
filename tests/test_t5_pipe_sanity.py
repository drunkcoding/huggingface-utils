import os
from transformers import T5ForConditionalGeneration
import torch

from hfutils.pipe.t5 import T5PyTorchPipe
from hfutils.sanity import test_parameters_consistency


device = "cuda:0"

home_dir = os.path.expanduser("~")
base_dir = os.path.join(home_dir, "HuggingFace")
model_dir = os.path.join(base_dir, "google", "t5-base-lm-adapt")

if __name__ == "__main__":
    model_gold = T5ForConditionalGeneration.from_pretrained(model_dir
    ).to(device)
    model_gold.eval()

    model_test = T5PyTorchPipe(model_gold)
    model_test.convert(device)
    model_test.eval()

    # print(model_test.layers)

    # test_parameters_consistency(model_gold, model_test)

    input_ids = torch.Tensor([[200, 200, 200, 200, 0, 0, 0, 0, 0],[200, 200, 200, 200, 0, 0, 0, 0, 0]]).to(device).to(torch.long)
    attention_mask = torch.Tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0],[1, 1, 1, 1, 0, 0, 0, 0, 0]]).to(device).to(torch.long)
    outputs_gold = model_gold.generate(input_ids=input_ids, attention_mask=attention_mask, do_sample=False, return_dict_in_generate=True, output_hidden_states=True)
    
    print(outputs_gold.keys())
    print(type(outputs_gold.encoder_hidden_states), len(outputs_gold.encoder_hidden_states), len(outputs_gold.decoder_hidden_states))

    # print(type(outputs_gold.decoder_hidden_states[0]), len(outputs_gold.decoder_hidden_states[0]))
    # print(type(outputs_gold.decoder_hidden_states[1]), len(outputs_gold.decoder_hidden_states[1]))

    hidden_states_gold = outputs_gold.encoder_hidden_states + outputs_gold.decoder_hidden_states[0]
    # sequences_gold = outputs_gold.sequences

    outputs_test = model_test((input_ids, attention_mask), output_hidden_states=True)
    hidden_states_test = outputs_test[-1]

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
