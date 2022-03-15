import encodings
import os
import torch

from transformers import GPT2Tokenizer, GPT2LMHeadModel

from hfutils.pipe.gpt import GPTLMHeadModelPipe
from hfutils.sanity import test_parameters_consistency
from hfutils.preprocess import split_train_test, vit_collate_fn, ViTFeatureExtractorTransforms

device = "cuda:0"

home_dir = os.path.expanduser("~")
base_dir = os.path.join(home_dir, "HuggingFace")
model_dir = os.path.join(base_dir, "gpt2")

if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    text = """
    Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny's Child. Managed by her father, Mathew Knowles, the group became one of the world's best-selling girl groups of all time. Their hiatus saw the release of Beyoncé's debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".
    """

    encodings = tokenizer((text,),
        max_length=384, 
        stride=128, 
        return_tensors="pt"
    )
    encodings = encodings.to(device)

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    print("input_ids", input_ids.shape)
    print("attention_mask", attention_mask.shape)

    model_gold = GPT2LMHeadModel.from_pretrained(model_dir)
    model_gold = model_gold.to(device)
    model_gold.eval()

    model_test = GPTLMHeadModelPipe(model_gold)
    model_test.convert(device)
    model_test.eval()

    test_parameters_consistency(model_gold, model_test)

    outputs_gold = model_gold(
    input_ids=input_ids,
    attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
    
    
    print(outputs_gold.keys())
    print(type(outputs_gold.hidden_states), len(outputs_gold.hidden_states))
    
    hidden_states_gold = outputs_gold.hidden_states

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
