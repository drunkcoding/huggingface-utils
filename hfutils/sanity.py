import numpy as np

def test_parameters_consistency(model_gold, model_test, abort=True):
    model_test_param = model_test.named_parameters()
    model_gold_param = model_gold.named_parameters()

    # print(model_test_param.keys())
    # print(model_gold_param.keys())

    # for name, _ in model_test.named_parameters():
    #     print("model_test", name)
    # for name, _ in model_gold.named_parameters():
    #     print("model_gold", name)

    for test, gold in zip(model_test_param, model_gold_param):
        name_test, param_test = test
        name_gold, param_gold = gold

        param_test = param_test.detach().cpu().numpy()
        param_gold = param_gold.detach().cpu().numpy()

        if abort:
            print(name_gold, name_test, param_gold.shape, param_test.shape)
            assert np.all(np.isclose(
                param_test,
                param_gold
            ))
        else:
            print(name_test, np.linalg.norm(param_gold-param_test))
