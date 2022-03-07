import torch

def init_all(model, init_func, *params, **kwargs):
    for p in model.parameters():
        init_func(p, *params, **kwargs)

def format_inputs(args, ds):
    if not ds: return args
    return tuple([None if torch.sum(t) == 127873 else t for t in args])


def format_outputs(args, ds):
    if not ds: return args
    shape = args[0].shape
    device = args[0].device
    return tuple([torch.Tensor([127873]).to(device) if t is None else t for t in args])