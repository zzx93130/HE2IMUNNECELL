"""Eight square symmetries of joint RGB/mask tensors; average probabilities."""
import torch


def d4(images):
    if images.shape[-2] != images.shape[-1]:
        raise ValueError('D4 averaging requires square cell crops')
    return [torch.rot90(images,k,(-2,-1)) for k in range(4)]+[
        torch.rot90(images.flip(-1),k,(-2,-1)) for k in range(4)]


def average_views(model,images):
    probabilities=model(torch.cat(d4(images),dim=0)).softmax(-1).reshape(8,len(images),4)
    return probabilities.mean(0),probabilities
