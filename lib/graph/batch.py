import torch


def to_torch_batch(batch):
    batch = torch.stack([batch.x[batch.batch == n] for n in range(batch.num_graphs)],axis=0)
    return batch