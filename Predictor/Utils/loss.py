import torch as t
import torch.nn.functional as F
import torch

# def calculate_loss(pred, gold):
#     vocabulary_size = pred.size()[-1]
#     inputs = pred.view(-1, vocabulary_size)
#     gold = gold.view(-1)
#     loss = t.nn.functional.cross_entropy(inputs, gold, ignore_index=0, reduction='mean')
#     return loss

#
def calculate_loss(pred, gold, smoothing=0.0):
    """Calculate cross entropy loss, apply label smoothing if needed.
    """
    pred = pred.view(-1, pred.size(2))
    gold = gold.contiguous().view(-1)

    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        # Generate one-hot matrix: N x C.
        # Only label position is 1 and all other positions are 0
        # gold include -1 value (IGNORE_ID) and this will lead to assert error
        gold_for_scatter = gold.ne(0).long() * gold
        one_hot = torch.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(0)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=0,
                               reduction='mean')

    return loss