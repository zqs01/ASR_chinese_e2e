import torch as t
import torch.nn.functional as F


def calculate_loss(pred, gold):
    vocabulary_size = pred.size()[-1]
    inputs = pred.view(-1, vocabulary_size)
    gold = gold.view(-1)
    loss = t.nn.functional.cross_entropy(inputs, gold, ignore_index=0)
    return loss

#
# def calculate_loss(pred, gold, input_lengths=None, target_lengths=None, smoothing=0.0, loss_type="ce", pad=0):
#     """
#     Calculate loss
#     args:
#         pred: B x T x C
#         gold: B x T
#         input_lengths: B (for CTC)
#         target_lengths: B (for CTC)
#         smoothing:
#         type: ce|ctc (ctc => pytorch 1.0.0 or later)
#         input_lengths: B (only for ctc)
#         target_lengths: B (only for ctc)
#     """
#     if loss_type == "ce":
#         pred = pred.view(-1, pred.size(2)) # (B*T) x C
#         gold = gold.contiguous().view(-1) # (B*T)
#         if smoothing > 0.0:
#             eps = smoothing
#             num_class = pred.size(1)
#
#             gold_for_scatter = gold.ne(pad).long() * gold
#             one_hot = t.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
#             one_hot = one_hot * (1-eps) + (1-one_hot) * eps / num_class
#             log_prob = F.log_softmax(pred, dim=1)
#
#             non_pad_mask = gold.ne(pad)
#             num_word = non_pad_mask.sum().item()
#             loss = -(one_hot * log_prob).sum(dim=1)
#             loss = loss.masked_select(non_pad_mask).sum() / num_word
#         else:
#             loss = F.cross_entropy(pred, gold, ignore_index=pad, reduction="mean")
#     elif loss_type == "ctc":
#         log_probs = pred.transpose(0, 1) # T x B x C
#         # print(gold.size())
#         targets = gold
#         # targets = gold.contiguous().view(-1) # (B*T)
#
#         """
#         log_probs: torch.Size([209, 8, 3793])
#         targets: torch.Size([8, 46])
#         input_lengths: torch.Size([8])
#         target_lengths: torch.Size([8])
#         """
#
#         # print("log_probs:", log_probs.size())
#         # print("targets:", targets.size())
#         # print("input_lengths:", input_lengths.size())
#         # print("target_lengths:", target_lengths.size())
#         # print(input_lengths)
#         # print(target_lengths)
#
#         log_probs = F.log_softmax(log_probs, dim=2)
#         loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction="mean")
#         # mask = loss.clone() # mask Inf loss
#         # # mask[mask != float("Inf")] = 1
#         # mask[mask == float("Inf")] = 0
#
#         # loss = mask
#         # print(loss)
#
#         # loss_size = len(loss)
#         # loss = loss.sum() / loss_size
#         # print(loss)
#     else:
#         print("loss is not defined")
#
#     return loss