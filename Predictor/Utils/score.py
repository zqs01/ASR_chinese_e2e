import Levenshtein as Lev


def calculate_cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

    Arguments:
        s1 (string): space-separated sentence (hyp)
        s2 (string): space-separated sentence (gold)
    """
    word_num = len(s2.split(' '))
    return Lev.distance(s1, s2) / word_num



