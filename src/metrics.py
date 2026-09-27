"""Macro F0.5 exactly as described in the challenge (per S1 entity, singletons included)."""


def f05(pred, true):
    if not true:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(pred_map, true_map):
    """pred_map / true_map: {s1_id: set(ids)}; averaged over the keys of true_map."""
    tot = 0.0
    for s1, t in true_map.items():
        tot += f05(pred_map.get(s1, set()), t)
    return tot / max(1, len(true_map))
