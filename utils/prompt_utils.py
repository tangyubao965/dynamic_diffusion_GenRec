

import random

def sample_identifiers(user_history, inverted_index, num_samples=5):
    """
    Sample identifiers from inverted index excluding user's history.
    """
    all_items = set(inverted_index.keys())
    history_items = set(user_history)
    candidate_items = list(all_items - history_items)

    sampled = random.sample(candidate_items, min(num_samples, len(candidate_items)))
    return sampled


def construct_prompt_sequence(user_history, metadata_map, sep_token='; '):
    """
    Construct prompt sequence from metadata of user history items.
    """
    prompt = []
    for item_id in user_history:
        meta = metadata_map.get(item_id, "")
        prompt.append(meta.strip())
    return sep_token.join(prompt)


def construct_masked_prompt(prompt_tokens, mask_ratio=0.3):
    """
    Randomly mask tokens in the prompt sequence for denoising objective.
    """
    tokens = prompt_tokens.split()
    n = len(tokens)
    num_mask = int(n * mask_ratio)

    if num_mask == 0:
        return prompt_tokens

    mask_indices = random.sample(range(n), num_mask)
    for idx in mask_indices:
        tokens[idx] = '[MASK]'
    return ' '.join(tokens)