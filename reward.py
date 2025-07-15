def default_reward(answer: str, oracle_answer: str):
    reward = 0
    if answer is not None:
        if answer == oracle_answer:
            reward = 1.0
        elif oracle_answer in answer:
            reward = 0.5
        else:
            reward = 0.01
    return reward


def soft_overlong_punishment(answer: str, L_max: int, L_cache: int):
    """
    Soft Overlong Punishment in DAPO paper
    
    Args
        L_max (int): maximum length a response can have
        L_cache (int): 
    """
    reward = 0
