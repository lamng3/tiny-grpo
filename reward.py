def default_reward(answer: str, oracle_answer: str):
    """default reward"""
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
    When response length exceeds the predefined maximum value, a punishment interval is applied
    
    Args
        L_max (int): maximum length a response can have
        L_cache (int): also mentioned as overlong buffer in verl/dapo implementation 
    """
    reward = 0
    answer_length = len(answer) if answer is not None else 0
    if answer_length <= L_max - L_cache:
        return 0
    elif L_max - L_cache < answer_length <= L_max:
        return ((L_max - L_cache) - answer_length) / L_cache
    else:
        return -1
