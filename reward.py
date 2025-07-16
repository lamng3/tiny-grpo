def default_reward(answer: str, oracle_answer: str) -> float:
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


def rule_based_reward(answer: str, oracle_answer: str) -> float:
    """introduced in DAPO to avoid reward hacking"""
    reward = 0
    if answer is not None:
        if answer == oracle_answer:
            reward = 1
        else:
            reward = -1
    return reward


def soft_overlong_punishment(
    answer: str, 
    max_resp_len: int,
    overlong_buffer_len: int,
    penalty_factor: float = 1.0,
) -> float:
    """
    Soft Overlong Punishment in DAPO paper
    When response length exceeds the predefined maximum value, a punishment interval is applied
    
    Args
        max_resp_len (int): maximum length a response can have (aka L_max in DAPO)
        overlong_buffer_len (int): overlong buffer length of a response (aka L_cache in DAPO)
    """
    valid_resp_len = len(answer) if answer is not None else 0
    expected_len = max_resp_len - overlong_buffer_len
    exceed_len = valid_resp_len - expected_len
    overlong_penalty_factor = penalty_factor
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
    return overlong_reward
