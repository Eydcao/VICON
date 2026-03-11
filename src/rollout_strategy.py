from pprint import pprint
import numpy as np
import torch


def random_single_step_strategy(gt_ref_steps, demo_num, total_steps):
    strategy_list = []

    # get demo_num*cond_t_idx, where each of them < gt_ref_steps
    # then get demo_num*qoi_t_idx, where each = cond_t_idx+1, and <= gt_ref_steps
    # the replace in np.random.choice is True to allow repeated sampling
    demo_cond = np.random.choice(gt_ref_steps, demo_num, replace=True)
    # sort the demo_cond
    demo_cond = np.sort(demo_cond)
    demo_qoi = demo_cond + 1

    for i in range(gt_ref_steps - 1, total_steps - 1):
        strategy_list.append((demo_cond, demo_qoi, i, i + 1))

    return strategy_list


def fix_single_step_strategy(gt_ref_steps, demo_num, total_steps):
    # assert gt_ref_steps == demo_num + 1
    strategy_list = []

    demo_cond = np.arange(demo_num)
    demo_qoi = demo_cond + 1
    for i in range(gt_ref_steps, total_steps):  # i is quest qoi, i.e., target
        strategy_list.append((demo_cond, demo_qoi, i - 1, i))

    return strategy_list


def fix_multi_step_strategy(gt_ref_steps, demo_num, max_stride, total_steps, full):
    # if full: demo_num means the maximum number of demos. Fewer demos may be used for large stride prediction
    # if not full: demo_num means the exact number of demos to use for each prediction, could use repeated demos
    assert gt_ref_steps >= max_stride + 1  # make sure even max_stride has at least one example

    strategy_list = []

    demo_cond = {}
    demo_qoi = {}

    for i in range(1, max_stride + 1):  # i is the current stride
        this_demo_num = min(demo_num, gt_ref_steps - i)
        this_demo_qoi = np.arange(gt_ref_steps - this_demo_num, gt_ref_steps)

        if full:  # add more repeated demos
            repeats = demo_num // this_demo_num + 1
            this_demo_qoi = np.tile(this_demo_qoi, repeats)[:demo_num]
            this_demo_qoi = np.sort(this_demo_qoi)  # increasing order

        demo_cond[i] = this_demo_qoi - i
        demo_qoi[i] = this_demo_qoi

    for i in range(gt_ref_steps, total_steps):  # i is quest qoi, i.e., target
        dist = i - gt_ref_steps + 1  # distance from the last available frame
        this_stride = min(dist, max_stride)
        strategy_list.append((demo_cond[this_stride], demo_qoi[this_stride], i - this_stride, i))

    return strategy_list


def gen_strategy_list(cfg, total_steps, dropped_mask):
    """
    Generate a list of strategies for rollout.

    Args:
        cfg: Configuration object
        total_steps: Total number of steps in the sequence
        dropped_mask: Optional mask indicating dropped frames, shape [t]
    """
    if dropped_mask is not None:
        # Find available frame indices
        available_indices = torch.where(~dropped_mask)[0].tolist()
        # only select indices that are less than gt_ref_steps
        # print("Available frames:", available_indices)
        available_indices = [idx for idx in available_indices if idx < cfg.gt_ref_steps]
        # print("Available frames after:", available_indices)

        if cfg.strategy == "fixed_single_step":
            strategy_list = fix_single_step_strategy_with_drops(
                cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, available_indices
            )
        elif cfg.strategy == "fixed_multi_step_norepeat":
            strategy_list = fix_multi_step_strategy_with_drops(
                cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, available_indices, full=False
            )
        elif cfg.strategy == "fixed_multi_step_repeat":
            strategy_list = fix_multi_step_strategy_with_drops(
                cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, available_indices, full=True
            )
        else:
            raise ValueError("Invalid strategy")
    else:
        if cfg.strategy == "fixed_single_step":
            strategy_list = fix_single_step_strategy(cfg.gt_ref_steps, cfg.demo_num, total_steps)
        elif cfg.strategy == "fixed_multi_step_norepeat":
            strategy_list = fix_multi_step_strategy(
                cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, full=False
            )
        elif cfg.strategy == "fixed_multi_step_repeat":
            strategy_list = fix_multi_step_strategy(
                cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, full=True
            )
        else:
            raise ValueError("Invalid strategy")
    return strategy_list


def get_available_pairs(demo_num, dt, available_indices):
    # Sort available indices
    available_indices = sorted(available_indices)

    # Find pairs of consecutive available frames
    available_pairs = []
    for i in range(len(available_indices) - dt):
        for j in range(i + 1, len(available_indices)):
            if available_indices[j] - available_indices[i] == dt:
                available_pairs.append((available_indices[i], available_indices[j]))

    if not available_pairs:
        return []

    # If we don't have enough unique pairs, repeat them
    if len(available_pairs) < demo_num:
        # Calculate how many times to repeat each pair
        repeats = [demo_num // len(available_pairs)] * len(available_pairs)
        # Add extra repeats to reach demo_num
        for i in range(demo_num % len(available_pairs)):
            repeats[i] += 1

        # Create the repeated pairs list
        repeated_pairs = []
        for i, pair in enumerate(available_pairs):
            repeated_pairs.extend([pair] * repeats[i])
        available_pairs = repeated_pairs

    return available_pairs


def fix_single_step_strategy_with_drops(gt_ref_steps, demo_num, fixed_stride, total_steps, available_indices):
    """
    Generate a single-step strategy accounting for dropped frames.

    Args:
        gt_ref_steps: Number of ground truth reference steps
        demo_num: Number of demonstrations to use
        total_steps: Total steps in sequence
        available_indices: List of indices of available frames

    Returns:
        List of strategies (demo_cond_idxs, demo_qoi_idxs, quest_cond_idx, quest_qoi_idx)
    """
    strategy_list = []

    # Sort available indices
    available_indices = sorted(available_indices)
    # Get the last available frame index
    last_avail_idx = available_indices[-1]
    # Start rollout from the last available frame or gt_ref_steps, whichever is smaller
    start_frame = min(gt_ref_steps, last_avail_idx)

    # prepare all available pairs for each stride
    available_pairs = get_available_pairs(demo_num, fixed_stride, available_indices)
    if len(available_pairs) == 0:
        raise ValueError("No available pairs found for stride {} and available_indices.".format(fixed_stride))

    # accumulated (pred) indices
    accumulated_indices = available_indices.copy()

    for i in range(start_frame + 1, total_steps):  # i + 1 is quest qoi, i.e., target
        this_stride = fixed_stride
        potential_cond_idx = i - this_stride
        # check if the potential cond idx is in the accumulated indices
        if not potential_cond_idx in accumulated_indices:
            continue

        # Use the selected demo pairs
        demo_cond_idxs = [pair[0] for pair in available_pairs[:demo_num]]
        demo_qoi_idxs = [pair[1] for pair in available_pairs[:demo_num]]
        # Get the quest cond and qoi
        quest_cond_idx = i - this_stride
        quest_qoi_idx = i
        # Append the strategy
        strategy_list.append((demo_cond_idxs, demo_qoi_idxs, quest_cond_idx, quest_qoi_idx))
        # Add the quest qoi idx to the accumulated indices
        accumulated_indices.append(quest_qoi_idx)

    return (strategy_list, accumulated_indices)


def fix_multi_step_strategy_with_drops(gt_ref_steps, demo_num, max_stride, total_steps, available_indices, full):
    """
    Generate a multi-step strategy accounting for dropped frames.

    Args:
        gt_ref_steps: Number of ground truth reference steps
        demo_num: Number of demonstrations to use
        max_stride: Maximum stride between consecutive steps
        total_steps: Total steps in sequence
        available_indices: List of indices of available frames
        full: Whether to use all strides or just the maximum stride

    Returns:
        List of strategies (demo_cond_idxs, demo_qoi_idxs, quest_cond_idx, quest_qoi_idx)
    """
    strategy_list = []

    # Sort available indices
    available_indices = sorted(available_indices)
    # Get the last available frame index
    last_avail_idx = available_indices[-1]
    # Start rollout from the last available frame or gt_ref_steps, whichever is smaller
    start_frame = min(gt_ref_steps, last_avail_idx)

    # prepare all available pairs for each stride
    available_pairs_dict = {}
    for dt in range(1, max_stride + 1):  # i is the current stride
        available_pairs = get_available_pairs(demo_num, dt, available_indices)
        if len(available_pairs) == 0:
            continue
        else:
            available_pairs_dict[dt] = available_pairs
    # if none raise error
    if len(available_pairs_dict) == 0:
        raise ValueError("No available pairs found for any stride and available_indices.")
    # find min available stride
    # min_avail_stride = min(available_pairs_dict.keys())
    max_avail_stride = max(available_pairs_dict.keys())

    # accumulated (pred) indices
    accumulated_indices = available_indices.copy()

    # # debug
    # print("Available pairs dict:")
    # pprint(available_pairs_dict)

    for i in range(start_frame + 1, total_steps):  # i + 1 is quest qoi, i.e., target
        dt = i - start_frame  # distance from the last available frame
        max_this_stride = min(dt, max_stride, max_avail_stride)
        # gradually decrease the max_this_stride until it is in the available pairs
        found_starting = False
        for this_stride in sorted(available_pairs_dict.keys(), reverse=True):
            if this_stride <= max_this_stride:
                potential_cond_idx = i - this_stride
                # check if the potential cond idx is in the accumulated indices
                if potential_cond_idx in accumulated_indices:
                    found_starting = True
                    break
        if not found_starting:
            continue
        available_pairs = available_pairs_dict[this_stride]
        # Use the selected demo pairs
        demo_cond_idxs = [pair[0] for pair in available_pairs[:demo_num]]
        demo_qoi_idxs = [pair[1] for pair in available_pairs[:demo_num]]
        # Get the quest cond and qoi
        quest_cond_idx = i - this_stride
        quest_qoi_idx = i
        # Append the strategy
        strategy_list.append((demo_cond_idxs, demo_qoi_idxs, quest_cond_idx, quest_qoi_idx))
        # Add the quest qoi idx to the accumulated indices
        accumulated_indices.append(quest_qoi_idx)

    return (strategy_list, accumulated_indices)


def print_strategy(strategy_list):
    for i, (cond_t_idx, qoi_t_idx, cond_t, qoi_t) in enumerate(strategy_list):
        print("# {}".format(i + 1), cond_t_idx, qoi_t_idx, cond_t, qoi_t)


if __name__ == "__main__":
    # # Regular strategy functions without dropped frames
    # print("=== Original strategies without drops ===")
    # print("Single step strategy:")
    # print_strategy(fix_single_step_strategy(10, 3, 20))
    # print("\nMulti-step strategy (max stride only):")
    # print_strategy(fix_multi_step_strategy(10, 3, 5, 20, full=False))
    # print("\nMulti-step strategy (all strides):")
    # print_strategy(fix_multi_step_strategy(10, 3, 5, 20, full=True))

    # print("\n\n=== Strategies with every other frame dropped ('halved') ===")
    # # Scenario 1: Every other frame is available (halved)
    # available_indices = list(range(0, 20, 2))  # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    # print(f"Available frames: {available_indices}")
    # print("\nSingle step strategy with drops:")
    # print_strategy(fix_single_step_strategy_with_drops(10, 3, 20, available_indices))
    # print("\nMulti-step strategy with drops (max stride only):")
    # print_strategy(fix_multi_step_strategy_with_drops(10, 3, 5, 20, available_indices, full=False))
    # print("\nMulti-step strategy with drops (all strides):")
    # print_strategy(fix_multi_step_strategy_with_drops(10, 3, 5, 20, available_indices, full=True))

    # print("\n\n=== Strategies with random frames dropped ===")
    # # Scenario 2: Random frames are available
    import random

    random.seed(10)  # For reproducibility
    # check random dropping
    print("Test random dropping")
    available_indices = sorted(random.sample(range(9), 7))  # 12 random frames out of 20
    print(f"\nAvailable frames: {available_indices}")
    print("\nSingle step strategy with drops:")
    (strategy_list, accumulated_indices) = fix_single_step_strategy_with_drops(10, 9, 1, 20, available_indices)
    print_strategy(strategy_list)
    # print("Accumulated indices:", accumulated_indices)
    print(f"\nMulti-step strategy with drops (max stride only), max stride:", 3)
    (strategy_list, accumulated_indices) = fix_multi_step_strategy_with_drops(
        10, 9, 3, 20, available_indices, full=False
    )
    print_strategy(strategy_list)
    # print("Accumulated indices:", accumulated_indices)

    # # check halve dropping
    # print("\nTest halved dropping")
    # available_indices = list(range(0, 10, 2))  # [0, 2, 4, 6, 8]
    # print(f"\nAvailable frames: {available_indices}")
    # print("\nSingle step strategy with drops:")
    # (strategy_list, accumulated_indices) = fix_single_step_strategy_with_drops(10, 9, 2, 20, available_indices)
    # print_strategy(strategy_list)
    # print("Accumulated indices:", accumulated_indices)
    # print("\nMulti-step strategy with drops (all strides):")
    # (strategy_list, accumulated_indices) = fix_multi_step_strategy_with_drops(
    #     10, 9, 3, 20, available_indices, full=False
    # )
    # print_strategy(strategy_list)
    # print("Accumulated indices:", accumulated_indices)
