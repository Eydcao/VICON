from pprint import pprint
import numpy as np


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


def gen_strategy_list(cfg, total_steps):
    if cfg.strategy == "fixed_single_step":
        strategy_list = fix_single_step_strategy(cfg.gt_ref_steps, cfg.demo_num, total_steps)
    elif cfg.strategy == "fixed_multi_step_norepeat":
        strategy_list = fix_multi_step_strategy(cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, full=False)
    elif cfg.strategy == "fixed_multi_step_repeat":
        strategy_list = fix_multi_step_strategy(cfg.gt_ref_steps, cfg.demo_num, cfg.max_stride, total_steps, full=True)
    else:
        raise ValueError("Invalid strategy")
    return strategy_list


def print_strategy(strategy_list):
    for i, (cond_t_idx, qoi_t_idx, cond_t, qoi_t) in enumerate(strategy_list):
        print("# {}".format(i + 1), cond_t_idx, qoi_t_idx, cond_t, qoi_t)


if __name__ == "__main__":
    print_strategy(fix_single_step_strategy(10, 9, 20))
    print("=====================================")
    print_strategy(fix_multi_step_strategy(10, 9, 5, 20, full=False))
    print("=====================================")
    print_strategy(fix_multi_step_strategy(10, 9, 5, 20, full=True))
