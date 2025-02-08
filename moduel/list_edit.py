import random

def random_select(input_list:list, num_elements:int, random_seed:int, allow_duplicates=False, keep_order=False):
    """
    从字符串列表中随机选择指定数量的元素。

    参数:
    - input_list: 输入的字符串列表
    - num_elements: 需要随机选择的元素数量
    - random_seed: 随机种子
    - allow_duplicates: 是否允许重复选择（布尔值）
    - keep_order: 是否保持原顺序（布尔值）

    返回:
    - 一个新列表，包含随机选择的元素
    """
    if len(input_list) <= 1:
        print("Warning: If the input is empty or only one, the original data will be returned !")
        return input_list

    if num_elements > len(input_list) and not allow_duplicates:
        print("Warning: Selecting more than the original data and not allowing duplication will return the original data !")
        return input_list

    # 创建独立的随机数生成器实例
    rng = random.Random(random_seed)

    if allow_duplicates:  # 允许重复选择
        selected_indices = [rng.randint(0, len(input_list) - 1) for _ in range(num_elements)]
    else:  # 不允许重复选择
        selected_indices = rng.sample(range(len(input_list)), num_elements)

    if keep_order:  # 保持原顺序
        selected_indices = sorted(selected_indices, key=lambda x: input_list.index(input_list[x]))

    # 根据索引选择元素
    output = [input_list[i] for i in selected_indices]
    return output

