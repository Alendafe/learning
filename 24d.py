def solve_24(nums):
    result = set()

    def dfs(current_nums, current_exprs):
        if len(current_nums) == 1:
            if abs(current_nums[0] - 24) < 1e-6:
                result.add(current_exprs[0])
            return

        # 遍历所有可能的两数组合，i < j 避免重复选择
        for i in range(len(current_nums)):
            for j in range(len(current_nums)):
                if i < j:
                    x = current_nums[i]
                    y = current_nums[j]
                    x_exp = current_exprs[i]
                    y_exp = current_exprs[j]

                    # 生成剩下的数字和表达式
                    remaining_nums = [current_nums[k] for k in range(len(current_nums)) if k != i and k != j]
                    remaining_exprs = [current_exprs[k] for k in range(len(current_exprs)) if k != i and k != j]

                    # 加法
                    new_num = x + y
                    new_expr = f"({x_exp}+{y_exp})"
                    dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

                    # 乘法
                    new_num = x * y
                    new_expr = f"({x_exp}*{y_exp})"
                    dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

                    # 减法 x - y
                    new_num = x - y
                    new_expr = f"({x_exp}-{y_exp})"
                    dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

                    # 减法 y - x
                    new_num = y - x
                    new_expr = f"({y_exp}-{x_exp})"
                    dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

                    # 除法 x / y，检查y不为0
                    if y != 0:
                        new_num = x / y
                        new_expr = f"({x_exp}/{y_exp})"
                        dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

                    # 除法 y / x，检查x不为0
                    if x != 0:
                        new_num = y / x
                        new_expr = f"({y_exp}/{x_exp})"
                        dfs(remaining_nums + [new_num], remaining_exprs + [new_expr])

    # 初始表达式列表为各数字的字符串形式
    initial_exprs = [str(num) for num in nums]
    dfs(nums, initial_exprs)
    return result

# 示例用法
if __name__ == "__main__":
    input_nums = [2,3,6,5]
    solutions = solve_24(input_nums)
    if solutions:
        print("找到的解法：")
        for sol in solutions:
            print(sol)
    else:
        print("无法构成24点")