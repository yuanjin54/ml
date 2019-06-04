# 16. 3Sum最近
# 给定一个数组nums的ñ整数和一个整数target，找出三个整数nums 使得总和最接近  target。
# 返回三个整数的总和。
# 您可以假设每个输入都只有一个解决方案。
# 例：
# 给定数组nums = [-1,2,1，-4]和target = 1。
#
# 最接近目标的总和是2.（-1 + 2 + 1 = 2）。
nums = [-1, 2, -1, 0, 1]


def three_sum_closest(nums, target):
    return 0


def three_sum(nums, target):
    return 0


def two_sum(nums, target):
    dict = {}
    for i in range(len(nums)):
        if target - nums[i] in dict:
            return dict[target - nums[i]], i
        else:
            dict[nums[i]] = i
    return None


print(two_sum(nums, 1))
