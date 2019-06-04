# 11 .大多数水的容器
# 给定n个非负整数a1，a2，...，an ，其中每个表示坐标（i，ai）处的点。
# 绘制n条垂直线，使得线i的两个端点位于（i，ai）和（i，0）。
# 找到两条线，它们与x轴一起形成一个容器，这样容器就含有最多的水。
#
# 注意： 您可能不会倾斜容器，n至少为2。
# 输入： [1,8,6,2,5,4,8,3,7]
# 产出： 49
# 思路： 定义两个指针i，j分别指向height的头和尾，计算当前的面积，在移动height[i]和height[j]最小的指针
#       一直迭代，直至j=i为止
height = [1, 1]


def max_area(height):
    i, j = 0, len(height) - 1
    result = min(height[i], height[j]) * (j - i)
    while i < j:
        area = min(height[i], height[j]) * (j - i)
        result = max(area, result)
        if height[i] < height[j]:
            i += 1
        else:
            j -= 1
    return result


print(max_area(height))
