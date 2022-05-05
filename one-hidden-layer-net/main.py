import numpy as np
import matplotlib.pyplot as plt


# 获取呈现 ‘几’ 字分布的豆豆
def get_beans(counts):
    xs = np.random.rand(counts) * 2
    xs = np.sort(xs)
    ys = np.zeros(counts)
    for i in range(counts):
        x = xs[i]
        yi = 0.7 * x + (0.5 - np.random.rand()) / 50 + 0.5
        if yi > 0.8 and yi < 1.4:
            ys[i] = 1

    return xs, ys


# s 形函数
def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))


# 初始化参数
# 第一层
w11_1 = np.random.rand()  # 第一层第一个神经元的第一个权重
b11 = np.random.rand()  # 第一层第一个神经元的第一个偏置

w12_1 = np.random.rand()  # 第一层第二个神经元的第一个权重
b12 = np.random.rand()

# 第二层
w21_1 = np.random.rand()
w21_2 = np.random.rand()
b21 = np.random.rand()


# 前向传播
def forward_propgation(xs):
    i11_1 = w11_1 * xs + b11  # 第一层第一个神经元的第一个输入
    o11_1 = sigmoid(i11_1)  # 第一层第一个神经元的第一个输出

    i12_1 = w12_1 * xs + b12
    o12_1 = sigmoid(i12_1)

    i21_1 = w21_1 * o11_1 + w21_2 * o12_1 + b21
    o21_1 = sigmoid(i21_1)

    return o21_1, i21_1, o12_1, i12_1, o11_1, i11_1


m = 100  # 获取豆豆的数量
xs, ys = get_beans(m)
plt.title("size-toxicity function", fontsize=12)
plt.xlabel("bean size")
plt.ylabel("toxicity")
plt.scatter(xs, ys)  # 绘制豆豆分布的散点图
o21_1, i21_1, o12_1, i12_1, o11_1, i11_1 = forward_propgation(xs)
plt.plot(xs, o21_1)  # 绘制初始预测函数形状
plt.show()

# 进行 10000次 随机样本梯度下降
for _ in range(10000):
    for i in range(100):  # 逐一进行
        x = xs[i]
        y = ys[i]
        # 先进行一次前向传播
        o21_1, i21_1, o12_1, i12_1, o11_1, i11_1 = forward_propgation(x)
        # 获取误差
        e = (y - o21_1) ** 2

        dedo21_1 = -2 * (y - o21_1)  # e对o21_1的偏导

        do21_1di21_1 = o21_1 * (1 - o21_1)  # o21_1对i21_1的偏导

        di21_1do11_1 = w21_1
        di21_1do12_1 = w21_2
        di21_1dw21_1 = o11_1
        di21_1dw21_2 = o12_1
        di21_1db21 = 1

        do11_1di11_1 = o11_1 * (1 - o11_1)
        di11_1dw11_1 = x
        di11_1db11 = 1

        do12_1di12_1 = o12_1 * (1 - o12_1)
        di12_1dw12_1 = x
        di12_1db12 = 1

        # 第一层部分求偏导（链式法则
        dedw11_1 = dedo21_1 * do21_1di21_1 * di21_1do11_1 * do11_1di11_1 * di11_1dw11_1
        dedb11 = dedo21_1 * do21_1di21_1 * di21_1do11_1 * do11_1di11_1 * di11_1db11

        dedw12_1 = dedo21_1 * do21_1di21_1 * di21_1do12_1 * do12_1di12_1 * di12_1dw12_1
        dedb12 = dedo21_1 * do21_1di21_1 * di21_1do12_1 * do12_1di12_1 * di12_1db12

        # 第二层部分求偏导（链式法则
        dedw21_1 = dedo21_1 * do21_1di21_1 * di21_1dw21_1
        dedw21_2 = dedo21_1 * do21_1di21_1 * di21_1dw21_2
        dedb21 = dedo21_1 * do21_1di21_1 * di21_1db21

        # 学习率 α = 0.01 梯度下降修正
        alpha = 0.01

        w11_1 = w11_1 - alpha * dedw11_1
        b11 = b11 - alpha * dedb11

        w12_1 = w12_1 - alpha * dedw12_1
        b12 = b12 - alpha * dedb12

        w21_1 = w21_1 - alpha * dedw21_1
        w21_2 = w21_2 - alpha * dedw21_2
        b21 = b21 - alpha * dedb21

    if _ % 100 == 0:  # 每学习一百次打印一次函数拟合程度
        plt.clf()
        plt.scatter(xs, ys)
        o21_1, i21_1, o12_1, i12_1, o11_1, i11_1 = forward_propgation(xs)
        plt.plot(xs, o21_1)
        plt.pause(0.02)
        print("第"+str(_)+"次训练")


plt.clf()
plt.scatter(xs, ys)
o21_1, i21_1, o12_1, i12_1, o11_1, i11_1 = forward_propgation(xs)
plt.plot(xs, o21_1)
plt.show()