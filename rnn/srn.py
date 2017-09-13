# -*- coding:utf-8 -*-
"""
@author: gxjun
@file: srn.py
@time: 17-9-7 下午3:24
"""
import numpy as np

# 定义参数

memory_length = 9  # 定义记忆时长
hidden_number = 100  # 定义隐藏层参数

# 加载数据
data = open('input.txt', 'r').read();
vocabs = list(set(data));  # 包含的字符个数
data_size, vocabs_size = len(data), len(vocabs)

# 定义各个层之间的关系

# 输入层
input_w = np.random.randn(hidden_number, vocabs_size) * 0.01
input_b = np.zeros((hidden_number, 1))
# 隐藏层
unroll_w = np.random.randn(hidden_number, hidden_number) * 0.01
# 输出层
output_w = np.random.randn(vocabs_size, hidden_number) * 0.01
output_b = np.zeros((vocabs_size, 1))


# 定义softmax
def softmax(feature):
    factor = np.exp(feature)
    return factor / np.sum(factor)


# 定义tanh

def forward(x, state):  # state is t-1 stage
    preds = []
    for ix in range(len(x)):
        xes = np.zeros((vocabs_size, 1));
        xes[x[ix]] = 1;
        state = np.tanh(np.dot(input_w, xes) + np.dot(unroll_w, state) + input_b);
        y = np.dot(output_w, state) + output_b;
        prob = softmax(y);
        pred = np.argmax(prob);
        preds.append(pred);
    return preds;


def backward(x, y, state):
    probs = {};
    bstate = {};
    ixs = {}
    loss = 0.
    binput_w = np.zeros_like(input_w)
    binput_b = np.zeros_like(input_b)
    bunroll_w = np.zeros_like(unroll_w)
    boutput_w = np.zeros_like(output_w)
    boutput_b = np.zeros_like(output_b)
    bstate[-1] = np.copy(state)

    for t in range(len(x)):
        xes = np.zeros((vocabs_size, 1))
        xes[x[t]] = 1
        ixs[t] = xes
        bstate[t] = np.tanh(np.dot(input_w, xes) + np.dot(unroll_w, bstate[t - 1]) + input_b)
        output = np.dot(output_w, bstate[t]) + output_b
        prob = softmax(output)
        probs[t] = prob
        loss += (-1 * np.log(prob[y[t], 0]))  # 求解交叉熵
    bstatenext = np.zeros_like(bstate[0])
    for t in reversed(range(len(x))):
        detla_prob = np.copy(probs[t])  # modify probs[t] -> np.copy(probs[t])
        detla_prob[y[t]] -= 1

        # 求output偏导数
        boutput_w += np.dot(detla_prob, bstate[t].T)
        boutput_b += detla_prob

        # 求state偏导数
        delta_bstate = np.dot(boutput_w.T, detla_prob) + bstatenext

        # 求tanh的偏导数
        delta_tanh = (1 - bstate[t]) * (1 + bstate[t]) * delta_bstate

        # 求unroll偏导数
        bunroll_w += (np.dot(delta_tanh, bstate[t - 1].T))

        # 求input偏导数
        xes = np.zeros((vocabs_size, 1))
        xes[x[t]] = 1
        binput_w += np.dot(delta_tanh, xes.T)
        binput_b += delta_tanh
        # 求bstatenext偏导数
        bstatenext = np.dot(bunroll_w, delta_tanh)
        # 防止爆炸
    for param in [boutput_b, boutput_w, bunroll_w, binput_b, binput_w]:
        np.clip(param, -5, 5, out=param);
    return loss, boutput_b, boutput_w, bunroll_w, binput_b, binput_w, bstate[len(x) - 1]


# 生成向量矩阵
ch_to_ix = {ch: i for i, ch in enumerate(vocabs)}
ix_to_ch = {i: ch for i, ch in enumerate(vocabs)}

mwinput, mwunroll, mwoutput = np.zeros_like(input_w), np.zeros_like(unroll_w), np.zeros_like(output_w)
mbinput, mboutput = np.zeros_like(input_b), np.zeros_like(output_b)  # memory variables for Adagrad
learning_rate = 0.1
E = 1e-8


def train(epochs):
    smooth_loss = - np.log((1. / vocabs_size)) * memory_length;
    _iter, _pos = 0, 0;
    state = np.zeros_like(input_b);
    for epoch in range(epochs):
        _pos = 0
        while _pos + memory_length < data_size:
            ##加载训练数据
            x = [ch_to_ix[ch] for ch in data[_pos:_pos + memory_length]]
            y = [ch_to_ix[ch] for ch in data[_pos + 1:_pos + memory_length + 1]]
            if _iter == 0 or _pos + memory_length >= data_size:
                state = np.zeros_like(input_b)
                _pos = 0

            if _iter % 10 == 0:
                preds = forward(x, state)
                print "predict the keys is : "
                print '_' * 30;
                print 'input -> target -> predict'
                for i, pred in enumerate(preds):
                    print '{} : {} -> {}'.format(ix_to_ch[x[i]], ix_to_ch[y[i]], ix_to_ch[pred])
                print '_' * 30
            loss, doutput_b, doutput_w, dunroll_w, dinput_b, dinput_w, state = backward(x, y, state);
            # 进行adagrad更新权重
            if _iter % 20 == 0:
                print 'iter:{}  loss: {}'.format(_iter, loss)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            for param, dparam, men in zip([output_w, output_b, input_w, input_b, unroll_w],
                                          [doutput_w, doutput_b, dinput_w, dinput_b, dunroll_w],
                                          [mwoutput, mboutput, mwinput, mbinput, mwunroll]):
                men += dparam * dparam;
                param += -((dparam * learning_rate) / np.sqrt(men + E))
            _pos += memory_length
            _iter += 1


if __name__ == '__main__':
    train(10000)
