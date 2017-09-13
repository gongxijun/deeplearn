# -*- coding:utf-8 -*-
"""
@author: gxjun
@file: elemnet.py
@time: 17-9-5 上午11:40
"""
import numpy as np

""""文本转换车成向量"""
text = open('input.txt', 'r').read()  # should be simple plain text file
# 将文本转换成向量
chars = list(set(text))
text_size, vocab_size = len(text), len(chars)
print 'text has %d characters, %d unique.' % (text_size, vocab_size)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

""""
helloworld
h -> [1,0,0,0,0,0,0]^T
e -> [0,1,0,0,0,0,0]^T
l -> [0,0,1,0,0,0,0]^T
o -> [0,0,0,1,0,0,0]^T
w -> [0,0,0,0,1,0,0]^T
r -> [0,0,0,0,0,1,0]^T
d -> [0,0,0,0,0,0,1]^T
""""定义网络"""

hidden_neuron_numbers = 100;
seq_length = 9  # 句子记忆长度
# 定义隐藏层
hidden_weight = np.random.randn(hidden_neuron_numbers, vocab_size) * 0.01;
hidden_biase = np.zeros((hidden_neuron_numbers, 1));
# 定义展开中间层
unroll_weight = np.random.randn(hidden_neuron_numbers, hidden_neuron_numbers) * 0.01;
# 定义输出层
output_weight = np.random.randn(vocab_size, hidden_neuron_numbers) * 0.01;
output_biase = np.zeros((vocab_size, 1))


def softmax(feature):
    factor = np.exp(feature)
    return factor / np.sum(factor)


def forward(state, x):
    """"定义前馈网络"""
    ixes = []
    for i in range(len(x)):
        vec_x = np.zeros((vocab_size, 1));
        vec_x[x[i]] = 1;
        state = np.tanh(np.dot(hidden_weight, vec_x) + np.dot(unroll_weight, state) + hidden_biase);
        output = np.dot(output_weight, state) + output_biase;
        prob = softmax(output);
        pred = np.argmax(prob)  # Y值
        ixes.append(pred)
    return ixes


def backward(x, y, state):
    loss = 0.
    bvec_x, bstate = {}, {}
    probs = {}
    bstate[-1] = np.copy(state)
    for i in xrange(len(x)):
        bvec_x = np.zeros((vocab_size, 1));
        bvec_x[x[i]] = 1;
        bstate[i] = np.tanh(np.dot(hidden_weight, bvec_x) + np.dot(unroll_weight, bstate[i - 1]) + hidden_biase);
        output = np.dot(output_weight, bstate[i]) + output_biase;
        prob = softmax(output);
        probs[i] = prob;
        pred = np.argmax(prob)  # Y值
        loss += (-1*np.log(prob[y[i], 0]));  # softmax 求交叉熵 sum(p*(np.log(q)))
    """计算梯度grad,对hidden_weight , unroll_weight , output_weight,hidden_biase , output_biase"""
    bhidden_weight, bunroll_weight, boutput_weight = np.zeros_like(hidden_weight), np.zeros_like(
        unroll_weight), np.zeros_like(output_weight)
    bhidden_biase = np.zeros_like(hidden_biase);
    boutput_biase = np.zeros_like(output_biase);
    bstate_next = np.zeros_like(bstate[0]);
    for t in reversed(xrange(len(x))):  # 剥洋葱大法
        prob = np.copy(probs[t]);
        prob[y[t]] -= 1;  # 要使目标不断接近1
        # 更新 output w,b 分别秋偏导，然后乘以误差
        boutput_weight += (np.dot(prob, bstate[t].T));
        boutput_biase += prob
        # 更新unroll部分的w
        delta_state = np.dot(boutput_weight.T, prob) + bstate_next
        #tanh求导
        qdtanh = (1 - bstate[t]) * (1 + bstate[t]) * delta_state
        bunroll_weight += (np.dot(qdtanh, bstate[t - 1].T))
        # 更新hidden w,b
        ix = np.zeros((vocab_size, 1));
        ix[x[t]] = 1;
        bhidden_weight += (np.dot(qdtanh, ix.T))
        bhidden_biase += qdtanh;
        # 更新bstate_next;
        bstate_next = np.dot(bunroll_weight, qdtanh);
        # 防止梯度爆炸
    for param in [hidden_weight, unroll_weight, output_weight, hidden_biase, output_biase]:
        np.clip(param, -5, 5, out=param)
    return loss, bhidden_weight, bunroll_weight, boutput_weight, bhidden_biase, boutput_biase, bstate[len(x) - 1]


mWxh, mWhh, mWhy = np.zeros_like(hidden_weight), np.zeros_like(unroll_weight), np.zeros_like(output_weight)
mbh, mby = np.zeros_like(hidden_biase), np.zeros_like(output_biase)  # memory variables for Adagrad
learning_rate = 0.1


def train(epochs):
    _iter = 0;
    state = np.zeros((hidden_neuron_numbers, 1));
    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # 初始化loss
    for epoch in range(epochs):
        start_pos = 0;
        while start_pos + seq_length < text_size:
            # 初始化 state
            if start_pos + seq_length >= text_size or _iter == 0:
                state = np.zeros((hidden_neuron_numbers, 1));
                start_pos = 0;
            # 装载训练数据
            x = [char_to_ix[ch] for ch in text[start_pos:start_pos + seq_length]]
            y = [char_to_ix[ch] for ch in text[start_pos + 1:start_pos + seq_length + 1]]
            if _iter % 10 == 0:
                preds = forward(state, x)
                print "predict the keys is : "
                print '_'*30;
                print 'input -> target -> predict'
                for i,pred in enumerate(preds):
                    print '{} : {} -> {}'.format(ix_to_char[x[i]],ix_to_char[y[i]],ix_to_char[pred])
                print '_'*30
            # 使用adagrad方法更新learn_rate
            # 计算loss
            loss, bhidden_weight, bunroll_weight, boutput_weight, bhidden_biase, boutput_biase, state = backward(x, y,
                                                                                                                 state);
            if _iter % 20==0:
                print 'iter:{}  loss: {}'.format(_iter, loss)
            smooth_loss = smooth_loss * 0.999 + loss * 0.001
            for param, dparam, mem in zip([hidden_weight, unroll_weight, output_weight, hidden_biase, output_biase],
                                          [bhidden_weight, bunroll_weight, boutput_weight, bhidden_biase,
                                           boutput_biase],
                                          [mWxh, mWhh, mWhy, mbh, mby]):
                mem += dparam * dparam
                param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

            start_pos += seq_length  # move data pointer
            _iter += 1  # iteration counter


if __name__ == '__main__':
    train(10000)
