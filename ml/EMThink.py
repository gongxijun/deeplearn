#-*- coding:utf-8 -*-
"""
@author: gxjun
@file: EMThink.py
@time: 17-9-13 下午5:37
"""
#http://www.jianshu.com/p/1121509ac1dc
import numpy as np
#每轮使用硬币以及统计情况
#参考
exm_list = [[1,3,2],[2,2,3],[1,1,4],[2,3,2],[1,2,3]];
debug=True;
def calc_prob(exm_list):
    p_1 =p_2=0;
    #分子，分母
    #from IPython.core.debugger import Tracer; Tracer()()
    pi_count=np.zeros((2,2));
    for col in exm_list:
        pi_count[col[0]-1][0]+=col[1];
        pi_count[col[0]-1][1]+=5;
    p_1 = pi_count[0][0]/pi_count[0][1];
    p_2 = pi_count[1][0]/pi_count[1][1]
    if debug:
        print pi_count
    return p_1 , p_2;

def mle(exm_list,pred_p1 ,pred_p2):
    #exm_list: 实验统计情况
    #pred_p1 : 预测的p1
    #pred_p2 : 预测的p2
    pred_pi = np.array([pred_p1,pred_p2])
    new_exm_list=np.zeros((5,4));
    for idx ,col in enumerate(exm_list):
        for ind , pred in enumerate(pred_pi):
            col_prob_p_coin=1.;
            for i in range(col[1]+col[2]):
                col_prob_p_coin*=pred if i < (col[1]) else (1 -pred);
            new_exm_list[idx][ind]=col_prob_p_coin;
        new_exm_list[idx][2]=exm_list[idx][1];
        new_exm_list[idx][3]=exm_list[idx][2];
    if debug:
        print new_exm_list;
        print '-'*60;
    for idx , col in enumerate(new_exm_list):
        new_exm_list[idx][0]/=(new_exm_list[idx][0]+new_exm_list[idx][1]);
        new_exm_list[idx][1]=1-new_exm_list[idx][0];
    if debug:
        print new_exm_list
        print '-'*60;
    prob_p= np.zeros((2,2));
    new_pro_list = np.zeros((5,2,4));
    for idx , col in enumerate(new_exm_list):
        for i in range(2):  #硬币
            for j in range(2): #正反面
                new_pro_list[idx][i][j]=(new_exm_list[idx][i]*new_exm_list[idx][2+j]);
                prob_p[i][j]+=new_pro_list[idx][i][j];
                new_pro_list[idx][i][2+j]=new_exm_list[idx][2+j]
    if debug:
        print new_pro_list;
        print '-'*60;
    pred_p1 =prob_p[0][0]/(prob_p[0][0]+prob_p[0][1]);
    pred_p2 =prob_p[1][0]/(prob_p[1][0]+prob_p[1][1]);
    return pred_p1 ,pred_p2;
if __name__ == '__main__':
    print calc_prob(exm_list);
    pred_p1=0.4 ;
    pred_p2=0.5;
    print pred_p1 , pred_p2;
    for epoch in range(10):
        pred_p1,pred_p2 = mle(exm_list,pred_p1,pred_p2)
        print pred_p1 , pred_p2;