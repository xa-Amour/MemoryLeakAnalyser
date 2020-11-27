# today = '{0:%Y-%m-%d}'.format(datetime.datetime.now())
# today = datetime.date.today()
# today = datetime.date.today()
# yesterday = datetime.date.today() - datetime.timedelta(days=1)
# print(today)
# print(yesterday)

# import statistics
# cache_usage = [5, 6, 2, 9]
# cache_usage_avl =statistics.mean(cache_usage)
# print(cache_usage_avl)
# lst = [20, 23, 23, 24, 24, 24, 24, 25, 23, 25, 25, 25, 25, 26, 26, 26, 26, 26, 24, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 25, 24, 25, 25, 25, 25, 25, 25, 25, 25, 26, 27, 25, 24, 24, 25, 25, 26, 26, 27, 23, 25, 25, 25, 26, 26, 26, 26, 26, 27, 28, 29, 29, 29, 29, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 23, 23, 25, 25, 24, 26, 26, 26, 25, 25, 25, 26, 25, 26, 25, 24, 24, 24, 24, 25, 25, 25, 26]
# print(lst)

import statistics
import argparse
from functools import reduce
import pprint
import numpy as np
import datetime
import json
import os
import time

import requests
from elasticsearch import Elasticsearch

ELASTIC_URI = "https://es-xxxx"
WECHAT_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"
# today = '{0:%Y-%m-%d}'.format(datetime.datetime.now())

# CONTEXT = {'index': "performance_smoke_test_",
#            'branch': "arsenal", 'platform': "linux", }


def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


today = '{0:%Y-%m-%d}'.format(datetime.datetime.now())
yday = datetime.date.today() - datetime.timedelta(days=1)
date = '{0:%Y_%m}'.format(datetime.datetime.now())
index = "sdk_ng_performance_{}".format(date)
# print(date)
# print(today)
# print(yday)

CONTEXT = {'index': index,
           'branch': "arsenal", 'platform': "windows", }


def search_data():
    pprint('[INFO]: Update memory usage info to elastic search')
    HEADERS = {"Content-Type": "application/json"}
    AUTH = ('elastic', 'auth_xxxx')
    url = "{}/{}/_doc".format(ELASTIC_URI, CONTEXT['index'])
    with requests.Session() as s:
        s.auth = AUTH
        s.headers.update(HEADERS)
        query1 = {'bool': {'must': [{'match': {'branch': "arsenal"}}, {
            'match': {'platform': "windows"}}, {'match': {'date': "2020-11-23"}}]}}

        try:
            res = s.get(ELASTIC_URI, data=json.dumps(query1))
        except Exception as ex:
            pass
    print(res.status_code)
    print(res.content)
    print('[INFO]: Update Done.')


# search_data()


def search():
    query1 = {'bool': {'must': [{'match': {'branch': "arsenal"}}, {
        'match': {'platform': "windows"}}, {'match': {'date': "2020-11-23"}}]}}

    es_query = json.dumps(query1)
    url = "{}/{}/_doc".format(ELASTIC_URI, CONTEXT['index'])
    print(es_query)
    print(url)
    r = requests.get(url, params=es_query)
    results = json.loads(r.text)
    print(results)
    # data = [res['_source']['api_id'] for res in results['hits']['hits'] ]
    # print("results: %d" % len(data))
    # pprint(data)

# search()


# def update_data_to_elastic_search(proc_mem_peak, cache_usage):
#     pprint('[INFO]: Update memory usage info to elastic search')
#     HEADERS = {"Content-Type": "application/json"}
#     AUTH = ('elastic', 'auth_xxxx')
#     url = "{}/{}/_doc".format(ELASTIC_URI, CONTEXT['index'])
#     with requests.Session() as s:
#         s.auth = AUTH
#         s.headers.update(HEADERS)
#         test_case_info = {
#             "branch": CONTEXT['branch'],
#             "platform": CONTEXT['platform'],
#             "date": "2020-11-23",
#             "timestamp": int(time.time() * 1000),
#             "build_num": os.environ.get('BUILD_NUMBER', 0),
#             "proc_mem_peak": proc_mem_peak,
#             "cache_usage": cache_usage,
#         }
#         try:
#             res = s.post(url, data=json.dumps(test_case_info))
#         except Exception as ex:
#             pass
#     print('[INFO]: Update Done.')


# update_data_to_elastic_search("1.1", "27")

# for key, value in  CONTEXT.items():
#     print(key, value)

# def func():
#     return 1,2

# print(func()[1])

# a = 10.33
# fact = 0
#
# print(abs((a-fact) % fact))

# data = np.random.rand(30, 50, 40, 20)
# # print(data)
# first_derivative = np.gradient(data)
# # print(first_derivative)
#
#
# def fct(x):
#     y = x ** 3 + 1
#     return y


# grad_fct = grad(fct)
# print(grad_fct(1.0))
# def get_data_from_elastic_search(branch, platform, date, doc_type):
#     pprint('[INFO]: Get memory usage info from elastic search')
#     es = Elasticsearch(
#         ["https://es-xxxx"],
#         http_auth=(
#             "username",
#             "123456"),
#         scheme="https")
#     try:
#         result = es.search(index=index, doc_type='doc_type')
#     except Exception as ex:
#         pass
#     print(json.dumps(result, indent=2, ensure_ascii=False))
#     print('[INFO]: Get Done.')
# es = Elasticsearch(
#     ["https://es-xxxx"],
#     http_auth=(
#         "username",
#         "password"),
#     scheme="https")
# body = {
#     "query":{
#         "bool":{
#             "filter": [
#                     {"match_all": {}},
#                     {"match_phrase": {"branch": "arsenal"}},
#                     {"match_phrase": {"platform": "linux"}}
#             ],
#             "must": [],
#             "must_not": [],
#             "should": []
#         },
#         "script_fields": {},
#         "size": 500
#     }
# }
# import pprint
#
# body = {
#     "query": {
#         "bool": {
#             "must": [
#                 {
#                     "match": {
#                         "branch": "arsenal"
#                     }
#                 },
#                 {
#                     "match": {
#                         "platform": "linux"
#                     }
#                 },
#                 {
#                     "match": {
#                         "date": "2020-11-23"
#                     }
#                 }
#
#             ]
#         }
#     }
# }
#
# print(body)
# res = es.search(index=index, body=body)
# pprint.pprint(res)
# print(type(res))
# hits = res.get("hits").get("hits")
# hits_len = len(hits)
# print(hits_len)
# for i in hits:
#     cache_usage = i.get("_source").get("cache_usage")
#     print(cache_usage)
# pprint.pprint(hits)
# cache_usage_avg = res.get("hits").get("hits")# .get("_source").get("cache_usage")
# pprint.pprint(cache_usage_avg)

# lst = [int(i.get("_source").get("cache_usage")) for i in hits]
# print(lst)
# lsat_cache_usage_avg = reduce(lambda x, y: x+y, [int(item.get("_source").get("cache_usage")) for item in hits]) / hits_len
# print(lsat_cache_usage_avg)
# pprint.pprint(res.get("hits").get("hits"))

# print({"1":"a", "2":"fact"}.get("1"))
# memory = [0,0,0]
# def memory_peak_analyser(memory):
#     if len(set(memory)) == 1:
#         return False
#     print("[Process Memory Usage Peak]: {}".format(memory))
#     return True
#
# memory_peak_analyser(memory)

# today = '{0:%Y-%m-%d}'.format(datetime.datetime.now())

import numpy as np
import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5]
# x = np.array(x)
# print('x is :\n', x)
# num = [22, 24, 25, 23, 22]
# y = np.array(num)
# print('y is :\n', y)
# # 用3次多项式拟合
# f1 = np.polyfit(x, y, 3)
# print('f1 is :\n', f1)
#
# p1 = np.poly1d(f1)
# print('p1 is :\n', p1)
#
# #也可使用yvals=np.polyval(f1, x)
# yvals = p1(x)  #拟合y值
# print('yvals is :\n',yvals)
# #绘图
# plot1 = plt.plot(x, y, 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4) #指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()



import numpy as np



def linear_regression(x, y):
    # y=bx+a，线性回归
    num = len(x)
    fact = (np.sum(x * y) - num * np.mean(x) * np.mean(y)) / (np.sum(x * x) - num * np.mean(x) ** 2)
    a = np.mean(y) - fact * np.mean(x)
    return np.array([fact, a])


def f(x):
    return 2 * x + 1


x = np.linspace(-5, 5)
print(x)
print(len(x))
y = f(x) + np.random.randn(len(x))  # 加入噪音
print(y)

# x = [1, 2, 3, 4, 5]
# y = f(x) + np.random.randn(len(x))
y_fit = np.polyfit(x, y, 1)  # 一次多项式拟合，也就是线性回归
print(linear_regression(x, y))
print(y_fit)

x_lst = list(range(10))
print(x_lst)


# X=[ 1 ,2  ,3 ,4 ,5 ,6]
Y=[19, 22, 23, 24, 24, 24, 24, 25, 24, 25, 25, 25, 25, 25, 25, 25, 24, 24, 23, 24, 26, 25, 29, 25, 30, 25, 25, 25, 25, 25, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 22, 24, 25, 25, 26, 26, 26, 26, 26, 23, 25, 25, 26, 26, 27, 26, 26, 26, 27, 28, 28, 29, 29, 29, 23, 23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 24, 25, 25, 26, 25, 25, 25, 26, 26, 26, 25, 26, 26, 25, 24, 24, 24, 25, 25, 25, 29]
X = list(range(len(Y)))
z1 = np.polyfit(X, Y, 1)  #一次多项式拟合，相当于线性拟合
p1 = np.poly1d(z1)
print(z1)  #[ 1.          1.49333333]

print("-"*100)

print(z1[0])
print(p1)  # 1 x + 1.493

print("*"*50)


def calc_peak(lst):
    xArr,yArr = [[1.0, space] for space in list(range(len(lst)))],lst
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    double_peak = (xTx.I * (xMat.T*yMat))[1,0]
    line_pred_avg = (_line_trend(lst) + _line_fit(lst)) / 2
    peak = double_peak if abs(  line_pred_avg - double_peak) <= 0.01 else  line_pred_avg
    return peak

def _line_trend(lst):
    x,y = list(range(len(lst))),lst
    [der,  fact] = np.polyfit(x, y, 1)
    return der

def _line_fit(lst):
    import math
    x,y = list(range(len(lst))),lst
    N = float(len(x))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for elec in range(0, int(N)):
        sx += x[elec]
        sy += y[elec]
        sxx += x[elec] * x[elec]
        syy += y[elec] * y[elec]
        sxy += x[elec] * y[elec]
    der= (sy * sx / N - sxy) / (sx * sx / N - sxx)
    fact = (sy - der * sx) / N
    r = abs(sy * sx / N - sxy) / math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    return der

lst = [19, 22, 23, 24, 24, 24, 24, 25, 24, 25, 25, 25, 25, 25, 25, 25, 24, 24, 23, 24, 26, 25, 29, 25, 30, 25, 25, 25, 25, 25, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 22, 24, 25, 25, 26, 26, 26, 26, 26, 23, 25, 25, 26, 26, 27, 26, 26, 26, 27, 28, 28, 29, 29, 29, 23, 23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 24, 25, 25, 26, 25, 25, 25, 26, 26, 26, 25, 26, 26, 25, 24, 24, 24, 25, 25, 25, 29]
print("calc",calc_peak(lst))


def linear_regression(x,y):
    #y=bx+a，线性回归
    num=len(x)
    b=(np.sum(x*y)-num*np.mean(x)*np.mean(y))/(np.sum(x*x)-num*np.mean(x)**2)
    a=np.mean(y)-b*np.mean(x)
    return np.array([b,a])
def f(x):
    return 0.01453 * x + 24.27
x=np.linspace(-5,5)
y=f(x)+np.random.randn(len(x))#加入噪音
y_fit=np.polyfit(x,y,1)#一次多项式拟合，也就是线性回归
print(y_fit)




# from scipy.optimize import leastsq
# import pylab as pl
#
# def func(x, p):
#     """
#     数据拟合所用的函数: A*sin(2*pi*k*x + theta)
#     """
#     A, k, theta = p
#     return A*np.sin(2*np.pi*k*x+theta)
#
# def residuals(p, y, x):
#     """
#     实验数据x, y和拟合函数之间的差，p为拟合需要找到的系数
#     """
#     return y - func(x, p)
#
# x = np.linspace(0, -2*np.pi, 100)
# A, k, theta = 10, 0.34, np.pi/6 # 真实数据的函数参数
# y0 = func(x, [A, k, theta]) # 真实数据
# y1 = y0 + 2 * np.random.randn(len(x)) # 加入噪声之后的实验数据
#
# p0 = [7, 0.2, 0] # 第一次猜测的函数拟合参数
#
# # 调用leastsq进行数据拟合
# # residuals为计算误差的函数
# # p0为拟合参数的初始值
# # args为需要拟合的实验数据
# plsq = leastsq(residuals, p0, args=(y1, x))
#
# print u"真实参数:", [A, k, theta]
# print u"拟合参数", plsq[0] # 实验数据拟合后的参数
#
# pl.plot(x, y0, label=u"真实数据")
# pl.plot(x, y1, label=u"带噪声的实验数据")
# pl.plot(x, func(x, plsq[0]), label=u"拟合数据")
# pl.legend()