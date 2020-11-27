import math
import numpy as np

def linefit(x , y):
    N = float(len(x))
    sx,sy,sxx,syy,sxy=0,0,0,0,0
    for i in range(0,int(N)):
        sx  += x[i]
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    a = (sy*sx/N -sxy)/( sx*sx/N -sxx)
    b = (sy - a*sx)/N
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N))
    return a,b,r


Y = [19, 22, 23, 24, 24, 24, 24, 25, 24, 25, 25, 25, 25, 25, 25, 25, 24, 24, 23, 24, 26, 25, 29, 25, 30, 25, 25, 25, 25, 25, 23, 23, 24, 24, 24, 24, 24, 24, 25, 25, 25, 22, 24, 25, 25, 26, 26, 26, 26, 26, 23, 25, 25, 26, 26, 27, 26, 26, 26, 27, 28, 28, 29, 29, 29, 23, 23, 23, 23, 23, 23, 24, 24, 25, 25, 25, 24, 25, 25, 26, 25, 25, 25, 26, 26, 26, 25, 26, 26, 25, 24, 24, 24, 25, 25, 25, 29]
X=list(range(len(Y)))
a,b,r=linefit(X,Y)
print("X=",X)
print("Y=",Y)
print("拟合结果: y = %10.5f x + %10.5f , r=%10.5f" % (a,b,r) )
print(a)
print(b)


print("-"*100)

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

xArr,yArr = loadDataSet('ex0.txt')
# print("xArr", xArr)
# print("yArr", yArr)

def standRegres(lst):
    xArr = [[1.0, i] for i in list(range(len(lst)))]
    yArr = lst
    print(xArr)
    print(yArr)
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print ("This matrix is singular, cannot do inverse")
        return
    ws = (xTx.I * (xMat.T*yMat))[1,0]
    return ws



# print("standRegres",standRegres([[1.0, 0.067732], [1.0, 0.42781], [1.0, 0.995731]], [3.176513, 3.816464, 4.550095]  ))
# print("standRegres",standRegres([[1.0, 1], [1.0, 2], [1.0, 3]], [3, 6, 6]  ))
#
# origin_lst = []
#
# xArr = [[1.0, i] for i in list(range(len(Y)))]
# print(xArr)
# yArr = Y

print("standRegres",standRegres(Y))
# standRegres [[2.42716179e+01]
#  [1.45302967e-02]]

print("-"*100)

# sample points
# X = [0, 5, 10, 15, 20]


# solve for a and b
def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# solution
a, b = best_fit(X, Y)
#best fit line:
#y = 0.80 + 0.92x

# plot points and fit line
import matplotlib.pyplot as plt
plt.scatter(X, Y)
yfit = [a + b * xi for xi in X]
plt.plot(X, yfit)
from functools import reduce

lst = [1,2,3,4,5,6,7,8,9]
# print(len(lst)//4)
# print(lst[-2:])

def if_overshoot(lst):
    qb = len(lst) // 4
    lst_avg = sum(lst) / len(lst)
    return abs(sum(lst[-qb:]) / qb - lst_avg) / lst_avg >= 0.2

print(if_overshoot(Y))

# print(sum(lst))
import requests
import datetime
import json
WECHAT_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"
today = '{0:%Y-%m-%d}'.format(datetime.datetime.now())

def notify_wechat(target_branch, target_os, event, job_name, build_number, wechat_url=WECHAT_URL):
    send_info = "<font color=\"red\">Smoke Test Memory Leak.</font> Please deal with them as soon as possible. \n " \
                ">** {gen_report_date} {target_branch} {target_os} **\n " \
                ">** Type : <font color=\"warning\">{event}</font> ** \n" \
                "\n[前往CI查看详情](http://localhost/job/SDK_CI/job/Daily-Test/job/{job_name}/{build_number}/console)\n".format(
                    gen_report_date=today, target_branch=target_branch, target_os=target_os, event=event, job_name=job_name, build_number=build_number)
    payload = {
        "msgtype": "markdown",
        "agentid": 1,
        "markdown": {
            "content": send_info
        },
        "safe": 0,
        "enable_id_trans": 0,
        "enable_duplicate_check": 0
    }
    requests.post(wechat_url, data=json.dumps(payload))

notify_wechat("arsenal","linux","Memory Overshoot Exception", "Smoke_Test_Linux_Analysis_release", "4060")

index = "sdk_ng_performance_"
print("[Index]:", index.upper())

from enum import Enum, unique


# class Threshold(Enum):
#     Excess_Ratio = 0.2 # Rate of excess between two days
#     Error_Ratio = 0.01 # Error rate of peak in different ways
#     Notify_Peak = 0.2  # Exceeding this value will notify wechat
#     Overshoot_Ratio = 0.25 # Rate of overshoot at one time
#
# print(Threshold.Excess_Ratio.value)
# print(Threshold.Error_Ratio.value)
# print(Threshold.Notify_Peak.value)
# print(Threshold.Overshoot_Ratio.value)

# from sys import platform
#
# print(platform)
