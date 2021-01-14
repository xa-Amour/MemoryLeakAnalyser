#!/usr/bin/env python3
#
"""Analyze and warn of memory leaks as well as uploading result on test machine."""
"""Usage: python3 ./memory_leak_analyser.py --file memory_info.log """

import argparse
import datetime
import json
import os
import statistics
import time
from enum import Enum
from functools import reduce
from sys import platform

import math
import numpy as np
import requests
from elasticsearch import Elasticsearch
from matplotlib import pyplot

ELASTIC_URI = "https://es-xxxx"
WECHAT_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=xxxx"

CONTEXT = {}
TODAY = '{0:%Y-%m-%d}'.format(datetime.datetime.now())
YESTERDAY = datetime.date.today() - datetime.timedelta(days=1)


class Threshold(Enum):
    Notify_Eigenvalue = 0.0285  # Exceeding this value will notify
    Excess_Ratio = 0.2  # Rate of excess between two days
    Error_Ratio = 0.01  # Error Rate of eigenvalue in different ways
    Overshoot_Ratio = 0.2  # Rate of overshoot at one analyser
    Invalid_Count = 150  # Number of data less than this is invalid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch', metavar='branch', default='arsenal',
                        help='test branch for current run')
    parser.add_argument('--platform', metavar='platform', default='',
                        help='override the smoke test platform')
    parser.add_argument('--jobName', metavar='jobName', default='',
                        help='override the smoke test job name')
    parser.add_argument('--buildNumber', metavar='buildNumber', default='',
                        help='override the smoke test build number')
    parser.add_argument('--index', metavar='index', default='',
                        help='override the elastic search index')
    parser.add_argument('--file', metavar='file', default='',
                        help='override the memory usage file of smoke test')
    args = parser.parse_args()
    date = '{0:%Y_%m}'.format(datetime.datetime.now())
    index = "sdk_ng_performance_{}".format(date) if not args.index else args.index

    # Save the config into context
    CONTEXT['branch'] = args.branch
    CONTEXT['platform'] = args.platform if args.platform else platform
    CONTEXT['jobName'] = args.jobName
    CONTEXT['buildNumber'] = args.buildNumber
    CONTEXT['index'] = index
    CONTEXT['file'] = args.file

    pprint("[Index]:", index.upper())

    if not CONTEXT['file']:
        pprint("No File Input")
        return
    memory_leak_analyser(CONTEXT['file'])


def memory_leak_analyser(file):
    with open(file, "r") as f:
        lines = f.readlines()
    cache, proc_mem = [], []
    for line in lines:
        if line.startswith("i420_cache_usage"):
            cache.append(int(line.split(":")[1].strip()))
            continue
        if line.startswith("process_mem_usage"):
            proc_mem.append(int(line.split(":")[1].strip()))
            continue

    upload_to_esearch(
        memory_eigenvalue_analyser(proc_mem)[1],
        memory_overshoot_analyser(cache)[1])

    exps = {
        "Memory Leak": memory_eigenvalue_analyser,
        "Abnormal Memory Fluctuation": memory_overshoot_analyser
    }

    for exp, analyser in exps.items():
        if (exp == "Memory Leak" and analyser(proc_mem)[0]) or (
                exp == "Abnormal Memory Fluctuation" and analyser(cache)[0]):
            notify_wechat(CONTEXT['branch'], CONTEXT['platform'], exp, CONTEXT['jobName'], CONTEXT['buildNumber'])


def notify_wechat(target_branch, target_os, event, job_name, build_number, wechat_url=WECHAT_URL):
    send_info = "<font color=\"red\">Smoke Test Memory Leak.</font> Please deal with them as soon as possible.\n " \
                ">** {gen_report_date} {target_branch} {target_os} **\n " \
                ">** Type : <font color=\"warning\">{event}</font> ** \n" \
                "\n[Details](http://localhost/job/SDK_CI/job/Daily-Test/job/{job_name}/{build_number}/console)\n".format(
        gen_report_date=TODAY, target_branch=target_branch, target_os=target_os, event=event, job_name=job_name,
        build_number=build_number)
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


def gen_memory_graph(series):
    xArr, yArr = list(range(len(series))), series
    pyplot.xlabel('step')
    pyplot.ylabel('memory_usage')
    pyplot.title('Memory Leak Analyser')
    pyplot.plot(xArr, yArr, "ro")
    fig = pyplot.gcf()
    fig.savefig('memory_usage_graph.png', dpi=100)


def plot_series(series, start=500, end=3000, freq=500):
    """This is a helper function to view memory usage"""
    breakpoint = np.arange(0, len(series))
    print(len(breakpoint))
    pyplot.plot(breakpoint, series)
    pyplot.xlabel('step')
    pyplot.ylabel('memory_usage')
    pyplot.title('Memory Leak Analyser')
    pyplot.yticks(list(range(start, end, freq)))
    pyplot.fill_between(breakpoint, series, 15, color='green', alpha=0.7)
    pyplot.show()


def memory_overshoot_analyser(mem, keywords="cache_usage"):
    if len(set(mem)) == 1 and mem[0] == 0:
        return (False, mem[0])
    tday_avg = statistics.mean(mem)
    yday_avg = search_from_esearch(
        CONTEXT['branch'], CONTEXT['platform'], keywords)
    pprint("[{} Usage]: {}".format(keywords, tday_avg))
    try:
        if abs((tday_avg - yday_avg) / yday_avg) >= Threshold.Excess_Ratio.value:
            return (True, tday_avg)
    except ZeroDivisionError:
        if tday_avg and (isinstance(tday_avg, int) or isinstance(tday_avg, float)):
            return (True, tday_avg)
    return (False, tday_avg)


def memory_eigenvalue_analyser(mem):
    if len(set(mem)) == 1:  # If the set elements are the same, it will not be analyzed
        return (False, "InvalidValue")
    if len(mem) <= Threshold.Invalid_Count.value:  # If the data set is too small, it will not be analyzed
        return (False, "InvalidValue")
    if if_overshoot(mem) or calc_eigenvalue(mem) >= Threshold.Notify_Eigenvalue.value:
        pprint("[Eigenvalue]: {}".format(calc_eigenvalue(mem)))
        return (True, calc_eigenvalue(mem))
    return (False, calc_eigenvalue(mem))


def calc_eigenvalue(mem):
    """
    Description:
        Ridge regression
    Note:
        The mem need to be converted to a special data structure like [[1.0, 1],[1.0, 2]...]
    Website:
        http://www.cuijiahua.com/
    """
    xArr, yArr = [[1.0, space] for space in list(range(len(mem)))], mem
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        pprint("This matrix is singular, cannot do inverse")
        return
    double_eigenvalue, line_pred_avg = (
                                               xTx.I * (xMat.T * yMat))[1, 0], (
                                               line_trend(mem) + line_fit(mem)) / 2  # Regression coefficient
    eigenvalue = double_eigenvalue if abs(line_pred_avg -
                                          double_eigenvalue) <= Threshold.Error_Ratio.value else line_pred_avg
    return round(eigenvalue, 4)


def upload_to_esearch(proc_mem_eigenvalue, cache):
    pprint('[INFO]: Upload to Elastic Search')
    HEADERS = {"Content-Type": "application/json"}
    AUTH = ('elastic', 'auth_xxxx')
    url = "{}/{}/_doc".format(ELASTIC_URI, CONTEXT['index'])
    with requests.Session() as s:
        s.auth = AUTH
        s.headers.update(HEADERS)
        send_info = {
            "branch": CONTEXT['branch'],
            "platform": CONTEXT['platform'],
            "date": TODAY,
            "timestamp": int(time.time() * 1000),
            "build_num": os.environ.get('BUILD_NUMBER', 0),
            "proc_mem_eigenvalue": proc_mem_eigenvalue,
            "cache_usage": cache,
        }
        try:
            res = s.post(url, data=json.dumps(send_info))
        except Exception as ex:
            pass
    pprint('[INFO]: Upload Done.')


def search_from_esearch(branch, platform, keywords):
    pprint('[INFO]: Search From Elastic Search')
    es = Elasticsearch(
        [ELASTIC_URI],
        http_auth=(
            "username",
            "password"),
        scheme="https")
    body = {'query': {'bool': {'must': [{'match': {'branch': branch}}, {
        'match': {'platform': platform}}, {'match': {'date': YESTERDAY}}]}}}
    try:
        res = es.search(index=CONTEXT['index'], body=body)
    except Exception as ex:
        pass

    try:
        hits = res.get("hits").get("hits")
        avg_val = reduce(lambda x, y: x + y,
                         [int(item.get("_source").get(keywords)) for item in hits]) / len(hits)
    except ZeroDivisionError:
        pass
    except Exception as ex:
        pprint(ex)
        return 0
    pprint('[INFO]: Search Done.')
    return avg_val


def line_fit(mem):
    xArr, yArr = list(range(len(mem))), mem
    N = float(len(xArr))
    sx, sy, sxx, syy, sxy = 0, 0, 0, 0, 0
    for elec in range(0, int(N)):
        sx += xArr[elec]
        sy += yArr[elec]
        sxx += xArr[elec] * xArr[elec]
        syy += yArr[elec] * yArr[elec]
        sxy += xArr[elec] * yArr[elec]
    der = (sy * sx / N - sxy) / (sx * sx / N - sxx)
    fact = (sy - der * sx) / N
    r = abs(sy * sx / N - sxy) / \
        math.sqrt((sxx - sx * sx / N) * (syy - sy * sy / N))
    return der


def line_trend(mem):
    xArr, yArr = list(range(len(mem))), mem
    [der, fact] = np.polyfit(xArr, yArr, 1)
    return der


def if_overshoot(mem):
    fb, eb = len(mem) // 4, len(mem) // 8
    mem_avg = sum(mem) / len(mem)

    def _inner(arry, qb, overshoot_ratio=Threshold.Overshoot_Ratio.value):
        return abs(
            sum(arry[:qb]) / qb - sum(arry[-qb:]) / qb) / (sum(arry[:qb]) / qb) >= overshoot_ratio

    return any([_inner(mem, tan) for tan in [fb, eb]])


def pprint(*args, **kwargs):
    print(*args, **kwargs, flush=True)


if __name__ == '__main__':
    main()
