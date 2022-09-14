import base64
import datetime
import itertools
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
import json
import jieba
import numpy as np
import pymysql
import simhash
from pymysql import cursors
from requests_html import HTML, HTMLSession

sys.setrecursionlimit(2000)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
reg1: re.Pattern = re.compile(r"[^\u4e00-\u9fa5]")
reg2: re.Pattern = re.compile(
    r"[^\u4e00-\u4e27\u4e29-\u9fa5\u000a\u0021-\u007B\u007D\u007E\u3002\uFF1F\uFF01\u3010\u3011\uFF0C\u3001\uFF1B\uFF1A\u300C\u300D\u300E\u300F\u2019\u201C\u201D\u2018\uFF08\uFF09\u3014\u3015\u2026\u2013\uFF0E\u2014\u300A\u300B\u3008\u3009]"
)

logging.getLogger('simhash').disabled = True
logger = logging.getLogger("spider")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(f'../log/{time.strftime("%Y%m%d%H%M%S", time.localtime())}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
signal.signal(signal.SIGINT, exit)
signal.signal(signal.SIGTERM, exit)
is_kill: bool = False

isPrv: bool
if sys.platform == "win32":
    isPrv = (os.system("ping -n 1 192.168.0.16") == 0)
else:
    isPrv = (os.system("ping -c 1 192.168.0.16") == 0)

with open("./setting.json", "r") as f:
    ua: dict = json.load(f)
headers: dict = ua["ua"][random.randint(0, 1)]
node: int = ua["node"]
session = None
blacklist: list = ["https://finance.sina.com.cn/oldnews", "https://www.gurufocus.cn", ".PDF"]


def exit(signum, frame):
    global is_kill
    is_kill = True
    logger.error("exiting...")
    exit()


def roll_error(db: pymysql.Connection, errno: int):
    # -1:no results
    # -2:bing search page prase error:
    # -3:other bing search error
    # -4:empty result page content
    # -5:results uploading error
    # -6:Bing search result time incorrect
    while True:
        try:
            db.ping(reconnect=True)
            with db.cursor(cursors.DictCursor) as cursor:
                cursor.execute(
                    f"update spider_head set status={errno},node={node},updateTime='{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}' where id={to_search['id']}"
                )
            db.commit()
            if db.open: db.close()
            break
        except Exception as e:
            logger.error("db connection error. retry...")
            logger.error(traceback.format_exc())
            if db.open: db.close()
            time.sleep(5)
            continue


if __name__ == '__main__':
    while True:
        try:
            db: pymysql.Connection = pymysql.connect(host=("192.168.0.16" if isPrv else "42.193.127.214"),
                                                     port=(3306 if isPrv else 2798),
                                                     db="spider",
                                                     user="root",
                                                     password="1q2w3e4rasdfspider",
                                                     charset='utf8')
            logger.info("db connectied.")
            break
        except:
            logger.error("db connection error. retry...")
            logger.error(traceback.format_exc())
            time.sleep(5)
            continue

    while True:
        # find a empty row
        try:
            db.ping(reconnect=True)
            with db.cursor(cursors.DictCursor) as cursor:
                cursor.execute("select * from spider_head where status = 0 limit 1")
                to_search: dict = (cursor.fetchall())[0]
            with db.cursor(cursors.DictCursor) as cursor:
                cursor.execute(
                    f"update spider_head set status=1,node={node},updateTime='{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}' where id={to_search['id']}"
                )
            db.commit()
            if db.open: db.close()
            logger.info("get item to search." + str(to_search))
        except IndexError as e:
            logger.warning("no item to search. exit..")
            if db.open: db.close()
            break
        except:
            logger.error("db connection error. retry...")
            logger.error(traceback.format_exc())
            if db.open: db.close()
            time.sleep(5)
            continue

        # begin serach
        params: dict = {
            "q": "\"" + to_search["shortName"] + "\"+\"" + "财务造假" + "\"",
            "filters": "ex1:\"ez5_" + str(to_search["startTS"]) + "_" + str(to_search["endTS"]) + "\""
        }
        url: str = "https://cn.bing.com/search"
        try:
            session: HTMLSession = HTMLSession(browser_args=['--no-sandbox', '--user-data-dir=../chrome_userdata'])
            result = session.get(url=url, params=params, headers=headers)
            result.encoding = 'utf-8'
            result.html.render(sleep=10)
            session.close()
            logger.info("bing page is " + str(result.url))
            raw_link_div: list = result.html.find("li.b_algo")
            if len(raw_link_div) == 0:
                if result.html.find("li.b_no"):
                    logger.warning("id " + str(to_search["id"]) + " has no result. skip.")
                    roll_error(db, -1)
                else:
                    logger.error("bing return NULL. reached limit?")
                    roll_error(db, -2)
                time.sleep(random.randint(2, 5))
                continue
        except Exception as e:
            logger.error("Other error while connect bing, rollback id " + str(to_search["id"]) + " and retry...")
            logger.error(traceback.format_exc())
            roll_error(db, -3)
            time.sleep(random.randint(2, 5))
            continue

        finally:
            if isinstance(session, HTMLSession):
                session.close()

        links_link: set = set()
        links_info: list = []
        links_time_err: int = 0
        for link_div in raw_link_div:
            page_time: int = 0
            try:
                page_time_p = link_div.find('div.b_caption p')
                assert len(page_time_p) > 0
                page_time_text: list = re.findall(r"^[0-9]{4}(?=-[0-9]{1,2}-[0-9]{1,2})", page_time_p[0].text)
                assert len(page_time_text) > 0
                page_time = int(page_time_text[0])
                if page_time != to_search['year']:
                    links_time_err += 1
                    logger.warning(f"Bing search result's time error, {page_time} should be {to_search['year']}")
            except Exception as e:
                links_time_err += 1
                logger.warning(f"Bing search result's time error, None should be {to_search['year']}")

            link: str = link_div.absolute_links.pop()
            flag: bool = True
            for i in blacklist:
                if i in link:
                    logger.info(f"{i} in blacklist, skip.")
                    flag = False
                    break
            if flag: links_link.add(link)

        if links_time_err >= len(raw_link_div):
            logger.error("Bing page time incorrect. rollback id " + str(to_search["id"]) + " and retry...")
            roll_error(db, -6)
            continue

        for i in links_link:
            links_info.append({"link": i})
        logger.info("id " + str(to_search["id"]) + " find " + str(len(links_info)) + " result(s)")

        for link_info_index in range(0, len(links_info)):
            link_info: dict = links_info[link_info_index]
            link_info["html"] = ""
            link_info["text"] = ""
            link_info["page_time"] = ""
            link_info["status"] = 0
            time.sleep(random.randint(1, 2))
            logger.info("downloading " + str(link_info["link"]) + " begin")
            session: HTMLSession = HTMLSession(browser_args=['--no-sandbox', '--user-data-dir=../chrome_userdata'])
            html: HTML = HTML(html="<html></html>")
            try:
                result = session.get(url=link_info["link"], headers=headers, timeout=10)
                result.html.html.encode("gb2312")
                html = result.html
            except UnicodeEncodeError:
                encoding = result.apparent_encoding
                if not encoding:
                    result.encoding
                if not encoding:
                    encoding = "utf-8"
                html = HTML(html=result.content, default_encoding=encoding, url=result.url)
            except Exception as e:
                logger.error("downloading " + str(link_info["link"]) + " failed, try next.")
                if isinstance(session, HTMLSession): session.close()
                continue
            try:
                if len(html.find('p')) < 10:
                    result.html.render(retries=3, timeout=10)
                    result.html.html.encode("gb2312")
                    html = result.html
            except UnicodeEncodeError:
                encoding = result.apparent_encoding
                if not encoding:
                    result.encoding
                if not encoding:
                    encoding = "utf-8"
                html = HTML(html=result.content, default_encoding=encoding, url=result.url)
            except RecursionError as e:
                logger.warning(str(link_info["link"]) + " parse failed, skip.")
                continue
            except Exception as e:
                logger.warning(str(link_info["link"]) + " render failed, using first fecth.")
            if isinstance(session, HTMLSession): session.close()
            link_info["html"] = html.html
            try:
                raw_text_tag: list = html.find("p")
            except RecursionError as e:
                logger.warning(str(link_info["link"]) + " parse failed, skip.")
                continue
            except Exception as e:
                logger.warning(str(link_info["link"]) + " unknown error, skip.")
                continue
            try:
                if len(raw_text_tag) > 500:
                    logger.warning(str(link_info["link"]) + " to many args, skip.")
                    raise ValueError
                if len(raw_text_tag) == 0:
                    logger.warning(str(link_info["link"]) + " has no content, skip.")
                    raise ValueError
            except ValueError as e:
                continue
            except Exception as e:
                logger.warning(str(link_info["link"]) + " unknown error, skip.")
                continue
            link_info["status"] = 1

            raw_text_all: list = []
            raw_text_chn: list = []
            raw_text_all_len: np.array = np.array([], dtype='uint16')
            raw_text_chn_len: np.array = np.array([], dtype='uint16')
            raw_text_len_ratio: np.ndarray = np.array([], dtype='float')
            raw_text_len_ma: np.ndarray = np.array([], dtype='float')
            raw_text_sel_bit: np.ndarray = np.array([], dtype='bool')

            # 获取正文
            for x in raw_text_tag:
                xt: str = x.text
                if xt not in raw_text_all:
                    xtl: list = [re.sub(reg2, "", i) for i in xt.split("\n")]
                    xtl_chn: list = [re.sub(reg1, "", i) for i in xtl]
                    raw_text_all.extend(xtl)
                    raw_text_chn.extend(xtl_chn)
                    raw_text_all_len = np.concatenate((raw_text_all_len, np.array([len(i) for i in xtl])))
                    raw_text_chn_len = np.concatenate((raw_text_chn_len, np.array([len(i) for i in xtl_chn])))
            raw_text_thr: int = np.mean(raw_text_all_len) * 0.5
            # ma(5)>thr & chn/all>0.5
            raw_text_len_ratio = np.divide(raw_text_chn_len,
                                           raw_text_all_len,
                                           out=np.zeros_like(raw_text_chn_len, dtype='float'),
                                           where=raw_text_all_len != 0)
            w: int = min(len(raw_text_all_len), 5)
            raw_text_len_ma = np.convolve(raw_text_all_len, np.ones(w) / w, mode='same')
            raw_text_sel_bit = (raw_text_len_ratio > 0.5) & (raw_text_len_ma >= raw_text_thr)
            text_sel = list(itertools.compress(raw_text_all, np.nditer(raw_text_sel_bit)))
            link_info["text"] = "".join(text_sel)
            #print(link_info["text"])
            logger.info("downloading " + str(link_info["link"]) + " succeed")

        # 检测失败
        check_null: int = 0
        for i in links_info:
            if i["text"] == "": check_null += 1
        if len(links_info) == 0 or len(links_info) == check_null:
            logger.error("no search page found, rollback id " + str(to_search["id"]) + " and retry...")
            roll_error(db, -4)
            continue

        # dedup
        logger.info("id " + str(to_search["id"]) + " dedup begin")
        for i in range(0, len(links_info)):
            voc_text: list = jieba.lcut(links_info[i]["text"])
            links_info[i]["simhash"] = simhash.Simhash(voc_text)
            logger.debug(str(links_info[i]["link"]) + " simhash is " + hex(links_info[i]["simhash"].value))
        dedup_index: list = []
        for i in range(0, len(links_info) - 1):
            for j in range(i + 1, len(links_info)):
                if links_info[i]["simhash"].value == 0 or (links_info[j]["simhash"].value != 0
                                                           and links_info[i]["simhash"].distance(links_info[j]["simhash"]) < 3):
                    dedup_index.append(i)
                    break
        if links_info[-1]["simhash"].value == 0: dedup_index.append(len(links_info) - 1)
        logger.debug("id " + str(to_search["id"]) + " has simliar item as " + str(dedup_index))
        links_info = [i for num, i in enumerate(links_info) if num not in dedup_index]
        logger.info("id " + str(to_search["id"]) + " dedup finished, uploading...")

        if is_kill: break

        while True:
            try:
                db.ping(reconnect=True)
                with db.cursor(cursors.DictCursor) as cursor:
                    for i in links_info:
                        cursor.execute("INSERT INTO spider.spider_links (link,link_html,link_text,head_id,node,updateTime) " +
                                       f"VALUES ('{i['link']}'," + f"'{str(base64.b64encode(i['html'].encode('utf-8')),encoding='utf-8')}'," +
                                       f"'{str(base64.b64encode(i['text'].encode('utf-8')),encoding='utf-8')}'," + f"{to_search['id']}," +
                                       f"{node}," + f"'{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}')")
                    cursor.execute(
                        f"update spider_head set status=2,node={node},updateTime='{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}' where id={to_search['id']}"
                    )
                db.commit()
                if db.open: db.close()
                logger.info("id " + str(to_search["id"]) + " upload finished.")
                break
            except Exception as e:
                if isinstance(e, pymysql.err.OperationalError) and e.args[0] in [1213, "1213"]:
                    logger.warning("id " + str(to_search["id"]) + " deadlock. retry...")
                    continue
                logger.error("id " + str(to_search["id"]) + " upload failed. retry...")
                logger.error(traceback.format_exc())
                roll_error(db, -5)
            break
    if db.open: db.close()
