# coding:utf-8
import base64
import os
import pickle
import re
import sys
from time import sleep
import traceback
import pandas as pd
import pymysql
import logging
import datetime
from bert_serving.client import BertClient
import numpy as np
from requests_html import HTML
import jieba.analyse

sys.setrecursionlimit(2000)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
isPrv: bool
if sys.platform == "win32":
    isPrv = (os.system("ping -n 1 192.168.0.16") == 0)
else:
    isPrv = (os.system("ping -c 1 192.168.0.16") == 0)
bc: BertClient = BertClient()

logging.getLogger('simhash').disabled = True
logger = logging.getLogger("spider")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s]:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
fh = logging.FileHandler(f'../log/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_bert.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

pattern_list: list = [
    "内幕", "内控", "审计", "商誉", "虚假", "减持", "违规", "会计师", "审计师", "索赔案", "控制", "爆雷", "调增", "索赔", "痼疾", "欺诈", "暴跌", "禁入", "违法", "违约", "下降", "风险", "分析程序",
    "缓刑", "处罚金", "罚金", "证监会", "核算", "确认", "评估", "证据", "虚构", "关联", "存在", "监管", "无法", "无法保证", "资金占用", "让渡", "索偿", "退市", "减值", "计提", "前后矛盾", "矛盾", "异常",
    "准则", "警告", "占用", "警示", "予以", "承诺", "稽查", "核查", "重大", "举报", "周转率", "现金流量", "现金流", "维权", "亏损", "大额", "质押", "抵押", "保留意见", "疑点", "资本运作", "爆料", "涉嫌",
    "事先", "骗得", "执行", "舞弊", "造假", "处罚", "虚增", "虚减", "预谋", "伪造", "变造", "虚高", "危机", "待解", "罕见", "困难", "合理", "特别", "典型", "触及", "掏空", "怀疑", "罚款", "删除",
    "修改", "虚低", "隐瞒", "给予", "处于", "立案", "罪状", "主观", "恶意", "账实", "不符", "查明", "关注", "犯罪", "违规披露", "不披露", "重要信息", "震惊", "圆谎", "撒谎", "做空", "刑事", "未计提",
    "未被", "披露", "对冲", "核销", "缺位", "混乱", "缺失", "漠视", "肆意", "严惩", "授意", "指挥", "手法", "消化", "配合", "掩盖", "隐蔽", "篡改", "策划", "欺骗", "代价", "暴露", "远高于", "远低于",
    "调查", "疑云", "存疑", "疑虑", "何以", "畸高", "畸低", "真实", "为何", "质疑", "陷入", "哗然", "漏洞", "迹象", "惊雷", "问询", "谈话", "采取", "鲜明", "对比", "相称", "猜疑", "通牒", "变脸",
    "紧张", "舆论", "终止", "调整", "疑", "隐秘", "警惕", "撤资", "逾期", "输送", "认定", "蹊跷", "野蛮", "反常", "不合理", "不专业", "挪用", "未按规定", "未披露", "不记账", "责令", "改正", "担忧",
    "困扰", "担心", "困惑", "巨额", "热议", "失实", "遭受", "损失", "超额", "不了", "负面", "趋紧", "恶化", "不确定", "起诉", "冻结", "冲击", "远远", "不真实", "远超", "颓势", "疑惑", "费解", "相差",
    "甚远", "难度", "归咎", "难辞其咎", "诡异", "忌讳", "疑问", "颇为", "不解", "过快", "过慢", "过大", "过小", "压力", "生存", "相干", "相关", "逻辑", "利益", "知悉", "组织", "实施", "暗藏", "猫腻",
    "叫板", "实名", "未真实", "安排", "无法表示意见", "指向", "注目", "疑似", "自杀", "究竟", "证伪", "整治", "深陷", "波动", "监控", "重点", "容忍", "查处", "巨亏", "很难", "令人", "引起", "下调",
    "遭遇", "强制", "惊天", "竟然", "抽身", "套现", "报案", "受理", "曝光", "诉讼", "受损", "至暗", "狙击", "遭到", "保真", "保证", "凭空", "涉及", "臭名昭著", "突然", "察觉", "拙劣", "把戏", "财技",
    "洗澡", "存贷", "谎言", "撤回", "震慑", "分辨", "消失", "一纸", "回避", "自查", "未能", "排查", "更正", "缺陷", "整改", "充分", "顶格", "一审", "二审", "终审", "姑息", "经查", "检查", "所涉",
    "处分", "涉", "合规", "有效", "判决", "不正当", "合法", "正当", "完全", "防范", "假", "跨期", "保留意见", "否定意见", "否定", "披露", "意见", "高于", "低于", "大于", "小于"
]
alias: pd.DataFrame = pd.read_csv("../model/alias.csv", index_col="symbol")
jieba.load_userdict("../model/word2vec/userdict.txt")
jieba.initialize()

if __name__ == '__main__':
    while True:
        try:
            db: pymysql.Connection = pymysql.connect(host=("192.168.0.16" if isPrv else "42.193.127.214"),
                                                     port=(3306 if isPrv else 2798),
                                                     db="spider_old_1",
                                                     user="root",
                                                     password="1q2w3e4rasdfspider",
                                                     charset='utf8')
            logger.info("db connectied.")
            break
        except:
            logger.error("db connection error. retry...")
            logger.error(traceback.format_exc())
            sleep(5)
            continue

    # 找到可分析的字符串
    while True:
        to_search_company: dict = dict()
        try:
            db.ping(reconnect=True)
            with db.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute("select * from spider_head where is_bert=0 and status=2 limit 1")
                #cursor.execute("select * from spider_head where id=24 limit 1")
                cursor_result = cursor.fetchall()
                if not cursor_result:
                    logger.warning("no item to search. exit..")
                    break
                to_search_company = cursor_result[0]
                head_id: int = to_search_company["id"]
                cursor.execute(f"update spider_head set is_bert=1 where id={head_id}")
                db.commit()
                cursor.execute(f"select link_id, head_id, link_text,link_html from spider_links where head_id={head_id}")
                to_search_links: list = cursor.fetchall()
            if db.open: db.close()
            logger.info("get item to bert_analy. " + str(to_search_company["id"]))
        except:
            logger.error("db connection error. retry...")
            logger.error(traceback.format_exc())
            if db.open: db.close()
            sleep(5)

        # 对找到的样本进行分析
        if len(to_search_company) and len(to_search_links):

            company_symbol: int = int(to_search_company["symbol"])
            company_name: str = str(to_search_company["shortName"])
            company_name = company_name.replace("S*ST", 'ST').replace("SST", 'ST').\
                                        replace("*", '').replace('G ', '').replace(' ', '')
            if company_symbol in list(alias.index):
                company_name = company_name.replace(alias.loc[company_symbol, "origin"], alias.loc[company_symbol, "alias"])
            company_year: str = str(to_search_company["year"])

            # 统计词频权重
            for i in range(0, len(to_search_links)):
                to_search_link = to_search_links[i]
                to_search_link["abs_w"] = 0
                to_search_link["w"] = 0

            for i in range(0, len(to_search_links)):
                to_search_link = to_search_links[i]
                link_text: str = base64.b64decode(to_search_link["link_text"]).decode(encoding="utf-8")
                link_text = link_text.replace("G ", "").replace(" ", "").\
                                      replace("\t", "").replace("*", '')
                try:
                    link_dom: HTML = HTML(html=base64.b64decode(to_search_link["link_html"]).decode(), default_encoding='utf8')
                    link_title: str = link_dom.find("title", first=True).text
                except:
                    link_title = ""
                if company_symbol in list(alias.index):
                    link_text = link_text.replace(alias.loc[company_symbol, "origin"], alias.loc[company_symbol, "alias"])
                    link_title = link_title.replace(alias.loc[company_symbol, "origin"], alias.loc[company_symbol, "alias"])
                to_search_link["link_text"] = link_text
                to_search_link["link_title"] = link_title

                abs_w: int = 0
                tfidf: list = jieba.analyse.extract_tags(to_search_link["link_text"], topK=50)
                if company_name in to_search_link["link_title"] or company_name in tfidf:
                    for i in pattern_list:
                        if i in tfidf or i in to_search_link["link_title"]:
                            abs_w += 1
                to_search_link["abs_w"] = abs_w

            sum_w: int = sum([i["abs_w"] for i in to_search_links])
            if sum_w:
                for i in range(0, len(to_search_links)):
                    to_search_link = to_search_links[i]
                    to_search_link["w"] = to_search_link["abs_w"] / sum_w

            # 计算bert
            dim: int = 768  # 向量维度是768
            bert_list: list = []
            total_bert: np.ndarray = np.zeros((1, dim))
            total_weight: list = []
            for i in range(0, len(to_search_links)):
                logger.info(f"{head_id}-{i}/{len(to_search_links)}")
                to_search_link = to_search_links[i]
                if to_search_link["w"] > 0:
                    text_list: list = re.split(r"[！？｡。＃＆＇＊＋－＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣〃「」『』【】〔〕〖〗〘〙〚〛〜〟〰–—‘’‛„…‧﹏!#&'*+\-<=>?@[\]\^_`{\|}~]",
                                               to_search_link["link_text"])
                    text_list = list(filter(lambda x: len(x) >= 10, text_list))
                    if len(text_list) > 0:
                        result: np.ndarray = np.array(bc.encode(text_list if len(text_list) <= 700 else text_list[0:701]))
                        result = np.average(result, axis=0)
                        to_search_link["bert"] = result
                        bert_list.append(result)
                        total_weight.append(to_search_link["w"])
            if bert_list:
                total_bert = np.stack(bert_list)
            avg_bert = np.average(total_bert, axis=0, weights=(total_weight if total_weight else None))
            to_search_company["bert"] = avg_bert

            avg_bert_dump: bytes = pickle.dumps(avg_bert)
            while True:
                try:
                    db.ping(reconnect=True)
                    with db.cursor(pymysql.cursors.DictCursor) as cursor:
                        cursor.execute("insert into spider_bert(head_id,vec) values (%s,%s);", [head_id, avg_bert_dump])
                        cursor.execute(f"update spider_head set is_bert=2 where id={head_id}")
                        db.commit()
                    if db.open: db.close()
                    logger.info("id " + str(to_search_company["id"]) + " bert_analy finished.")
                    break
                except Exception as e:
                    if isinstance(e, pymysql.err.OperationalError) and e.args[0] in [1213, "1213"]:
                        logger.warning("id " + str(to_search_company["id"]) + " deadlock. retry...")
                        if db.open: db.close()
                        break
                    logger.error("id " + str(to_search_company["id"]) + " bert_analy failed. retry...")
                    logger.error(traceback.format_exc())
                    if db.open: db.close()
                    break
