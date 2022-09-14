import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import pandas as pd
import pandas.core.groupby.generic
from tensorflow import keras
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,roc_curve
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

os.chdir(os.path.dirname(os.path.abspath(__file__)))
spilt_column: str = "industryCode1"
normal_columns: list = [
    '流动比率', '速动比率', '利息保障倍数', '现金流动负债比', '资产负债率', '总资产周转率', '固定资产周转率', '存货周转率', '应收账款周转率（不含票）', '应收账款与收入比', '总资产净利率(平均)', '净资产收益率ROE', '营业毛利率',
    '财务费用率', '销售费用率', '净利润率1', '资产减值损失/营业总收入', '总资产增长率', '固定资产增长率', '每股净资产增长率', '净利润增长率', '营业收入增长率', '净利润现金净含量', '销售收现比', '折旧摊销', '资本支出与折旧摊销比',
    '应收类资产比率', '营业外收入占比', '投资收益率', "董事会规模", "独立董事占比", "监事会规模", "管理层女性占比", "未领取薪酬管理层占比", "管理层持股比例", "管理层薪酬总额对数", "兼任情况", "董事会会议次数", "监事会会议次数",
    "股东大会会议次数", "审计委员会会议次数", "审计意见", "是否换所", "是否大所", "第一大股东占比", "前十大股东占比", "Z指数", "H指数", "国有股占比", "两权分离率", "内控是否有缺陷", "缺陷级别", "缺陷个数", "困境"
]
all_financial_columns: list = [
    '流动比率', '速动比率', '利息保障倍数', '现金流动负债比', '资产负债率', '总资产周转率', '固定资产周转率', '存货周转率', '应收账款周转率（不含票）', '应收账款与收入比', '总资产净利率(平均)', '净资产收益率ROE', '营业毛利率',
    '财务费用率', '销售费用率', '净利润率1', '资产减值损失/营业总收入', '总资产增长率', '固定资产增长率', '每股净资产增长率', '净利润增长率', '营业收入增长率', '净利润现金净含量', '销售收现比', '折旧摊销', '资本支出与折旧摊销比',
    '应收类资产比率', '营业外收入占比', '投资收益率', 'indust_应收账款与收入比', 'indust_总资产净利率(平均)', 'indust_营业毛利率', 'indust_财务费用率', 'indust_销售费用率', 'indust_总资产增长率',
    'indust_营业收入增长率', 'indust_流动比率', 'indust_资产负债率', 'indust_营业外收入占比', 'prev_总资产周转率', 'prev_应收账款与收入比', 'prev_营业毛利率', 'prev_销售费用率',
    'prev_资产减值损失/营业总收入', 'prev_销售收现比', '董事会规模', '独立董事占比', '监事会规模', '管理层女性占比', '未领取薪酬管理层占比', '管理层持股比例', '管理层薪酬总额对数', '兼任情况', '董事会会议次数', '监事会会议次数',
    '股东大会会议次数', '审计委员会会议次数', '审计意见', '是否换所', '是否大所', '第一大股东占比', '前十大股东占比', 'Z指数', 'H指数', '国有股占比', '两权分离率', '内控是否有缺陷', '缺陷级别', '缺陷个数', '困境'
]


def smote_data(X: pd.DataFrame, Y: pd.DataFrame, sm=SMOTE(random_state=42)):
    g_array_X: np.ndarray = X.values
    g_array_Y: np.ndarray = Y.values
    try:
        g_array_X_smote, g_array_Y_smote = sm.fit_resample(g_array_X, g_array_Y)
    except:
        try:
            sm.k_neighbors = 2
            g_array_X_smote, g_array_Y_smote = sm.fit_resample(g_array_X, g_array_Y)
        except:
            g_array_X_smote, g_array_Y_smote = g_array_X, g_array_Y
    finally:
        row_smote: int = g_array_X_smote.shape[0]
    return g_array_X_smote, g_array_Y_smote, row_smote


def tf_model():
    input_ = keras.Input(shape=train_set_X.shape[1:])
    hidden1 = keras.layers.Dense(1000, activation="relu")(input_)
    hidden2 = keras.layers.Dense(500, activation="relu")(hidden1)
    hidden3 = keras.layers.Dense(100, activation="relu")(hidden2)
    hidden4 = keras.layers.Dense(50, activation="relu")(hidden3)
    hidden5 = keras.layers.Dense(10, activation="relu")(hidden4)
    output = keras.layers.Dense(1, activation="sigmoid")(hidden5)
    model = keras.Model(inputs=[input_], outputs=[output])
    model.compile(loss='mse', optimizer=keras.optimizers.SGD(learning_rate=0.001), metrics=["accuracy"])
    return model


if __name__ == '__main__':
    store: pd.HDFStore = pd.HDFStore(r"../proced/financial.h5", mode="r")
    fraud_posit_all: pd.DataFrame = store["fraud_posit_all"]
    fraud_nega_all: pd.DataFrame = store["fraud_nega_all"]
    all_financial_data: pd.DataFrame = pd.concat([fraud_posit_all, fraud_nega_all])
    store.close()

    store: pd.HDFStore = pd.HDFStore(r"../proced/data_text.h5", mode="r")
    all_text_data: pd.DataFrame = store["data_word2vec"]
    all_text_data.drop(columns=['id', 'isfraud'], inplace=True)
    store.close()

    all_data = pd.merge(all_financial_data, all_text_data, on=['symbol', 'year'])
    all_data = all_data[all_data[spilt_column].isin(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'R'])].reset_index(drop=True)

    # smote过采样，解决样本不平衡
    spilt_column: str = "industryCode1"
    all_data_smote: pd.DataFrame = pd.DataFrame()
    a: pandas.core.groupby.generic.DataFrameGroupBy = all_data.groupby(spilt_column)
    for k, v in a:
        #data_meta: pd.DataFrame = v[["symbol", "year", "shortName", "industryName", "industryCode1", "industryCode2"]]
        data_financial: pd.DataFrame = v[all_financial_columns]
        data_isfraud: pd.DataFrame = v[['isfraud']]
        #data_text: pd.DataFrame = v.loc[:, 0:]
        data_financial_smote_X, data_financial_smote_Y, rows = smote_data(data_financial, data_isfraud)
        data_financial_smote_X = pd.DataFrame(data_financial_smote_X)
        data_financial_smote_Y = pd.DataFrame(data_financial_smote_Y)
        data_financial_smote_X.columns = all_financial_columns
        data_financial_smote_Y.columns = ['isfraud']
        #data_text_smote_X, data_text_smote_Y, rows = smote_data(data_text, data_isfraud)
        #data_text_smote_X = pd.DataFrame(data_text_smote_X)
        data_smote: pd.DataFrame = pd.DataFrame(np.empty((rows, )), columns=[spilt_column])
        data_smote[spilt_column] = k
        data_smote = pd.concat([data_smote, data_financial_smote_Y, data_financial_smote_X], axis=1)
        all_data_smote = pd.concat([all_data_smote, data_smote], axis=0)
    all_data_smote = all_data_smote.reset_index(drop=True)

    # 对中处理，主要针对NN、SVM、SGD
    robust_scaler = RobustScaler()
    robust_scaler.fit(all_data_smote[normal_columns])
    all_data_smote[normal_columns] = robust_scaler.transform(all_data_smote[normal_columns])

    # one-hot编码，处理行业标签
    all_data_smote_dummy: pd.DataFrame = pd.get_dummies(all_data_smote[spilt_column], prefix="ind")
    all_data_smote = pd.concat([all_data_smote, all_data_smote_dummy], axis=1)
    temp = all_data_smote.pop("isfraud")
    all_data_smote.insert(value=temp, loc=all_data_smote.shape[1], column="isfraud", allow_duplicates=False)

    result: pd.DataFrame = pd.DataFrame()
    for i in range(1, 2):
        print(f"{i}/10 start.")

        # 层次分类，划分测试集和训练集
        a: pandas.core.groupby.generic.DataFrameGroupBy = all_data_smote.groupby(spilt_column)
        train_set: pd.DataFrame = pd.DataFrame()
        test_set: pd.DataFrame = pd.DataFrame()
        test_size: float = 0.2
        for k, v in a:
            v0: pd.DataFrame = v[v['isfraud'] == 0]
            v1: pd.DataFrame = v[v['isfraud'] == 1]
            train_set_0 = v0.sample(frac=1 - test_size)
            train_set_1 = v1.sample(frac=1 - test_size)
            test_set_0 = pd.concat([v0, train_set_0]).drop_duplicates(keep=False)
            test_set_1 = pd.concat([v1, train_set_1]).drop_duplicates(keep=False)
            train_set = pd.concat([train_set, train_set_0, train_set_1])
            test_set = pd.concat([test_set, test_set_0, test_set_1])
        train_set = shuffle(train_set)
        train_set.drop(columns=[spilt_column], inplace=True)
        train_set.reset_index(drop=True, inplace=True)
        test_set = shuffle(test_set)
        test_set.drop(columns=[spilt_column], inplace=True)
        test_set.reset_index(drop=True, inplace=True)

        train_set_X: np.ndarray = train_set.iloc[:, 0:-1].values
        train_set_Y: np.ndarray = train_set.iloc[:, -1].values
        test_set_X: np.ndarray = test_set.iloc[:, 0:-1].values
        test_set_Y: np.ndarray = test_set.iloc[:, -1].values

        # # 临时：生成纵向数据集
        # train_set_host: pd.DataFrame = pd.concat([train_set.loc[:, "index"],
        #                                           train_set.iloc[:, 1:int(train_set.shape[1] / 2)]], axis=1)
        # train_set_guest: pd.DataFrame = pd.concat([train_set.loc[:, "index"],
        #                                            train_set.loc[:, "isfraud"],
        #                                            train_set.iloc[:, (int(train_set.shape[1] / 2) + 1):-1]], axis=1)
        # test_set_host: pd.DataFrame = pd.concat([test_set.loc[:, "index"],
        #                                          test_set.iloc[:, 1:int(test_set.shape[1] / 2)]], axis=1)
        # test_set_guest: pd.DataFrame = pd.concat([test_set.loc[:, "index"],
        #                                           test_set.loc[:, "isfraud"],
        #                                           test_set.iloc[:, (int(test_set.shape[1] / 2) + 1):-1]], axis=1)
        # train_set_host.to_csv(
        #     "../proced/test_hetero_secureboot_only_financial/train_set_host.csv", index=False)
        # train_set_guest.to_csv(
        #     "../proced/test_hetero_secureboot_only_financial/train_set_guest.csv", index=False)
        # test_set_host.to_csv(
        #     "../proced/test_hetero_secureboot_only_financial/test_set_host.csv", index=False)
        # test_set_guest.to_csv(
        #     "../proced/test_hetero_secureboot_only_financial/test_set_guest.csv", index=False)

        # # 临时：生成横向数据集
        # train_set_host: pd.DataFrame = train_set.iloc[:int(
        #     train_set_X.shape[0]/2), :]
        # train_set_host.insert(value=train_set_host.pop(
        #     "isfraud"), loc=1, column="y", allow_duplicates=False)

        # train_set_guest: pd.DataFrame = train_set.iloc[int(
        #     train_set_X.shape[0]/2):, :]
        # train_set_guest.insert(value=train_set_guest.pop(
        #     "isfraud"), loc=1, column="y", allow_duplicates=False)

        # test_set_host: pd.DataFrame = test_set.iloc[:int(
        #     train_set_X.shape[0]/2), :]
        # test_set_host.insert(value=test_set_host.pop(
        #     "isfraud"), loc=1, column="y", allow_duplicates=False)

        # test_set_guest: pd.DataFrame = test_set.iloc[:int(
        #     train_set_X.shape[0]/2), :]
        # test_set_guest.insert(value=test_set_guest.pop(
        #     "isfraud"), loc=1, column="y", allow_duplicates=False)

        # train_set_host.to_csv(
        #     "../proced/test_homo_nn_only_financial/train_set_host.csv", index=False)
        # train_set_guest.to_csv(
        #     "../proced/test_homo_nn_only_financial/train_set_guest.csv", index=False)
        # test_set_host.to_csv(
        #     "../proced/test_homo_nn_only_financial/test_set_host.csv", index=False)
        # test_set_guest.to_csv(
        #     "../proced/test_homo_nn_only_financial/test_set_guest.csv", index=False)

        # # 支持向量机
        # svm_clf = SVC(kernel="poly", degree=1, coef0=1, C=5)  # 效率问题？
        # svm_clf.fit(train_set_X, train_set_Y)
        # train_set_Y_pred = cross_val_predict(
        #     svm_clf, train_set_X, train_set_Y, cv=10)
        # scores = confusion_matrix(train_set_Y, train_set_Y_pred)
        # print("SVM在训练集交叉验证结果")
        # print(scores)
        # pred_test = svm_clf.predict(test_set_X)
        # scores = confusion_matrix(test_set_Y, pred_test)
        # print("SVM在测试集验证结果")
        # print(scores)

        # # 随机梯度下降
        # sgd_clf = SGDClassifier(random_state=42, loss="log")
        # sgd_clf.fit(train_set_X, train_set_Y)
        # train_set_Y_pred = cross_val_predict(
        #     sgd_clf, train_set_X, train_set_Y, cv=10)
        # scores = confusion_matrix(train_set_Y, train_set_Y_pred)
        # print("SGD在训练集交叉验证结果")
        # print(scores)
        # pred_test = sgd_clf.predict(test_set_X)
        # scores = confusion_matrix(test_set_Y, pred_test)
        # print("SGD在测试集验证结果")
        # print(scores)

        # 决策树
        tree_clf = DecisionTreeClassifier(max_depth=4)
        tree_clf.fit(train_set_X, train_set_Y)
        pred_test = tree_clf.predict(test_set_X)
        scores = confusion_matrix(test_set_Y, pred_test)
        print("决策树在测试集验证结果")
        print(scores)
        pre: float = precision_score(test_set_Y, pred_test)
        rec: float = recall_score(test_set_Y, pred_test)
        f1: float = f1_score(test_set_Y, pred_test)
        acc: float = accuracy_score(test_set_Y, pred_test)
        auc: float = roc_auc_score(test_set_Y, pred_test)
        print(f"pre:{pre},rec:{rec},f1:{f1},acc:{acc},auc:{auc}")
        result = pd.concat(
            [result, pd.DataFrame([{
                "type": "base",
                "method": "rnd",
                "iter": i,
                "pre": pre,
                "rec": rec,
                "f1": f1,
                "acc": acc,
                "auc": auc
            }])])
        fpr, tpr, _=roc_curve(test_set_Y, tree_clf.predict_proba(test_set_X)[:,1],drop_intermediate=False)
        tree_roc:pd.DataFrame=pd.DataFrame({"fpr":fpr,"tpr":tpr})

        # 随机森林
        rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=4, oob_score=True, n_jobs=-1)
        rnd_clf.fit(train_set_X, train_set_Y)
        pred_test = rnd_clf.predict(test_set_X)
        scores = confusion_matrix(test_set_Y, pred_test)
        print("随机森林在测试集验证结果")
        print(scores)
        pre: float = precision_score(test_set_Y, pred_test)
        rec: float = recall_score(test_set_Y, pred_test)
        f1: float = f1_score(test_set_Y, pred_test)
        acc: float = accuracy_score(test_set_Y, pred_test)
        auc: float = roc_auc_score(test_set_Y, pred_test)
        print(f"pre:{pre},rec:{rec},f1:{f1},acc:{acc},auc:{auc}")
        result = pd.concat(
            [result, pd.DataFrame([{
                "type": "base",
                "method": "rnd",
                "iter": i,
                "pre": pre,
                "rec": rec,
                "f1": f1,
                "acc": acc,
                "auc": auc
            }])])
        fpr, tpr, _=roc_curve(test_set_Y, rnd_clf.predict_proba(test_set_X)[:,1],drop_intermediate=False)
        rnd_roc:pd.DataFrame=pd.DataFrame({"fpr":fpr,"tpr":tpr})

        # adaboost
        # ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=50)
        # rnd_clf.fit(train_set_X, train_set_Y)
        # pred_test = rnd_clf.predict(test_set_X)
        # scores = confusion_matrix(test_set_Y, pred_test)
        # print("adaboost在测试集验证结果")
        # print(scores)
        # pre: float = precision_score(test_set_Y, pred_test)
        # rec: float = recall_score(test_set_Y, pred_test)
        # f1: float = f1_score(test_set_Y, pred_test)
        # acc: float = accuracy_score(test_set_Y, pred_test)
        # auc: float = roc_auc_score(test_set_Y, pred_test)
        # print(f"pre:{pre},rec:{rec},f1:{f1},acc:{acc},auc:{auc}")
        # result = pd.concat(
        #     [result, pd.DataFrame([{
        #         "type": "base",
        #         "method": "ada",
        #         "iter": i,
        #         "pre": pre,
        #         "rec": rec,
        #         "f1": f1,
        #         "acc": acc,
        #         "auc": auc
        #     }])])

        #GBRT
        gbrt_clf = GradientBoostingClassifier(max_depth=2, n_estimators=4, learning_rate=1.0)
        gbrt_clf.fit(train_set_X, train_set_Y)
        pred_test = gbrt_clf.predict(test_set_X)
        scores = confusion_matrix(test_set_Y, pred_test)
        print("gbrt在测试集验证结果")
        print(scores)
        pre: float = precision_score(test_set_Y, pred_test)
        rec: float = recall_score(test_set_Y, pred_test)
        f1: float = f1_score(test_set_Y, pred_test)
        acc: float = accuracy_score(test_set_Y, pred_test)
        auc: float = roc_auc_score(test_set_Y, pred_test)
        print(f"pre:{pre},rec:{rec},f1:{f1},acc:{acc},auc:{auc}")
        result = pd.concat([
            result,
            pd.DataFrame([{
                "type": "base",
                "method": "gbrt",
                "iter": i,
                "pre": pre,
                "rec": rec,
                "f1": f1,
                "acc": acc,
                "auc": auc
            }])
        ])
        fpr, tpr, _=roc_curve(test_set_Y, gbrt_clf.predict_proba(test_set_X)[:,1],drop_intermediate=False)
        gbrt_roc:pd.DataFrame=pd.DataFrame({"fpr":fpr,"tpr":tpr})

        #nn
        tf_clf = keras.wrappers.scikit_learn.KerasRegressor(tf_model, epochs=20)
        history = tf_clf.fit(train_set_X, train_set_Y)
        pred_test_raw: np.array = np.array(tf_clf.predict(test_set_X))
        pred_test = np.where(pred_test_raw >= 0.5, 1, 0)
        scores = confusion_matrix(test_set_Y, pred_test)
        print("tf在测试集验证结果")
        print(scores)
        pre: float = precision_score(test_set_Y, pred_test)
        rec: float = recall_score(test_set_Y, pred_test)
        f1: float = f1_score(test_set_Y, pred_test)
        acc: float = accuracy_score(test_set_Y, pred_test)
        auc: float = roc_auc_score(test_set_Y, pred_test)
        print(f"pre:{pre},rec:{rec},f1:{f1},acc:{acc},auc:{auc}")
        result = pd.concat(
            [result, pd.DataFrame([{
                "type": "base",
                "method": "nn",
                "iter": i,
                "pre": pre,
                "rec": rec,
                "f1": f1,
                "acc": acc,
                "auc": auc
            }])])
        fpr:list=[]
        tpr:list=[]
        for i in np.arange(0,1.01,0.01):
            pred_test_roc = np.where(pred_test_raw >= i, 1, 0)
            scores_roc = confusion_matrix(test_set_Y, pred_test_roc)
            tpr.append(recall_score(test_set_Y,pred_test_roc))
            fpr.append(scores_roc[0][1]/(scores_roc[0][0]+scores_roc[0][1]))
        tf_roc:pd.DataFrame=pd.DataFrame({"fpr":fpr,"tpr":tpr})
        
        store: pd.HDFStore = pd.HDFStore(r"../proced/roc.h5", mode="a")
        store["base_tree"]=tree_roc
        store["base_rnd"]=rnd_roc
        store["base_gbrt"]=gbrt_roc
        store["base_tf"]=tf_roc
        store.close()

    result.to_csv(r"../result/base.csv", index=False)
