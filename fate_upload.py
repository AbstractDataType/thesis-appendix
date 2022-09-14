import os

from pipeline.backend.pipeline import PipeLine

os.chdir(os.path.dirname(os.path.abspath(__file__)))
data_base: str = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':
    partition = 16
    pipeline: PipeLine = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=10000)
    data_tags = [{
        "name": "test_set_guest_word2vec_base",
        "namespace": "word2vec"
    }, {
        "name": "test_set_host_word2vec",
        "namespace": "word2vec"
    }, {
        "name": "train_set_guest_word2vec_base",
        "namespace": "word2vec"
    }, {
        "name": "train_set_host_word2vec",
        "namespace": "word2vec"
    }, {
        "name": "test_set_guest_bert_base",
        "namespace": "bert"
    }, {
        "name": "test_set_host_bert",
        "namespace": "bert"
    }, {
        "name": "train_set_guest_bert_base",
        "namespace": "bert"
    }, {
        "name": "train_set_host_bert",
        "namespace": "bert"
    }]
    for i in data_tags:
        pipeline.add_upload_data(
            file=os.path.join(data_base, f"../proced/test_data/{i['name']}.csv"),
            table_name=i["name"],  # table name
            namespace=i["namespace"],  # namespace
            head=1,
            partition=partition)  # data info
    pipeline.upload(drop=1)  # 1是覆盖
