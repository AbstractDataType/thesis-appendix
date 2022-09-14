from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HeteroSecureBoost, Evaluation
from pipeline.interface import Data

data_type: str = 'word2vec'
test_guest: dict = {"name": f"test_set_guest_{data_type}_base", "namespace": f"{data_type}"}
test_host: dict = {"name": f"test_set_host_{data_type}", "namespace": f"{data_type}"}
train_guest: dict = {"name": f"train_set_guest_{data_type}_base", "namespace": f"{data_type}"}
train_host: dict = {"name": f"train_set_host_{data_type}", "namespace": f"{data_type}"}

if __name__ == '__main__':

    pipeline = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=10000)

    reader_0: Reader = Reader(name=f"reader_HSB_train_{data_type}")
    reader_0.get_party_instance(role='guest', party_id=9999).component_param(table=train_guest)  # set guest parameter
    reader_0.get_party_instance(role='host', party_id=10000).component_param(table=train_host)  # set host parameter

    data_transform_0: DataTransform = DataTransform(name=f"dataTransform_HSB_train_{data_type}")
    data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=10000).component_param(with_label=False)

    intersect_0: Intersection = Intersection(name=f"intersect_HSB_train_{data_type}")

    hetero_secureboost_0 = HeteroSecureBoost(name=f"hetero_secureboost_train_{data_type}",
                                             num_trees=5,
                                             bin_num=16,
                                             task_type="classification",
                                             objective_param={"objective": "cross_entropy"},
                                             encrypt_param={"method": "paillier"},
                                             tree_param={"max_depth": 3})
    evaluation_0 = Evaluation(name=f"evaluation_HSB_train_{data_type}", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersect_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))
    pipeline.compile()
    pipeline.fit()

    reader_1 = Reader(name=f"reader_HSB_test_{data_type}")
    reader_1.get_party_instance(role="guest", party_id=9999).component_param(table=test_guest)
    reader_1.get_party_instance(role="host", party_id=10000).component_param(table=test_host)
    evaluation_1 = Evaluation(name=f"evaluation_HSB_test_{data_type}", eval_type="binary")

    if data_type == 'word2vec':
        a = pipeline.dataTransform_HSB_train_word2vec
        b = pipeline.intersect_HSB_train_word2vec
        c = pipeline.hetero_secureboost_train_word2vec
    else:
        a = pipeline.dataTransform_HSB_train_bert
        b = pipeline.intersect_HSB_train_bert
        c = pipeline.hetero_secureboost_train_bert

    pipeline.deploy_component([a, b, c])
    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_1)
    predict_pipeline.add_component(pipeline, data=Data(predict_input={a.input.data: reader_1.output.data}))
    predict_pipeline.add_component(evaluation_1, data=Data(data=c.output.data))
    predict_pipeline.predict()
