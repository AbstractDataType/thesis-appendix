import os

from pipeline.backend.pipeline import PipeLine
from pipeline.component import (DataTransform, Evaluation, HeteroNN,
                                Intersection, Reader)
from pipeline.interface import Data
from tensorflow.keras import layers, optimizers

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if __name__ == '__main__':
    pipeline: PipeLine = PipeLine().set_initiator(role='guest', party_id=9999).set_roles(guest=9999, host=10000)
    data_type: str = 'bert'
    test_guest: dict = {"name": f"test_set_guest_{data_type}_base", "namespace": f"{data_type}"}
    test_host: dict = {"name": f"test_set_host_{data_type}", "namespace": f"{data_type}"}
    train_guest: dict = {"name": f"train_set_guest_{data_type}_base", "namespace": f"{data_type}"}
    train_host: dict = {"name": f"train_set_host_{data_type}", "namespace": f"{data_type}"}

    reader_0: Reader = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=9999).component_param(table=train_guest)
    reader_0.get_party_instance(role='host', party_id=10000).component_param(table=train_host)

    data_transform_0: DataTransform = DataTransform(name=f"data_transform_HsNN_{data_type}")
    data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=10000).component_param(with_label=False)

    intersection_0 = Intersection(name=f"intersection_HsNN_{data_type}")

    hetero_nn_0: HeteroNN = HeteroNN(name=f"hetero_nn_{data_type}", epochs=50, batch_size=-1)
    guest_nn_0: HeteroNN = hetero_nn_0.get_party_instance(role='guest', party_id=9999)
    guest_nn_0.add_bottom_model(layers.Dense(units=50, input_shape=(70, ), activation="relu"))
    guest_nn_0.add_bottom_model(layers.Dense(units=15, input_shape=(50, ), activation="relu"))
    guest_nn_0.set_interactve_layer(layers.Dense(units=15, input_shape=(50, )))
    guest_nn_0.add_top_model(layers.Dense(units=10, input_shape=(40, ), activation="relu"))
    guest_nn_0.add_top_model(layers.Dense(units=1, input_shape=(10, ), activation="sigmoid"))
    host_nn_0: HeteroNN = hetero_nn_0.get_party_instance(role='host', party_id=10000)
    dim: int = 327 if data_type == 'word2vec' else 768
    host_nn_0.add_bottom_model(layers.Dense(units=100, input_shape=(dim, ), activation="relu"))
    host_nn_0.add_bottom_model(layers.Dense(units=50, input_shape=(100, ), activation="relu"))
    host_nn_0.set_interactve_layer(layers.Dense(units=25, input_shape=(50, )))

    hetero_nn_0.compile(optimizer=optimizers.Adam(lr=0.01), loss='mse')
    evaluation_0 = Evaluation(name=f"evaluation_HsNN_{data_type}", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_nn_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_nn_0.output.data))
    pipeline.compile()
    pipeline.fit()

    if data_type == 'word2vec':
        a = pipeline.data_transform_HsNN_word2vec
        b = pipeline.intersection_HsNN_word2vec
        c = pipeline.hetero_nn_word2vec
    else:
        a = pipeline.data_transform_HsNN_bert
        b = pipeline.intersection_HsNN_bert
        c = pipeline.hetero_nn_bert

    pipeline.deploy_component([a, b, c])

    reader_1 = Reader(name="reader_1")
    reader_1.get_party_instance(role="guest", party_id=9999).component_param(table=test_guest)
    reader_1.get_party_instance(role="host", party_id=10000).component_param(table=test_host)

    predict_pipeline = PipeLine()
    predict_pipeline.add_component(reader_1)
    predict_pipeline.add_component(pipeline, data=Data(predict_input={a.input.data: reader_1.output.data}))
    predict_pipeline.add_component(evaluation_0, data=Data(data=c.output.data))
    predict_pipeline.predict()
