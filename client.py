import sys
from paddle_serving_client import Client
from paddle_serving_client.utils import benchmark_args
from paddle_serving_app.reader import ChineseBertReader
import numpy as np
import time



def data_preprocess_1():
    path = 'test.pickle3'
#    path = 'test_basic.pickle'
    f1 = open(path, 'rb')
    T =  ['token_ids', 'type_ids', 'pos_ids', 'tgt_ids', 'tgt_pos', 'parent_idx', 'latent_id', 'data_id']
    data = pickle.load(f1)
    print (data)
    tmp_data = {}
    data['init_score'] = np.array(data['init_score'])
    data['tgt_ids'] = np.array(data['tgt_ids'])
    data['tgt_pos'] = np.array(data['tgt_pos'])
    for key in data:
#        tmp_data[key] = np.array([data[key][0]]).squeeze(0)
        data[key] = np.array(data[key])
    return data

from paddle_serving_client import Client

client = Client()
client.load_client_config("serving_client.raw/serving_client_conf.prototxt")
#client.load_client_config("serving_client_conf.prototxt")
client.connect(["127.0.0.1:9200"])

#client.load_client_config("server_basic/serving_client/serving_client_conf.prototxt")
#client.connect(["127.0.0.1:9300"])
#data_generator = data_preprocess()



fetch_map = {}
total_time=0
begin_time = time.time()
for i in range(10):
    data = data_preprocess_1()
    print (data)
#    break
    fetch_map = client.predict(feed= data, fetch=["save_infer_model/scale_0.tmp_0"])#, batch=False)
    #fetch_map = client.predict(feed=data, fetch=["stack_0.tmp_0"])
    print(fetch_map)
    break

end_time = time.time()
total_time += end_time - begin_time


print(fetch_map)

print('time:', float(total_time)/1000)
