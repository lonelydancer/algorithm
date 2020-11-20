from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import BertTokenizer
import sys
import torch
import time
import copy
#import torchscipt
voc_path='../data/vocab_small.txt'
tokenizer = BertTokenizer(vocab_file=voc_path)
model_path = sys.argv[1]
model = GPT2LMHeadModel.from_pretrained(model_path, torchscript=True).to('cuda')
model.eval()

input_ids = [tokenizer.cls_token_id]
history_str=['hellowrdooa','what is your name']
for history_id, history_utr in enumerate(history_str):
    text = tokenizer.convert_ids_to_tokens(history_utr)
    id = tokenizer.encode(history_utr)
    input_ids.extend(id)
    input_ids.append(100)
    input_ids.append(tokenizer.sep_token_id)
input_ids = torch.tensor(input_ids).to('cuda')

traced_model = torch.jit.trace(model, input_ids)
torch.jit.save(traced_model, 'trace_gpt2.pt')

loaded_model = torch.jit.load('trace_gpt2.pt').to('cuda')
loaded_model.eval()

#print (loaded_model)
start = time.time()
for i in range(100):
    with torch.no_grad():
        loaded_model(input_ids)
#traced_model([copy.deepcopy(input_ids) for _ in range(100)])
end = time.time()
print ('traced model',(end-start))
start = time.time()
  
for i in range(100):
#    print ('input_ids', input_ids)
    with torch.no_grad():
        model(input_ids)
#    model([copy.deepcopy(input_ids) for _ in range(100)])
end = time.time()

print ('origin model',(end-start))
