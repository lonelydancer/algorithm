import time
import argparse

import torch
import lightseq.inference as lsi
from transformers import BartTokenizer, BartForConditionalGeneration, BertTokenizer


def ls_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
#generated_ids = model.infer(inputs)
#generated_ids = model.infer(inputs, multiple_output=True, sampling_method='topk',topk=8, topp=0.9, length_penalty=1.3)
#generated_ids = model.infer(inputs,  sampling_method='topk_greedy',topk=8, topp=0.9, length_penalty=1.3)
    print (model.infer)
    generated_ids = model.infer(inputs,  sampling_method='topk',topk=8, topp=0, length_penalty=0.9)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def hf_bart(model, inputs):
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    generated_ids = model.generate(inputs.to("cuda:0"), max_length=128, do_sample=True,top_k=8, repetition_penalty=1.3, num_return_sequences=10)
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    return generated_ids, end_time - start_time


def ls_generate(model, tokenizer, inputs_id):
    print("=========lightseq=========")
    print("lightseq generating...")
    ls_res_ids, ls_time = ls_bart(model, inputs_id)
    ls_res_ids = [ids[0] for ids in ls_res_ids[0]]
    ls_res = tokenizer.batch_decode(ls_res_ids, skip_special_tokens=True)
    candidate_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in ls_res_ids]
    print ('\n'.join(candidate_list))
    print(f"lightseq time: {ls_time}s")
    print("lightseq results:")
    for sent in ls_res:
        print(sent)


def hf_generate(model, tokenizer, inputs_id):
    print("=========huggingface=========")
    print("huggingface generating...")
    hf_res_ids, hf_time = hf_bart(model, inputs_id)
    candidate_list = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in hf_res_ids]
    print ('\n'.join(candidate_list))
    hf_res = tokenizer.batch_decode(hf_res_ids, skip_special_tokens=True)
    print(f"huggingface time: {hf_time}s")
    print("huggingface results:")
    for sent in hf_res:
        print(sent)


def warmup(tokenizer, ls_model, hf_model, sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True)
    inputs_id = inputs["input_ids"]

    ls_generate(ls_model, tokenizer, inputs_id)
    hf_generate(hf_model, tokenizer, inputs_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_input", action="store_true")
    args = parser.parse_args()

    print("initializing bart tokenizer...")
    # change to "facebook/bart-large" for large model
    model_name = 'uer/bart-base-chinese-cluecorpussmall'
    model_name = '/workspace/lightseq/examples/inference/python/test/checkpoint-341000'

    tokenizer = BertTokenizer.from_pretrained(model_name)

    print("creating lightseq model...")
    # change to "lightseq_bart_large.hdf5" for large model
#ls_model = lsi.Transformer("lightseq_bart_base.hdf5", 128)
    ls_model = lsi.Transformer("lightseq_bart_base.hdf5", 64)
    print("creating huggingface model...")
    # change to "facebook/bart-large" for large model
    hf_model = BartForConditionalGeneration.from_pretrained("/workspace/lightseq/examples/inference/python/test/checkpoint-341000")
    hf_model.to("cuda:0")

    sentences = [
        "I love that girl, but <mask> does not <mask> me.",
        "She is so <mask> that I can not help glance at <mask>.",
        "Nothing's gonna <mask> my love for you.",
        "Drop everything now. Meet me in the pouring <mask>. Kiss me on the sidewalk.",
    ]
    sentences = 10*["患者:后背右侧中间儿，就是胸部对应的那个区域，一直很酸痛，有一周了，一点儿都没有缓解，很痛。"]
    print("====================START warmup====================")
    warmup(tokenizer, ls_model, hf_model, sentences)
    print("====================END warmup====================")

    while True:
        if args.user_input:
            sentences = [input("input the masked sentence:\n")]

        print("tokenizing the sentences...")
        inputs = tokenizer(sentences, return_tensors="pt", padding=True)
        inputs_id = inputs["input_ids"]
        for i in range(1):
            ls_generate(ls_model, tokenizer, inputs_id)
            inputs = tokenizer(sentences[0], return_tensors="pt", padding=True)
            inputs_id = inputs["input_ids"]
            hf_generate(hf_model, tokenizer, inputs_id)

        if not args.user_input:
            break


if __name__ == "__main__":
    main()
