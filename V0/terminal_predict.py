import time
import sys
sys.path.append('../')
from load_data import *
from paths import *
import torch
import sys
from fastNLP.core.predictor import Predictor

if __name__ == '__main__':
    vocabs = {}
    char_vocab = Vocabulary().load(os.path.join(vocab_path, 'char_vocab.txt'))
    bigram_vocab = Vocabulary().load(os.path.join(vocab_path, 'bigram_vocab.txt'))
    label_vocab = Vocabulary().load(os.path.join(vocab_path, 'label_vocab.txt'))
    lattice_vocab = Vocabulary().load(os.path.join(vocab_path, 'lattice_vocab.txt'))
    vocabs["char"] = char_vocab
    vocabs["bigram"] = bigram_vocab
    vocabs["label"] = label_vocab
    vocabs["lattice"] = lattice_vocab
    device = torch.device('cuda:0')
    refresh_data = False
    yangjie_rich_pretrain_word_path = lk_word_path_2
    w_list = load_yangjie_rich_pretrain_word_list(yangjie_rich_pretrain_word_path,
                                                  _refresh=refresh_data,
                                                  _cache_fp='cache/{}'.format("yj"))

    model_dir = "./save/weibo/2023_05_06_20_12_47"  # 你的模型保存路径，加载自己的就好
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = "{}/model_ckpt.pkl".format(model_dir)
    model = torch.load(model_path)
    model.to(device)
    predictor = Predictor(model)   # 这里的model是加载权重之后的model
    predictor.batch_size = 1

    while True:
        line = input("input sentence, please:")
        datasets = pack_predict_data([line], vocabs, w_list, max_seq=128)
        test_label_list = predictor.predict(datasets['predict'])['pred']  # 预测结果
        pred_tags = []
        for test_label in test_label_list:
            for item in test_label:
                pred_tags.append(item)
        test_raw_char = datasets['predict']['raw_chars']     # 原始文字
        for sentence, tags in zip(test_raw_char, pred_tags):
            tag_text = [vocabs["label"].to_word(tags[i]) for i in range(len(sentence))]
            print(tag_text)