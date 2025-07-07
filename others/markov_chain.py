# 参考：https://techtrends.jp/tips/python-generative-ai-text-generation-tutorial/
import MeCab
import unidic
from random import randint

# テキストのトークン化
text_data = 'むかしむかし、あるところに、とても可愛らしい女の子がいました。あるとき、その女の子のおばあさんが、赤いビロードのきれで、女の子のかぶるずきんを作ってくれました。'
# text_data = '数理工学科にはたくさんの学生が所属しています。その学生はみんなオクラが好きです。'

tagger = MeCab.Tagger("-Owakati")
txt_data = tagger.parse(text_data).replace('。', '。\n').rstrip()
print(txt_data)

"""
マルコフ連鎖モデル
「現在の状態」から「次の状態」を確立的に予測，生成する方法のこと．
テキスト生成では「この単語は次にどの単語に繋がりやすいか」を確立で予測し，自然な文章を生成する．
"""
def make_1state_model(txt_data):
    model = {}
    txt_data = txt_data.split('\n')
    for sentence in txt_data:
        if not sentence:# 空白行は処理しない
            break
        eos_mark = "。！？"# 1文の終わりを示す記号
        if sentence[-1] not in eos_mark:
            print(f'not processed: {sentence}')
            continue
        words = sentence.split(' ')
        previous_word = 'BoS'# 1文の最初の単語
        for word in words:
            if previous_word in model:
                model[previous_word].append(word)
            else:
                model[previous_word] = [word]
            previous_word = word
    return model

model = make_1state_model(txt_data)
# print(model)

# 文章生成
def generate_sentence(model):
    eos_mark = "。！？"
    key_list = model['BoS']
    key = key_list[randint(0, len(key_list) - 1)]
    result = key
    while key not in eos_mark:
        key_list = model[key]
        key = key_list[randint(0, len(key_list) - 1)]
        result += key
    return result

print("\n--- Generated Sentences ---")
for _ in range(5):
    print(generate_sentence(model))