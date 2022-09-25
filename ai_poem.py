import chinese_converter
import torch
from simplet5 import SimpleT5
from transformers import T5Tokenizer, T5ForConditionalGeneration


class MengziSimpleT5(SimpleT5):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cuda:1")

    def load_my_model(self, local_path, use_gpu: bool = True):
        self.tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(local_path)


AUTHOR_PROMPT = "模仿："
TITLE_PROMPT = "作诗："
EOS_TOKEN = '</s>'

general_poem_model = MengziSimpleT5()
general_poem_model.load_my_model(
    local_path='/home/me/ai/chinese-ai-writing-share/data/simplet5-epoch-3-train-loss-3.597/')
general_poem_model.model = general_poem_model.model.to('cuda:1')

MAX_AUTHOR_CHAR = 4
MAX_TITLE_CHAR = 12
MIN_CONTENT_CHAR = 10
MAX_CONTENT_CHAR = 64


def poem(title_str, opt_author=None, is_input_traditional_chinese=False):
    if opt_author:
        in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR] + EOS_TOKEN + AUTHOR_PROMPT + opt_author[
                                                                                             :MAX_AUTHOR_CHAR]
    else:
        in_request = TITLE_PROMPT + title_str[:MAX_TITLE_CHAR]
    if is_input_traditional_chinese:
        in_request = chinese_converter.to_simplified(in_request)

    out = general_poem_model.predict(in_request,
                                     max_length=MAX_CONTENT_CHAR)[0].replace(",", "，")
    if is_input_traditional_chinese:
        out = chinese_converter.to_traditional(out)
        print(f"標題： {in_request.replace('</s>', ' ')}\n詩歌： {out}")
    else:
        print(f"标题： {in_request.replace('</s>', ' ')}\n诗歌： {out}")


if __name__ == '__main__':
    for title in ['秋日', "柿子", '美人林下', "游船", "品茶"]:
        # Empty author means general style
        for author in ["", "杜甫", "李白", "李清照", "苏轼", "钱力", "辛弃疾", "陆游", "王维", "李商隐", "杜牧"]:
            poem(title, author)
        print()
