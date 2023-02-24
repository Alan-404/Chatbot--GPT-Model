from argparse import ArgumentParser
from preprocessing.text import TextProcessor
import io
import pickle
import numpy as np
from util import load_model_config

parser = ArgumentParser()

parser.add_argument("--type", type=str, default="pretrain")
parser.add_argument("--data_path", type=str)
parser.add_argument("--length_seq", type=int)
parser.add_argument("--tokenizer", type=str)

parser.add_argument("--clean_folder", type=str)
parser.add_argument("--delim", type=str)

parser.add_argument("--saved_pretrain_path", type=str)


args = parser.parse_args()

def save_data(data: np.ndarray, path: str):
    with open(f"{path}", 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def build_pretrain_dataset(data_path: str, tokenizer_path: str, length_seq: int, saved_path: str):
    if data_path.endswith('.txt') == False:
        print("Invalid type of Data")
        return
    data = io.open(data_path, encoding='utf-8').read().strip().split('\n')
    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    processed_data = text_processor.process(sequences=data, max_len=length_seq, start_token=True, end_token=True)

    save_data(processed_data, saved_path)

    print("Pretrain Processing Data Finished")

def build_finetune_dataset(data_path: str, tokenizer_path: str, length_seq:int, saved_path: str, delim_token: str):
    if data_path.endswith('.txt') == False:
        print("Invalid type of Data")
        return
    
    data = io.open(data_path, encoding='utf-8').read().strip().split('\n')

    questions = []
    answers = []

    for item in data:
        qa = item.split(delim_token)
        questions.append(qa[0])
        answers.append(qa[1])

    text_processor = TextProcessor(tokenizer_path=tokenizer_path)

    questions = text_processor.process(sequences=questions, max_len=length_seq, start_token=True)
    answers = text_processor.process(sequences=answers, max_len=length_seq ,end_token=True)

    save_data(questions, path=f'{saved_path}/question.pkl')
    save_data(answers, path=f"{saved_path}/answer.pkl")

    print('Fine-tune Processing Data Finished')



if __name__ == "__main__":
    flag = False
    if args.data_path is None or args.tokenizer is None:
        print("Missing Information")
    if args.length_seq is None:
        flag = True
        config = load_model_config(path='./config.yml')
        args.length_seq = config['util']['length_seq']
    if args.type == "pretrain":
        if args.saved_pretrain_path is None:
            print("Missing Information")
        else:
            build_pretrain_dataset(
                data_path=args.data_path,
                tokenizer_path=args.tokenizer,
                length_seq=args.length_seq,
                saved_path=args.args.saved_pretrain_path
            )
    elif args.type == "finetune":
        if args.clean_folder is None:
            print("Missing Information")
        else:
            if flag == True:
                args.length_seq = args.length_seq - 1
            build_finetune_dataset(
                data_path=args.data_path,
                tokenizer_path=args.tokenizer,
                length_seq=args.length_seq,
                saved_path=args.clean_folder,
                delim_token=args.delim
            )
    