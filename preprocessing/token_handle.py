import re

class TokenHandler:
    def __init__(self):
        self.START_TOKEN = "__start__"
        self.END_TOKEN = "__end__"
        self.SEP_TOKEN = "__sep__"
        self.DELIM_TOKEN = "__delim__"
        self.LINE_TOKEN = "__line__"
        self.TRIPLE_DOT_TOKEN = "__triple_dot__"
        self.COMMA_TOKEN = "__comma__"
        self.CODE_TOKEN = "__code__"
        self.OR_TOKEN = "__or__"
        self.QUESTION_TOKEN = "__question__"
        self.QA_TOKEN = "__qa__"
        self.GENETIVE_TOKEN = "__genetive__"

    def change_token(self, sequence: str):
        sequence = re.sub(r"\.\.\.", f" {self.TRIPLE_DOT_TOKEN}", sequence)
        sequence = re.sub(r"\.", f" {self.SEP_TOKEN}", sequence)
        sequence = re.sub(r'\?', f" {self.QUESTION_TOKEN}", sequence)
        sequence = re.sub(r"\,", f" {self.COMMA_TOKEN}", sequence)
        sequence = re.sub("\n", f" {self.LINE_TOKEN}", sequence)
        return sequence

    def decode_token(self, sequence: str):
        sequence = re.sub(r"\.\.\.", f" {self.TRIPLE_DOT_TOKEN}", sequence)
        sequence = re.sub(r"\.", f" {self.SEP_TOKEN}", sequence)
        sequence = re.sub(r'\?', f" {self.QUESTION_TOKEN}", sequence)
        sequence = re.sub(r"\,", f" {self.COMMA_TOKEN}", sequence)
        sequence = re.sub("\n", f" {self.LINE_TOKEN}", sequence)
        return sequence
    def process(self, sequences: list):
        result = []
        for sequence in sequences:
            sequence = f"{self.START_TOKEN} " + self.change_token(sequence) + f" {self.END_TOKEN}"
            result.append(sequence)

        return result
    def handle(self, sequences: list, type: str = "input"):
        result = []
        for sequence in sequences:
            if type == "input":
                sequence = f"{self.START_TOKEN} " + self.change_token(sequence) + f" {self.DELIM_TOKEN}"
            else:
                sequence = self.change_token(sequence) + f" {self.END_TOKEN}"
            result.append(sequence)

        return result