import re

class TokenHandler:
    def __init__(self):
        self.START_TOKEN = "__START__"
        self.END_TOKEN = "__END__"
        self.SEP_TOKEN = "__SEP__"
        self.DELIM_TOKEN = "__DELIM__"
        self.LINE_TOKEN = "__LINE__"
        self.TRIPLE_DOT_TOKEN = "__TRIPLE_DOT__"
        self.COMMA_TOKEN = "__COMMA__"
        self.CODE_TOKEN = "__CODE__"
        self.OR_TOKEN = "__OR__"
        self.QUESTION_TOKEN = "__QUESTION__"
        self.QA_TOKEN = "__QA__"

    def change_token(self, sequence: str):
        sequence = re.sub("\u2026", f" {self.TRIPLE_DOT_TOKEN}", sequence)
        sequence = re.sub(r"(\b[A-Z])\.(?=[A-Z]\b|\s|$)", f" {self.SEP_TOKEN}", sequence)
        sequence = re.sub('[?]', f" {self.QUESTION_TOKEN}", sequence)
        sequence = re.sub(",", f" {self.COMMA_TOKEN}", sequence)
        sequence = re.sub("\n", f" {self.LINE_TOKEN}", sequence)
        return sequence

    def handle(self, sequences: list):
        result = []
        for sequence in sequences:
            sequence = self.change_token(sequence)
            result.append(sequence)

        return result