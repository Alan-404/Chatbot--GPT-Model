import re

R_patterns = [
    (r"won\'t", 'will not'),
    (r"can\'t", 'cannot'),
    (r"i\'m", 'i am'),
    (r"(\w+)\'ll", '\g<1> will'),
    (r"(\w+)n\'t", '\g<1> not'),
    (r"(\w+)\'ve", '\g<1> have'),
    (r"(\w+)\'re", '\g<1> are'),
]


class Replacer:
    def __init__(self, patterns = R_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return 