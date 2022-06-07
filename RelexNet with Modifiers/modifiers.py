import numpy as np
import pandas as pd
import torch
from preprocessor import Preprocessor
from collections import Counter, defaultdict
from sklearn.preprocessing import OneHotEncoder
import torch.utils.data as Data

class Modifiers:
    def __init__(self):
        C_INCR = 0.733
        B_INCR = 0.293
        B_DECR = -0.293

        N_SCALAR = -0.74
        self.NEGATE = \
            ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
             "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
             "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
             "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
             "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
             "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
             "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
             "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

        # booster/dampener 'intensifiers' or 'degree adverbs'
        #

        self.BOOSTER_DICT = \
            {"absolutely": B_INCR, "amazingly": B_INCR, "awfully": B_INCR, "completely": B_INCR, "considerably": B_INCR,
             "decidedly": B_INCR, "deeply": B_INCR, "effing": B_INCR, "enormously": B_INCR,
             "entirely": B_INCR, "especially": B_INCR, "exceptionally": B_INCR, "extremely": B_INCR,
             "fabulously": B_INCR, "flipping": B_INCR, "flippin": B_INCR,
             "fricking": B_INCR, "frickin": B_INCR, "frigging": B_INCR, "friggin": B_INCR, "fully": B_INCR,
             "fucking": B_INCR,
             "greatly": B_INCR, "hella": B_INCR, "highly": B_INCR, "hugely": B_INCR, "incredibly": B_INCR,
             "intensely": B_INCR, "majorly": B_INCR, "more": B_INCR, "most": B_INCR, "particularly": B_INCR,
             "purely": B_INCR, "quite": B_INCR, "really": B_INCR, "remarkably": B_INCR,
             "so": B_INCR, "substantially": B_INCR,
             "thoroughly": B_INCR, "totally": B_INCR, "tremendously": B_INCR,
             "uber": B_INCR, "unbelievably": B_INCR, "unusually": B_INCR, "utterly": B_INCR,
             "very": B_INCR,
             "almost": B_DECR, "barely": B_DECR, "hardly": B_DECR, "just enough": B_DECR,
             "kind of": B_DECR, "kinda": B_DECR, "kindof": B_DECR, "kind-of": B_DECR,
             "less": B_DECR, "little": B_DECR, "marginally": B_DECR, "occasionally": B_DECR, "partly": B_DECR,
             "scarcely": B_DECR, "slightly": B_DECR, "somewhat": B_DECR,
             "sort of": B_DECR, "sorta": B_DECR, "sortof": B_DECR, "sort-of": B_DECR}

# M = Modifiers()
# print(M.BOOSTER_DICT)
