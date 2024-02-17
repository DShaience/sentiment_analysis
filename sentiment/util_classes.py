from abc import ABC


class SentimentResult:
    def __init__(self, positive: float = None, negative: float = None, neutral: float = None,
                 polarity: float = None, raw_scores=None, raw_text: str = None):
        self.positive: float = positive
        self.negative: float = negative
        self.neutral: float = neutral
        self.polarity: float = polarity
        self.raw_scores = raw_scores   # this is the non-standardized output from the SentimentAnalyzer class
        self.raw_text = raw_text

    @staticmethod
    def _get_score_validation(score, strict: bool = False):
        if (not strict) or (score is not None):
            return score
        else:
            raise AssertionError("Score is None. This can happen when the attribute for it wasn't initialized, "
                                 "which can happen as not all sentiment analysis classes have the same classifications")

    def get_score(self, score_type: str, strict: bool = False):
        """
        :param score_type:
        :param strict: provide protection against None. Default is false.
        'all' and 'raw' are intentionally left exempt from strict
        """
        if score_type == 'pos':
            return self._get_score_validation(self.positive, strict)
        elif score_type == 'neg':
            return self._get_score_validation(self.negative, strict)
        elif score_type == 'neutral':
            return self._get_score_validation(self.neutral, strict)
        elif score_type == 'polarity':
            return self._get_score_validation(self.polarity, strict)
        elif score_type == 'all':
            return self.positive, self.negative, self.negative, self.polarity
        elif score_type == 'raw':
            return self.raw_scores
        else:
            raise ValueError(f"Undefined score type: {score_type}")

    def __str__(self):
        text = f", raw-text: {self.raw_text}" if self.raw_text is not None else None
        return (f"positive: {self.positive}, negative: {self.negative}, "
                f"neutral: {self.neutral}, polarity: {self.polarity}{text}")


class TextInputGetter(ABC):
    def __getitem__(self, index):
        raise NotImplementedError


class ManualText(TextInputGetter):
    def __init__(self, texts: list[str]):
        self.texts = texts

    def __getitem__(self, index):
        return self.texts[index]

    def __len__(self):
        return len(self.texts)


class SentimentAnalyzerWrapper(ABC):
    def __call__(self, text: str):
        raise NotImplementedError


class VaderSentimentAnalyzerWrapper(SentimentAnalyzerWrapper):
    def __init__(self):
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        self.sid = SentimentIntensityAnalyzer()

    def __call__(self, text) -> SentimentResult:
        raw_res = self.sid.polarity_scores(text)
        return SentimentResult(raw_res['pos'], raw_res['neg'], polarity=raw_res['compound'],
                               raw_scores=raw_res, raw_text=text)

