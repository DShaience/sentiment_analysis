from abc import ABC
from typing import TypedDict

from tabulate import tabulate


class ReportInfo(TypedDict):
    negative: float
    neutral: float
    positive: float
    polarity: float
    text: str


class SentimentResult:
    def __init__(self, positive: float = None, negative: float = None, neutral: float = None,
                 polarity: float = None, raw_scores=None, raw_text: str = None):
        self.result = ReportInfo(negative=negative, neutral=neutral, positive=positive, polarity=polarity, text=raw_text)
        self._raw_scores = raw_scores  # this is the non-standardized output from the SentimentAnalyzer class

    def get_score(self, score_type: str):
        """
        :param score_type:
        :param strict: provide protection against None. Default is false.
        'all' and 'raw' are intentionally left exempt from strict
        """
        if score_type == 'pos':
            return self.result['positive']
        elif score_type == 'neg':
            return self.result['negative']
        elif score_type == 'neutral':
            return self.result['neutral']
        elif score_type == 'polarity':
            return self.result['polarity']
        elif score_type == 'all':
            return self.result['positive'], self.result['negative'], self.result['negative'], self.result['polarity']
        elif score_type == 'raw':
            return self._raw_scores
        else:
            raise ValueError(f"Undefined score type: {score_type}")

    def __str__(self):
        formatted_pairs = ", ".join([f"{key}: {value}" for key, value in self.result.items()])
        return formatted_pairs


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


class ReportProducer:
    def __init__(self, scores: list[ReportInfo]):
        if len(scores) == 0:
            self.scores = []
        self.scores = scores

    def to_dict(self) -> list[ReportInfo]:
        return self.scores

    def __len__(self):
        return len(self.scores)

    def __str__(self):
        if len(self) == 0:
            return ""

        headers = [c for c in self.scores[0].keys()]
        table_data = [[row[field] for field in headers] for row in self.scores]  # type: ignore
        return tabulate(table_data, headers=headers)

