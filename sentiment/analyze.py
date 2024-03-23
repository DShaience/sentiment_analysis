# Taken from:
# https://huggingface.co/avichr/heBERT_sentiment_analysis
# https://huggingface.co/avichr/heBERT
# https://medium.com/@umarsmuhammed/how-to-perform-sentiment-analysis-using-python-step-by-step-tutorial-with-code-snippets-4ac3e9747fff

# from transformers import AutoTokenizer, AutoModel, pipeline
from itertools import chain
import nltk
import re
# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
from sentiment.util_classes import SentimentAnalyzerWrapper, SentimentResult, ReportProducer
from sentiment.reader_classes import TextInputGetter, ManualText


def contains_only_digits_or_punctuation(input_str):
    return bool(re.match(r'^[\d\W]+$', input_str))


def clean_text_lines(lines: list[str]):
    lines = [line.replace('\n', ' ').strip() for line in lines]
    lines = [line.replace('\n', ' ').strip() for line in lines if line != '']

    split_lines_to_sentences = [nltk.sent_tokenize(line) for line in lines]
    flattened_sentences = list(chain.from_iterable(split_lines_to_sentences))
    flattened_sentences = [sentence for sentence in flattened_sentences if
                           not contains_only_digits_or_punctuation(sentence)]
    return flattened_sentences


def read_and_analyze_polarity(text_getter: TextInputGetter, sentiment_analyzer: SentimentAnalyzerWrapper) -> list[SentimentResult]:
    scores = []
    for line in text_getter:
        cleaned_lines = clean_text_lines([line])
        for cleaned_line in cleaned_lines:
            cleaned_line_score = sentiment_analyzer(cleaned_line)
            scores.append(cleaned_line_score)
    return scores


if __name__ == '__main__':
    from sentiment.util_classes import VaderSentimentAnalyzerWrapper

    # text_sample = ["The jailer's name was Aug. It had once been Augustus, but the boy had none of his mother's pretensions, and no one still living knew the longer appellation that made sense of the shorter—not even the jailer's wife. Right now Aug wished he'd taken the higher road his mother'd urged and had ended up an officer in the army perhaps, or a local magistrate or solicitor. For that matter, he would gratefully trade places with the merchants or farmers, the ones sent to destroy the Beast's family. Even being the executioner would be better, for his duty would be carried out tomorrow in the sane light of day and amidst the fear-salving crowd at the scaffold. He'd rather be anyone and anywhere other than the village jailer sitting where he was sitting at this moment—alone, at night. "]
    sentiment_analyzer = VaderSentimentAnalyzerWrapper()
    text_sample = ["And still the cries went on. The screams became deeper and gruffer; the words were taken over by snarls... For what seemed like an eternity, there was only the sound of Aug's own terrified heartbeat. Still, she was a quite attractive blonde, despite her air of having walked these halls for eons."]
    text_getter = ManualText(text_sample)
    scores = read_and_analyze_polarity(text_getter, sentiment_analyzer)
    report = ReportProducer([score.result for score in scores])
    print(report)











