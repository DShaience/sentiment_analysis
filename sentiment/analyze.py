# Taken from:
# https://huggingface.co/avichr/heBERT_sentiment_analysis
# https://huggingface.co/avichr/heBERT
import os.path
# https://medium.com/@umarsmuhammed/how-to-perform-sentiment-analysis-using-python-step-by-step-tutorial-with-code-snippets-4ac3e9747fff

# from transformers import AutoTokenizer, AutoModel, pipeline
from itertools import chain
import docx
import glob
import nltk
import pandas as pd
import re
# nltk.download('punkt', quiet=True)
# nltk.download('averaged_perceptron_tagger')
# nltk.download('vader_lexicon')
from sentiment.util_classes import TextInputGetter, SentimentAnalyzerWrapper, SentimentResult


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


def read_text_file_to_lines(file_path: str):
    text_file = open(file_path, "r", encoding='utf8')
    lines = text_file.readlines()
    return lines


# def init_model():
#     # tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT_sentiment_analysis")  # same as 'avichr/heBERT' tokenizer
#     # model = AutoModel.from_pretrained("avichr/heBERT_sentiment_analysis")
#     # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
#     # model = AutoModel.from_pretrained('bert-base-uncased')
#
#     # sentiment_analysis = pipeline(
#         # "sentiment-analysis",
#         # model="avichr/heBERT_sentiment_analysis",
#         # tokenizer="avichr/heBERT_sentiment_analysis",
#         # return_all_scores=True
#     # )
#     sentiment_analysis = pipeline(
#         "sentiment-analysis",
#         model="bert-base-uncased",
#         tokenizer="bert-base-uncased",
#         return_all_scores=True
#     )
#     return sentiment_analysis


# def analyze_polarity(text_inputs: list[str]):
#     sentiment_analysis = init_model()
#     results = [sentiment_analysis(text_input) for text_input in text_inputs]
#     return results


def converty_polarity_dicts_to_records(polarity_dicts: list[dict], org_input: str = None):
    res = {d['label']: d['score'] for d in polarity_dicts}
    if org_input is not None:
        res['org_input'] = org_input

    return res


def produce_report(polarity_analysis: list[dict], original_input: list[str] or None):
    if original_input is None:
        original_input = [None] * len(polarity_analysis)

    polarity_records_per_input = [converty_polarity_dicts_to_records(polarity_analysis[i][0], original_input[i])
                                  for i in range(0, len(polarity_analysis))]

    polarity_report_base = pd.DataFrame(polarity_records_per_input)
    polarity_report_base['max_polarity'] = polarity_report_base[['neutral', 'positive', 'negative']].idxmax(axis=1)

    summary_report = pd.DataFrame(polarity_report_base[['max_polarity']].groupby('max_polarity').size()).transpose()
    summary_report.reset_index(drop=True, inplace=True)
    summary_report['sum_polarity'] = summary_report[['negative', 'positive']].sum(axis=1)

    return polarity_report_base, summary_report


def get_text_from_docx(filename: str):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return fullText


# def read_and_analyze_polarity(input_file: str, output_dir=None):
#     lines = get_text_from_docx(input_file)
    # clean_lines = clean_text_lines(lines)
    # polarity = analyze_polarity(clean_lines)
    # polarity_report_base, summary_report = produce_report(polarity, clean_lines)
    # output_dir = os.path.dirname(input_file) if output_dir is None else output_dir
    # file_basename = os.path.basename(input_file)
    # polarity_report_base.to_csv(output_dir + os.sep + file_basename + '.full_report.csv', index=False,
    #                             encoding='utf-8-sig')
    # summary_report.to_csv(output_dir + os.sep + file_basename + '.summary_report.csv', index=False,
    #                       encoding='utf-8-sig')


class ReportProducer:
    def __init__(self, scores: list[SentimentResult]):
        self.scores = scores


def read_and_analyze_polarity(text_getter: TextInputGetter, sentiment_analyzer: SentimentAnalyzerWrapper) -> list[SentimentResult]:
    scores = []
    for line in text_getter:
        cleaned_lines = clean_text_lines([line])
        for cleaned_line in cleaned_lines:
            cleaned_line_score = sentiment_analyzer(cleaned_line)
            scores.append(cleaned_line_score)
    return scores




if __name__ == '__main__':
    from sentiment.util_classes import ManualText, VaderSentimentAnalyzerWrapper, TextInputGetter

    # text_sample = ["The jailer's name was Aug. It had once been Augustus, but the boy had none of his mother's pretensions, and no one still living knew the longer appellation that made sense of the shorter—not even the jailer's wife. Right now Aug wished he'd taken the higher road his mother'd urged and had ended up an officer in the army perhaps, or a local magistrate or solicitor. For that matter, he would gratefully trade places with the merchants or farmers, the ones sent to destroy the Beast's family. Even being the executioner would be better, for his duty would be carried out tomorrow in the sane light of day and amidst the fear-salving crowd at the scaffold. He'd rather be anyone and anywhere other than the village jailer sitting where he was sitting at this moment—alone, at night. "]
    sentiment_analyzer = VaderSentimentAnalyzerWrapper()
    text_sample = ["And still the cries went on. The screams became deeper and gruffer; the words were taken over by snarls... For what seemed like an eternity, there was only the sound of Aug's own terrified heartbeat. Still, she was a quite attractive blonde, despite her air of having walked these halls for eons."]
    text_getter = ManualText(text_sample)
    read_and_analyze_polarity(text_getter, sentiment_analyzer)















