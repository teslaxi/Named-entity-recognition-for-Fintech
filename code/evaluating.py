from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from utils import flatten_lists
import seaborn as sns


class Metrics(object):
    def __init__(self, golden_tags, predict_tags):
        self.golden_tags = flatten_lists(golden_tags)
        self.predict_tags = flatten_lists(predict_tags)
        self.tagset = set(self.golden_tags)
        self.correct_tags_number = self.count_correct_tags()
        self.predict_tags_counter = Counter(self.predict_tags)
        self.golden_tags_counter = Counter(self.golden_tags)
        self.precision_scores = self.cal_precision()
        self.recall_scores = self.cal_recall()
        self.f1_scores = self.cal_f1()

    def cal_precision(self):
        precision_scores = {}
        for tag in self.tagset:
            if self.predict_tags_counter[tag] == 0:
                precision_scores[tag] = 0
                continue
            precision_scores[tag] = self.correct_tags_number.get(tag, 0) / self.predict_tags_counter[tag]

        return precision_scores

    def cal_recall(self):
        recall_scores = {}
        for tag in self.tagset:
            recall_scores[tag] = self.correct_tags_number.get(tag, 0) / \
                                 self.golden_tags_counter[tag]
        return recall_scores

    def cal_f1(self):
        f1_scores = {}
        for tag in self.tagset:
            p, r = self.precision_scores[tag], self.recall_scores[tag]
            f1_scores[tag] = 2 * p * r / (p + r + 1e-10)
        return f1_scores

    def average_precision_value(self):
        return self._cal_weighted_average()['precision']

    def average_precision(self):
        avg_metrics = self._cal_weighted_average()
        print(f"Average Precision: {avg_metrics['precision']:.4f}")
        print(f"Average Recall: {avg_metrics['recall']:.4f}")
        print(f"Average F1-score: {avg_metrics['f1_score']:.4f}")
        print(f"Total Support: {len(self.golden_tags)}")

    def report_scores(self, tag):
        labels = list(self.tagset)
        precision = [self.precision_scores[tag] for tag in labels]
        recall = [self.recall_scores[tag] for tag in labels]
        f1 = [self.f1_scores[tag] for tag in labels]
        # Calculate and print the average values
        plt.figure(figsize=(12, 8))
        x = np.arange(len(labels))
        width = 0.25
        plt.bar(x - width, precision, width, label='Precision')
        plt.bar(x, recall, width, label='Recall')
        plt.bar(x + width, f1, width, label='F1-score')
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.xlabel('Tag')
        plt.ylabel('Score')
        plt.title('Evaluation Metrics')
        plt.legend()
        base_filename = "Score for "
        filename = f"./pic/{base_filename}{tag}.png"
        plt.savefig(filename, dpi=600)
        plt.show()

    def count_correct_tags(self):
        correct_dict = {}
        for gold_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            if gold_tag == predict_tag:
                if gold_tag not in correct_dict:
                    correct_dict[gold_tag] = 1
                else:
                    correct_dict[gold_tag] += 1
        return correct_dict

    def _cal_weighted_average(self):

        weighted_average = {}
        total = len(self.golden_tags)
        # weighted precisions:
        weighted_average['precision'] = 0.
        weighted_average['recall'] = 0.
        weighted_average['f1_score'] = 0.
        for tag in self.tagset:
            size = self.golden_tags_counter[tag]
            weighted_average['precision'] += self.precision_scores[tag] * size
            weighted_average['recall'] += self.recall_scores[tag] * size
            weighted_average['f1_score'] += self.f1_scores[tag] * size
        for metric in weighted_average.keys():
            weighted_average[metric] /= total
        return weighted_average

    def report_confusion_matrix(self, tag):
        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        tags_size = len(tag_list)
        matrix = []
        for i in range(tags_size):
            matrix.append([0] * tags_size)
        # Iterate through the golden_tags and predict_tags
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:
                continue
        row_format_ = '{:>7} ' * (tags_size + 1)
        print(row_format_.format("", *tag_list))
        for i, row in enumerate(matrix):
            print(row_format_.format(tag_list[i], *row))

    def report_confusion_matrix_visual(self, tag):
        """Calculate and visualize the confusion matrix"""
        print("\nConfusion Matrix:")
        tag_list = list(self.tagset)
        tags_size = len(tag_list)
        # Initialize the confusion matrix
        matrix = np.zeros((tags_size, tags_size), dtype=int)
        # Iterate through golden_tags and predict_tags to update the confusion matrix
        for golden_tag, predict_tag in zip(self.golden_tags, self.predict_tags):
            try:
                row = tag_list.index(golden_tag)
                col = tag_list.index(predict_tag)
                matrix[row][col] += 1
            except ValueError:  # Some tags may appear in predict_tags but not in golden_tags, skip these cases
                continue
        # Plot the confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix, annot=True, cmap='Blues', xticklabels=tag_list, yticklabels=tag_list)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Tag')
        plt.ylabel('True Tag')
        base_filename = "Confusion Matrix "
        filename = f"./pic/{base_filename}{tag}.png"
        plt.savefig(filename, dpi=600)
        plt.show()
