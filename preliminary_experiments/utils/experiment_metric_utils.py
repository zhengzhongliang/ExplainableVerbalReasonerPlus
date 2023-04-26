import re


class ExpMetricUtils:

    @classmethod
    def get_cartesian_hit(cls, prediction, target):

        prediction = prediction.replace(".", ",")
        prediction = [re.sub(r'\s+', "", p) for p in prediction.split(",") if not p.isspace() and p != ""]

        target = target.replace(".", ",")
        target = [re.sub(r'\s+', "", t) for t in target.split(",") if not t.isspace() and t != ""]

        if set(prediction) == set(target):
            return 1
        else:
            return 0

    @classmethod
    def get_seq2seq_em(cls, prediction, target, data_pattern):

        """
        Return 1 if the prediction matches the target, otherwise return 0. It should handle different data patterns.
        :param prediction:
        :param target:
        :param data_pattern:
        :return:
        """

        data_pattern_root = data_pattern[:-5]

        if data_pattern_root == "chaining":
            if prediction == target:
                return 1
            else:
                return 0
        elif data_pattern_root == "cartesian":
            return cls.get_cartesian_hit(prediction, target)

        elif data_pattern_root == "tree_search":
            if prediction == target:
                return 1
            else:
                return 0
        elif data_pattern_root == "chaining_tree_search":
            if prediction == target:
                return 1
            else:
                return 0
        elif data_pattern_root == "cartesian_tree_search":
            if prediction == target:
                return 1
            else:
                return 0
        elif data_pattern_root == "chaining_cartesian_tree_search":
            if prediction == target:
                return 1
            else:
                return 0
        else:
            if prediction == target:
                return 1
            else:
                return 0

    @classmethod
    def get_edit_distance(cls, prediction, target):

        pass

    @classmethod
    def get_evr_em(cls, prediction, target):

        if prediction.replace(" ", "") == target.replace(" ", ""):
            return 1
        else:
            return 0
