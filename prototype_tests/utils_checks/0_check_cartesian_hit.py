from preliminary_experiments.utils.experiment_metric_utils import ExpMetricUtils


class CheckCartesianHit:

    @classmethod
    def check_cartesian_hit(cls):

        cases = [
            ("Eugene Castillo had 5 puppies, Eugene Castillo had 17 owls, Scott Carter had 5 toy cars.",
             "Eugene Castillo had 5 puppies,Eugene Castillo had 17 owls,Scott Carter had 5 toy cars."),

            ("Eugene Castillo had 5 puppies, Eugene Castillo had 17 owls, Scott Carter had 5 toy cars.",
             "Eugene Castillo had   5 puppies,Eugene Castillo had 17 owls,ScottCarter had 5 toy cars."),

            ("Eugene Castillo had 5 puppies, Eugene Castillo had 17 owls, Scott Carter had 5 toy cars.",
             "Eugene   Castillo   had 5 puppies,    Eugene Castillo had 17 owls,   Scott Carter had 5 toy cars."),

            ("Eugene Castillo had 5 puppies. Eugene Castillo had 17 owls, Scott Carter had 5 toy cars.",
             "Eugene   Castillo   had 5 puppies,    Eugene Castillo had 17 owls,   Scott Carter had 5 toy cars."),

            ("Eugene Castillo , had 5 puppies. Eugene Castillo had 17 owls, Scott Carter had 5 toy cars.",
             "Eugene   Castillo   had 5 puppies,    Eugene Castillo had 17 owls,   Scott Carter had 5 toy cars."),
        ]

        targets = [
            1, 1, 1, 1, 0
        ]

        for idx in range(len(cases)):
            pred_hit = ExpMetricUtils.get_cartesian_hit(cases[idx][0], cases[idx][1])

            assert pred_hit == targets[idx]


if __name__ == "__main__":

    CheckCartesianHit.check_cartesian_hit()
