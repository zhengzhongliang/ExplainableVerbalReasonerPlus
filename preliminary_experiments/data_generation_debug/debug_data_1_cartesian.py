import json

from preliminary_experiments.data_generation.data_1_cartesian import GenerateCartesianData


class GenerateCartesianDataDebug(GenerateCartesianData):

    @classmethod
    def check_one_instance(cls, depth=2):

        for i in range(100):
            instance = cls.generate_one_example(depth=depth)

            print("=" * 40)
            print(json.dumps(instance, indent=2))
            input("--")


if __name__ == "__main__":
    GenerateCartesianDataDebug.check_one_instance(depth=4)
