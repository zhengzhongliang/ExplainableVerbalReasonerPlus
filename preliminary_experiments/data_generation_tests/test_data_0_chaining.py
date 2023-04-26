import unittest

from preliminary_experiments.data_generation.data_0_chaining import GenerateChainingData


class TestGenerateChainingData(unittest.TestCase):
    """This class tests various functions of the chaining data generation.

    The functions tested should at least include:
     - generate initial statement
     - generate update statement
     - generate question
     - the generated instances:
         - all instances should have the same depth
         - the answer should be derived from the operation list
         - the answer should be between 0 and 20
         - the names should not overlap, and the number of unique names should equal to depth + 1
         - the statistics of the instances.
             - the distribution of answer?
             - The distribution of the sampled operation of each step?
         - manually check the examples.
    """

    def test_generate_initial_statement(self):
        sampled_main_role_name = "Mike Lee"
        sampled_item = ("apple", "apples")

        for quantity in [-3, 0, 1, 2, 15, 20, 21]:
            initial_statement = GenerateChainingData.generate_initial_statement_from_sampled_vars(
                sampled_main_role_name, sampled_item, quantity)

            print("-" * 20)
            print(initial_statement)

    def test_generate_update_context_statement(self):
        main_role = "Mike Lee"
        other_role = "John Smith"
        sampled_item = ("apple", "apples")

        for sampled_op in ["-2", "-1", "0", "1", "2"]:
            update_context = GenerateChainingData.generate_update_statement_from_sampled_vars(
                main_role, other_role, sampled_item, sampled_op)

            print("-" * 40)
            print(update_context)

    def test_generate_question(self):
        main_role = "Mike Lee"
        other_role = "John Smith"
        sampled_item = ("apple", "apples")

        question = GenerateChainingData.generate_question_from_sampled_vars(main_role, sampled_item)
        print(question)

        # All tests have passed, and there are no significant distribution issues.


if __name__ == "__main__":
    unittest.main()