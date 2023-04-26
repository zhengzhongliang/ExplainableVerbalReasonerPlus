import json
import numpy as np
import random
import hashlib
import os
import copy

from preliminary_experiments.data_generation.data_utils import DataUtils, DatasetUtils
from preliminary_experiments.data_generation.data_base_class import DataBase


from transformers import T5Tokenizer


class GenerateTreeSearchData(DataBase):

    @classmethod
    def generate_update_statement_from_sampled_vars(cls, sampled_main_role_name, sampled_other_role_name, sampled_item,
                                                    sampled_op):

        """
        The function is to generate statements such as "A has k items. "
        :param sampled_main_role_name:
        :param sampled_other_role_name:
        :param sampled_item:
        :param sampled_op:
        :return:
        """

        # TODO: later we might want to add more variations to the dataset.
        op_num = int(sampled_op)
        noun_form_index = 0 if op_num in [-1, 1] else 1  # This determines whether to use single or plural form
        if op_num > 0:
            update_statement = sampled_other_role_name + " gave " + sampled_main_role_name + " " + str(op_num) + " " + \
                               sampled_item[noun_form_index] + "."
        elif op_num == 0:
            update_statement = sampled_other_role_name + " did not give " + sampled_main_role_name + " any " + \
                               sampled_item[
                                   1] + "."
        else:
            update_statement = sampled_main_role_name + " gave " + sampled_other_role_name + " " + str(
                abs(op_num)) + " " + sampled_item[
                                   noun_form_index] + "."

        return update_statement

    @classmethod
    def generate_grounded_statements(
            cls,
            n_statements_to_gen,
            existing_grounded_chara_item,
            existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item,
            cand_names_set,
            debug_flag=False
        ):
        """
        This function and a few following functions generates statements, and can optionally avoid some statements in
        the generation. There are different variants of this function.

        There are 6 places in total that need the statement generation:
         - initial grounded statements
         - initial ungrounded statements [avoid the grounded (c, q, i) of the first step]
         For each reasoning step:
          - conclusion of grounded rule: [avoid the grounded (c, i) and ungrounded (c, q, i) of any previous steps]
          - conclusion of ungrounded rule: [avoid the grounded and ungrounded (c, q, i) of any previous steps]
          - ungrounded statements: used for ungrounded rule preconditions. [avoid the grounded and ungrounded (c, q, i) of any previous steps]
          - distractors: [avoid the grounded and ungrounded (c, q, i) of any previous steps]

        We need three statement sampling function:
         - Sample grounded statements: needs to avoid overlapped statements.
         - Sample ungrounded statements/distractor: need to avoid overlapped statements.

        We need to maintain three sets so that we can keep track of the existing statements:
         - One storing grounded (q, i). This is used for the grounded rule conclusion generation
         - One storing grounded (q, c, i), used for ungrounded statement/distractor generation
         - One storing ungrounded (q, c, i), used for ungrounded statement/distractor generation

        :param n_statements_to_gen: number of statements to generate
        :param statements_to_avoid: what character name that should not be generated in the process, should be a set.
        :return:
        """

        statements = []

        # A/B is the person name, X/Y is the item. AX is the original statement's person name and item name.
        statement_types = ["AmY", "AnY", "BmX", "BnX", "BmY", "BnY"]

        while len(statements) < n_statements_to_gen:

            # When there are more than 1 statement, give the option to choose harder conclusions.
            if len(statements) >= 1:
                statement_type = random.choice(statement_types)
                sampled_statement = random.choice(statements)
                if statement_type[0] == "A":
                    main_chara = sampled_statement[0]
                else:
                    main_chara = random.choice(list(cand_names_set))

                if statement_type[1] == "m":
                    item_quantity = sampled_statement[1]
                else:
                    item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])

                if statement_type[2] == "X":
                    item = sampled_statement[2]
                else:
                    item = random.choice(cls.items)

                if debug_flag:
                    print("=" * 40)
                    print("existing statements:", statements)
                    print("sampled statement type:", statement_type)
                    print("sampled statement:", sampled_statement)

            else:
                main_chara = random.choice(list(cand_names_set))   # choice: randomly select one element from the list
                item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])
                item = random.choice(cls.items)

                if debug_flag:
                    print("=" * 40)
                    print("existing statements:", statements)

            statement_tuple = (main_chara, item_quantity, item)
            if ((main_chara, item) not in existing_grounded_chara_item and
                    statement_tuple not in existing_ungrounded_chara_quant_item):
                statements.append(statement_tuple)
                existing_grounded_chara_item.add((main_chara, item))
                existing_grounded_chara_quant_item.add(statement_tuple)

            if debug_flag:
                print("generated statement:", statement_tuple)
                print("new statements:", statements)
                print("grounded chara item:", existing_grounded_chara_item)
                print("grounded chara quant item:", existing_grounded_chara_quant_item)
                input("-" * 40)

        return (statements,
                existing_grounded_chara_item,
                existing_grounded_chara_quant_item)

    @classmethod
    def generate_ungrounded_statements(cls,
                                       n_statements_to_gen,
                                       step_grounded_statements,
                                       existing_grounded_chara_quant_item,
                                       existing_ungrounded_chara_quant_item,
                                       debug_flag=False
                                       ):
        """
        This samples certain numbers of ungrounded statements, given the grounded statements.
        Previously the ungrounded statements are generated so that the character names have no overlap with the grounded statements.
        However, here we change this, so that the ungrounded statements are diverse enough, so that (hopeful) it is not
        too easy for the model to learn these patterns.
        :param n_statements_to_gen: number of ungrounded statements to generate
        :param step_grounded_statements: this is the grounded statements that we might use in the generation
        :param all_statements: this is used to make sure we don't sample repeated/contradicted things.
        :return:

        This function is used in two places (three places originally, but the distractor sampling is now obsolete):
         - ungrounded rule conclusion sampling,
         - ungrounded statement sampling,
        """

        # Assuming the original grounded query is "AmX", A for chara name, m for quantity, X for item.
        statement_types = ["AmY", "AnX", "AnY", "BmX", "BmY", "BnX", "BnY"]

        ungrounded_statements = []

        while len(ungrounded_statements) < n_statements_to_gen:
            statement_type = random.choice(statement_types)
            sampled_grounded_statement = random.choice(step_grounded_statements)

            if debug_flag:
                print("=" * 40)
                print("step grounded statements:", step_grounded_statements)
                print("sampled statement type:", statement_type)
                print("sampled grounded statement:", sampled_grounded_statement)

            if statement_type[0] == "A":
                main_chara = sampled_grounded_statement[0]
            else:  # distractor_type[0] == "B"
                main_chara = random.choice(cls.full_names)

            if statement_type[1] == "m":
                item_quantity = sampled_grounded_statement[1]
            else:
                item_quantity = random.randint(cls.num_item_bound[0], cls.num_item_bound[1])

            if statement_type[2] == "X":
                item = sampled_grounded_statement[2]
            else:
                item = random.choice(cls.items)

            statement_tuple = (main_chara, item_quantity, item)
            if statement_tuple not in existing_grounded_chara_quant_item and statement_tuple not in existing_ungrounded_chara_quant_item:
                ungrounded_statements.append(statement_tuple)
                existing_ungrounded_chara_quant_item.add(statement_tuple)

            if debug_flag:
                print("generated ungrounded statement:", statement_tuple)
                print("existing ungounded statements:", existing_ungrounded_chara_quant_item)
                input("-" * 40)

        return ungrounded_statements, existing_ungrounded_chara_quant_item

    @classmethod
    def generate_statement_combination(cls, statements):
        """
        E.g., if the statements is [a, b, c], the combination should be [a, b], [a, c], [b, c],
        :return:
        """

        statements_combination = []

        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):

                if random.random() > 0.5:
                    statements_combination.append((statements[j], statements[i]))
                else:
                    statements_combination.append((statements[i], statements[j]))

        return statements_combination

    @classmethod
    def generate_initial_grounded_ungrounded_statements(
            cls,
            initial_statements_grounded, n_initial_statements_grounded,
            initial_statements_ungrounded, n_initial_statements_ungrounded,
            existing_grounded_chara_item, existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item,
            cand_names_set
    ):
        """Handles different cases of generating initial grounded and ungrounded statements.

        First handles the grounded statements.
         - If the initial grounded statements are empty, generate it.
         - If the initial grounded statements are not empty, use them, and update the existing_grounded_chara_item and
            existing_grounded_char_quant_item

        Then handles the ungrounded statements.
         - If the initial ungrounded statements are empty, generate it.
         - If the initial ungrounded statements are not empty, use them and update existing_ungrounded_chara_quant_item
        """

        if len(initial_statements_grounded) == 0:
            (initial_statements_grounded, existing_grounded_chara_item,
             existing_grounded_chara_quant_item) = cls.generate_grounded_statements(
                n_initial_statements_grounded, existing_grounded_chara_item,
                existing_grounded_chara_quant_item, existing_ungrounded_chara_quant_item,
                cand_names_set
            )
        else:
            if len(initial_statements_grounded) > n_initial_statements_grounded:
                initial_statements_grounded = random.sample(
                    initial_statements_grounded, n_initial_statements_grounded)
            # Update the existing_grounded_chara_item and existing_grounded_chara_quant_item
            existing_grounded_chara_item_ = set([(x[0], x[2]) for x in initial_statements_grounded])
            existing_grounded_chara_quant_item_ = set([(x[0], x[1], x[2]) for x in initial_statements_grounded])
            existing_grounded_chara_item.update(existing_grounded_chara_item_)
            existing_grounded_chara_quant_item.update(existing_grounded_chara_quant_item_)

        if len(initial_statements_ungrounded) == 0:
            (initial_statements_ungrounded,
             existing_ungrounded_chara_quant_item) = cls.generate_ungrounded_statements(
                n_initial_statements_ungrounded, initial_statements_grounded, existing_grounded_chara_quant_item,
                existing_ungrounded_chara_quant_item)
        else:
            if len(initial_statements_ungrounded) > n_initial_statements_ungrounded:
                initial_statements_ungrounded = random.sample(
                    initial_statements_ungrounded, n_initial_statements_ungrounded)
            existing_ungrounded_chara_quant_item_ = [(x[0], x[1], x[2]) for x in initial_statements_ungrounded]
            existing_ungrounded_chara_quant_item.update(existing_ungrounded_chara_quant_item_)

        return (initial_statements_grounded, initial_statements_ungrounded,
                existing_grounded_chara_item, existing_grounded_chara_quant_item,
                existing_ungrounded_chara_quant_item)

    @classmethod
    def generate_one_structured_example(cls, depth, k,
                                        initial_statements_grounded,
                                        initial_statements_ungrounded,
                                        existing_grounded_chara_item,
                                        existing_grounded_chara_quant_item,
                                        existing_ungrounded_chara_quant_item,
                                        names_to_subtract,
                                        debug_flag=False):
        """
        This function recursively generates the tree-search data. For simplicity, we use forward chaining here, so that
        later it might be combined with the pattern 0 chaining data.

        The task is to determine whether a statement is true of false. If we could not prove a statement is true, it is false.

        It returns the structured data (formal representations in each reasoning step). The translation from the
        structured data to natural language expression is handled by a separate function.

        Generation process:
         - The depth determines the reasoning steps.
         - In each reasoning step, we have
             - (grounded facts) -> (grounded rules) -> (grounded facts)
             - (ungrounded facts) -> (ungrounded rules) -> (ungrounded facts)
         - The final query is done by either sampling from the last step's grounded facts or the last step's ungrounded facts.
        :return:
        """

        # This is a set of full names from which we sample the names for the statements.
        # To avoid the confusion of the context, the statements sampled later should have different names than the
        # names in the statements that are sampled earlier.
        cand_names_set = set(cls.full_names) - names_to_subtract

        # 1: determine how many initial statements to generate:
        n_initial_statements_grounded = random.randint(2, k)  # sample number of initial statements, both boundaries included.
        n_initial_statements_ungrounded = n_initial_statements_grounded

        # 2: sample whether the final query can be proved or not (i.e., positive class or negative class)
        instance_label = 1 if random.random() > 0.5 else 0

        # Sample the initial statements (subject, number and items), both grounded and ungrounded.
        (initial_statements_grounded, initial_statements_ungrounded,
         existing_grounded_chara_item, existing_grounded_chara_quant_item,
         existing_ungrounded_chara_quant_item) = cls.generate_initial_grounded_ungrounded_statements(
            initial_statements_grounded, n_initial_statements_grounded,
            initial_statements_ungrounded, n_initial_statements_ungrounded,
            existing_grounded_chara_item, existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item,
            cand_names_set=cand_names_set
        )

        if debug_flag:
            print("=" * 40)
            print("initial statements:")
            print(initial_statements_grounded)
            print("initial ungrounded statements:")
            print(initial_statements_ungrounded)

        # 2: use recursive function to generate each step's run and get the entailed statements.
        # Forms of rules: (one var, grounded), (one var, ungrounded), (two var grounded), (two var ungrounded)
        # Temporarily sample rules at each step.
        # The work can be done by a for loop not necessarily recursive function.

        # Workflow: each step we sample:
        #  - 1 grounded 1 var rule + 2 grounded 2 var rule + 1 ungrounded 1 var rule + 2 ungrounded 2 var rule
        # Also note that in the generation process, we want there to be back-tracking cases, so that in some steps we
        #   might need to have the cases where we have both A -> C and B -> C, but only A is grounded.
        # This means we need another probability where at each step, some grounded and ungrounded rules lead
        # to the same conclusion

        # 2.1 First sample the grounded 1 var, grounded 2 var, ungrounded 1 var, ungrounded 2 var with certain probs:
        grounded_rule_types_to_sample = [["grounded 1 var", "grounded 1 var"],
                                         ["grounded 1 var", "grounded 2 var"]]
        ungrounded_rule_types_to_sample = [["ungrounded 1 var", "ungrounded 1 var"],
                                           ["ungrounded 1 var", "ungrounded 2 var"]]

        # This stores the entailed or non-entailed statements at each step.
        generated_statements = {
            "grounded": [copy.deepcopy(initial_statements_grounded)],
            "ungrounded": [copy.deepcopy(initial_statements_ungrounded)],
            "distractors": []
        }

        # This stores the sampled rules at each step:
        sampled_rules = {
            "grounded 1 var": [],
            "grounded 2 var": [],
            "ungrounded 1 var": [],
            "ungrounded 2 var": [],
            "backtracking": []
        }

        # Initialize the final question triple:
        question_triple = ()

        # This is to store what statements we have sampled. We should not sample these again when creating new rules
        for reasoning_step in range(1, depth + 1):
            # At least one ungrounded rule should serve as the distractor to trigger backtracking.
            # I.e., that ungrounded rule should have the same right-hand argument as one of the grounded rules.

            # This stores the entailed statements for grounded and ungrounded at each step.
            step_statements = {
                "grounded 1 var": [],
                "grounded 2 var": [],
                "ungrounded 1 var": [],
                "ungrounded 2 var": []
            }

            step_rules = {
                "grounded 1 var": [],
                "grounded 2 var": [],
                "ungrounded 1 var": [],
                "ungrounded 2 var": []
            }

            # Get the valid and invalid statements generated from the last step. including all of the 2-var combinations
            # E.g., [a, b, c] -> [[a, a], [a, b], [a, c], ... , [c, c]]
            # The sampled rules rely on the valid and invalid statements generated from the last step.
            last_step_grounded_statements = generated_statements["grounded"][-1]
            last_step_grounded_statements_cartesian = cls.generate_statement_combination(last_step_grounded_statements)

            # Sample what types of rules to generate at each reasoning step
            # The following sampling technique makes sure that there is at least one grounded rule and one ungrounded
            #   rule at each step.
            sampled_rule_types = random.choice(grounded_rule_types_to_sample) + \
                                 random.choice(ungrounded_rule_types_to_sample)

            # First handle the grounded rules.
            n_grounded_1_var = sampled_rule_types.count("grounded 1 var")

            # Sample the preconditions of grounded 1 variable rules, without replacement.
            n_grounded_1_var_sampled_precondition = random.sample(last_step_grounded_statements, n_grounded_1_var)

            # Sample the conclusions of grounded 1 variable rules.
            (grounded_s_1var, existing_grounded_chara_item,
             existing_grounded_chara_quant_item) = cls.generate_grounded_statements(
                n_grounded_1_var,
                existing_grounded_chara_item=existing_grounded_chara_item,
                existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item,
                cand_names_set=cand_names_set
            )

            for idx_gr in range(len(grounded_s_1var)):
                step_rules["grounded 1 var"].append(
                    ((n_grounded_1_var_sampled_precondition[idx_gr], ),
                     grounded_s_1var[idx_gr])
                )
                step_statements["grounded 1 var"].append(grounded_s_1var[idx_gr])

            # Sample the preconditions and conclusions for grounded 2 variable rules.
            n_grounded_2_var = sampled_rule_types.count("grounded 2 var")
            n_grounded_2_var_sampled_precondition = random.sample(last_step_grounded_statements_cartesian, n_grounded_2_var)

            (grounded_s_2var, existing_grounded_chara_item,
             existing_grounded_chara_quant_item) = cls.generate_grounded_statements(
                n_grounded_2_var,
                existing_grounded_chara_item=existing_grounded_chara_item,
                existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item,
                cand_names_set=cand_names_set
            )

            for idx_gr in range(len(grounded_s_2var)):
                step_rules["grounded 2 var"].append(
                    (n_grounded_2_var_sampled_precondition[idx_gr],
                     grounded_s_2var[idx_gr])
                )
                step_statements["grounded 2 var"].append(grounded_s_2var[idx_gr])

            # Now handle the ungrounded rules at each step
            # The ungrounded rules come from two places:
            # - one of the preconditions are new facts.
            # - one of the preconditions are based on previously entailed invalid statements.
            # How to do this: first generate a pool of all invalid statements. Then the generation process is similar to the grounded ones.
            # At each step, make sure there might be one ungrounded rule that lead to the same conclusion as the grounded rules
            #   so that we can make sure solving the problem requires backtracking.

            # TODO: the ungrounded 2 var: does it include the situation where 1 of the 2 preconditions is provable? No.
            #   But we might not need it in the near future.

            n_ungrounded_1_var = sampled_rule_types.count("ungrounded 1 var")
            n_ungrounded_2_var = sampled_rule_types.count("ungrounded 2 var")

            # The ungrounded statement pool comes from both the generated ones and the previous entailed ungrounded ones.
            ungrounded_precondition_candidates = [s for s in generated_statements["ungrounded"][-1]]
            ungrounded_precondition_candidates_cartesian = cls.generate_statement_combination(ungrounded_precondition_candidates)

            # Sample the preconditions of ungrounded 1 variable rules.
            n_ungrounded_1_var_sampled_precondition = random.sample(ungrounded_precondition_candidates, n_ungrounded_1_var)

            # Sample the conclusions of grounded 1 variable rules.
            ungrounded_s_1var, existing_ungrounded_chara_quant_item = \
                cls.generate_ungrounded_statements(
                    n_ungrounded_1_var,
                    step_grounded_statements=step_statements["grounded 1 var"] + step_statements["grounded 2 var"],
                    existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                    existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

            for idx_gr in range(len(ungrounded_s_1var)):
                step_rules["ungrounded 1 var"].append(
                    ((n_ungrounded_1_var_sampled_precondition[idx_gr], ),
                     ungrounded_s_1var[idx_gr])
                )
                step_statements["ungrounded 1 var"].append(ungrounded_s_1var[idx_gr])

            # Sample the preconditions and conclusions for grounded 2 variable rules.
            # We sample n_ungrounded_2_var + 1 preconditions, reserve one for the back-tracking branch
            n_ungrounded_2_var_sampled_precondition = random.sample(ungrounded_precondition_candidates_cartesian,
                                                                  n_ungrounded_2_var)

            ungrounded_s_2var, existing_ungrounded_chara_quant_item = \
                cls.generate_ungrounded_statements(
                    n_ungrounded_2_var,
                    step_grounded_statements=step_statements["grounded 1 var"] + step_statements["grounded 2 var"],
                    existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
                    existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item)

            for idx_gr in range(len(ungrounded_s_2var)):
                step_rules["ungrounded 2 var"].append(
                    (n_ungrounded_2_var_sampled_precondition[idx_gr],
                     ungrounded_s_2var[idx_gr])
                )
                step_statements["ungrounded 2 var"].append(ungrounded_s_2var[idx_gr])

            # Store the generated rules and statements for each step.
            generated_statements["grounded"].append(
                step_statements["grounded 1 var"] + step_statements["grounded 2 var"])
            generated_statements["ungrounded"].append(step_statements["ungrounded 1 var"] + step_statements[
                "ungrounded 2 var"])

            sampled_rules["grounded 1 var"].append(step_rules["grounded 1 var"])
            sampled_rules["grounded 2 var"].append(step_rules["grounded 2 var"])
            sampled_rules["ungrounded 1 var"].append(step_rules["ungrounded 1 var"])
            sampled_rules["ungrounded 2 var"].append(step_rules["ungrounded 2 var"])

            # Generate one rule for potential back-tracking at each step
            # We generate three types of candidate backtracking branches:
            #  - ungrounded fact 1 -> ungrounded fact 2
            #  - ungrounded fact 1 + ungrounded fact 2 -> grounded fact 3
            #  - grounded fact 1 + ungrounded fact 2 -> grounded fact 3

            # One problem found: for the grounded statement to prove, usually the backtracking rule will contribute some additional information.
            # The solutions is just to add backtracking rule for ungrounded statement as well.

            # Use a random number to determine which type of backtracking branch we want
            backtracking_branch_type = random.choice([0, 1, 2])

            # Get the candidate grounded and ungrounded statements
            prev_step_grounded_statements = generated_statements["grounded"][reasoning_step - 1]
            prev_step_ungrounded_statements = generated_statements["ungrounded"][reasoning_step - 1]

            step_grouneded_n_ungrounded_statements = generated_statements["grounded"][reasoning_step] + \
                                                     generated_statements["ungrounded"][reasoning_step]

            if backtracking_branch_type == 0:
                backtrack_rule_precondition = random.choice(prev_step_ungrounded_statements)
                backtrack_rule_conclusion = random.choice(step_grouneded_n_ungrounded_statements)
                backtrack_rule = ((backtrack_rule_precondition, ), backtrack_rule_conclusion)
            elif backtracking_branch_type == 1:
                backtrack_rule_preconditions = tuple(random.sample(prev_step_ungrounded_statements, 2))
                backtrack_rule_conclusion = random.choice(step_grouneded_n_ungrounded_statements)
                backtrack_rule = (backtrack_rule_preconditions, backtrack_rule_conclusion)
            else:
                backtrack_rule_precondition1 = random.choice(prev_step_ungrounded_statements)
                backtrack_rule_precondition2 = random.choice(prev_step_grounded_statements)
                backtrack_rule_conclusion = random.choice(step_grouneded_n_ungrounded_statements)

                # The order of the preconditions should not be determined.
                if random.random() > 0.5:
                    backtrack_rule = (
                        (backtrack_rule_precondition1, backtrack_rule_precondition2),
                        backtrack_rule_conclusion
                    )
                else:
                    backtrack_rule = (
                        (backtrack_rule_precondition2, backtrack_rule_precondition1),
                        backtrack_rule_conclusion
                    )

            if backtrack_rule not in sampled_rules["backtracking"] and \
                    backtrack_rule not in step_rules["ungrounded 1 var"] and \
                    backtrack_rule not in step_rules["ungrounded 2 var"]:
                sampled_rules["backtracking"].append(backtrack_rule)
            else:
                sampled_rules["backtracking"].append(None)

            generated_statements["distractors"].append([])

            if debug_flag:
                print("-" * 40)
                print("grounded r 1 var:")
                for r in step_rules["grounded 1 var"]:
                    print("\t", r)
                print("grounded r 2 var:")
                for r in step_rules["grounded 2 var"]:
                    print("\t", r)
                print("grounded s 1 var:")
                for r in step_statements["grounded 1 var"]:
                    print("\t", r)
                print("grounded s 2 var:")
                for r in step_statements["grounded 2 var"]:
                    print("\t", r)

                print("\n")

                print("ungrounded r 1 var:")
                for r in step_rules["ungrounded 1 var"]:
                    print("\t", r)
                print("ungrounded r 2 var:")
                for r in step_rules["ungrounded 2 var"]:
                    print("\t", r)
                print("ungrounded s 1 var:")
                for r in step_statements["ungrounded 1 var"]:
                    print("\t", r)
                print("ungrounded s 2 var:")
                for r in step_statements["ungrounded 2 var"]:
                    print("\t", r)

                print("\n")

                print("step backtrack rule")
                print("\t", sampled_rules["backtracking"][-1])

                print("\n")

                input("-" * 40)

        # Finally generate the question and answer.
        if instance_label == 1:
            question_triple = random.choice(
                generated_statements["grounded"][-1])
            answer = "Yes"
        else:
            question_triple = random.choice(
                generated_statements["ungrounded"][-1])
            answer = "No"

        instance = {
            "depth": depth,
            "provable": instance_label,
            "statements": generated_statements,
            "rules": sampled_rules,
            "question": question_triple,
            "answer": answer,
        }

        return instance

    @classmethod
    def convert_formal_statement_to_natural_language(cls, statement):

        if statement[1] == 0:
            statement_nl = statement[0] + " had no " + str(statement[2][1]) + "."
        elif statement[1] == 1:
            statement_nl = statement[0] + " had 1 " + str(statement[2][0]) + "."
        else:
            statement_nl = statement[0] + " had " + str(statement[1]) + " " + str(statement[2][1]) + "."

        return statement_nl

    @classmethod
    def convert_formal_query_to_natural_language(cls, query_rep):

        if query_rep[1] > 1:
            question_nl = "Did " + query_rep[0] + " have " + str(query_rep[1]) + " " + query_rep[2][1] + "?"
        else:
            question_nl = "Did " + query_rep[0] + " have " + str(query_rep[1]) + " " + query_rep[2][0] + "?"

        return question_nl

    @classmethod
    def convert_formal_rules_to_natural_language(cls, rule):

        preconds_nl = [cls.convert_formal_statement_to_natural_language(precond) for precond in rule[0]]
        conclusion_nl = cls.convert_formal_statement_to_natural_language(rule[1])

        if len(preconds_nl) == 1:
            rule_nl = "If " + preconds_nl[0][:-1] + " then " + conclusion_nl
        else:
            rule_nl = "If " + preconds_nl[0][:-1] + " and " + preconds_nl[1][:-1] + " then " + conclusion_nl

        return rule_nl

    @classmethod
    def generate_natural_language_expressions_from_structured_example(cls, instance):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]
                         # Only the initial grounded statement should be added

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r != None]

        query = instance["question"]

        all_text = []

        all_statements = [cls.convert_formal_statement_to_natural_language(s) for s in all_statements]
        all_statements_indices = list(range(len(all_statements)))
        random.shuffle(all_statements_indices)  # The indices are now shuffled.
        all_statements = [all_statements[i] for i in all_statements_indices]   # The statements are now shuffled too.
        all_statements_indices_shuffle_map = {old_idx: new_idx for new_idx, old_idx in enumerate(all_statements_indices)}

        all_rules = [cls.convert_formal_rules_to_natural_language(r) for r in all_rules]
        all_rules_indices = list(range(len(all_rules)))
        random.shuffle(all_rules_indices)  # The indices are now shuffled.
        all_rules = [all_rules[i] for i in all_rules_indices]  # The statements are now shuffled too.
        all_rules_indices_shuffle_map = {old_idx: new_idx for new_idx, old_idx in
                                              enumerate(all_rules_indices)}

        question = cls.convert_formal_query_to_natural_language(query)

        all_text.extend(all_statements)
        all_text.extend(all_rules)
        all_text.append(question)

        return (all_text, all_statements, all_rules, question, all_statements_indices_shuffle_map,
                all_rules_indices_shuffle_map)

    @classmethod
    def generate_id_from_context_using_hash(cls, context_string):
        cls.hash_module.update(context_string.encode("utf-8"))

        return cls.hash_module.hexdigest()

    @classmethod
    def generate_statement_and_rule_indices_shuffle_map(cls, instance, debug_flag=False):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]
        # Only the initial grounded statement should be added

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r != None]

        all_statements_nl = [cls.convert_formal_statement_to_natural_language(s) for s in all_statements]

        all_rules_nl = [cls.convert_formal_rules_to_natural_language(r) for r in all_rules]

        n_statements = len(all_statements)
        n_rules = len(all_rules)

        statement_indices_shuffle_map = {all_statements_nl.index(instance["context_list"][new_idx]): new_idx
                                         for new_idx in range(n_statements)}

        rule_indices_shuffle_map = {all_rules_nl.index(instance["context_list"][new_idx + n_statements]): new_idx
                                    for new_idx in range(n_rules)}

        all_sents_nl = [all_statements_nl[idx] for idx in statement_indices_shuffle_map.keys()] + \
                       [all_rules_nl[idx] for idx in rule_indices_shuffle_map.keys()]

        if debug_flag:
            print("=" * 40)
            for idx in range(len(all_sents_nl)):
                print(instance["context_list"][idx], all_sents_nl[idx])
            input("-" * 40)

        assert instance["context_list"] == all_sents_nl

        return statement_indices_shuffle_map, rule_indices_shuffle_map

    @classmethod
    def generate_tree_search_backward_select_evidence_one_instance(cls, instance, tokenizer, debug_flag=False):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]
        # Only the initial grounded statement should be added

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r != None]

        all_statements = [all_statements[int(old_idx)]
                          for old_idx in instance["statement_indices_shuffle_map"].keys()]

        all_rules = [all_rules[int(old_idx)]
                     for old_idx in instance["rule_indices_shuffle_map"].keys()]

        all_sents = all_statements + all_rules

        pred, traversal_history, depth_history = cls._backward_recursive(
            query=instance["question"],
            all_statements=all_statements,
            all_rules=all_rules,
            traversal_history=[],
            depth_history=[]
        )

        target_text = " ".join([instance["context_list"][ctx_idx] for ctx_idx in traversal_history]) + \
                      " answer: " + instance["answer"]
        target_text_len = len(tokenizer(target_text)["input_ids"])

        if debug_flag:
            print("=" * 40)

            cls.debug_print_traversal_history(
                instance["question_string"], instance["context_list"], pred, instance["answer"],
                traversal_history, depth_history, instance)

            print("\n")

            print(target_text)
            print("target text len:", target_text_len)

            input("-" * 40)

        assert pred == (True if instance["answer"] == "Yes" else False)

        # TODO: might also consider the data in the way that includes the sent1, sent2, ... indicator and represent the
        # target text by the position of the evidence sentence

        return target_text, target_text_len, traversal_history, depth_history

    @classmethod
    def generate_one_example(
            cls,
            depth,
            k,
            tokenizer,
            initial_statements_grounded,
            initial_statements_ungrounded,
            existing_grounded_chara_item,
            existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item,
            names_to_subtract,
            debug_flag=False
    ):
        instance = cls.generate_one_structured_example(
            depth=depth, k=k, initial_statements_grounded=initial_statements_grounded,
            initial_statements_ungrounded=initial_statements_ungrounded,
            existing_grounded_chara_item=existing_grounded_chara_item,
            existing_grounded_chara_quant_item=existing_grounded_chara_quant_item,
            existing_ungrounded_chara_quant_item=existing_ungrounded_chara_quant_item,
            names_to_subtract=names_to_subtract,
            debug_flag=debug_flag)

        all_text, all_statements, all_rules, question, statement_indices_shuffle_map, rule_indices_shuffle_map = \
            cls.generate_natural_language_expressions_from_structured_example(instance)
        instance["context_list"] = all_text[:-1]  # context = all statements and rules, without question
        instance["context_string"] = " ".join(instance["context_list"])
        instance["question_string"] = question
        instance["statement_indices_shuffle_map"] = statement_indices_shuffle_map
        instance["rule_indices_shuffle_map"] = rule_indices_shuffle_map

        (target_text_w_inter, target_text_w_inter_len, traversal_history,
         depth_history) = GenerateTreeSearchData.generate_tree_search_backward_select_evidence_one_instance(
            instance, tokenizer,
        )

        instance["target_text_w_inter"] = target_text_w_inter

        GenerateTreeSearchDataRuntimeChecks.runtime_check_generated_tree_search_data_one_example(instance, depth)
        GenerateTreeSearchDataRuntimeChecks.runtime_check_generated_tree_search_data_with_backward_chaining_one_example(
            instance)

        return instance

    @classmethod
    def generate_data_with_certain_depth(cls, depth, num_train, num_dev, num_test, debug_flag=False):
        '''
        We should first use tree search (recursive search) to get the formal representations, then translate them to
        natural language expressions.
        :param depth:
        :param num_train:
        :param num_dev:
        :param num_test:
        :param debug_flag:
        :return:
        '''

        random.seed(depth)

        splits = ["train", "dev", "test"]

        num_instances = {
            "train": num_train,
            "dev": num_dev,
            "test": num_test
        }

        instances_all_splits = {split: [] for split in splits}

        existing_instances = {}  # This is used to make sure we don't generate repeating examples.

        tokenizer = T5Tokenizer.from_pretrained("t5-small")  # This is used to check the length of the input

        for split in splits:

            while len(instances_all_splits[split]) < num_instances[split]:

                instance = cls.generate_one_example(
                    depth=depth, k=3, tokenizer=tokenizer,
                    initial_statements_grounded=[],
                    initial_statements_ungrounded=[],
                    existing_grounded_chara_item=set([]),
                    existing_grounded_chara_quant_item=set([]),
                    existing_ungrounded_chara_quant_item=set([]),
                    names_to_subtract=set([]),
                    debug_flag=debug_flag
                )

                if instance["context_string"] not in existing_instances:
                    instances_all_splits[split].append(instance)
                    existing_instances[instance["context_string"]] = 1

                    instance["id"] = cls.generate_id_from_context_using_hash(instance["context_string"])

                    len_tokenized_input = len(tokenizer(instance["context_string"])["input_ids"])
                    instance["context_len"] = len_tokenized_input

        return instances_all_splits

    @classmethod
    def generate_data_all_depths(cls):

        """
        This function generates data with various depth
        :return:
        """

        n_train = 10000
        n_dev = 1000
        n_test = 1000

        data_folder_dir = os.path.join(cls.project_data_folder_path, "tree_search_v1.0/")

        if not os.path.exists(data_folder_dir):
            os.mkdir(data_folder_dir)

        data_with_various_depth_raw = {}
        for d in [0, 1, 2, 3, 4]:
            print("=" * 40)
            print(f"Generating tree search data with depth {d}")
            chaining_data = cls.generate_data_with_certain_depth(depth=d,
                                                                 num_train=n_train,
                                                                 num_dev=n_dev,
                                                                 num_test=n_test)

            data_with_various_depth_raw[d] = chaining_data

        chaining_data_by_du = {}
        for du in [2, 4]:
            n_train_per_depth = int(n_train / (du + 1))
            n_dev_per_depth = int(n_dev / (du + 1))
            n_test_per_depth = int(n_test / (du + 1))

            chaining_data_by_du[du] = {"train": [], "dev": [], "test": []}
            for d in range(du + 1):
                chaining_data_by_du[du]["train"].extend(data_with_various_depth_raw[d]["train"][:n_train_per_depth])
                chaining_data_by_du[du]["dev"].extend(data_with_various_depth_raw[d]["dev"][:n_dev_per_depth])
                chaining_data_by_du[du]["test"].extend(data_with_various_depth_raw[d]["test"][:n_test_per_depth])

            chaining_data_by_du[du]["statistics"] = DatasetUtils.get_dataset_statistics(chaining_data_by_du[du])

            print("=" * 40)
            print("du ", du)
            print("statistics:")
            print(json.dumps(chaining_data_by_du[du]["statistics"], indent=2))

            with open(data_folder_dir + "tree_search_data_du" + str(du) + ".json", "w") as handle:
                json.dump(chaining_data_by_du[du], handle)

    @classmethod
    def update_statement_rule_indices_shuffle_map(cls, debug_flag=False):

        for du in [2, 5]:

            data_dir = "/Users/zhengzhongliang/NLP_Research/2022_ThinkInNaturalLanguage/data_generated/" + \
                "tree_search_v0.5/tree_search_data_du" + str(du) + ".json"

            instances_all_splits = DataUtils.load_json(data_dir)

            for split in ["train", "dev", "test"]:

                for instance in instances_all_splits[split]:

                    statement_indices_shuffle_map, rule_indices_shuffle_map = \
                        cls.generate_statement_and_rule_indices_shuffle_map(instance, debug_flag)

                    instance["statement_indices_shuffle_map"] = statement_indices_shuffle_map
                    instance["rule_indices_shuffle_map"] = rule_indices_shuffle_map

            with open(data_dir, "w") as handle:
                json.dump(instances_all_splits, handle)

    @classmethod
    def _backward_recursive(cls, query, all_statements, all_rules, traversal_history, depth_history, d=0,
                            debug_flag=False):

        if query in all_statements:

            q_idx = all_statements.index(query)
            traversal_history.append(q_idx)
            depth_history.append(d)

            if debug_flag:
                print("\t" * d, query)

            return True, traversal_history, depth_history

        else:
            for r_idx_, rule in enumerate(all_rules):

                rule_match_flag = rule[1] == query

                if rule_match_flag:
                    # The true r_idx that should be added to history is adjusted by the offset
                    r_idx = len(all_statements) + r_idx_
                    traversal_history.append(r_idx)
                    depth_history.append(d)

                    all_precond_sat = []

                    if debug_flag:
                        print("\t" * d, rule)

                    for precond in rule[0]:
                        s_proved_flag, traversal_history, depth_history = \
                            cls._backward_recursive(
                                precond, all_statements, all_rules, traversal_history, depth_history,
                                d=d + 1, debug_flag=debug_flag
                            )

                        all_precond_sat.append(s_proved_flag)

                        if False in all_precond_sat:
                            break

                    if False not in all_precond_sat:
                        return True, traversal_history, depth_history
            # No rule is matched for this query.
            return False, traversal_history, depth_history

    @classmethod
    def debug_print_traversal_history(cls, query, all_context, pred, answer, traversal_history, depth_history,
                                      instance):

        print("question:", query)
        print("pred:", pred, " answer:", answer)

        print("\n")

        for step_idx in range(len(instance["context_list"])):
            print(str(step_idx) + ": " + instance["context_list"][step_idx])

        print("\n")

        for step_idx in range(len(traversal_history)):
            print("\t" * depth_history[step_idx] + str(traversal_history[step_idx]) + ": " + all_context[
                traversal_history[step_idx]])


class GenerateTreeSearchDataRuntimeChecks(GenerateTreeSearchData):

    @classmethod
    def runtime_check_generated_tree_search_data_one_example(cls, instance, depth):

        # First check the length of the statements and rules
        assert (len(instance["statements"]["grounded"]) == depth + 1)

        assert (len(instance["statements"]["ungrounded"]) == depth + 1)

        for rules_all_steps in instance["rules"].values():
            assert (len(rules_all_steps) == depth)

        for reasoning_depth in range(1, depth + 1):

            # Each 1 var rule should have 1 precondition
            for step_rules in instance["rules"]["grounded 1 var"] + instance["rules"]["ungrounded 1 var"]:
                for rule in step_rules:
                    assert (len(rule[0]) == 1)

            # Each 2 var rule should have 2 preconditions
            for step_rules in instance["rules"]["grounded 2 var"] + instance["rules"]["ungrounded 2 var"]:
                for rule in step_rules:
                    assert (len(rule[0]) == 2)

            # The preconditions of grounded rules should come from last steps grounded statements
            preconditions = [rule[0][0] for rule in instance["rules"]["grounded 1 var"][reasoning_depth - 1]] + \
                            [s for rule in instance["rules"]["grounded 2 var"][reasoning_depth - 1] for s in rule[0]]
            grounded_statements_last_step = [s for s in instance["statements"]["grounded"][reasoning_depth - 1]]
            for precond in preconditions:
                assert (precond in grounded_statements_last_step)

            # The grounded statements should come from the conclusions of the grounded rules.
            conclusions = [rule[1] for rule in instance["rules"]["grounded 1 var"][reasoning_depth - 1] + \
                           instance["rules"]["grounded 2 var"][reasoning_depth - 1]]
            for s in instance["statements"]["grounded"][reasoning_depth]:
                assert (s in conclusions)

            # The preconditions of ungrounded rules should come from the last step's ungrounded statements
            preconditions = [rule[0][0] for rule in instance["rules"]["ungrounded 1 var"][reasoning_depth - 1]] + \
                            [s for rule in instance["rules"]["ungrounded 2 var"][reasoning_depth - 1] for s in
                             rule[0]]
            ungrounded_statements_prev_step = [s for s in instance["statements"]["ungrounded"][reasoning_depth - 1]]
            for precond in preconditions:
                assert (precond in ungrounded_statements_prev_step)

            # The conclusions of the ungrounded rules should go to the ungrounded statements.
            ungrounded_rule_conclusions = [rule[1] for rule in
                                           instance["rules"]["ungrounded 1 var"][reasoning_depth - 1] + \
                                           instance["rules"]["ungrounded 2 var"][reasoning_depth - 1]]
            for ungrounded_conclusion in ungrounded_rule_conclusions:
                assert (ungrounded_conclusion in instance["statements"]["ungrounded"][reasoning_depth])

        statement_str_grounded = [
            s[0] + str(s[1]) + s[2][0] \
            for step_statements in instance["statements"]["grounded"] \
            for s in step_statements
        ]

        statement_str_ungrounded = [
            s[0] + str(s[1]) + s[2][0] \
            for step_statements in instance["statements"]["ungrounded"] \
            for s in step_statements
        ]

        # The grounded statements and ungrounded statements should not overlap
        assert (len(set(statement_str_grounded).intersection(statement_str_ungrounded)) == 0)

        # There should be no repeating facts and rules
        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]
        # Only the initial grounded statement should be added

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r != None]

        assert len(all_statements) == len(set(all_statements))
        assert len(all_rules) == len(set(all_rules))

    @classmethod
    def runtime_check_generated_tree_search_data_with_backward_chaining_one_example(cls, instance, debug_flag=False):

        all_statements = [s for s in instance["statements"]["grounded"][0]] + \
                         [s for step_s in instance["statements"]["distractors"] for s in step_s]

        all_rules = [r for step_r in instance["rules"]["grounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["grounded 2 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 1 var"] for r in step_r] + \
                    [r for step_r in instance["rules"]["ungrounded 2 var"] for r in step_r] + \
                    [r for r in instance["rules"]["backtracking"] if r != None]

        all_statements = [all_statements[int(old_idx)]
                          for old_idx in instance["statement_indices_shuffle_map"].keys()]

        all_rules = [all_rules[int(old_idx)]
                     for old_idx in instance["rule_indices_shuffle_map"].keys()]

        if debug_flag:
            for r in all_rules:
                print(r)
            print("\n")

        # This is to make sure even when the conclusion can be proved eventually, backtracking is needed in some cases.
        query = instance["question"]

        answer = instance["answer"]

        pred_ = cls._recursive_check(query, all_statements, all_rules, debug_flag=debug_flag)
        pred = "No" if pred_ == False else "Yes"

        if debug_flag:
            print("pred:", pred, " answer:", answer)

        assert (pred == answer)

        return pred, answer, all_statements, all_rules

    @classmethod
    def _recursive_check(cls, query, all_statements, all_rules, d=0, debug_flag=False):

        # TODO: we should also probably keep track of the depth of the proof.

        if query in all_statements:

            if debug_flag:
                print("\t" * d, query)

            return True
        else:
            for rule in all_rules:

                rule_match_flag = rule[1] == query

                if rule_match_flag:
                    all_precond_sat = []

                    if debug_flag:
                        print("\t" * d, rule)

                    for precond in rule[0]:
                        s_proved_flag = cls._recursive_check(precond, all_statements, all_rules, d=d + 1,
                                                             debug_flag=debug_flag)
                        all_precond_sat.append(s_proved_flag)
                    if False not in all_precond_sat:
                        return True
            # No rule is matched for this query.
            return False


if __name__ == "__main__":
    GenerateTreeSearchData.generate_data_all_depths()


