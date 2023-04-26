from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent


class CheckGetVariableValue:

    @classmethod
    def check_get_variable_value(cls):
        snippets = [
            {
                "v": " '123' ",
                "l": {},
                "e": {}
            },

            {
                "v": " '  ' ",
                "l": {},
                "e": {}
            },

            {
                "v": "None",
                "l": {},
                "e": {}
            },

            {
                "v": "True",
                "l": {},
                "e": {}
            },

            {
                "v": "False",
                "l": {},
                "e": {}
            },

            {
                "v": "  123   ",
                "l": {},
                "e": {}
            },

            {
                "v": "  #1   ",
                "l": {"#1": 123},
                "e": {}
            },

            {
                "v": "  episodic_buffer_1   ",
                "l": {},
                "e": {"episodic_buffer_1": 123}
            },
        ]

        print("=" * 40)
        print("check get local variable name")
        print("=" * 40)

        evr_agent = EVRAgent(neural_module=None)

        for s in snippets:
            print(s["v"])
            print(evr_agent.get_variable_value(s["v"], s["l"], s["e"]))
            print("-" * 40)

    @classmethod
    def check_get_list_values(cls):

        local_variable_dict = {"a": "This is a"}

        episodic_buffer_dict = {
            "episodic_buffer_2": "This is the value"
        }

        evr_agent = EVRAgent(neural_module=None)

        cases = [
            "[]",
            "[,]",
            "[a]",
            "[0]",
            "['x']",
            "['x', ]",
            "[ 'x'  ,  ]",
            "[  'x', True ,'True',  0 , episodic_buffer_2,  5]",
            "  [  'x', True ,'True',  0 , episodic_buffer_2,  5]   "
        ]

        targets = [
            [],
            [],
            ['This is a'],
            [0],
            ['x'],
            ['x'],
            ['x'],
            ['x', True, 'True', 0, 'This is the value', 5],
            ['x', True, 'True', 0, 'This is the value', 5],
        ]

        for i, case in enumerate(cases):
            list_of_vars = evr_agent.get_variable_value(case, local_variable_dict, episodic_buffer_dict)
            print(list_of_vars)
            assert list_of_vars == targets[i]


if __name__ == "__main__":
    #CheckGetVariableValue.check_get_variable_value()
    CheckGetVariableValue.check_get_list_values()
