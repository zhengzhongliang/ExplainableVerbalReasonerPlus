from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent


class CheckForHandler:

    evr_agent = EVRAgent(neural_module=None)

    @classmethod
    def check_get_for_span(cls):

        programs = [
            [
                "a = 1",
                "b = 2",
                "for a in b",
                    "c = 1",
                    "d = 2",
                    "e = 3",
                "end_for",
                "f = 1",
                "g = 2"
            ],
            [
                "a = 1",
                "b = 2",
                "for a in b",
                    "c = 1",
                    "d = 2",
                    "e = 3",
                    "for c in d",
                        "a = 1",
                        "b = 2",
                    "end_for",
                    "a = 1",
                "end_for",
                "f = 1",
                "g = 2"
            ],
            [
                "a = 1",
                "b = 2",
                "for a in b",
                    "c = 1",
                    "d = 2",
                    "e = 3",
                    "while b < a",
                        "a = 1",
                        "b = 2",
                    "end_while",
                    "a = 1",
                "end_for",
                "f = 1",
                "g = 2"
            ],
            [
                "a = 1",
                "b = 2",
                "for a in b",
                    "c = 1",
                    "d = 2",
                    "e = 3",
                    "if a < d",
                        "a = 1",
                    "else",
                        "b = 2",
                    "end_if",
                    "a = 1",
                "end_for",
                "f = 1",
                "g = 2"
            ]
        ]

        program_start_counters = [
            2, 2, 2, 2
        ]

        for idx in range(len(programs)):
            for_span_start, for_span_end, for_program_lines = \
                 cls.evr_agent.get_for_span(for_span_start=program_start_counters[idx], program_lines=programs[idx])

            print("=" * 40)
            print("start and end:", for_span_start, for_span_end)
            print("program:", "\n".join(for_program_lines))

    @classmethod
    def check_for_loop(cls):
        programs = [
            [
                "#1 = 0",
                "for #2 in [1, 2, 3, 4]",
                "add_to_episodic(#2)",
                "end_for"
            ],
            [
                "#1 = 0",
                "for #3 in #2",
                "add_to_episodic(#3)",
                "end_for"
            ],
            [
                "#3 = [1, #1, #2]",
                "for #4 in #3",
                "add_to_episodic(#4)",
                "end_for"
            ],
            [
                "#2 = ['prove A', 'prove B', 'prove C']",
                "for #1 in #2",
                "add_to_episodic(#1)",
                "end_for"
            ]
        ]

        local_variable_dicts = [
            {},
            {"#2": [1, 2, 3, 4]},
            {"#1": 2, "#2": 3},
            {}
        ]

        for idx in range(len(programs)):
            local_variable_dict, episodic_buffer_dict = cls.evr_agent.program_handler(
                program_lines=programs[idx],
                local_variable_dict=local_variable_dicts[idx],
                episodic_buffer_dict={},
                external_textual_buffer_dict={}
            )

            print("=" * 40)
            print(episodic_buffer_dict)


if __name__ == "__main__":
    CheckForHandler.check_get_for_span()

    CheckForHandler.check_for_loop()
