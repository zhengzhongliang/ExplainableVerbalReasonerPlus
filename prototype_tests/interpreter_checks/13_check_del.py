import json

from preliminary_experiments.experiments_evr.evr_class.evr_agent import EVRAgent


class CheckDel:

    @classmethod
    def check_del(cls):

        cases = [
            {
                "pg_lines": ["del('#0')"],
                "pg_counter": 0,
                "l_dict": {"#0": 0, "#1": 1},
                "ep_dict": {},
                "ex_dict": {}
            },

            {
                "pg_lines": ["del('episodic_buffer_0')"],
                "pg_counter": 0,
                "l_dict": {},
                "ep_dict": {"episodic_buffer_0": "abc"},
                "ex_dict": {}
            },

            {
                "pg_lines": ["del('chunk_0')"],
                "pg_counter": 0,
                "l_dict": {},
                "ep_dict": {},
                "ex_dict": {"chunk_0": "abc"}
            },

            {
                "pg_lines": ["del('#1', 'chunk_0')"],
                "pg_counter": 0,
                "l_dict": {"#1": "abc"},
                "ep_dict": {},
                "ex_dict": {"chunk_0": "def"}
            },
        ]

        evr_agent = EVRAgent(neural_module=None)

        for case in cases:
            print("=" * 40)
            print(json.dumps(case))

            print("-" * 40)

            returned = evr_agent.del_handler(
                program_lines=case["pg_lines"],
                program_counter=case["pg_counter"],
                local_variable_dict=case["l_dict"],
                episodic_buffer_dict=case["ep_dict"],
                external_textual_buffer_dict=case["ex_dict"]
            )

            print(json.dumps(returned))
            input("-" * 20)


if __name__ == "__main__":

    CheckDel.check_del()
