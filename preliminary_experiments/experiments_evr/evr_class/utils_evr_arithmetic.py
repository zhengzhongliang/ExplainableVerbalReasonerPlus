class ArithmeticOperators:

    @classmethod
    def get_arithmetic_operators(cls):

        arithmetic_operators = {
            "==": cls.equal_to,
            "!=": cls.not_equal_to,
            ">": cls.greater_than,
            ">=": cls.greater_than_or_equal_to,
            "<": cls.less_than,
            "<=": cls.less_than_or_equal_to
        }

        return arithmetic_operators

    @classmethod
    def equal_to(cls, a, b):
        return True if a == b else False

    @classmethod
    def not_equal_to(cls, a, b):
        return True if a != b else False

    @classmethod
    def greater_than(cls, a, b):
        return True if a > b else False

    @classmethod
    def greater_than_or_equal_to(cls, a, b):
        return True if a >= b else False

    @classmethod
    def less_than(cls, a, b):
        return True if a < b else False

    @classmethod
    def less_than_or_equal_to(cls, a, b):
        return True if a <= b else False
