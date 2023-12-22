import re
import sys


class SymbolicExpression:
    class InstancesDict(dict):
        def __init__(self, parent):
            self.parent = parent

        def __setitem__(self, __k, __v):
            self.parent.collect_garbage()
            return super().__setitem__(__k, __v)

    @classmethod
    def initialise(cls):
        cls.count = 0
        cls.instances = SymbolicExpression.InstancesDict(cls)

    @staticmethod
    def clear_instances():
        SymbolicExpression.count = 0
        SymbolicExpression.instances.clear()

    @staticmethod
    def collect_garbage():
        num_items = len(SymbolicExpression.instances)
        n = 0
        while len(SymbolicExpression.instances) > 0:
            symbol = list(SymbolicExpression.instances)[n]
            item = SymbolicExpression.instances[symbol]

            external_ref_count = sys.getrefcount(item)

            internal_ref_count = 0
            for se in list(SymbolicExpression.instances):
                if re.search(
                    rf"\b{re.escape(symbol)}\b",
                    SymbolicExpression.instances[se].expression,
                ):
                    internal_ref_count += 1

            if not internal_ref_count and external_ref_count < 4:
                # DEBUG SymbolicExpression Garbage Collector Activity
                # print(f'deleting {symbol} with refcount {external_ref_count}')
                del SymbolicExpression.instances[symbol]
                num_items = len(SymbolicExpression.instances)
                n = 0
            else:
                n += 1
            if n == num_items:
                break

    @staticmethod
    def print_reference_counts():
        for symbol, se in SymbolicExpression.instances.items():
            print(symbol, sys.getrefcount(se))

    class CircularReferenceError(Exception):
        pass

    class InvalidExpressionError(Exception):
        pass

    supported_types = (int, float)

    def __init__(
        self, type: type, expression=0, symbol: str = ""
    ):  # (self, expression: str|{self.type}|SymbolicExpression = 0, symbol: str = ''):
        if type not in self.supported_types:
            raise TypeError(
                f"type argument should be int or float, got {type.__name__}"
            )
        self.type = type

        self._symbol: str = ""
        self._expression: str = ""

        if symbol == "":
            while True:
                symbol = "_sym_" + type.__name__ + str(SymbolicExpression.count)
                SymbolicExpression.count += 1
                if symbol not in SymbolicExpression.instances.keys():
                    break

        self.expression = expression
        self.symbol = symbol

        ################################ Settable Interface (Quantify)

        self.label: str = ""
        self.unit: str = ""

    def set(self, value):
        self.value = value

    @property
    def name(self):
        return self._symbol

    ################################ Settable Interface (Quantify)

    @property
    def symbol(self) -> str:
        return self._symbol

    @symbol.setter
    def symbol(self, symbol: str):
        if not isinstance(symbol, str):
            raise TypeError(
                f"symbol argument type should be str, got {type(symbol).__name__}"
            )
        if symbol in SymbolicExpression.instances.keys():
            pass  # Allows overwriting
            # raise KeyError(f"symbol should be unique, there is already a SymbolicExpression with symbol {symbol}: {SymbolicExpression.instances[symbol]}")
        if self._symbol == "":
            # Creation
            SymbolicExpression.instances[symbol] = self
        else:
            # Renaming
            SymbolicExpression.instances[
                symbol
            ] = self  # Add a new reference with the new symbol
            if not self._symbol == symbol:
                del SymbolicExpression.instances[
                    self._symbol
                ]  # Remove the previous reference

            for (
                se
            ) in (
                SymbolicExpression.instances.values()
            ):  # Update all SymbolicExpressions with the symbol change
                match_string = rf"\b{re.escape(self._symbol)}\b"
                replacement = symbol
                try:
                    se.expression = re.sub(match_string, replacement, se.expression)
                except (
                    SymbolicExpression.InvalidExpressionError,
                    SymbolicExpression.CircularReferenceError,
                ) as e:
                    pass
        self._symbol = symbol
        # test for CircularReferenceError
        self.evaluate(self._expression, self._symbol)

    def __getitem__(self, symbol):
        self.symbol = symbol
        return self

    @property
    def expression(self):
        return self._expression

    @expression.setter
    def expression(
        self, expression
    ):  # (self, expression: str|{self.type}|SymbolicExpression):
        if isinstance(expression, str):
            # evaluate the str expression to confirm it is valid before assigning it
            self.evaluate(expression)
            self._expression = self._replace_internal_symbols(expression)
        elif isinstance(expression, self.type):
            self._expression = str(expression)
        elif isinstance(expression, SymbolicExpression):
            self._expression = expression._expression
            # self.symbol = expression._symbol #(TODO find a solution so that intermediate operations are not stored as instances)
        else:
            raise TypeError(
                f"expression argument type should be {self.type.__name__} or SymbolicExpression, got {type(expression).__name__}"
            )

    @property
    def value(self):  # -> {self.type}:
        return self.evaluate(self._expression, self._symbol)

    @value.setter
    def value(self, value):  # (self, value: {self.type})
        if isinstance(value, self.type):
            self._expression = str(value)
        else:
            raise TypeError(
                f"value argument type should be {self.type.__name__}, got {type(value).__name__}"
            )

    @property
    def is_constant(self) -> bool:
        try:
            return str(self.type(self._expression)) == self._expression
        except:
            return False

    def __repr__(self):
        try:
            response = f"{self._symbol}: {self._expression} = {self.value}"
        except SymbolicExpression.CircularReferenceError:
            response = f"{self._symbol}: {self._expression} = CircularReferenceError"
        except SymbolicExpression.InvalidExpressionError:
            response = f"{self._symbol}: {self._expression} = InvalidExpressionError"
        return response

    def evaluate(self, expression: str, *previous_evaluations):  # -> {self.type}:
        for symbol in SymbolicExpression.instances.keys():
            if symbol in expression:
                if symbol in previous_evaluations:
                    raise SymbolicExpression.CircularReferenceError(
                        f"Circular Reference evaluating {expression}, variable {symbol} found in {previous_evaluations}"
                    )
                match_string = rf"\b{re.escape(symbol)}\b"
                replacement = str(
                    self.evaluate(
                        SymbolicExpression.instances[symbol]._expression,
                        *previous_evaluations,
                        SymbolicExpression.instances[symbol]._symbol,
                    )
                )
                expression = re.sub(match_string, replacement, expression)
        try:
            invalid_characters = "abcdefghijklmnopqrstuvwxyz_"
            expression = expression.lower()
            for char in invalid_characters:
                if char in expression:
                    raise Exception()
            result = eval(expression)
        except:
            raise SymbolicExpression.InvalidExpressionError(
                f"Unable to evaluate expression: {expression}"
            )

        if not isinstance(result, self.type):
            raise SymbolicExpression.InvalidExpressionError(
                f"The evaluation of the expression: {expression} does not return {self.type.__name__}"
            )
        return result

    def _replace_internal_symbols(
        self, expression: str, *previous_evaluations
    ):  # -> {self.type}:
        symbol: str
        for symbol in SymbolicExpression.instances.keys():
            if symbol.startswith("_sym_") and symbol in expression:
                if symbol in previous_evaluations:
                    raise SymbolicExpression.CircularReferenceError(
                        f"Circular Reference evaluating {expression}, variable {symbol} found in {previous_evaluations}"
                    )
                match_string = rf"\b{re.escape(symbol)}\b"
                replacement = str(
                    self._replace_internal_symbols(
                        SymbolicExpression.instances[symbol]._expression,
                        *previous_evaluations,
                        SymbolicExpression.instances[symbol]._symbol,
                    )
                )
                expression = re.sub(match_string, replacement, expression)

        return expression

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __str__(self):
        return str(self.value)

    def __lt__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value < other.value
        if isinstance(other, self.supported_types):
            return self.value < other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __gt__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value > other.value
        if isinstance(other, self.supported_types):
            return self.value > other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __le__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value <= other.value
        if isinstance(other, self.supported_types):
            return self.value <= other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __ge__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value >= other.value
        if isinstance(other, self.supported_types):
            return self.value >= other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __eq__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value == other.value
        if isinstance(other, self.supported_types):
            return self.value == other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __ne__(self, other):
        if isinstance(other, SymbolicExpression):
            return self.value != other.value
        if isinstance(other, self.supported_types):
            return self.value != other
        raise TypeError(
            f"Comparison operators expect SymbolicExpression or {self.type.__name__} arguments, got {type(other).__name__}"
        )

    def __add__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({self.symbol} + {other.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({self.symbol} + {str(other)})")

    def __radd__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({other.symbol} + {self.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({str(other)} + {self.symbol})")

    def __sub__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({self.symbol} - {other.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({self.symbol} - {str(other)})")

    def __rsub__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({other.symbol} - {self.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({str(other)} - {self.symbol})")

    def __mul__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({self.symbol} * {other.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({self.symbol} * {str(other)})")

    def __rmul__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({other.symbol} * {self.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({str(other)} * {self.symbol})")

    def __floordiv__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({self.symbol} // {other.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({self.symbol} // {str(other)})")

    def __rfloordiv__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({other.symbol} // {self.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({str(other)} // {self.symbol})")

    def __mod__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({self.symbol} % {other.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({self.symbol} % {str(other)})")

    def __rmod__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            return self.__class__(f"({other.symbol} % {self.symbol})")
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            return self.__class__(f"({str(other)} % {self.symbol})")

    def __iadd__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            self.expression = f"({self.expression} + {other.symbol})"
            return self
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            if self.is_constant:
                self.expression = str(self.type(self.expression) + other)
            else:
                self.expression = f"({self.expression} + {str(other)})"
            return self

    def __isub__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            self.expression = f"({self.expression} - {other.symbol})"
            return self
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            if self.is_constant:
                self.expression = str(self.type(self.expression) - other)
            else:
                self.expression = f"({self.expression} - {str(other)})"
            return self

    def __imul__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            self.expression = f"({self.expression} * {other.symbol})"
            return self
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            if self.is_constant:
                self.expression = str(self.type(self.expression) * other)
            else:
                self.expression = f"({self.expression} * {str(other)})"
            return self

    def __ifloordiv__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            self.expression = f"({self.expression} // {other.symbol})"
            return self
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            if self.is_constant:
                self.expression = str(self.type(self.expression) // other)
            else:
                self.expression = f"({self.expression} // {str(other)})"
            return self

    def __imod__(self, other):  # -> SymbolicExpression:
        if isinstance(other, SymbolicExpression):
            self.expression = f"({self.expression} % {other.symbol})"
            return self
        if (
            self.type == float
            and isinstance(other, self.supported_types)
            or self.type == int
            and isinstance(other, int)
        ):
            if self.is_constant:
                self.expression = str(self.type(self.expression) % other)
            else:
                self.expression = f"({self.expression} % {str(other)})"
            return self

    def __neg__(self):  # -> SymbolicExpression:
        return self.__class__(f"-{self.symbol}")

    def __hash__(self):
        return hash(self._symbol)


class intSymbolicExpression(SymbolicExpression):
    def __init__(self, expression=0, symbol: str = ""):
        super().__init__(int, expression, symbol)


class floatSymbolicExpression(SymbolicExpression):
    def __init__(self, expression=0, symbol: str = ""):
        super().__init__(float, expression, symbol)


SymbolicExpression.initialise()

# TODO
# t0, t1 = SymbolicExpression(0, 't0', 't1') # t0 = 0, t1 = 0
# t0, t1 = SymbolicExpression([0, 5], ['t0', 't1']) # t0 = 0, t1 = 5
# or even better with a dictionary
# tv_dict = {}
# tv_dict = SymbolicExpression({t0: 0, t1: 5}) # t0 = 0, t1 = 5

# TODO
# add all integer and float operators

# TODO
# make compatible with numpy types int32, float, etc....
