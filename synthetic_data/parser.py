import ast
import math
import operator as op


class MathParser:
    """Basic parser with local variable and math functions

    Args:
       vars (mapping): mapping object where obj[name] -> numerical value
       math (bool, optional): if True (default) all math function are added in the same name space

    Example:
       data = {'r': 3.4, 'theta': 3.141592653589793}
       parser = MathParser(data)
       assert parser.parse('r*cos(theta)') == -3.4
       data['theta'] =0.0
       assert parser.parse('r*cos(theta)') == 3.4
    """

    _op_to_method = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.BitXor: op.xor,
        ast.Or: op.or_,
        ast.And: op.and_,
        ast.Mod: op.mod,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.FloorDiv: op.floordiv,
        ast.USub: op.neg,
        ast.UAdd: lambda a: a,
    }

    def __init__(self, vars, math=True):
        self._vars = vars
        if not math:
            self._alt_name = self._no_alt_name

    @staticmethod
    def _alt_name(name):
        if name.startswith("_"):
            raise NameError(f"{name!r}")
        try:
            return getattr(math, name)
        except AttributeError:
            raise NameError(f"{name!r}")

    @staticmethod
    def _no_alt_name(name):
        raise NameError(f"{name!r}")

    def eval_(self, node):
        if isinstance(node, ast.Expression):
            return self.eval_(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Name):
            return (
                self._vars[node.id]
                if node.id in self._vars
                else self._alt_name(node.id)
            )
        if isinstance(node, ast.BinOp):
            method = self._op_to_method[type(node.op)]
            return method(self.eval_(node.left), self.eval_(node.right))
        if isinstance(node, ast.UnaryOp):
            method = self._op_to_method[type(node.op)]
            return method(self.eval_(node.operand))
        if isinstance(node, ast.Attribute):
            return getattr(self.eval_(node.value), node.attr)
        if isinstance(node, ast.Call):
            return self.eval_(node.func)(
                *(self.eval_(a) for a in node.args),
                **{k.arg: self.eval_(k.value) for k in node.keywords},
            )
            # return self.Call(
            #     self.eval_(node.func), tuple(self.eval_(a) for a in node.args)
            # )

        raise TypeError(node)

    def parse(self, expr):
        return self.eval_(ast.parse(expr, mode="eval"))
