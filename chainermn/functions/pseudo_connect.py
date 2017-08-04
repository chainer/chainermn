import chainer
from chainer import cuda
import chainer.utils


class PseudoConnect(chainer.Function):
    """Connect a variable with delegating variable."""

    def forward(self, inputs):
        # delegate_variable = inputs[0]
        actual_variables = inputs[1:]
        return actual_variables

    def backward(self, inputs, grad_outputs):
        delegate_variable = inputs[0]
        # actual_variables = inputs[1:]
        xp = cuda.get_array_module(*inputs)

        # delegate_variable do not need backward gradients, instead sending
        # back dummy grads in order to take consistency of shapes of grads.
        grad_delegate_variable = xp.zeros_like(delegate_variable)

        # grad_outputs corresponds to grads of actual_variables.
        return tuple([grad_delegate_variable] + list(grad_outputs))


def pseudo_connect(delegate_variable, *actual_variables):
    """Connect independent connected graph component.

    In model-parallel framework, models sometimes have many non-connected
    components. When some additional components follow model outputs,
    outputs of the last component must be merged with model outputs.
    Otherwise backprop does not work well, got stuck into dead lock.

    Args:
        delegate_variable (chainer.Variable):
            Pointer to the previous non-connected graph component.
        actual_variables (tuple of chainer.Variable):
            Actual values which ``delegate_variable`` imitate.

    Returns:
        ~chainer.Variable:
            A variable with the given values combined with delegating variable.
    """
    chainer.utils.experimental('chainermn.functions.pseudo_connect')
    return PseudoConnect()(delegate_variable, *actual_variables)
