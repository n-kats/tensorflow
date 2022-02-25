# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient support for Composite Tensors."""

import abc

from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


class HasCompositeTensorGradient(object):
  """Protocol for adding gradient support to CompositeTensors."""
  __composite_gradient__: "CompositeTensorGradient"


class CompositeTensorGradient(object, metaclass=abc.ABCMeta):
  """Class used to help compute gradients for CompositeTensors.

  This abstract base class defines two methods: `get_gradient_components`, which
  returns the components of a value that should be included in gradients; and
  `replace_gradient_components`, which replaces the gradient components in a
  value.  These methods can be used to compute the gradient of a `y` with
  respect to `x` (`grad(y, x)`) as follows:

  * If `y` is a `CompositeTensor` with `TypeSpec` `s`, then `grad(y, x)` =
    `grad(s.get_gradient_components(y), x)`.

  * If `x` is a `CompositeTensor` with `TypeSpec` `s`, then `grad(y, x)` =
    `s.replace_gradient_components(x, grad(y, s.get_gradient_components(x))`.
  """

  @abc.abstractmethod
  def get_gradient_components(self, value):
    """Returns the components of `value` that should be included in gradients.

    This method may not call TensorFlow ops, since any new ops added to the
    graph would not be propertly tracked by the gradient mechanisms.

    Args:
      value: A `CompositeTensor` value.

    Returns:
      A nested structure of `Tensor` or `CompositeTensor`.
    """
    raise NotImplementedError("%s.get_gradient_components()" %
                              type(self).__name__)

  @abc.abstractmethod
  def replace_gradient_components(self, value, component_grads):
    """Replaces the gradient components in `value` with `component_grads`.

    This method may not call TensorFlow ops, since any new ops added to the
    graph would not be propertly tracked by the gradient mechanisms.

    Args:
      value: A value compatible with this `TypeSpec`.
      component_grads: A nested structure of `Tensor` or `CompositeTensor` or
        `None` (for unconnected gradients).

    Returns:
      A copy of `value`, where the components that should be included in
      gradients have been replaced by `component_grads`; or `None` (if
      `component_grads` includes `None`).
    """
    raise NotImplementedError("%s.replace_gradient_components()" %
                              type(self).__name__)


class WithValuesCompositeTensorGradient(CompositeTensorGradient):
  """CompositeTensorGradient based on `T.values` and `T.with_values`."""

  def get_gradient_components(self, value):
    return value.values

  def replace_gradient_components(self, value, component_grads):
    if component_grads is None:
      return None
    if isinstance(component_grads, indexed_slices.IndexedSlices):
      component_grads = ops.convert_to_tensor(component_grads)
    return value.with_values(component_grads)


def get_tensors_for_gradient(x):
  """Returns the Tensors in `x` that should be differentiated.

  Args:
    x: A `Tensor` or `CompositeTensor`.

  Returns:
    A `Tensor` or a nested structure of `Tensor`.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return x

  composite_gradient = getattr(x, "__composite_gradient__", None)
  if composite_gradient is None:
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source.")
  return nest.map_structure(get_tensors_for_gradient,
                            composite_gradient.get_gradient_components(x))


def replace_tensors_for_gradient(x, grad):
  """Replaces the tensors in `x` that should be differentiated with `grad`.

  Args:
    x: A Tensor or CompositeTensor.
    grad: A nested structure of `Tensor`, with the same structure as the value
      returned by `get_tensors_for_gradient(x)`.

  Returns:
    A Tensor or CompositeTensor.
  """
  if not isinstance(x, composite_tensor.CompositeTensor):
    return grad

  composite_gradient = getattr(x, "__composite_gradient__", None)
  if composite_gradient is None:
    raise ValueError(
        f"Type {type(x).__name__} is not supported as a gradient source.")
  x_components = composite_gradient.get_gradient_components(x)
  grad_components = nest.map_structure_up_to(x_components,
                                             replace_tensors_for_gradient,
                                             x_components, grad)
  return composite_gradient.replace_gradient_components(x, grad_components)


def get_flat_tensors_for_gradients(xs):
  """Returns a flat list of Tensors that should be differentiated for `xs`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.

  Returns:
    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are
    left as-is, and `CompositeTensor`s are replaced with
    `get_tensors_for_gradient(x)`.
  """
  # Note: we could just return
  # nest.flatten([get_tensors_for_gradient(x) for x in xs]), but we
  # manually walk over the results to give better warning messages.
  result = []
  for x in xs:
    if not isinstance(x, composite_tensor.CompositeTensor):
      result.append(x)
    else:
      x_tensors = nest.flatten(get_tensors_for_gradient(x))
      for t in x_tensors:
        if not backprop_util.IsTrainable(t):
          logging.log_first_n(
              logging.WARN, f"The dtype of differentiable component {t} in "
              f"{x} must be floating (e.g., tf.float32), got {t.dtype}")
      result.extend(x_tensors)
  return result


def replace_flat_tensors_for_gradients(xs, flat_grads):
  """Replaces Tensors that should be differentiated in `xs` with `flat_grads`.

  Args:
    xs: A list of `Tensor`s or `CompositeTensor`s.
    flat_grads: A list of `Tensor`.

  Returns:
    A list of `Tensor` or `CompositeTensor`.
  """
  xs_structure = [get_tensors_for_gradient(x) for x in xs]
  grads = nest.pack_sequence_as(xs_structure, flat_grads)
  return [replace_tensors_for_gradient(x, grad) for x, grad in zip(xs, grads)]
