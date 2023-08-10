import ivy
from typing import List


class LoRALayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = ivy.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Embedding(ivy.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        ivy.Embedding.__init__(
            self, num_embeddings, embedding_dim, weight_initializer=ivy.Zeros, **kwargs
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=0,
            merge_weights=merge_weights,
        )

        if r > 0:
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()

    # Actual trainable parameters
    def _create_variables(self, device, dtype=None):
        if self.r > 0:
            self.v = dict(
                **self.v,
                lora_A=self._weight_initializer.create_variables(
                    (self.r, self._num_embeddings),
                    device,
                    self.r,
                    self._num_embeddings,
                    dtype,
                ),
                lora_B=self._weight_initializer.create_variables(
                    (self._embedding_dim, self.r),
                    device,
                    self._embedding_dim,
                    self.r,
                    dtype,
                ),
            )
        self.reset_parameters()
        return self.v

    def reset_parameters(self):
        if hasattr(self, "v"):
            # initialize A the same way as the default for ivy.Linear and B to zero
            ivy.inplace_update(self.v.lora_A, ivy.zeros(self.v.lora_A))
            ivy.inplace_update(self.lora_B, ivy.l2_normalize(self.v.lora_B))

    def train(self, mode: bool = True):
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.v.w -= (self.v.lora_B @ self.v.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.v.w += (self.v.lora_B @ self.v.lora_A).transpose(
                        0, 1
                    ) * self.scaling
                self.merged = True

    def _forward(self, x: ivy.Array):
        if self.r > 0 and not self.merged:
            result = ivy.Embedding._forward(self, x)
            after_A = ivy.embedding(
                x, self.v.lora_A.transpose(0, 1), self._padding_idx, self._max_norm
            )
            result += (after_A @ self.v.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return ivy.Embedding._forward(self, x)


class Linear(ivy.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        ivy.Linear.__init__(
            self, in_features, out_features, weight_initializer=ivy.Zeros, **kwargs
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
        self.reset_parameters()

        if fan_in_fan_out:
            self.v.w = self.v.w.transpose(0, 1)

    def _create_variables(self, device, dtype=None):
        if self.r > 0:
            self.v = dict(
                **self.v,
                lora_A=self._weight_initializer.create_variables(
                    (self.r, self._in_features),
                    device,
                    self.r,
                    self._in_features,
                    dtype,
                ),
                lora_B=self._weight_initializer.create_variables(
                    (self._out_features, self.r),
                    device,
                    self._out_features,
                    self.r,
                    dtype,
                ),
            )
        self.reset_parameters()
        return self.v

    def reset_parameters(self):
        if hasattr(self, "v"):
            # initialize A the same way as the default for ivy.Linear and B to zero
            ivy.inplace_update(
                # self.v.lora_A, kaiming_uniform_(self.v.lora_A, a=ivy.sqrt(5))
            )
            ivy.inplace_update(self.lora_B, ivy.zeros(self.v.lora_B))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.v.w -= T(self.v.lora_B @ self.v.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.v.w += T(self.v.lora_B @ self.v.lora_A) * self.scaling
                self.merged = True

    def _forward(self, x: ivy.Array):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = ivy.linear(x, T(self.v.w), bias=self.v.b)
            result += (
                self.lora_dropout(x)
                @ self.v.lora_A.transpose(0, 1)
                @ self.v.lora_B.transpose(0, 1)
            ) * self.scaling
            return result
        else:
            return ivy.linear(x, T(self.v.w), bias=self.v.b)


class MergedLinear(ivy.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        **kwargs
    ):
        ivy.Linear.__init__(
            self, in_features, out_features, weight_initializer=ivy.Zeros, **kwargs
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        assert (
            out_features % len(enable_lora) == 0
        ), "The length of enable_lora must divide out_features"
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.weight.requires_grad = False
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.reshape(-1)
        self.reset_parameters()
        if fan_in_fan_out:
            self.v.w = self.v.w.transpose(0, 1)

    def _create_variables(self, device, dtype=None):
        if self.r > 0 and any(self.enable_lora):
            self.v = dict(
                **self.v,
                lora_A=self._weight_initializer.create_variables(
                    (self.r * sum(self.enable_lora), self._input_channels),
                    device,
                    self.r * sum(self.enable_lora),
                    self._input_channels,
                    dtype,
                ),
                lora_B=self._weight_initializer.create_variables(
                    (
                        self._output_channels
                        // len(self.enable_lora)
                        * sum(self.enable_lora),
                        self.r,
                    ),
                    device,
                    self._output_channels
                    // len(self.enable_lora)
                    * sum(self.enable_lora),
                    self.r,
                    dtype,
                ),
                lora_ind=self._weight_initializer.create_variables(
                    (self._output_channels,),
                    device,
                    self._output_channels,
                    dtype=ivy.bool,
                ).reshape(len(self.enable_lora), -1),
            )
        self.reset_parameters()
        return self.v

    def reset_parameters(self):
        if hasattr(self, "v"):
            # initialize A the same way as the default for ivy.Linear and B to zero
            ivy.inplace_update(
                # self.v.lora_A, kaiming_uniform_(self.lora_A, a=ivy.sqrt(5))
            )
            ivy.inplace_update(self.v.lora_B, ivy.zeros(self.lora_B))

    def zero_pad(self, x):
        result = ivy.inplace_update(x, ivy.zeros((len(self.lora_ind), *x.shape[1:])))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = ivy.conv1d(
            self.v.lora_A.unsqueeze(0),
            self.v.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora),
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    self.v.w -= self.merge_AB() * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    self.v.w += self.merge_AB() * self.scaling
                self.merged = True

    def _forward(self, x: ivy.Array):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return ivy.linear(x, T(self.v.w), bias=self.v.b)
        else:
            result = ivy.linear(x, T(self.v.w), bias=self.v.b)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result


class ConvLoRA(ivy.Module, LoRALayer):
    def __init__(
        self,
        conv_module,
        in_channels,
        out_channels,
        kernel_size,
        r=0,
        lora_alpha=1,
        lora_dropout=0.0,
        merge_weights=True,
        **kwargs
    ):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            weight_initializer=ivy.Zeros,
            **kwargs,
        )
        LoRALayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )
        assert isinstance(kernel_size, int)

        # Actual trainable parameters
        if r > 0:
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            # self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def _create_variables(self, *, device=None, dtype=None):
        if self.r > 0:
            self.v = dict(
                **self.v,
                lora_A=self.conv._weight_initializer.create_variables(
                    (
                        self.r * self.conv.kernel_size[0],
                        self.conv.in_channels * self.conv.kernel_size[0],
                    ),
                    device,
                    self.r * self.conv.kernel_size[0],
                    self.conv.in_channels * self.conv.kernel_size[0],
                    dtype,
                ),
                lora_B=self.conv._weight_initializer.create_variables(
                    (
                        self.conv.out_channels
                        // self.conv.groups
                        * self.conv.kernel_size[0],
                        self.r,
                    ),
                    device,
                    self.conv.out_channels
                    // self.conv.groups
                    * self.conv.kernel_size[0],
                    self.r,
                    dtype,
                ),
            )
        self.reset_parameters()
        return self.v

    def reset_parameters(self):
        if hasattr(self, "v"):
            # initialize A the same way as the default for ivy.Linear and B to zero
            ivy.inplace_update(
                self.v.lora_A, ivy.kaiming_uniform_(self.v.lora_A, a=ivy.sqrt(5))
            )
            ivy.inplace_update(self.v.lora_B, ivy.zeros(self.v.lora_B))

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.v.w -= (
                        ivy.reshape(
                            (self.v.lora_B @ self.v.lora_A), self.conv.v.w.shape
                        )
                        * self.scaling
                    )
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.v.w += (
                        ivy.reshape(
                            (self.v.lora_B @ self.v.lora_A), self.conv.v.w.shape
                        )
                        * self.scaling
                    )
                self.merged = True

    def _forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.v.w
                + ivy.reshape((self.v.lora_B @ self.v.lora_A), self.conv.v.w.shape)
                * self.scaling,
                self.conv.v.b,
            )
        return self.conv(x)


class Conv2D(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2D, self).__init__(ivy.Conv2D, *args, **kwargs)


class Conv1D(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(ivy.Conv1D, *args, **kwargs)


# Can Extend to other ones like this


class Conv3D(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3D, self).__init__(ivy.Conv3D, *args, **kwargs)
