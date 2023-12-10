import torch
from typing import Optional, Union
from lm_eval.base import BaseLM

from accessory.model.meta import MetaModel

def _get_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class AccessoryModel(BaseLM):

    _DEFAULT_MAX_LENGTH = 2048

    def __init__(
        self,
        model: MetaModel,
        batch_size=1,
        max_batch_size=512,
        max_length=None,
    ):
        super().__init__()


        self.model = model
        self.model_name = model.llama_type
        self.tokenizer = model.tokenizer

        self.model.eval()

        self.vocab_size = self.tokenizer.n_words

        # Validate batch_size
        assert isinstance(batch_size, (int, str))

        self.batch_size_per_gpu = 1
        self.max_batch_size = max_batch_size

        self._max_length = max_length

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self._DEFAULT_MAX_LENGTH

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return list(self.model.parameters())[0].device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, bos=True, eos=False)  # todo ldy

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model.llma(inps, None)

    def _model_generate(self, context, max_length, eos_token_id):
        generation_kwargs = {"do_sample": False, "max_length": max_length}
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id
            generation_kwargs[
                "pad_token_id"
            ] = eos_token_id  # setting eos_token_id as pad token
        return self.model.generate(context, **generation_kwargs)
