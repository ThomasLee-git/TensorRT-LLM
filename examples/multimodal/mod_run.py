import typing
import os
import sys
import platform
from pathlib import Path
import dataclasses
from types import SimpleNamespace

# import importlib


# isort: off
import torch
import tensorrt as trt
# isort: on

import einops
from cuda import cudart
from transformers import BertTokenizer, AutoConfig

import tensorrt_llm
from tensorrt_llm.lora_manager import LoraManager
import tensorrt_llm.profiler as profiler
from tensorrt_llm import logger
from tensorrt_llm.runtime.model_runner import (
    ModelRunner,
    Engine,
    MpiComm,
    trt_gte_10,
    get_engine_version,
)
from tensorrt_llm.runtime.generation import (
    GenerationSession,
    ModelConfig,
    Mapping,
    SamplingConfig,
    RuntimeTensor,
    StoppingCriteria,
    LogitsProcessor,
    CUASSERT,
    _tile_beam_width,
    _contiguous_tile_beam_width,
)


sys.path.append(str(Path(__file__).parent.parent))


@dataclasses.dataclass
class ModSamplingConfig(SamplingConfig):
    cfg_coef: float = dataclasses.field(default=-1)


class ModGenerationSession(GenerationSession):
    def __init__(
        self,
        model_config: ModelConfig,
        engine_buffer,
        mapping: Mapping,
        debug_mode=False,
        debug_tensors_to_save=None,
        cuda_graph_mode=False,
        stream: torch.cuda.Stream = None,
    ):
        super().__init__(
            model_config,
            engine_buffer,
            mapping,
            debug_mode,
            debug_tensors_to_save,
            cuda_graph_mode,
            stream,
        )
        # reinit dynamic_decode_op
        del self.dynamic_decoder
        self.dynamic_decoder = torch.classes.trtllm.CustomDynamicDecodeOp(
            model_config.max_batch_size,
            model_config.max_beam_width,
            self.vocab_size,
            self.vocab_size_padded,
            self.mapping.tp_size,
            self.mapping.pp_size,
            self.decoder_logits_dtype,
        )

    def handle_per_step(
        self,
        cache_indirections: list,
        step: int,
        batch_size: int,
        max_context_length: int,
        beam_width: int,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        scfg: SamplingConfig,
        kv_cache_block_offsets: torch.Tensor,
        host_kv_cache_block_offsets: torch.Tensor,
        cross_kv_cache_block_offsets: torch.Tensor,
        host_cross_kv_cache_block_offsets: torch.Tensor,
        prompt_embedding_table: torch.Tensor,
        tasks: torch.Tensor,
        context_lengths: torch.Tensor,
        host_context_lengths,
        attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        prompt_vocab_size: torch.Tensor,
        ite: int,
        sequence_limit_lengths: torch.Tensor,
        sequence_lengths: torch.Tensor,
        next_step_tensors: typing.Dict[str, RuntimeTensor],
        stop_words_data,
        bad_words_data,
        encoder_output: torch.Tensor,
        encoder_input_lengths: torch.Tensor,
        stopping_criteria: StoppingCriteria,
        logits_processor: LogitsProcessor,
        **kwargs,
    ):
        if step % 2:
            context = self.runtime.context_0
            this_src_cache_indirection = cache_indirections[1]
            this_tgt_cache_indirection = cache_indirections[0]
            next_src_cache_indirection = cache_indirections[0]
        else:
            context = self.runtime.context_1
            this_src_cache_indirection = cache_indirections[0]
            this_tgt_cache_indirection = cache_indirections[1]
            next_src_cache_indirection = cache_indirections[1]

        if step == 0:
            model_inputs = self._prepare_context_inputs(
                batch_size=batch_size,
                context_lengths=context_lengths,
                host_context_lengths=host_context_lengths,
                use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                remove_input_padding=self.remove_input_padding,
                max_context_length=max_context_length,
                input_ids=input_ids,
                pad_id=scfg.pad_id,
                eos_id=scfg.end_id,
            )

            position_ids = model_inputs.get("position_ids", None)
            last_token_ids = model_inputs.get("last_token_ids")
            attention_mask = model_inputs.get("attention_mask", None)

            if self.paged_kv_cache and self.has_attn_layers:
                host_kv_cache_block_offsets = self.kv_cache_manager.get_block_offsets(
                    beam_width=1
                )
                kv_cache_block_offsets = host_kv_cache_block_offsets.to("cuda")
                if self.cross_attention:
                    host_cross_kv_cache_block_offsets = (
                        self.cross_kv_cache_manager.get_block_offsets(beam_width=1)
                    )
                    cross_kv_cache_block_offsets = host_cross_kv_cache_block_offsets.to(
                        "cuda"
                    )

            ctx_tensors = self._get_context_shape_buffer(
                input_ids,
                context_lengths,
                host_context_lengths,
                position_ids,
                last_token_ids,
                attention_mask,
                cross_attention_mask,
                this_src_cache_indirection,
                kv_cache_block_offsets,
                host_kv_cache_block_offsets,
                cross_kv_cache_block_offsets,
                host_cross_kv_cache_block_offsets,
                hidden_states,
                prompt_embedding_table,
                tasks,
                prompt_vocab_size,
                encoder_output,
                encoder_input_lengths,
            )

            context = self.runtime.ctx_context
            self.runtime._set_tensors(context, ctx_tensors)
            if self.debug_mode:
                self.debug_buffer = {
                    name: tensor.to_torch() for name, tensor in ctx_tensors.items()
                }
            if self.cuda_graph_mode:
                # context mode, clean cuda graph instances
                self.runtime.cuda_graph_instances = [None for _ in range(2)]

        if self.debug_mode:
            self.runtime._check_tensors(context)
        # dynamic_decoder currently use torch's current stream, so must let TRT enqueue use same stream here
        stream = torch.cuda.current_stream().cuda_stream
        instance_idx = step % 2
        if (
            self.cuda_graph_mode
            and self.runtime.cuda_graph_instances[instance_idx] is not None
        ):
            # launch cuda graph
            CUASSERT(
                cudart.cudaGraphLaunch(
                    self.runtime.cuda_graph_instances[instance_idx], stream
                )
            )
            ok = True
        else:
            ok = self.runtime._run(context, stream)

        if not ok:
            raise RuntimeError(f"Executing TRT engine failed step={step}!")

        if platform.system() == "Windows" or self.debug_mode:
            torch.cuda.synchronize()

        context_logits = None
        if self.mapping.is_last_pp_rank():
            if step == 0 and self.gather_context_logits:
                assert not self.is_medusa_mode
                context_logits = self.buffer["logits"].detach().clone()
                # gather last token of context
                if self.remove_input_padding:
                    # reshape self.buffer['logits'] from [bs, max_context_length, vocab]
                    # to [1, bs * max_context_length, vocab]
                    # Note that the data are put in the buffer without padding although
                    # the allocated buffer has padding.
                    self.buffer["logits"] = self.buffer["logits"].reshape(
                        [1, -1, self.vocab_size_padded]
                    )
                    self.buffer["logits"] = torch.index_select(
                        self.buffer["logits"], 1, last_token_ids - 1
                    ).view(batch_size, self.vocab_size_padded)
                else:
                    last_token_ids = last_token_ids.reshape(batch_size, 1, 1)
                    last_token_ids = (
                        last_token_ids.expand(batch_size, 1, self.vocab_size_padded) - 1
                    )
                    self.buffer["logits"] = torch.gather(
                        self.buffer["logits"],
                        dim=1,
                        index=last_token_ids.to(dtype=torch.int64),
                    ).view(batch_size, self.vocab_size_padded)

        if step == 0 and beam_width > 1:
            assert not self.is_medusa_mode
            assert not self.has_rnn_layers
            # these tiled tensors are returned by handle_per_step(), so they can relay to the next generation calls
            if not self.use_gpt_attention_plugin:
                attention_mask = _tile_beam_width(attention_mask, beam_width)
            context_lengths = _tile_beam_width(context_lengths, beam_width)
            host_context_lengths = _tile_beam_width(host_context_lengths, beam_width)
            if encoder_input_lengths is not None:
                encoder_input_lengths = _tile_beam_width(
                    encoder_input_lengths, beam_width
                )

            if tasks is not None:
                tasks = _tile_beam_width(tasks, beam_width)

            # Move tiling before logit computing of context
            if not self.paged_kv_cache:
                for key in self.buffer:
                    # Note: this tiles both self attn cache and cross attn
                    # cache! both names contain "present_key_value"
                    if "present_key_value" in key:
                        if self.use_gpt_attention_plugin:
                            self.buffer[key] = _tile_beam_width(
                                self.buffer[key], beam_width
                            )
                        else:
                            # In the OOTB path, KV cache should be contiguously
                            # tiled since TRT engine allocates past_kv cache of
                            # length context_length, i.e., we need a buffer of
                            # shape (batch * beam, 2, heads, context_length, head_size).
                            b, _, h, _, d = self.buffer[key].shape
                            numel = 2 * b * h * (max_context_length + step) * d
                            self.buffer[key] = _contiguous_tile_beam_width(
                                self.buffer[key], numel, beam_width
                            )

            if self.mapping.is_last_pp_rank():
                self.buffer["logits"] = _tile_beam_width(
                    self.buffer["logits"], beam_width
                )

        generation_logits = None
        if self.mapping.is_last_pp_rank():
            if self.gather_generation_logits:
                generation_logits = self.buffer["logits"].detach().clone()

        # Initialize sequence_lengths (no paddings) for the generation phase.
        if step == 0:
            self.sequence_length_buffer = context_lengths.detach().clone()

        # NOTE: handle next step.
        if not step == self.max_new_tokens - 1:
            # Set shape and address for the next step
            model_inputs = self._prepare_generation_inputs(
                batch_size=batch_size,
                context_lengths=context_lengths,
                use_gpt_attention_plugin=self.use_gpt_attention_plugin,
                remove_input_padding=self.remove_input_padding,
                step=step,
                num_beams=beam_width,
                attention_mask=attention_mask,
            )

            position_ids = model_inputs.get("position_ids", None)
            last_token_ids = model_inputs.get("last_token_ids")
            attention_mask = model_inputs.get("attention_mask", None)

            # Prepare for the next step, and always allocate 1 token slot.
            if self.paged_kv_cache and self.has_attn_layers:
                # Iterate to the next step in KV cache manager.
                # Increase number of tokens for all unfinished sequences.
                # And allocate new blocks if needed.
                # We set this to False for all sequences, since we use only length criterion to stop now
                # OPTIMIZE: find a better of adding multiple tokens for paged kv cache.
                if self.is_medusa_mode and self.num_medusa_tokens > 0:
                    # Allocate kv cache token slots for next step.
                    # Make sure there are always > (num_medusa_tokens + 1) free token slots.
                    # Allocate (num_medusa_tokens + 1) * 2 for safety as we don't know the current step or next step's accepted lengths.
                    add_token_count = (
                        (self.num_medusa_tokens + 1) * 2
                        if step == 0
                        else torch.max(self.accept_lengths).item()
                    )
                    assert add_token_count > 0
                    for new_tokens in range(add_token_count):
                        self.kv_cache_manager.step([False] * batch_size)
                else:
                    self.kv_cache_manager.step([False] * batch_size)
                host_kv_cache_block_offsets = self.kv_cache_manager.get_block_offsets(
                    beam_width
                )
                kv_cache_block_offsets = host_kv_cache_block_offsets.to("cuda")
                if self.cross_attention:
                    host_cross_kv_cache_block_offsets = (
                        self.cross_kv_cache_manager.get_block_offsets(beam_width)
                    )
                    cross_kv_cache_block_offsets = host_cross_kv_cache_block_offsets.to(
                        "cuda"
                    )

            next_context = (
                self.runtime.context_1 if step % 2 else self.runtime.context_0
            )
            next_step_tensors = self._get_next_step_shape_buffer(
                batch_size,
                beam_width,
                max_context_length,
                step,
                context_lengths,
                host_context_lengths,
                position_ids,
                last_token_ids,
                attention_mask,
                cross_attention_mask,
                next_src_cache_indirection,
                kv_cache_block_offsets,
                host_kv_cache_block_offsets,
                cross_kv_cache_block_offsets,
                host_cross_kv_cache_block_offsets,
                hidden_states,
                prompt_embedding_table,
                tasks,
                prompt_vocab_size,
                encoder_output,
                encoder_input_lengths,
            )

            # there are some tensors created inside the _get_next_step_shape_buffer, not owned by any object
            # needs to pro-long the life time of the tensors inside the next_step_tensors array
            # otherwise, it maybe released before the next step actually enqueued
            # one way to prolong it is to return the list, and destroy it in next step by assigning new values
            self.runtime._set_tensors(next_context, next_step_tensors)

            if self.cuda_graph_mode:
                self._capture_cuda_graph_and_instantiate(next_context, stream, step)

        should_stop = None
        logits = None
        if self.mapping.is_last_pp_rank():
            logits = self.buffer["logits"]
            if logits is not None:
                if self.is_medusa_mode:
                    should_stop = self.process_logits_for_medusa_mode(
                        step,
                        batch_size,
                        input_ids,
                        logits,
                        False,
                        next_step_tensors,
                        context_lengths,
                    )
                else:
                    if logits_processor is not None:
                        final_output_ids = self.finalize_decoder(
                            context_lengths,
                            batch_size,
                            beam_width,
                            scfg,
                            in_progress=True,
                        )
                        # keep the shape as same as huggingface stopping_criteria
                        final_output_ids_ = final_output_ids.reshape(
                            -1, final_output_ids.size(-1)
                        )
                        logits = logits_processor(step, final_output_ids_, logits)
                        self.buffer["logits"] = logits
                    # [batch_size x beam_width, vocab_size_padded] -> [batch_size, beam_width, vocab_size_padded]
                    next_token_logits = logits.reshape((batch_size, beam_width, -1)).to(
                        self.decoder_logits_dtype
                    )
                    decode_step = step + max_context_length

                    stop_words_list_ptrs, stop_words_lens, max_stop_words_len = (
                        stop_words_data
                    )
                    bad_words_list_ptrs, bad_words_lens, max_bad_words_len = (
                        bad_words_data
                    )

                    # NOTE: (ThomasLee) apply cfg
                    assert isinstance(
                        scfg, ModSamplingConfig
                    ), "invalid sampling config"
                    # cond_logits, uncond_logits = next_token_logits.chunk(
                    #     chunks=2, dim=0
                    # )
                    # cond_logits = (
                    #     scfg.cfg_coef * cond_logits
                    #     + (1 - scfg.cfg_coef) * uncond_logits
                    # )
                    # next_token_logits = torch.cat((cond_logits, uncond_logits), dim=0)
                    # sampling
                    should_stop = self.dynamic_decoder.forward(
                        next_token_logits,
                        scfg.cfg_coef,
                        decode_step,
                        max_context_length,
                        self.max_attention_window_size,
                        self.sink_token_length,
                        ite,
                        batch_size,
                        self.end_ids,
                        self.embedding_bias_opt,
                        context_lengths,
                        sequence_limit_lengths,
                        stop_words_list_ptrs,
                        stop_words_lens,
                        max_stop_words_len,
                        bad_words_list_ptrs,
                        bad_words_lens,
                        max_bad_words_len,
                        this_src_cache_indirection,
                        self.output_ids,
                        self.new_tokens,
                        self.finished,
                        self.finished,
                        self.sequence_length_buffer,
                        self.cum_log_probs,
                        self.log_probs,
                        self.log_probs_tiled,
                        self.parent_ids,
                        this_tgt_cache_indirection,
                        self.beam_hyps_output_ids_cba,
                        self.beam_hyps_seq_len_cba,
                        self.beam_hyps_cum_log_probs_cba,
                        self.beam_hyps_normed_scores_cba,
                        self.beam_hyps_log_probs_cba,
                        self.beam_hyps_min_normed_scores,
                        self.beam_hyps_num_beams,
                        self.beam_hyps_is_done,
                        scfg.use_beam_hyps,
                    )
                    logger.info(
                        f"{decode_step=} {should_stop=} {self.new_tokens=} {self.finished=}"
                    )

                    if stopping_criteria is not None and not should_stop.item():
                        final_output_ids = self.finalize_decoder(
                            context_lengths,
                            batch_size,
                            beam_width,
                            scfg,
                            in_progress=True,
                        )
                        # keep the shape as same as huggingface stopping_criteria
                        final_output_ids_ = final_output_ids.reshape(
                            -1, final_output_ids.size(-1)
                        )
                        should_stop[0] = stopping_criteria(
                            step, final_output_ids_, logits
                        )

        if self.runtime._is_profiling():
            if not context.report_to_profiler():
                logger.warning("Runtime report to profiler failed.")
            self.runtime._insert_step_to_profiler(step)

        if self.mapping.has_pp():
            should_stop = self.pp_communicate_new_tokens(
                should_stop, this_tgt_cache_indirection, self.sequence_length_buffer
            )

        if self.paged_kv_cache and self.has_attn_layers:
            if (step >= self.max_new_tokens - 1) or (
                should_stop is not None and should_stop.item()
            ):
                # Free all blocks in all sequences.
                # With in-flight batching and while loop we'll free some sequences, when they are done
                self.kv_cache_manager.step([True] * batch_size)
                if self.cross_attention:
                    self.cross_kv_cache_manager.step([True] * batch_size)

        if self.debug_mode:
            # self.dump_debug_buffers(step)

            if next_step_tensors is not None:
                self.debug_buffer = {
                    name: tensor.to_torch()
                    for name, tensor in next_step_tensors.items()
                }

        return (
            should_stop,
            next_step_tensors,
            tasks,
            context_lengths,
            host_context_lengths,
            attention_mask,
            context_logits,
            generation_logits,
            encoder_input_lengths,
        )


class ModModelRunner(ModelRunner):
    def __init__(
        self,
        session: ModGenerationSession,
        max_batch_size: int,
        max_input_len: int,
        max_seq_len: int,
        max_beam_width: int,
        lora_manager: LoraManager | None = None,
    ) -> None:
        super().__init__(
            session,
            max_batch_size,
            max_input_len,
            max_seq_len,
            max_beam_width,
            lora_manager,
        )

    @classmethod
    def from_engine(
        cls,
        engine: Engine,
        lora_dir: typing.Optional[typing.List[str]] = None,
        rank: int = 0,
        debug_mode: bool = False,
        lora_ckpt_source: str = "hf",
        medusa_choices: typing.List[typing.List[int]] = None,
        stream: torch.cuda.Stream = None,
        gpu_weights_percent: float = 1,
    ) -> "ModModelRunner":
        pretrained_config = engine.config.pretrained_config
        build_config = engine.config.build_config

        tp_size = pretrained_config.mapping.tp_size
        num_heads = pretrained_config.num_attention_heads // tp_size
        num_kv_heads = pretrained_config.num_key_value_heads
        num_kv_heads = (num_kv_heads + tp_size - 1) // tp_size
        hidden_size = pretrained_config.hidden_size // tp_size
        head_size = pretrained_config.head_size

        rnn_config_items = [
            "conv_kernel",
            "layer_types",
            "rnn_hidden_size",
            "state_size",
            "state_dtype",
        ]
        rnn_configs_kwargs = {}
        for item in rnn_config_items:
            if hasattr(pretrained_config, item):
                rnn_configs_kwargs[item] = getattr(pretrained_config, item)

        model_config = ModelConfig(
            max_batch_size=build_config.max_batch_size,
            max_beam_width=build_config.max_beam_width,
            vocab_size=pretrained_config.vocab_size,
            num_layers=pretrained_config.num_hidden_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_size=head_size,
            gpt_attention_plugin=bool(build_config.plugin_config.gpt_attention_plugin),
            mamba_conv1d_plugin=bool(build_config.plugin_config.mamba_conv1d_plugin),
            remove_input_padding=build_config.plugin_config.remove_input_padding,
            paged_kv_cache=build_config.plugin_config.paged_kv_cache,
            paged_state=build_config.plugin_config.paged_state,
            tokens_per_block=build_config.plugin_config.tokens_per_block,
            quant_mode=pretrained_config.quant_mode,
            gather_context_logits=build_config.gather_context_logits,
            gather_generation_logits=build_config.gather_generation_logits,
            dtype=pretrained_config.dtype,
            max_prompt_embedding_table_size=build_config.max_prompt_embedding_table_size,
            lora_plugin=build_config.plugin_config.lora_plugin,
            lora_target_modules=build_config.lora_config.lora_target_modules,
            trtllm_modules_to_hf_modules=build_config.lora_config.trtllm_modules_to_hf_modules,
            max_medusa_tokens=pretrained_config.max_draft_len
            if hasattr(pretrained_config, "max_draft_len")
            else 0,
            num_medusa_heads=pretrained_config.num_medusa_heads
            if hasattr(pretrained_config, "num_medusa_heads")
            else 0,
            use_custom_all_reduce=build_config.plugin_config.use_custom_all_reduce,
            **rnn_configs_kwargs,
            gpu_weights_percent=gpu_weights_percent,
        )
        max_batch_size = build_config.max_batch_size
        max_input_len = build_config.max_input_len
        max_seq_len = build_config.max_seq_len
        max_beam_width = build_config.max_beam_width
        assert not (
            pretrained_config.architecture == "ChatGLMForCausalLM"
            and pretrained_config.chatglm_version in ["glm", "chatglm"]
        )
        session_cls = ModGenerationSession
        engine_buffer = engine.engine
        runtime_mapping = pretrained_config.mapping

        if medusa_choices is not None:
            assert (
                session_cls == GenerationSession
            ), "Medusa is only supported by GenerationSession"

            assert (
                model_config.max_medusa_tokens > 0
            ), "medusa_chioce is specified but model_config.max_medusa_tokens is 0."

        if MpiComm.size() > runtime_mapping.gpus_per_node:
            assert MpiComm.local_size() == runtime_mapping.gpus_per_node
        torch.cuda.set_device(rank % runtime_mapping.gpus_per_node)
        session = session_cls(
            model_config,
            engine_buffer,
            runtime_mapping,
            debug_mode=debug_mode,
            stream=stream,
        )
        if trt_gte_10() and session.runtime.engine.streamable_weights_size:
            session.runtime._set_weight_streaming(gpu_weights_percent)

        if session.use_lora_plugin:
            lora_manager = LoraManager()
            if lora_dir is not None:
                lora_manager.load_from_ckpt(
                    model_dir=lora_dir,
                    model_config=model_config,
                    runtime_mapping=runtime_mapping,
                    ckpt_source=lora_ckpt_source,
                )
        else:
            lora_manager = None

        return cls(
            session=session,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            max_beam_width=max_beam_width,
            lora_manager=lora_manager,
        )

    @classmethod
    def from_dir(
        cls,
        engine_dir: str,
        lora_dir: typing.Optional[typing.List[str]] = None,
        rank: int = 0,
        debug_mode: bool = False,
        lora_ckpt_source: str = "hf",
        medusa_choices: typing.List[typing.List[int]] = None,
        stream: torch.cuda.Stream = None,
        gpu_weights_percent: float = 1,
    ) -> "ModModelRunner":
        engine_version = get_engine_version(engine_dir)
        assert engine_version is not None, "new format engine_version MUST NOT be NONE"
        profiler.start("load tensorrt_llm engine")
        engine = Engine.from_dir(engine_dir, rank)
        if lora_dir is None:
            config_lora_dir = engine.config.build_config.lora_config.lora_dir
            if len(config_lora_dir) > 0:
                lora_dir = [f"{engine_dir}/{dir}" for dir in config_lora_dir]
                lora_ckpt_source = (
                    engine.config.build_config.lora_config.lora_ckpt_source
                )
        runner = ModModelRunner.from_engine(
            engine,
            lora_dir,
            rank,
            debug_mode,
            lora_ckpt_source,
            medusa_choices,
            stream,
            gpu_weights_percent,
        )
        profiler.stop("load tensorrt_llm engine")
        loading_time = profiler.elapsed_time_in_sec("load tensorrt_llm engine")
        logger.info(f"Load engine takes: {loading_time} sec")
        return runner


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    else:
        raise TypeError("%s is not supported" % dtype)


class MultimodalModelRunner:
    def __init__(self, args):
        from transformers.models.llama import LlamaConfig
        from safetensors.torch import load_file as safetensors_load_file

        class PrefixWrapper(torch.nn.Module):
            def __init__(self, config: LlamaConfig) -> None:
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(
                    config.vocab_size, config.hidden_size, config.pad_token_id
                )
                self.project_out = torch.nn.Linear(512, config.hidden_size, bias=False)
                self.motif = torch.nn.Embedding(2, config.hidden_size)
                self.config = config

            @torch.inference_mode()
            def get_prefix(
                self,
                input_ids: torch.Tensor,
                clap: torch.Tensor,
                motif_ids: torch.Tensor,
            ) -> torch.Tensor:
                """return prefix embeddings
                input_ids, clap and motif_ids all contain conditional(first half) and unconditional(last half) components
                [batch_size, seq_len, emb_dim]
                """
                cast_dtype = self.config.torch_dtype
                with torch.autocast(
                    self.embed_tokens.weight.device.type,
                    dtype=cast_dtype,
                    enabled=True,
                ):
                    # NOTE: embedding layer DOESNOT convert dtype
                    input_emb = self.embed_tokens(input_ids).to(cast_dtype)
                    clap = clap.to(input_emb.dtype)

                    clap_emb = self.project_out(clap)
                    input_emb = torch.cat([clap_emb, input_emb], dim=1)

                    if motif_ids is not None:
                        input_len = input_emb.size(1)
                        motif_len = motif_ids.size(1)

                        if input_len != motif_len:
                            # prepend zeros
                            motif_ids = torch.nn.functional.pad(
                                motif_ids, pad=(input_len - motif_len, 0), value=0
                            )
                        motif_emb = (
                            self.motif(motif_ids).to(cast_dtype) * motif_ids[:, :, None]
                        )
                        input_emb = input_emb + motif_emb

                    # split conditional and unconditional
                    cond_emb, uncond_emb = torch.chunk(input_emb, chunks=2, dim=0)
                    # to list
                    cond_prefix_list = torch.split(
                        cond_emb, split_size_or_sections=1, dim=0
                    )
                    uncond_prefix_list = torch.split(
                        uncond_emb, split_size_or_sections=1, dim=0
                    )
                return cond_prefix_list, uncond_prefix_list

            @torch.inference_mode()
            def get_token_emb(self, token_ids_list: list):
                import einops

                cast_dtype = self.config.torch_dtype
                with torch.autocast(
                    self.embed_tokens.weight.device.type,
                    dtype=cast_dtype,
                    enabled=True,
                ):
                    token_ids_tensor_list = [
                        torch.tensor(token_ids, dtype=torch.long)
                        if isinstance(token_ids, list)
                        else token_ids
                        for token_ids in token_ids_list
                    ]
                    packed_ids, packed_shape = einops.pack(token_ids_tensor_list, "*")

                    # NOTE: embedding layer DOESNOT convert dtype
                    input_emb = self.embed_tokens(
                        packed_ids.to(self.embed_tokens.weight.device)
                    ).to(cast_dtype)

                    input_emb_list = einops.unpack(input_emb, packed_shape, "* d")
                    # input_emb_list = [x.unsqueeze(0) for x in input_emb_list]
                return input_emb_list

            @torch.inference_mode()
            def get_emb(self, token_ids: torch.Tensor):
                return self.get_token_emb([token_ids])[0]

            @torch.inference_mode()
            def get_clap_emb(self, clap: torch.Tensor):
                cast_dtype = self.config.torch_dtype
                with torch.autocast(
                    self.embed_tokens.weight.device.type,
                    dtype=cast_dtype,
                    enabled=True,
                ):
                    clap_emb = self.project_out(clap)
                return clap_emb

        def get_prefix_wrapper(
            model_dir: Path,
            config_dir: Path,
            device: typing.Union[torch.device, str],
            default_dtype: torch.dtype,
        ):
            config: LlamaConfig = AutoConfig.from_pretrained(config_dir)
            # convert dtype
            if config.torch_dtype != default_dtype:
                config.torch_dtype = default_dtype
            prefix_processor = PrefixWrapper(config).to(device)
            prefix_processor.eval()
            prefix_checkpoint_path = model_dir.joinpath("model_prefix.safetensors")
            prefix_processor.load_state_dict(
                safetensors_load_file(prefix_checkpoint_path)
            )
            logger.info(
                f"done loading prefix_processor from {prefix_checkpoint_path.as_posix()}"
            )
            return prefix_processor

        self.args = args
        self.runtime_rank = tensorrt_llm.mpi_rank()
        device_id = self.runtime_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_id)
        self.device = "cuda:%d" % (device_id)

        self.stream = torch.cuda.Stream(torch.cuda.current_device())
        torch.cuda.set_stream(self.stream)

        # set params
        default_dtype = torch.bfloat16
        self.model_type = "CustomLlamaModel"

        self.profiling_iterations = 20

        self.prefix_wrapper = get_prefix_wrapper(
            args.prefix_model_dir,
            args.prefix_config_dir,
            device=self.device,
            default_dtype=default_dtype,
        )
        tokenizer_name = "bert-base-multilingual-cased"
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        sys.path.append(args.prefix_model_dir.as_posix())
        import dataset as import_dataset

        self.import_dataset = import_dataset

        # init llm
        self.model = ModModelRunner.from_dir(
            args.llm_engine_dir,
            rank=tensorrt_llm.mpi_rank(),
            debug_mode=True,
            #   debug_mode=False,
            stream=self.stream,
        )
        self.model_config = self.model.session._model_config
        self.runtime_mapping = self.model.session.mapping

    def get_input_seq(
        self,
        text_list: typing.List[str],
        tokenizer: BertTokenizer,
        import_dataset: typing.Type,
    ):
        text_token = (
            tokenizer(
                text_list, add_special_tokens=True, return_tensors="pt", padding=False
            )["input_ids"]
            + import_dataset.FIRST_TEXT
        )
        batch_size, text_len = text_token.size()

        null_token = (
            torch.full_like(text_token, fill_value=import_dataset.MASK_TEXT)
            + import_dataset.FIRST_TEXT
        )
        null_token[:, 0] = text_token[:, 0]
        null_token[:, -1] = text_token[:, -1]

        # init motif tokens
        motif_tokens = torch.full(
            (batch_size, 1), fill_value=import_dataset.EOS_MOTIF + 1
        )
        eos_motif_tokens = torch.full(
            (batch_size, 1), fill_value=import_dataset.EOS_MOTIF
        )

        duration = 0
        duration_token = torch.full(
            (batch_size, 1), fill_value=import_dataset.DURATION_SHIFT + duration
        )
        null_duration_token = torch.full_like(
            duration_token, fill_value=import_dataset.DURATION_SHIFT
        )
        eos_duration_token = torch.full(
            (batch_size, 1), fill_value=import_dataset.EOS_DURATION
        )
        bos_audio_token = torch.full(
            (batch_size, 1), fill_value=import_dataset.BOS_AUDIO
        )
        cond_input_seq = torch.concat(
            [
                text_token,
                motif_tokens,
                eos_motif_tokens,
                duration_token,
                eos_duration_token,
                bos_audio_token,
            ],
            dim=1,
        )
        uncond_input_seq = torch.concat(
            [
                null_token,
                motif_tokens,
                eos_motif_tokens,
                null_duration_token,
                eos_duration_token,
                bos_audio_token,
            ],
            dim=1,
        )
        input_seq = torch.cat((cond_input_seq, uncond_input_seq), dim=0).to(self.device)
        return input_seq

    def setup_fake_prompts(self, clap_list: typing.List[torch.Tensor]):
        # convert clap_list to embeddings
        clap_tensor = torch.cat(clap_list, dim=0).to(self.device)
        # [batch_size, 1, emb_dim]
        clap_emb = self.prefix_wrapper.get_clap_emb(clap_tensor)
        # init prompt id
        vocab_size = self.model_config.vocab_size
        # [batch_size * 1] -> [batch_size, 1]
        fake_prompt_id = torch.arange(
            vocab_size,
            vocab_size + clap_emb.size(0) * clap_emb.size(1),
            device=self.device,
        )
        fake_prompt_id = einops.rearrange(
            fake_prompt_id, "(b s) -> b s", b=clap_emb.size(0)
        )
        # emb table
        clap_emb = einops.rearrange(clap_emb, "b s d -> (b s) d")
        return fake_prompt_id, clap_emb

    def generate(
        self,
        text_list: typing.List[str],
        clap_list: typing.List[torch.Tensor],
        max_new_tokens,
        warmup,
        lora_uids: typing.Optional[list] = None,
    ):
        if not warmup:
            profiler.start("Generate")

        # construct input_ids, input_lengths, ptuning_args
        input_seq = self.get_input_seq(text_list, self.tokenizer, self.import_dataset)
        assert input_seq.size(0) == len(
            clap_list
        ), f"mismatched inputs {text_list} vs {clap_list}"
        fake_prompt_id, prompt_emb_table = self.setup_fake_prompts(clap_list)
        input_ids = torch.cat((fake_prompt_id, input_seq), dim=1)
        input_lengths = torch.IntTensor([input_ids.shape[1]] * input_ids.shape[0]).to(
            torch.int32
        )
        ptuning_args = [prompt_emb_table]

        if warmup:
            return None

        streaming = True
        profiler.start("LLM")
        ptuning_args[0] = torch.stack([ptuning_args[0]])
        scfg = ModSamplingConfig(end_id=None, pad_id=None, cfg_coef=self.args.cfg_coef)
        output_ids = self.model.generate(
            input_ids,
            sampling_config=scfg,
            prompt_table=ptuning_args[0],
            lora_uids=lora_uids,
            max_new_tokens=max_new_tokens,
            # ThomasLee
            # end_id=self.import_dataset.BOS_AUDIO,
            # pad_id=self.import_dataset.EOS_AUDIO,
            end_id=32768,
            pad_id=-1,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            num_beams=self.args.num_beams,
            output_sequence_lengths=False,
            return_dict=False,
            streaming=streaming,
        )
        profiler.stop("LLM")

        if tensorrt_llm.mpi_rank() == 0:
            # Extract a list of tensors of shape beam_width x output_ids.
            if streaming:
                for idx, output in enumerate(output_ids):
                    pass
                logger.info(
                    f"{self.model.session.finished=} {self.model.session.sequence_length_buffer=}"
                )
                output_token_list = [
                    output[
                        batch_idx,
                        :,
                        input_lengths[
                            batch_idx
                        ] : self.model.session.sequence_length_buffer[batch_idx].item(),
                    ]
                    for batch_idx in range(self.args.batch_size // 2)
                ]
            else:
                output_token_list = [
                    output_ids[batch_idx, :, input_lengths[batch_idx] :]
                    for batch_idx in range(self.args.batch_size // 2)
                ]
            profiler.stop("Generate")
            return output_token_list
        else:
            profiler.stop("Generate")
            return None

    def ptuning_setup(self, prompt_table, input_ids, input_lengths):
        hidden_size = self.model_config.hidden_size * self.runtime_mapping.tp_size
        if prompt_table is not None:
            task_vocab_size = torch.tensor(
                [prompt_table.shape[1]],
                dtype=torch.int32,
            ).cuda()
            prompt_table = prompt_table.view(
                (prompt_table.shape[0] * prompt_table.shape[1], prompt_table.shape[2])
            )

            assert (
                prompt_table.shape[1] == hidden_size
            ), "Prompt table dimensions do not match hidden size"

            prompt_table = prompt_table.cuda().to(
                dtype=tensorrt_llm._utils.str_dtype_to_torch(self.model_config.dtype)
            )
        else:
            prompt_table = torch.empty([1, hidden_size]).cuda()
            task_vocab_size = torch.zeros([1]).cuda()

        if self.model_config.remove_input_padding:
            tasks = torch.zeros([torch.sum(input_lengths)], dtype=torch.int32).cuda()
            if self.decoder_llm:
                tasks = tasks.unsqueeze(0)
        else:
            tasks = torch.zeros(input_ids.shape, dtype=torch.int32).cuda()

        return [prompt_table, tasks, task_vocab_size]

    def run(
        self,
        text_list: typing.List[str],
        clap_list: typing.List[torch.Tensor],
        max_new_tokens: int,
        lora_uids: typing.Optional[list] = None,
    ):
        import time

        self.generate(
            text_list, clap_list, max_new_tokens, warmup=True, lora_uids=lora_uids
        )
        s_time = time.perf_counter()
        output_token_list = self.generate(
            text_list, clap_list, max_new_tokens, warmup=False, lora_uids=lora_uids
        )
        e_time = time.perf_counter()
        max_len = max([x.size(-1) for x in output_token_list])
        elapsed_time = e_time - s_time
        rtf = elapsed_time * 25 / max_len
        logger.info(f"{elapsed_time=} {max_len=} {rtf=}")
        return output_token_list


def infer():
    import datetime
    import numpy as np

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tensorrt_llm.logger.set_level("debug")

    # init model
    num_samples = 3
    batch_size = num_samples * 2
    args = SimpleNamespace(
        prefix_model_dir=Path("/mnt/workspace/users/xingda.li/s1_vllm/v534-crazy"),
        prefix_config_dir=Path(
            "/mnt/workspace/users/xingda.li/s1_vllm/v534-crazy/vllm"
        ),
        # llm_engine_dir=Path("/dev/shm/v534-crazy_engine/1gpu_bf16"),
        # llm_engine_dir=Path("/dev/shm/v534-crazy_engine/1gpu_bf16_bs8"),
        llm_engine_dir=Path("/dev/shm/v534-crazy_engine/1gpu_bf16_bs8_lora"),
        top_k=-1,
        top_p=1.0,
        temperature=0.75,
        repetition_penalty=1.1,
        cfg_coef=1.3,
        num_beams=1,
        batch_size=batch_size,
    )
    model = MultimodalModelRunner(args)
    # init inputs
    text_list = [
        # """[VERSE]In a world of sidewalks and traffic lights;We used to chase the sun, we used to take flight;With paper planes in hand, we'd let them soar;Dreaming of places we had never explored;[CHORUS]Oh, paper planes and dreams, they're all I need;To take me to a place where my heart is free;In a world that's filled with noise and screams;I find solace in my paper planes and dreams;"""
        """[verse]请少年别躲开她颊边一抹红；那是她未曾宣之于口的情衷；[chorus]海棠挽起云髻唤名作钗头红；应于妆镜前衬着佳人如梦；"""
    ] * num_samples
    clap_all = np.load("sampled_clap.npy")[:batch_size]
    # clap_list = [torch.zeros((1, 1, 512), device=model.device) + 1e-7] * batch_size
    clap_list = [
        torch.from_numpy(x).to(model.device).unsqueeze(0).unsqueeze(0).fill_(1e-7)
        for x in clap_all
    ]
    lora_uids = [0, 1, 4] * 2
    # NOTE:
    lora_uids = [str(uid) for uid in lora_uids]
    # lora_uids = ["0"] * batch_size
    # lora_uids = None
    # run model
    output_token_list = model.run(
        text_list, clap_list, max_new_tokens=3400, lora_uids=lora_uids
    )
    # save
    output_root = Path("tmp_trtllm_output_lora")
    tmp_output = output_root.joinpath(
        f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")}'
    )
    if not tmp_output.exists():
        tmp_output.mkdir(parents=True, exist_ok=True)
    for idx, output_token in enumerate(output_token_list):
        tmp_path = tmp_output.joinpath(f"token_{idx}.npy")
        tmp_array = output_token.squeeze(0).cpu().numpy()
        logger.info(f"token_{idx} {tmp_array.shape=}")
        np.save(tmp_path, tmp_array)


if __name__ == "__main__":
    infer()
