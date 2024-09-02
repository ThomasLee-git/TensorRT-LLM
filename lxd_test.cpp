#include "tensorrt_llm/runtime/torch.h"
#include "tensorrt_llm/runtime/utils/numpyUtils.h"
#include "tensorrt_llm/thop/dynamicDecodeOp.h"

#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>

th::Tensor from_path(std::string const& path, tensorrt_llm::runtime::BufferManager& manager)
{
    auto npy_data = tensorrt_llm::runtime::utils::loadNpy(manager, path, tensorrt_llm::runtime::MemoryType::kGPU);
    std::shared_ptr<tensorrt_llm::runtime::ITensor> dummy = std::move(npy_data);
    return tensorrt_llm::runtime::Torch::tensor(dummy);
}

/* deprecated
int test()
{
    // set logger
    // tensorrt_llm::common::Logger::getLogger()->setLevel(tensorrt_llm::common::Logger::Level::TRACE);
    // construct
    int64_t const max_batch_size = 2;
    int64_t const max_beam_width = 1;
    int64_t const vocab_size = 153600;
    int64_t const vocab_size_padded = 153600;
    int64_t const tensor_para_size = 1;
    int64_t const pipeline_para_size = 1;
    at::ScalarType const scaler_type = at::ScalarType::Float;
    torch_ext::DynamicDecodeOp op(max_batch_size, max_beam_width, vocab_size, vocab_size_padded, tensor_para_size,
        pipeline_para_size, scaler_type);
    // setup
    int64_t const batch_size = 2;
    int64_t const beam_width = 1;
    th::Tensor const top_k = torch::full({batch_size}, -1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const top_p = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const temperature = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const repetition_penalty
        = torch::full({batch_size}, 1.1, torch::dtype(torch::kFloat).requires_grad(false));
    // print
    std::cout << "top_k: " << top_k << std::endl;
    std::cout << "top_p: " << top_p << std::endl;
    std::cout << "temperature: " << temperature << std::endl;
    std::cout << "repetition_penalty: " << repetition_penalty << std::endl;
    //
    auto const presence_penalty = std::nullopt;
    auto const frequency_penalty = std::nullopt;
    th::Tensor const min_length = torch::full({batch_size}, 1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const length_penalty = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const early_stopping = torch::full({batch_size}, 1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const beam_search_diversity
        = torch::full({batch_size}, 0, torch::dtype(torch::kFloat).requires_grad(false));
    auto const random_seed = std::nullopt;
    auto const topp_decay = std::nullopt;
    auto const topp_min = std::nullopt;
    auto const topp_reset_ids = std::nullopt;
    auto const no_repeat_ngrams_size = std::nullopt;
    bool const output_log_probs = false;
    bool const cum_log_probs = false;
    op.setup(batch_size, beam_width, top_k, top_p, temperature, repetition_penalty, presence_penalty, frequency_penalty,
        min_length, length_penalty, early_stopping, beam_search_diversity, random_seed, topp_decay, topp_min,
        topp_reset_ids, no_repeat_ngrams_size, output_log_probs, cum_log_probs);
    // forward
    // inputs
    auto const device = at::Device{at::kCUDA, 0};
    // auto const cpu_device = at::Device{at::kCPU, 0};
    auto manager = tensorrt_llm::runtime::BufferManager(std::make_shared<tensorrt_llm::runtime::CudaStream>());
    std::string const logits_path = "tmp_inputs/next_token_logits_bs2.npy";
    auto logits = from_path(logits_path, manager);
    int64_t const decode_step = 65;
    int64_t const max_context_length = 65;
    int64_t const max_attention_window_size = 2565;
    int64_t const sink_token_length = 0;
    int64_t const ite = 0;
    // int64_t const batch_size = 1;
    th::Tensor const end_ids
        = torch::full({batch_size}, 32768, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const embedding_bias_opt = std::nullopt;
    th::Tensor const context_lengths
        = torch::full({batch_size * beam_width}, 65, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const sequence_limit_lengths
        = torch::full({batch_size, 1}, 2565, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const stop_words_list_ptrs = std::nullopt;
    auto const stop_words_lens = std::nullopt;
    int64_t const max_stop_words_len = 0;
    auto const bad_words_list_ptrs = std::nullopt;
    auto const bad_words_lens = std::nullopt;
    int64_t const max_bad_words_len = 0;
    std::string const this_src_cache_indirection_path = "tmp_inputs/this_src_cache_indirection_bs2.npy";
    th::Tensor const this_src_cache_indirection = from_path(this_src_cache_indirection_path, manager);
    // outputs
    std::string const output_ids_path = "tmp_inputs/output_ids_bs2.npy";
    th::Tensor const output_ids = from_path(output_ids_path, manager);
    auto output_ids_dummy = output_ids.clone().to(at::kCPU);
    th::Tensor const new_tokens
        = torch::full({batch_size, beam_width, 1}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const finished_input
        = torch::full({batch_size, beam_width}, 0, torch::dtype(torch::kUInt8).requires_grad(false).device(device));
    th::Tensor const finished_output
        = torch::full({batch_size, beam_width}, 0, torch::dtype(torch::kUInt8).requires_grad(false).device(device));
    th::Tensor const sequence_length_buffer
        = torch::full({batch_size * beam_width}, 65, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const cum_log_probs_opt = std::nullopt;
    auto const log_probs = std::nullopt;
    auto const log_probs_tiled = std::nullopt;
    std::string const parent_ids_path = "tmp_inputs/parent_ids_bs2.npy";
    th::Tensor const parent_ids = from_path(parent_ids_path, manager);
    std::string const this_tgt_cache_indirection_path = "tmp_inputs/this_tgt_cache_indirection_bs2.npy";
    th::Tensor const this_tgt_cache_indirection = from_path(this_src_cache_indirection_path, manager);
    std::string const beam_hyps_output_ids_cba_path = "tmp_inputs/beam_hyps_output_ids_cba_bs2.npy";
    th::Tensor const beam_hyps_output_ids_cba = from_path(beam_hyps_output_ids_cba_path, manager);
    th::Tensor const beam_hyps_seq_len_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const beam_hyps_cum_log_probs_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    th::Tensor const beam_hyps_normed_scores_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    std::string const beam_hyps_log_probs_cba_path = "tmp_inputs/beam_hyps_log_probs_cba_bs2.npy";
    th::Tensor const beam_hyps_log_probs_cba = from_path(beam_hyps_log_probs_cba_path, manager);
    th::Tensor const beam_hyps_min_normed_scores
        = torch::full({batch_size}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    th::Tensor const beam_hyps_num_beams
        = torch::full({batch_size}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const beam_hyps_is_done
        = torch::full({batch_size}, false, torch::dtype(torch::kBool).requires_grad(false).device(device));
    bool const use_beam_hyps = true;
    // print
    // std::cout << "logits: " << logits << std::endl;
    std::cout << "end_ids: " << end_ids << std::endl;
    std::cout << "context_lengths: " << context_lengths << std::endl;
    std::cout << "sequence_limit_lengths: " << sequence_limit_lengths << std::endl;
    std::cout << "repetition_penalty: " << repetition_penalty << std::endl;
    // forward
    std::cout << "before_outout_ids_dummy: "
              //   << output_ids_dummy.index(
              //          {"...", torch::indexing::Slice{torch::indexing::None, 70, torch::indexing::None}})
              << output_ids_dummy << std::endl;
    std::cout << "before_new_tokens:" << new_tokens << std::endl;
    auto result = op.forward(logits, // [BS, BM, VP]
        decode_step,                 //
        max_context_length,          //
        max_attention_window_size,   //
        sink_token_length,           //
        ite,                         //
        batch_size,                  //
        end_ids,                     //
        embedding_bias_opt,          //
        context_lengths,             //
        sequence_limit_lengths,      //
        stop_words_list_ptrs,        //
        stop_words_lens,             //
        max_stop_words_len,          //
        bad_words_list_ptrs,         //
        bad_words_lens,              //
        max_bad_words_len,           //
        this_src_cache_indirection,  //
        output_ids,                  //
        new_tokens,                  //
        finished_input,              //
        finished_output,             //
        sequence_length_buffer,      //
        cum_log_probs_opt,           //
        log_probs,                   //
        log_probs_tiled,             //
        parent_ids,                  //
        this_tgt_cache_indirection,  //
        beam_hyps_output_ids_cba,    //
        beam_hyps_seq_len_cba,       //
        beam_hyps_cum_log_probs_cba, //
        beam_hyps_normed_scores_cba, //
        beam_hyps_log_probs_cba,     //
        beam_hyps_min_normed_scores, //
        beam_hyps_num_beams,         //
        beam_hyps_is_done,           //
        use_beam_hyps                //
    );

    // check output
    output_ids_dummy = output_ids.clone().to(at::kCPU);
    std::cout << "after_output_ids: "
              //   << output_ids_dummy.index(
              //          {"...", torch::indexing::Slice{torch::indexing::None, 70, torch::indexing::None}})
              << output_ids_dummy << std::endl;
    std::cout << "after_new_tokens: " << new_tokens << std::endl;
    return 0;
}
*/
int test_custom_op()
{
    // set logger
    // tensorrt_llm::common::Logger::getLogger()->setLevel(tensorrt_llm::common::Logger::Level::TRACE);
    // construct
    int64_t const max_batch_size = 8;
    int64_t const max_beam_width = 1;
    int64_t const vocab_size = 153600;
    int64_t const vocab_size_padded = 153600;
    int64_t const tensor_para_size = 1;
    int64_t const pipeline_para_size = 1;
    at::ScalarType const scaler_type = at::ScalarType::Float;
    torch_ext::CustomDynamicDecodeOp op(max_batch_size, max_beam_width, vocab_size, vocab_size_padded, tensor_para_size,
        pipeline_para_size, scaler_type);
    // setup
    int64_t const batch_size = 6;
    int64_t const beam_width = 1;
    th::Tensor const top_k = torch::full({batch_size}, -1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const top_p = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const temperature = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const repetition_penalty
        = torch::full({batch_size}, 1.1, torch::dtype(torch::kFloat).requires_grad(false));
    // print
    std::cout << "top_k: " << top_k << std::endl;
    std::cout << "top_p: " << top_p << std::endl;
    std::cout << "temperature: " << temperature << std::endl;
    std::cout << "repetition_penalty: " << repetition_penalty << std::endl;
    //
    auto const presence_penalty = std::nullopt;
    auto const frequency_penalty = std::nullopt;
    th::Tensor const min_length = torch::full({batch_size}, 1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const length_penalty = torch::full({batch_size}, 1.0, torch::dtype(torch::kFloat).requires_grad(false));
    th::Tensor const early_stopping = torch::full({batch_size}, 1, torch::dtype(torch::kInt32).requires_grad(false));
    th::Tensor const beam_search_diversity
        = torch::full({batch_size}, 0, torch::dtype(torch::kFloat).requires_grad(false));
    auto const random_seed = std::nullopt;
    auto const topp_decay = std::nullopt;
    auto const topp_min = std::nullopt;
    auto const topp_reset_ids = std::nullopt;
    auto const no_repeat_ngrams_size = std::nullopt;
    bool const output_log_probs = false;
    bool const cum_log_probs = false;
    op.setup(batch_size, beam_width, top_k, top_p, temperature, repetition_penalty, presence_penalty, frequency_penalty,
        min_length, length_penalty, early_stopping, beam_search_diversity, random_seed, topp_decay, topp_min,
        topp_reset_ids, no_repeat_ngrams_size, output_log_probs, cum_log_probs);
    // forward
    // inputs
    auto const device = at::Device{at::kCUDA, 0};
    // auto const cpu_device = at::Device{at::kCPU, 0};
    auto manager = tensorrt_llm::runtime::BufferManager(std::make_shared<tensorrt_llm::runtime::CudaStream>());
    std::string const logits_path = "tmp_inputs/next_token_logits_bs6.npy";
    auto logits = from_path(logits_path, manager);
    auto const cfg_coef = 1.3;
    int64_t const decode_step = 65;
    int64_t const max_context_length = 65;
    int64_t const max_attention_window_size = 2565;
    int64_t const sink_token_length = 0;
    int64_t const ite = 0;
    // int64_t const batch_size = 1;
    th::Tensor const end_ids
        = torch::full({batch_size}, 32768, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const embedding_bias_opt = std::nullopt;
    th::Tensor const context_lengths
        = torch::full({batch_size * beam_width}, 65, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const sequence_limit_lengths
        = torch::full({batch_size, 1}, 2565, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const stop_words_list_ptrs = std::nullopt;
    auto const stop_words_lens = std::nullopt;
    int64_t const max_stop_words_len = 0;
    auto const bad_words_list_ptrs = std::nullopt;
    auto const bad_words_lens = std::nullopt;
    int64_t const max_bad_words_len = 0;
    std::string const this_src_cache_indirection_path = "tmp_inputs/this_src_cache_indirection_bs6.npy";
    th::Tensor const this_src_cache_indirection = from_path(this_src_cache_indirection_path, manager);
    // outputs
    std::string const output_ids_path = "tmp_inputs/output_ids_bs6.npy";
    th::Tensor const output_ids = from_path(output_ids_path, manager);
    auto output_ids_dummy = output_ids.clone().to(at::kCPU);
    th::Tensor const new_tokens
        = torch::full({batch_size, beam_width, 1}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const finished_input
        = torch::full({batch_size, beam_width}, 0, torch::dtype(torch::kUInt8).requires_grad(false).device(device));
    th::Tensor const finished_output
        = torch::full({batch_size, beam_width}, 0, torch::dtype(torch::kUInt8).requires_grad(false).device(device));
    th::Tensor const sequence_length_buffer
        = torch::full({batch_size * beam_width}, 65, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    auto const cum_log_probs_opt = std::nullopt;
    auto const log_probs = std::nullopt;
    auto const log_probs_tiled = std::nullopt;
    std::string const parent_ids_path = "tmp_inputs/parent_ids_bs6.npy";
    th::Tensor const parent_ids = from_path(parent_ids_path, manager);
    std::string const this_tgt_cache_indirection_path = "tmp_inputs/this_tgt_cache_indirection_bs6.npy";
    th::Tensor const this_tgt_cache_indirection = from_path(this_src_cache_indirection_path, manager);
    std::string const beam_hyps_output_ids_cba_path = "tmp_inputs/beam_hyps_output_ids_cba_bs6.npy";
    th::Tensor const beam_hyps_output_ids_cba = from_path(beam_hyps_output_ids_cba_path, manager);
    th::Tensor const beam_hyps_seq_len_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const beam_hyps_cum_log_probs_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    th::Tensor const beam_hyps_normed_scores_cba
        = torch::full({batch_size, beam_width * 2}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    std::string const beam_hyps_log_probs_cba_path = "tmp_inputs/beam_hyps_log_probs_cba_bs8.npy";
    th::Tensor const beam_hyps_log_probs_cba = from_path(beam_hyps_log_probs_cba_path, manager);
    th::Tensor const beam_hyps_min_normed_scores
        = torch::full({batch_size}, 0, torch::dtype(torch::kFloat).requires_grad(false).device(device));
    th::Tensor const beam_hyps_num_beams
        = torch::full({batch_size}, 0, torch::dtype(torch::kInt32).requires_grad(false).device(device));
    th::Tensor const beam_hyps_is_done
        = torch::full({batch_size}, false, torch::dtype(torch::kBool).requires_grad(false).device(device));
    bool const use_beam_hyps = true;
    // print
    // std::cout << "logits: " << logits << std::endl;
    std::cout << "end_ids: " << end_ids << std::endl;
    std::cout << "context_lengths: " << context_lengths << std::endl;
    std::cout << "sequence_limit_lengths: " << sequence_limit_lengths << std::endl;
    std::cout << "repetition_penalty: " << repetition_penalty << std::endl;
    // forward
    std::cout << "before_outout_ids_dummy: "
              //   << output_ids_dummy.index(
              //          {"...", torch::indexing::Slice{torch::indexing::None, 70, torch::indexing::None}})
              << output_ids_dummy << std::endl;
    std::cout << "before_new_tokens:" << new_tokens << std::endl;
    try
    {

        auto result = op.forward(logits, // [BS, BM, VP]
            cfg_coef,                    //
            decode_step,                 //
            max_context_length,          //
            max_attention_window_size,   //
            sink_token_length,           //
            ite,                         //
            batch_size,                  //
            end_ids,                     //
            embedding_bias_opt,          //
            context_lengths,             //
            sequence_limit_lengths,      //
            stop_words_list_ptrs,        //
            stop_words_lens,             //
            max_stop_words_len,          //
            bad_words_list_ptrs,         //
            bad_words_lens,              //
            max_bad_words_len,           //
            this_src_cache_indirection,  //
            output_ids,                  //
            new_tokens,                  //
            finished_input,              //
            finished_output,             //
            sequence_length_buffer,      //
            cum_log_probs_opt,           //
            log_probs,                   //
            log_probs_tiled,             //
            parent_ids,                  //
            this_tgt_cache_indirection,  //
            beam_hyps_output_ids_cba,    //
            beam_hyps_seq_len_cba,       //
            beam_hyps_cum_log_probs_cba, //
            beam_hyps_normed_scores_cba, //
            beam_hyps_log_probs_cba,     //
            beam_hyps_min_normed_scores, //
            beam_hyps_num_beams,         //
            beam_hyps_is_done,           //
            use_beam_hyps                //
        );
    }
    catch (std::exception const& e)
    {
        std::cout << "exception: " << e.what() << std::endl;
    }

    // check output
    output_ids_dummy = output_ids.clone().to(at::kCPU);
    std::cout << "after_output_ids: "
              //   << output_ids_dummy.index(
              //          {"...", torch::indexing::Slice{torch::indexing::None, 70, torch::indexing::None}})
              << output_ids_dummy << std::endl;
    std::cout << "after_new_tokens: " << new_tokens << std::endl;
    return 0;
}

/*
int test_npy()
{
    auto manager = tensorrt_llm::runtime::BufferManager(std::make_shared<tensorrt_llm::runtime::CudaStream>());
    auto const path = "tmp_inputs/next_token_logits.npy";
    auto data = tensorrt_llm::runtime::utils::loadNpy(manager, path, tensorrt_llm::runtime::MemoryType::kCPU);
    // std::cout << "data: " << data << std::endl;
    return 0;
}
*/
int main(int argc, char* argv[])
{
    // return test();
    return test_custom_op();
    // return test_npy();
}