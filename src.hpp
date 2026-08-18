#pragma once

#include "simulator.hpp"

#include <cassert>
#include <string>
#include <vector>

namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());

  Matrix *cumulative_transposed_keys = nullptr;
  Matrix *cumulative_values = nullptr;

  for (size_t i = 0; i < keys.size(); ++i) {
    Matrix *current_query = rater.GetNextQuery();
    const bool is_last_round = (i + 1 == keys.size());

    /*
     * Transfer the newly required inputs to SRAM. Keys are transferred first
     * because the calculation queue can begin transposing and concatenating
     * them while the query and value transfers continue.
     */
    gpu_sim.MoveMatrixToSharedMem(keys[i]);
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(values[i]);

    /*
     * Keep the cumulative key matrix in transposed form:
     *
     *   [K_0^T K_1^T ... K_i^T]
     *
     * Each incoming key has shape 1 x d. After transposition it has shape
     * d x 1, so it can be appended along axis 1. This avoids copying and
     * transposing the entire cumulative key matrix on every round.
     */
    gpu_sim.Transpose(keys[i], kInSharedMemory);

    if (cumulative_transposed_keys == nullptr) {
      cumulative_transposed_keys = keys[i];
    } else {
      Matrix *previous_keys = cumulative_transposed_keys;
      Matrix *combined_keys = matrix_memory_allocator.Allocate(
          "cumulative_transposed_keys_" + std::to_string(i));

      gpu_sim.Concat(previous_keys, keys[i], combined_keys, 1,
                     kInSharedMemory);
      gpu_sim.ReleaseMatrix(previous_keys);
      gpu_sim.ReleaseMatrix(keys[i]);

      cumulative_transposed_keys = combined_keys;
    }

    /*
     * Compute the unnormalized attention scores:
     *
     *   scores = Q K^T
     */
    Matrix *scores =
        matrix_memory_allocator.Allocate("scores_" + std::to_string(i));
    gpu_sim.MatMul(current_query, cumulative_transposed_keys, scores);

    if (is_last_round) {
      gpu_sim.ReleaseMatrix(cumulative_transposed_keys);
    }
    gpu_sim.ReleaseMatrix(current_query);

    /*
     * Compute exp(scores). Softmax must be performed independently for every
     * row, so each row is extracted, summed, divided by its sum, and then
     * concatenated back into the complete probability matrix.
     */
    Matrix *exponentials =
        matrix_memory_allocator.Allocate("exponentials_" + std::to_string(i));
    gpu_sim.MatExp(scores, exponentials);
    gpu_sim.ReleaseMatrix(scores);

    Matrix *softmax = nullptr;
    const size_t query_row_count = current_query->GetRowNum();

    for (size_t row_index = 0; row_index < query_row_count; ++row_index) {
      const std::string row_suffix =
          std::to_string(i) + "_" + std::to_string(row_index);

      Matrix *exponential_row =
          matrix_memory_allocator.Allocate("exponential_row_" + row_suffix);
      Matrix *row_sum =
          matrix_memory_allocator.Allocate("row_sum_" + row_suffix);
      Matrix *normalized_row =
          matrix_memory_allocator.Allocate("normalized_row_" + row_suffix);

      gpu_sim.GetRow(exponentials, row_index, exponential_row,
                     kInSharedMemory);
      gpu_sim.Sum(exponential_row, row_sum);
      gpu_sim.MatDiv(exponential_row, row_sum, normalized_row);

      gpu_sim.ReleaseMatrix(exponential_row);
      gpu_sim.ReleaseMatrix(row_sum);

      if (softmax == nullptr) {
        softmax = normalized_row;
      } else {
        Matrix *previous_softmax = softmax;
        Matrix *combined_softmax =
            matrix_memory_allocator.Allocate("softmax_" + row_suffix);

        gpu_sim.Concat(previous_softmax, normalized_row, combined_softmax, 0,
                       kInSharedMemory);
        gpu_sim.ReleaseMatrix(previous_softmax);
        gpu_sim.ReleaseMatrix(normalized_row);

        softmax = combined_softmax;
      }
    }

    gpu_sim.ReleaseMatrix(exponentials);

    /*
     * Build the cumulative value matrix:
     *
     *   [V_0]
     *   [V_1]
     *   [...]
     *   [V_i]
     *
     * This is scheduled after softmax construction so the score calculation
     * can overlap with the incoming value transfer.
     */
    if (cumulative_values == nullptr) {
      cumulative_values = values[i];
    } else {
      Matrix *previous_values = cumulative_values;
      Matrix *combined_values = matrix_memory_allocator.Allocate(
          "cumulative_values_" + std::to_string(i));

      gpu_sim.Concat(previous_values, values[i], combined_values, 0,
                     kInSharedMemory);
      gpu_sim.ReleaseMatrix(previous_values);
      gpu_sim.ReleaseMatrix(values[i]);

      cumulative_values = combined_values;
    }

    /*
     * Complete attention:
     *
     *   answer = softmax(Q K^T) V
     */
    Matrix *answer =
        matrix_memory_allocator.Allocate("answer_" + std::to_string(i));
    gpu_sim.MatMul(softmax, cumulative_values, answer);
    gpu_sim.ReleaseMatrix(softmax);

    if (is_last_round) {
      gpu_sim.ReleaseMatrix(cumulative_values);
    }

    gpu_sim.MoveMatrixToGpuHbm(answer);

    /*
     * All scheduled operations, including the final SRAM-to-HBM transfer,
     * must finish before the answer is committed.
     */
    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*answer);
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu
