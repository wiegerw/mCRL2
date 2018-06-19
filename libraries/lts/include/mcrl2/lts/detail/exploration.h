// Author(s): Ruud Koolen
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef MCRL2_LTS_DETAIL_EXPLORATION_NEW_H
#define MCRL2_LTS_DETAIL_EXPLORATION_NEW_H

#include <string>
#include <limits>
#include <memory>
#include <unordered_set>

#include "mcrl2/atermpp/indexed_set.h"
#include "mcrl2/trace/trace.h"
#include "mcrl2/lps/next_state_generator.h"
#include "mcrl2/lts/lts_lts.h"
#include "mcrl2/lts/detail/bithashtable.h"
#include "mcrl2/lts/detail/queue.h"
#include "mcrl2/lts/detail/lts_generation_options.h"
#include "mcrl2/lts/detail/exploration_strategy.h"


namespace mcrl2
{
namespace lts
{
namespace detail
{

    template <class COUNTER_EXAMPLE_GENERATOR>
    class state_index_pair
    {
      protected:
        lps::state m_state;
        typename COUNTER_EXAMPLE_GENERATOR::index_type m_index;
    
      public:
        state_index_pair(const lps::state& state, typename COUNTER_EXAMPLE_GENERATOR::index_type index)
         : m_state(state),
           m_index(index)
        {}
    
        lps::state state() const
        {
          return m_state;
        }
    
        typename COUNTER_EXAMPLE_GENERATOR::index_type index() const
        {
          return m_index;
        }
    };
} // end namespace detail

class lps2lts_algorithm
{
  private:
    typedef lps::next_state_generator next_state_generator;

  private:
    lts_generation_options m_options;
    next_state_generator *m_generator;
    next_state_generator::summand_subset *m_main_subset;

    atermpp::indexed_set<lps::state> m_state_numbers;
    bit_hash_table m_bit_hash_table;

    probabilistic_lts_lts_t m_output_lts;
    atermpp::indexed_set<process::action_list> m_action_label_numbers; 
    std::ofstream m_aut_file;

    bool m_maintain_traces;

    std::vector<bool> m_detected_action_summands;

    std::map<lps::state, lps::state> m_backpointers;
    std::size_t m_traces_saved;

    std::size_t m_num_states;
    std::size_t m_num_transitions;
    next_state_generator::transition::state_probability_list m_initial_states;
    std::size_t m_level;

    volatile bool m_must_abort;

  public:
    lps2lts_algorithm() :
      m_generator(nullptr),
      m_must_abort(false)
    {
      m_action_label_numbers.put(action_label_lts::tau_action().actions());  // The action tau has index 0 by default.
    }

    ~lps2lts_algorithm()
    {
      delete m_generator;
    }

    bool initialise_lts_generation(lts_generation_options* options);
    bool generate_lts();
    bool finalise_lts_generation();

    void abort()
    {
      // Stops the exploration algorithm if it is running by making sure
      // not a single state can be generated anymore.
      if (!m_must_abort)
      {
        m_must_abort = true;
        mCRL2log(log::warning) << "state space generation was aborted prematurely" << std::endl;
      }
    }

  private:
    bool save_trace(const lps::state& state1, const std::string& filename);
    bool save_trace(const lps::state& state1, const next_state_generator::transition& transition, const std::string& filename);
    void construct_trace(const lps::state& state1, mcrl2::trace::Trace& trace);

    bool is_nondeterministic(std::vector<lps2lts_algorithm::next_state_generator::transition>& transitions,
                             next_state_generator::transition& nondeterminist_transition);
    void save_actions(const lps::state& state, const next_state_generator::transition& transition);
    void save_deadlock(const lps::state& state);
    void save_nondeterministic_state(const lps::state& state, const next_state_generator::transition& nondeterminist_transition);
    void save_error(const lps::state& state);
    std::pair<std::size_t, bool> add_target_state(const lps::state& source_state, const lps::state& target_state);
    bool add_transition(const lps::state& source_state, const next_state_generator::transition& transition);
    void get_transitions(const lps::state& state,
                         std::vector<lps2lts_algorithm::next_state_generator::transition>& transitions,
                         next_state_generator::enumerator_queue& enumeration_queue
    );
    void generate_lts_breadth_todo_max_is_npos();
    void generate_lts_breadth_todo_max_is_not_npos(const next_state_generator::transition::state_probability_list& initial_states);
    void print_target_distribution_in_aut_format(
               const lps::next_state_generator::transition::state_probability_list& state_probability_list,
               const std::size_t last_state_number,
               const lps::state& source_state);
    void print_target_distribution_in_aut_format(
                const lps::next_state_generator::transition::state_probability_list& state_probability_list,
                const lps::state& source_state);
    probabilistic_state<std::size_t, lps::probabilistic_data_expression> transform_initial_probabilistic_state_list
                 (const next_state_generator::transition::state_probability_list& initial_states);
    probabilistic_state<std::size_t, lps::probabilistic_data_expression> create_a_probabilistic_state_from_target_distribution(
               const std::size_t base_state_number,
               const next_state_generator::transition::state_probability_list& other_probabilities,
               const lps::state& source_state);


};

} // namespace lps

} // namespace mcrl2

#endif // MCRL2_LTS_DETAIL_EXPLORATION_H
