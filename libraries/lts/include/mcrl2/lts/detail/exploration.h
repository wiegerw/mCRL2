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
#include "mcrl2/lts/detail/queue.h"
#include "mcrl2/lts/detail/lts_generation_options.h"
#include "mcrl2/lts/detail/exploration_strategy.h"

namespace mcrl2 {

namespace lts {

class lps2lts_algorithm
{
  private:
    typedef lps::next_state_generator next_state_generator;

  private:
    lts_generation_options m_options;
    next_state_generator* m_generator = nullptr;
    next_state_generator::summand_subset* m_main_subset = nullptr;

    atermpp::indexed_set<lps::state> m_state_numbers;

    lts_lts_t m_output_lts;
    atermpp::indexed_set<process::action_list> m_action_label_numbers; 
    std::ofstream m_aut_file;

    std::size_t m_number_of_states = 0;
    std::size_t m_number_of_transitions = 0;
    size_t m_initial_state_number = 0;
    std::size_t m_level = 0;

    volatile bool m_must_abort = false;

  public:
    lps2lts_algorithm()
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
    bool is_nondeterministic(std::vector<next_state_generator::transition>& transitions, next_state_generator::transition& nondeterminist_transition);
    std::pair<std::size_t, bool> add_target_state(const lps::state& source_state, const lps::state& target_state);
    bool add_transition(const lps::state& source_state, const next_state_generator::transition& transition);
    void generate_transitions(const lps::state& state,
                              std::vector<next_state_generator::transition>& transitions,
                              next_state_generator::enumerator_queue& enumeration_queue
    );
    void generate_lts_breadth_first();
};

} // namespace lps

} // namespace mcrl2

#endif // MCRL2_LTS_DETAIL_EXPLORATION_H
