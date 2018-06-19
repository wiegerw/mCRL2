// Author(s): Ruud Koolen
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <iomanip>
#include <ctime>

#include "mcrl2/utilities/logger.h"
#include "mcrl2/lps/resolve_name_clashes.h"
#include "mcrl2/lps/detail/instantiate_global_variables.h"
#include "mcrl2/lps/probabilistic_data_expression.h"
#include "mcrl2/lps/one_point_rule_rewrite.h"
#include "mcrl2/lts/detail/exploration.h"
#include "mcrl2/lts/detail/counter_example.h"
#include "mcrl2/lts/lts_io.h"

using namespace mcrl2;
using namespace mcrl2::log;
using namespace mcrl2::lps;
using namespace mcrl2::lts;

probabilistic_state<std::size_t, probabilistic_data_expression>
lps2lts_algorithm::transform_initial_probabilistic_state_list
        (const next_state_generator::transition::state_probability_list& initial_states)
{
  assert(!initial_states.empty());
  if (++initial_states.begin() == initial_states.end()) // Means initial_states.size()==1
  {
    std::size_t state_number = m_state_numbers.put(initial_states.front().state()).first;
    return probabilistic_state<std::size_t, probabilistic_data_expression>(state_number);
  }
  std::vector<state_probability_pair<std::size_t, lps::probabilistic_data_expression>> result;
  for (const auto& initial_state: initial_states)
  {
    std::size_t state_number = m_state_numbers.put(initial_state.state()).first;
    result.emplace_back(state_number, initial_state.probability());
  }
  return probabilistic_state<std::size_t, probabilistic_data_expression>(result.begin(), result.end());
}

probabilistic_state<std::size_t, probabilistic_data_expression>
lps2lts_algorithm::create_a_probabilistic_state_from_target_distribution(
        const std::size_t base_state_number,
        const next_state_generator::transition::state_probability_list& other_probabilities,
        const lps::state& source_state)
{
  if (other_probabilities.empty())
  {
    return probabilistic_state<std::size_t, probabilistic_data_expression>(base_state_number);
  }

  std::vector<state_probability_pair<std::size_t, probabilistic_data_expression> > result;

  probabilistic_data_expression residual_probability = probabilistic_data_expression::one();

  for (const auto& probability: other_probabilities)
  {
    if (is_application(probability.probability()) &&
        atermpp::down_cast<data::application>(probability.probability()).head().size() != 3)
    {
      throw mcrl2::runtime_error(
              "The probability " + data::pp(probability.probability()) + " is not a proper rational number.");
    }
    residual_probability = data::sort_real::minus(residual_probability, probability.probability());
    const std::pair<std::size_t, bool> probability_destination_state_number = add_target_state(source_state,
                                                                                               probability.state());
    result.emplace_back(probability_destination_state_number.first, probability.probability());
  }

  residual_probability = (m_generator->rewriter())(residual_probability);
  result.emplace_back(base_state_number, residual_probability);
  return probabilistic_state<std::size_t, probabilistic_data_expression>(result.begin(), result.end());
}

bool is_hidden_summand(const mcrl2::process::action_list& l,
                       const std::set<core::identifier_string>& internal_action_labels)
{
  // Note that if l is empty, true is returned, as desired.
  for (const mcrl2::process::action& a: l)
  {
    if (internal_action_labels.count(a.label().name()) ==
        0) // Not found, s has a visible action among its multi-actions.
    {
      return false;
    }
  }
  return true;
}

bool lps2lts_algorithm::initialise_lts_generation(lts_generation_options* options)
{
  m_options = *options;

  assert(!(false && m_options.outformat != lts_aut && m_options.outformat != lts_none));

  m_state_numbers = atermpp::indexed_set<lps::state>(m_options.initial_table_size, 50);

  m_num_states = 0;
  m_num_transitions = 0;
  m_level = 1;
  m_traces_saved = 0;

  m_maintain_traces = m_options.trace || m_options.save_error_trace;

  lps::stochastic_specification specification(m_options.specification);
  resolve_summand_variable_name_clashes(specification);

  if (m_options.outformat == lts_aut)
  {
    mCRL2log(verbose) << "writing state space in AUT format to '" << m_options.lts << "'." << std::endl;
    m_aut_file.open(m_options.lts.c_str());
    if (!m_aut_file.is_open())
    {
      mCRL2log(error) << "cannot open '" << m_options.lts << "' for writing" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  else if (m_options.outformat == lts_none)
  {
    mCRL2log(verbose) << "not saving state space." << std::endl;
  }
  else
  {
    mCRL2log(verbose) << "writing state space in " << mcrl2::lts::detail::string_for_type(m_options.outformat)
                      << " format to '" << m_options.lts << "'." << std::endl;
    m_output_lts.set_data(specification.data());
    m_output_lts.set_process_parameters(specification.process().process_parameters());
    m_output_lts.set_action_label_declarations(specification.action_labels());
  }


  if (m_options.usedummies)
  {
    lps::detail::instantiate_global_variables(specification);
  }

  data::rewriter rewriter;
  if (m_options.removeunused)
  {
    mCRL2log(verbose) << "removing unused parts of the data specification." << std::endl;
    std::set<data::function_symbol> extra_function_symbols = lps::find_function_symbols(specification);
    extra_function_symbols.insert(data::sort_real::minus(data::sort_real::real_(), data::sort_real::real_()));

    rewriter = data::rewriter(specification.data(),
                              data::used_data_equation_selector(specification.data(), extra_function_symbols,
                                                                specification.global_variables()), m_options.strat);
  }
  else
  {
    rewriter = data::rewriter(specification.data(), m_options.strat);
  }

  // Apply the one point rewriter to the linear process specification. 
  // This simplifies expressions of the shape exists x:X . (x == e) && phi to phi[x:=e], enabling
  // more lps's to generate lts's. The overhead of this rewriter is limited.
  one_point_rule_rewrite(specification);

  stochastic_action_summand_vector prioritised_summands;
  stochastic_action_summand_vector nonprioritised_summands;

  stochastic_action_summand_vector tau_summands;

  bool compute_actions =
          m_options.outformat != lts_none || !m_options.trace_multiactions.empty() || m_maintain_traces;
  if (!compute_actions)
  {
    for (auto& summand: specification.process().action_summands())
    {
      summand.multi_action().actions() = process::action_list();
    }
  }
  m_generator = new next_state_generator(specification, rewriter, m_options.use_enumeration_caching, false);

  m_main_subset = &m_generator->full_subset();

  if (m_options.detect_deadlock)
  {
    mCRL2log(verbose) << "Detect deadlocks.\n";
  }

  if (m_options.detect_nondeterminism)
  {
    mCRL2log(verbose) << "Detect nondeterministic states.\n";
  }

  return true;
}

bool lps2lts_algorithm::generate_lts()
{
  // First generate a vector of initial states from the initial distribution.
  m_initial_states = m_generator->initial_states();
  assert(!m_initial_states.empty());

  // Count the number of states.
  m_num_states = 0;
  for (auto i = m_initial_states.begin(); i != m_initial_states.end(); ++i)
  {
    m_num_states++;
  }

  // Store the initial states in the indexed set.
  for (auto i = m_initial_states.begin(); i != m_initial_states.end(); ++i)
  {
    if (m_state_numbers.put(i->state()).second && m_options.outformat != lts_aut) // The state is new.
    {
      m_output_lts.add_state(state_label_lts(i->state()));
    }
  }

  if (m_options.outformat == lts_aut)
  {
    m_aut_file << "des(";
    print_target_distribution_in_aut_format(m_initial_states, lps::state());
    // HACK: this line will be overwritten once generation is finished.
    m_aut_file << ",0,0)                                          " << std::endl;
  }
  else if (m_options.outformat != lts_none)
  {
    m_output_lts.set_initial_probabilistic_state(transform_initial_probabilistic_state_list(m_initial_states));
  }

  mCRL2log(verbose) << "generating state space with '" << es_breadth << "' strategy...\n";

  if (m_options.max_states == 0)
  {
    return true;
  }

  if (m_options.todo_max == std::string::npos)
  {
    generate_lts_breadth_todo_max_is_npos();
  }
  else
  {
    generate_lts_breadth_todo_max_is_not_npos(m_initial_states);
  }

  mCRL2log(verbose) << "done with state space generation ("
                    << m_level - 1 << " level" << ((m_level == 2) ? "" : "s") << ", "
                    << m_num_states << " state" << ((m_num_states == 1) ? "" : "s")
                    << " and " << m_num_transitions << " transition" << ((m_num_transitions == 1) ? "" : "s") << ")"
                    << std::endl;

  return true;
}

bool lps2lts_algorithm::finalise_lts_generation()
{
  if (m_options.outformat == lts_aut)
  {
    m_aut_file.flush();
    m_aut_file.seekp(0);
    m_aut_file << "des (";
    print_target_distribution_in_aut_format(m_initial_states, lps::state());
    m_aut_file << "," << m_num_transitions << "," << m_num_states << ")";
    m_aut_file.close();
  }
  else if (m_options.outformat != lts_none)
  {
    if (!m_options.outinfo)
    {
      m_output_lts.clear_state_labels();
    }

    switch (m_options.outformat)
    {
      case lts_lts:
      {
        m_output_lts.save(m_options.lts);
        break;
      }
      case lts_fsm:
      {
        probabilistic_lts_fsm_t fsm;
        detail::lts_convert(m_output_lts, fsm);
        fsm.save(m_options.lts);
        break;
      }
      case lts_dot:
      {
        probabilistic_lts_dot_t dot;
        detail::lts_convert(m_output_lts, dot);
        dot.save(m_options.lts);
        break;
      }
      default:
        assert(0);
    }
  }

  return true;
}

void lps2lts_algorithm::construct_trace(const lps::state& state1, mcrl2::trace::Trace& trace)
{
  lps::state state = state1;
  std::deque<lps::state> states;
  std::map<lps::state, lps::state>::iterator source;
  while ((source = m_backpointers.find(state)) != m_backpointers.end())
  {
    states.push_front(state);
    state = source->second;
  }

  trace.setState(state);
  next_state_generator::enumerator_queue enumeration_queue;
  for (auto& s: states)
  {
    for (auto j = m_generator->begin(state, &enumeration_queue); j != m_generator->end(); j++)
    {
      lps::state destination = j->target_state;
      if (destination == s)
      {
        trace.addAction(j->action);
        break;
      }
    }
    enumeration_queue.clear();
    state = s;
    trace.setState(state);
  }

}

// Contruct a trace to state1 and store in in filename.
bool lps2lts_algorithm::save_trace(const lps::state& state1, const std::string& filename)
{
  mcrl2::trace::Trace trace;
  lps2lts_algorithm::construct_trace(state1, trace);
  m_traces_saved++;

  try
  {
    trace.save(filename);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

// Contruct a trace to state1, then add transition to it and store in in filename.
bool lps2lts_algorithm::save_trace(const lps::state& state1,
                                   const next_state_generator::transition& transition,
                                   const std::string& filename)
{
  mcrl2::trace::Trace trace;
  lps2lts_algorithm::construct_trace(state1, trace);
  trace.addAction(transition.action);
  trace.setState(transition.target_state);
  m_traces_saved++;

  try
  {
    trace.save(filename);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

void lps2lts_algorithm::save_actions(const lps::state& state, const next_state_generator::transition& transition)
{
  auto state_number = m_state_numbers.index(state);
  mCRL2log(info) << "Detected action '" << pp(transition.action) << "' (state index " << state_number << ")";
  if (m_options.trace && m_traces_saved < m_options.max_traces)
  {
    std::string filename = m_options.trace_prefix + "_act_" + std::to_string(m_traces_saved);
    if (m_options.trace_multiactions.find(transition.action) != m_options.trace_multiactions.end())
    {
      filename = filename + "_" + pp(transition.action);
    }
    for (const process::action& a: transition.action.actions())
    {
      if (m_options.trace_actions.count(a.label().name()) > 0)
      {
        filename = filename + "_" + core::pp(a.label().name());
      }
    }
    filename = filename + ".trc";
    if (save_trace(state, transition, filename))
      mCRL2log(info) << " and saved to '" << filename << "'";
    else mCRL2log(info) << " but it could not saved to '" << filename << "'";
  }
  mCRL2log(info) << std::endl;
}

void lps2lts_algorithm::save_nondeterministic_state(const lps::state& state,
                                                    const next_state_generator::transition& nondeterminist_transition)
{
  auto state_number = m_state_numbers.index(state);
  if (m_options.trace && m_traces_saved < m_options.max_traces)
  {
    std::string filename = m_options.trace_prefix + "_nondeterministic_" + std::to_string(m_traces_saved) + ".trc";
    if (save_trace(state, nondeterminist_transition, filename))
    {
      mCRL2log(info) << "Nondeterministic state found and saved to '" << filename
                     << "' (state index: " << state_number << ").\n";
    }
    else
    {
      mCRL2log(info) << "Nondeterministic state found, but its trace could not be saved to '" << filename
                     << "' (state index: " << state_number << ").\n";
    }
  }
  else
  {
    mCRL2log(info) << "Nondeterministic state found (state index: " << state_number << ").\n";
  }
}

void lps2lts_algorithm::save_deadlock(const lps::state& state)
{
  auto state_number = m_state_numbers.index(state);
  if (m_options.trace && m_traces_saved < m_options.max_traces)
  {
    std::string filename = m_options.trace_prefix + "_dlk_" + std::to_string(m_traces_saved) + ".trc";
    if (save_trace(state, filename))
    {
      mCRL2log(info) << "deadlock-detect: deadlock found and saved to '" << filename
                     << "' (state index: " << state_number << ").\n";
    }
    else
    {
      mCRL2log(info) << "deadlock-detect: deadlock found, but its trace could not be saved to '" << filename
                     << "' (state index: " << state_number << ").\n";
    }
  }
  else
  {
    mCRL2log(info) << "deadlock-detect: deadlock found (state index: " << state_number << ").\n";
  }
}

void lps2lts_algorithm::save_error(const lps::state& state)
{
  if (m_options.save_error_trace)
  {
    std::string filename = m_options.trace_prefix + "_error.trc";
    if (save_trace(state, filename))
    {
      mCRL2log(verbose) << "saved trace to error in '" << filename << "'.\n";
    }
    else
    {
      mCRL2log(verbose) << "trace to error could not be saved in '" << filename << "'.\n";
    }
  }
}

// Add the target state to the transition system, and if necessary store it to be investigated later.
// Return the number of the target state.
std::pair<std::size_t, bool>
lps2lts_algorithm::add_target_state(const lps::state& source_state, const lps::state& target_state)
{
  std::pair<std::size_t, bool> destination_state_number;
  destination_state_number = m_state_numbers.put(target_state);
  if (destination_state_number.second) // The state is new.
  {
    m_num_states++;
    if (m_maintain_traces)
    {
      assert(m_backpointers.count(target_state) == 0);
      m_backpointers[target_state] = source_state;
    }

    if (m_options.outformat != lts_none && m_options.outformat != lts_aut)
    {
      m_output_lts.add_state(state_label_lts(target_state));
    }
  }
  return destination_state_number;
}

void lps2lts_algorithm::print_target_distribution_in_aut_format(
        const lps::next_state_generator::transition::state_probability_list& state_probability_list,
        const std::size_t last_state_number,
        const lps::state& source_state)
{
  for (const auto& state_probability: state_probability_list)
  {
    if (m_options.outformat == lts_aut)
    {
      const lps::state& probability_destination = state_probability.state();
      const std::pair<std::size_t, bool> probability_destination_state_number = add_target_state(source_state, probability_destination);
      if (is_application(state_probability.probability()) &&
          atermpp::down_cast<data::application>(state_probability.probability()).head().size() != 3)
      {
        if (m_options.outformat == lts_aut)
        {
          m_aut_file.flush();
        }
        throw mcrl2::runtime_error(
                "The probability " + data::pp(state_probability.probability()) + " is not a proper rational number.");
      }
      const auto& prob = atermpp::down_cast<data::application>(state_probability.probability());
      if (prob.head() != data::sort_real::creal())
      {
        throw mcrl2::runtime_error(
                "Probability is not a closed expression with a proper enumerator and denominator: " + pp(
                        state_probability.probability()) + ".");
      }
      m_aut_file << probability_destination_state_number.first << " " << (prob[0]) << "/"
                 << (prob[1]) << " ";
    }
  }
  m_aut_file << last_state_number;
}

void lps2lts_algorithm::print_target_distribution_in_aut_format(
        const lps::next_state_generator::transition::state_probability_list& state_probability_list,
        const lps::state& source_state)
{
  assert(!state_probability_list.empty());
  const std::pair<std::size_t, bool> a_destination_state_number = add_target_state(source_state,
                                                                                   state_probability_list.front().state());
  lps::next_state_generator::transition::state_probability_list temporary_list = state_probability_list;
  temporary_list.pop_front();
  print_target_distribution_in_aut_format(temporary_list, a_destination_state_number.first, source_state);
}


bool
lps2lts_algorithm::add_transition(const lps::state& source_state, const next_state_generator::transition& transition)
{

  std::size_t source_state_number;
  source_state_number = m_state_numbers[source_state];

  const lps::state& destination = transition.target_state;
  const std::pair<std::size_t, bool> destination_state_number = add_target_state(source_state, destination);

  if (m_options.outformat == lts_aut || m_options.outformat == lts_none)
  {

    if (m_options.outformat == lts_aut)
    {
      m_aut_file << "(" << source_state_number << ",\"" << lps::pp(transition.action) << "\",";
    }

    print_target_distribution_in_aut_format(transition.m_other_target_states, destination_state_number.first,
                                            source_state);

    // Close transition.
    if (m_options.outformat == lts_aut)
    {
      m_aut_file << ")\n"; // Intentionally do not use std::endl to avoid flushing.
    }
  }
  else
  {
    std::pair<std::size_t, bool> action_label_number = m_action_label_numbers.put(transition.action.actions());
    if (action_label_number.second)
    {
      assert(!transition.action.actions().empty());
      std::size_t action_number = m_output_lts.add_action(action_label_lts(transition.action));
      assert(action_number == action_label_number.first);
      static_cast <void>(action_number); // Avoid a warning when compiling in non debug mode.
    }
    std::size_t number_of_a_new_probabilistic_state = m_output_lts.add_probabilistic_state(
            create_a_probabilistic_state_from_target_distribution(
                    destination_state_number.first,
                    transition.m_other_target_states,
                    source_state)); // Add a new probabilistic state.
    m_output_lts.add_transition(mcrl2::lts::transition(source_state_number, action_label_number.first,
                                                       number_of_a_new_probabilistic_state));
  }

  m_num_transitions++;

  for (const auto& a: m_options.trace_multiactions)
  {
    if (a == transition.action)
    {
      save_actions(source_state, transition);
    }
  }

  return destination_state_number.second;
}

// The function below checks whether in the set of outgoing transitions,
// there are two transitions with the same label, going to different states.
// If this is the case, true is delivered and one nondeterministic transition 
// is returned in the variable nondeterministic_transition.
bool
lps2lts_algorithm::is_nondeterministic(std::vector<lps2lts_algorithm::next_state_generator::transition>& transitions,
                                       lps2lts_algorithm::next_state_generator::transition& nondeterministic_transition)
{
  // Below a mapping from transition labels to target states is made. 
  static std::map<lps::multi_action, lps::state> sorted_transitions; // The set is static to avoid repeated construction.
  assert(sorted_transitions.empty());
  for (const lps2lts_algorithm::next_state_generator::transition& t: transitions)
  {
    const std::map<lps::multi_action, lps::state>::const_iterator i = sorted_transitions.find(t.action);
    if (i != sorted_transitions.end())
    {
      if (i->second != t.target_state)
      {
        // Two transitions with the same label and different targets states have been found. This state is nondeterministic.
        sorted_transitions.clear();
        nondeterministic_transition = t;
        return true;
      }
    }
    else
    {
      sorted_transitions[t.action] = t.target_state;
    }
  }
  sorted_transitions.clear();
  return false;
}

void lps2lts_algorithm::get_transitions(const lps::state& state,
                                        std::vector<lps2lts_algorithm::next_state_generator::transition>& transitions,
                                        next_state_generator::enumerator_queue& enumeration_queue
)
{
  assert(transitions.empty());
  try
  {
    enumeration_queue.clear();
    next_state_generator::iterator it(m_generator->begin(state, *m_main_subset, &enumeration_queue));
    while (it)
    {
      transitions.push_back(*it++);
    }
  }
  catch (mcrl2::runtime_error& e)
  {
    mCRL2log(error) << "Error while exploring state space: " << e.what() << "\n";
    save_error(state);
    if (m_options.outformat == lts_aut)
    {
      m_aut_file.flush();
    }
    exit(EXIT_FAILURE);
  }

  if (m_options.detect_deadlock && transitions.empty())
  {
    save_deadlock(state);
  }

  if (m_options.detect_nondeterminism)
  {
    lps2lts_algorithm::next_state_generator::transition nondeterministic_transition;
    if (is_nondeterministic(transitions, nondeterministic_transition))
    {
      // save the trace to the nondeterministic state and one transition to indicate
      // which transition is nondeterministic. 
      save_nondeterministic_state(state, nondeterministic_transition);
    }
  }
}

void lps2lts_algorithm::generate_lts_breadth_todo_max_is_npos()
{
  assert(m_options.todo_max == std::string::npos);
  std::size_t current_state = 0;
  std::size_t start_level_seen = 1;
  std::size_t start_level_transitions = 0;
  std::vector<next_state_generator::transition> transitions;
  time_t last_log_time = time(nullptr) - 1, new_log_time;
  next_state_generator::enumerator_queue enumeration_queue;

  while (!m_must_abort && (current_state < m_state_numbers.size()) &&
         (current_state < m_options.max_states) && (!m_options.trace || m_traces_saved < m_options.max_traces))
  {
    lps::state state = m_state_numbers.get(current_state);
    get_transitions(state, transitions, enumeration_queue);
    for (const next_state_generator::transition& t: transitions)
    {
      add_transition(state, t);
    }
    transitions.clear();

    current_state++;
    if (current_state == start_level_seen)
    {
      mCRL2log(debug) << "Number of states at level " << m_level << " is " << m_num_states - start_level_seen << "\n";
      m_level++;
      start_level_seen = m_num_states;
      start_level_transitions = m_num_transitions;
    }

    if (!m_options.suppress_progress_messages && time(&new_log_time) > last_log_time)
    {
      last_log_time = new_log_time;
      std::size_t lvl_states = m_num_states - start_level_seen;
      std::size_t lvl_transitions = m_num_transitions - start_level_transitions;
      mCRL2log(status) << std::fixed << std::setprecision(2)
                       << m_num_states << "st, " << m_num_transitions << "tr"
                       << ", explored " << 100.0 * ((float) current_state / m_num_states)
                       << "%. Last level: " << m_level << ", " << lvl_states << "st, " << lvl_transitions
                       << "tr.\n";
    }
  }

  if (current_state == m_options.max_states)
  {
    mCRL2log(verbose) << "explored the maximum number (" << m_options.max_states << ") of states, terminating."
                      << std::endl;
  }
}

void lps2lts_algorithm::generate_lts_breadth_todo_max_is_not_npos(
        const next_state_generator::transition::state_probability_list& initial_states)
{
  assert(m_options.todo_max != std::string::npos);
  std::size_t current_state = 0;
  std::size_t start_level_seen = 1;
  std::size_t start_level_explored = 0;
  std::size_t start_level_transitions = 0;
  time_t last_log_time = time(nullptr) - 1, new_log_time;

  queue<lps::state> state_queue;
  state_queue.set_max_size(m_options.max_states < m_options.todo_max ? m_options.max_states : m_options.todo_max);
  for (const auto& initial_state: initial_states)
  {
    state_queue.add_to_queue(initial_state.state());
  }
  state_queue.swap_queues();
  std::vector<next_state_generator::transition> transitions;
  next_state_generator::enumerator_queue enumeration_queue;

  while (!m_must_abort && (state_queue.remaining() > 0) &&
         (current_state < m_options.max_states) && (!m_options.trace || m_traces_saved < m_options.max_traces))
  {
    const lps::state state = state_queue.get_from_queue();
    get_transitions(state, transitions, enumeration_queue);

    for (auto& tr: transitions)
    {
      if (add_transition(state, tr))
      {
        lps::state removed = state_queue.add_to_queue(tr.target_state);
        if (removed != lps::state())
        {
          m_num_states--;
        }
      }
    }
    transitions.clear();

    if (state_queue.remaining() == 0)
    {
      state_queue.swap_queues();
    }

    current_state++;
    if (current_state == start_level_seen)
    {
      if (!m_options.suppress_progress_messages)
      {
        mCRL2log(verbose) << "monitor: level " << m_level << " done."
                          << " (" << (current_state - start_level_explored) << " state"
                          << ((current_state - start_level_explored) == 1 ? "" : "s") << ", "
                          << (m_num_transitions - start_level_transitions) << " transition"
                          << ((m_num_transitions - start_level_transitions) == 1 ? ")\n" : "s)\n");
      }

      m_level++;
      start_level_seen = m_num_states;
      start_level_explored = current_state;
      start_level_transitions = m_num_transitions;
    }

    if (!m_options.suppress_progress_messages && time(&new_log_time) > last_log_time)
    {
      last_log_time = new_log_time;
      std::size_t lvl_states = m_num_states - start_level_seen;
      std::size_t lvl_transitions = m_num_transitions - start_level_transitions;
      mCRL2log(status) << std::fixed << std::setprecision(2)
                       << m_num_states << "st, " << m_num_transitions << "tr"
                       << ", explored " << 100.0 * ((float) current_state / m_num_states)
                       << "%. Last level: " << m_level << ", " << lvl_states << "st, " << lvl_transitions
                       << "tr.\n";
    }
  }

  if (current_state == m_options.max_states)
  {
    mCRL2log(verbose) << "explored the maximum number (" << m_options.max_states << ") of states, terminating."
                      << std::endl;
  }
}
