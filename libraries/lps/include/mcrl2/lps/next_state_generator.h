// Author(s): Ruud Koolen
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file next_state_generator.h

#ifndef MCRL2_LPS_NEXT_STATE_GENERATOR_H
#define MCRL2_LPS_NEXT_STATE_GENERATOR_H

#include <boost/iterator/iterator_facade.hpp>
#include <forward_list>
#include <iterator>
#include <string>
#include <vector>

#include "mcrl2/atermpp/detail/shared_subset.h"
#include "mcrl2/data/enumerator.h"
#include "mcrl2/lps/probabilistic_data_expression.h"
#include "mcrl2/lps/state.h"
#include "mcrl2/lps/state_probability_pair.h"
#include "mcrl2/lps/stochastic_specification.h"

namespace mcrl2 {

namespace lps {

class next_state_generator
{
  public:
    typedef atermpp::term_appl<data::data_expression> enumeration_cache_key;
    typedef std::list<data::data_expression_list> enumeration_cache_value;

    typedef data::enumerator_algorithm_with_iterator<> enumerator;
    typedef std::deque<data::enumerator_list_element_with_substitution<>> enumerator_queue;

    typedef data::rewriter::substitution_type rewriter_substitution;

  protected:
    struct next_state_action_label
    {
      process::action_label label;
      data::data_expression_vector arguments;
    };

    struct next_state_summand
    {
      stochastic_action_summand* summand;
      data::variable_list variables;
      data::data_expression condition;
      stochastic_distribution distribution;
      data::data_expression_vector result_state;
      std::vector<next_state_action_label> action_label;
      data::data_expression time;

      // enumeration caching
      std::vector<std::size_t> condition_parameters;
      atermpp::function_symbol condition_arguments_function;
      std::map<enumeration_cache_key, enumeration_cache_value> enumeration_cache;
    };

    struct pruning_tree_node
    {
      atermpp::detail::shared_subset<next_state_summand> summand_subset;
      std::map<data::data_expression, pruning_tree_node> children;
    };

  public:
    class iterator;

    class no_pruning_summand_subset
    {
        friend class next_state_generator;
        friend class next_state_generator::iterator;

      private:
        std::vector<std::size_t> m_summands;

      public:
        /// \brief Trivial constructor. Constructs an invalid command subset.
        no_pruning_summand_subset() = default;

        /// \brief Constructs the full summand subset for the given generator.
        no_pruning_summand_subset(next_state_generator *generator)
        {
          for (std::size_t i = 0; i < generator->m_summands.size(); i++)
          {
            m_summands.push_back(i);
          }
        }

        /// \brief Constructs the summand subset containing the given commands.
        no_pruning_summand_subset(next_state_generator* generator, const stochastic_action_summand_vector& summands)
        {
          std::set<stochastic_action_summand> summand_set(summands.begin(), summands.end());
          for (std::size_t i = 0; i < generator->m_summands.size(); i++)
          {
            if (summand_set.find(*generator->m_summands[i].summand) != summand_set.end())
            {
              m_summands.push_back(i);
            }
          }
        }
    };

    class pruning_summand_subset
    {
        friend class next_state_generator;
        friend class next_state_generator::iterator;

      private:
        next_state_generator *m_generator;
        std::vector<std::size_t> m_summands;

        pruning_tree_node m_pruning_tree;
        std::vector<std::size_t> m_pruning_parameters;
        rewriter_substitution m_pruning_substitution;

      public:
        /// \brief Trivial constructor. Constructs an invalid command subset.
        pruning_summand_subset() = default;

        /// \brief Constructs the full summand subset for the given generator.
        pruning_summand_subset(next_state_generator* generator, bool use_summand_pruning)
                : m_generator(generator)
        {
          m_pruning_tree.summand_subset = atermpp::detail::shared_subset<next_state_summand>(generator->m_summands);
          build_pruning_parameters(generator->m_specification.process().action_summands());
        }

        /// \brief Constructs the summand subset containing the given commands.
        pruning_summand_subset(next_state_generator* generator, const stochastic_action_summand_vector& summands)
                : m_generator(generator)
        {
          std::set<stochastic_action_summand> summand_set(summands.begin(), summands.end());

          atermpp::detail::shared_subset<next_state_summand> full_set(generator->m_summands);
          m_pruning_tree.summand_subset = atermpp::detail::shared_subset<next_state_summand>(full_set, std::bind(
                  next_state_generator::summand_subset::summand_set_contains, summand_set, std::placeholders::_1));
          build_pruning_parameters(summands);
        }

      private:
        struct parameter_score
        {
          std::size_t parameter_id;
          float score;
          parameter_score() = default;
          parameter_score(std::size_t id, float score_)
                  : parameter_id(id), score(score_)
          { }

          bool operator<(const parameter_score& other) const
          {
            return score > other.score;
          }
        };

        float condition_selectivity(const data::data_expression& e, const data::variable& v)
        {
          if (data::sort_bool::is_and_application(e))
          {
            return condition_selectivity(data::binary_left(atermpp::down_cast<data::application>(e)), v)
                   + condition_selectivity(data::binary_right(atermpp::down_cast<data::application>(e)), v);
          }
          else if (data::sort_bool::is_or_application(e))
          {
            float sum = 0;
            std::size_t count = 0;
            std::list<data::data_expression> terms;
            terms.push_back(e);
            while (!terms.empty())
            {
              data::data_expression expression = terms.front();
              terms.pop_front();
              if (data::sort_bool::is_or_application(expression))
              {
                terms.push_back(data::binary_left(atermpp::down_cast<data::application>(e)));
                terms.push_back(data::binary_right(atermpp::down_cast<data::application>(e)));
              }
              else
              {
                sum += condition_selectivity(expression, v);
                count++;
              }
            }
            return sum / count;
          }
          else if (is_equal_to_application(e))
          {
            const data::data_expression& left = data::binary_left(atermpp::down_cast<data::application>(e));
            const data::data_expression& right = data::binary_right(atermpp::down_cast<data::application>(e));

            if (data::is_variable(left) && data::variable(left) == v)
            {
              return 1;
            }
            else if (data::is_variable(right) && data::variable(right) == v)
            {
              return 1;
            }
            else
            {
              return 0;
            }
          }
          else
          {
            return 0;
          }
        }

        bool summand_set_contains(const std::set<stochastic_action_summand>& summand_set, const next_state_summand& summand)
        {
          return summand_set.count(*summand.summand) > 0;
        }

        void build_pruning_parameters(const stochastic_action_summand_vector& summands)
        {
          std::vector<parameter_score> parameters;

          for (std::size_t i = 0; i < m_generator->m_process_parameters.size(); i++)
          {
            parameters.emplace_back(i, 0);
            for (const auto& summand : summands)
            {
              parameters[i].score += condition_selectivity(summand.condition(), m_generator->m_process_parameters[i]);
            }
          }

          std::sort(parameters.begin(), parameters.end());

          for (std::size_t i = 0; i < m_generator->m_process_parameters.size(); i++)
          {
            if (parameters[i].score > 0)
            {
              m_pruning_parameters.push_back(parameters[i].parameter_id);
              mCRL2log(log::verbose) << "using pruning parameter "
                                     << m_generator->m_process_parameters[parameters[i].parameter_id].name() << std::endl;
            }
          }
        }

        atermpp::detail::shared_subset<next_state_summand>::iterator begin(const lps::state& state)
        {
          for (std::size_t m_pruning_parameter: m_pruning_parameters)
          {
            const data::variable& v = m_generator->m_process_parameters[m_pruning_parameter];
            m_pruning_substitution[v] = v;
          }

          pruning_tree_node* node = &m_pruning_tree;
          for (std::size_t parameter: m_pruning_parameters)
          {
            const data::data_expression& argument = state.element_at(parameter, m_generator->m_process_parameters.size());
            m_pruning_substitution[m_generator->m_process_parameters[parameter]] = argument;
            auto position = node->children.find(argument);
            if (position == node->children.end())
            {
              pruning_tree_node child;
              child.summand_subset = atermpp::detail::shared_subset<next_state_summand>(node->summand_subset,
                      [&](const next_state_summand& summand) { return m_generator->m_rewriter(summand.condition, m_pruning_substitution) != data::sort_bool::false_(); }
                      );
              node->children[argument] = child;
              node = &node->children[argument];
            }
            else
            {
              node = &position->second;
            }
          }

          return node->summand_subset.begin();
        }
    };

    class summand_subset
    {
      friend class next_state_generator;
      friend class next_state_generator::iterator;

      public:
        /// \brief Trivial constructor. Constructs an invalid command subset.
        summand_subset() = default;

        /// \brief Constructs the full summand subset for the given generator.
        summand_subset(next_state_generator *generator, bool use_summand_pruning);

        /// \brief Constructs the summand subset containing the given commands.
        summand_subset(next_state_generator* generator, const stochastic_action_summand_vector& summands, bool use_summand_pruning);

      private:
        next_state_generator *m_generator;
        bool m_use_summand_pruning;

        std::vector<std::size_t> m_summands;

        pruning_tree_node m_pruning_tree;
        std::vector<std::size_t> m_pruning_parameters;
        rewriter_substitution m_pruning_substitution;

        static bool summand_set_contains(const std::set<stochastic_action_summand>& summand_set, const next_state_summand& summand);
        void build_pruning_parameters(const stochastic_action_summand_vector& summands);
        bool is_not_false(const next_state_summand& summand);
        atermpp::detail::shared_subset<next_state_summand>::iterator begin(const lps::state& state);
    };

    typedef mcrl2::lps::state_probability_pair<lps::state, lps::probabilistic_data_expression> state_probability_pair;

    struct transition
    {
      typedef std::forward_list<state_probability_pair> state_probability_list;

      lps::multi_action action;
      lps::state target_state;
      std::size_t summand_index;

      // The following list contains all but one target states with their probabity.
      // m_target_state is the other state, with the residual probability, such
      // that all probabilities add up to 1.
      state_probability_list m_other_target_states;
    };

    class iterator: public boost::iterator_facade<iterator, const transition, boost::forward_traversal_tag>
    {
      protected:
        transition m_transition;
        next_state_generator* m_generator = nullptr;
        lps::state m_state;
        rewriter_substitution* m_substitution;

        bool m_single_summand;
        std::size_t m_single_summand_index;
        bool m_use_summand_pruning;
        std::vector<std::size_t>::iterator m_summand_iterator;
        std::vector<std::size_t>::iterator m_summand_iterator_end;
        atermpp::detail::shared_subset<next_state_summand>::iterator m_summand_subset_iterator;
        next_state_summand *m_summand;

        bool m_cached;
        enumeration_cache_value::iterator m_enumeration_cache_iterator;
        enumeration_cache_value::iterator m_enumeration_cache_end;
        enumerator::iterator m_enumeration_iterator;
        bool m_caching;
        enumeration_cache_key m_enumeration_cache_key;
        enumeration_cache_value m_enumeration_log;

        enumerator_queue* m_enumeration_queue;

        /// \brief Enumerate <variables, phi> with substitution sigma.
        void enumerate(const data::variable_list& variables, const data::data_expression& phi, data::mutable_indexed_substitution<>& sigma)
        {
          m_enumeration_queue->clear();
          m_enumeration_queue->push_back(data::enumerator_list_element_with_substitution<>(variables, phi));
          try
          {
            m_enumeration_iterator = m_generator->m_enumerator.begin(sigma, *m_enumeration_queue);
          }
          catch (mcrl2::runtime_error &e)
          {
            throw mcrl2::runtime_error(std::string(e.what()) + "\nProblem occurred when enumerating variables " + data::pp(variables) + " in " + data::pp(phi));
          }
        }


      public:
        iterator() = default;

        iterator(next_state_generator* generator, const lps::state& state, rewriter_substitution* substitution, summand_subset& summand_subset, enumerator_queue* enumeration_queue);

        iterator(next_state_generator* generator, const lps::state& state, rewriter_substitution* substitution, std::size_t summand_index, enumerator_queue* enumeration_queue);

        explicit operator bool() const
        {
          return m_generator != nullptr;
        }

      private:
        friend class boost::iterator_core_access;

        bool equal(const iterator& other) const
        {
          return (!(bool)*this && !(bool)other) || (this == &other);
        }

        const transition& dereference() const
        {
          return m_transition;
        }

        void increment();
    };

  protected:
    stochastic_specification m_specification;
    data::rewriter m_rewriter;
    rewriter_substitution m_substitution;
    data::enumerator_identifier_generator m_id_generator;
    enumerator m_enumerator;

    bool m_use_enumeration_caching;

    data::variable_vector m_process_parameters;
    std::vector<next_state_summand> m_summands;
    transition::state_probability_list m_initial_states;

    summand_subset m_all_summands;

  public:
    /// \brief Constructor
    /// \param spec The process specification
    /// \param rewriter The rewriter used
    /// \param use_enumeration_caching Cache intermediate enumeration results
    /// \param use_summand_pruning Preprocess summands using pruning strategy.
    next_state_generator(const stochastic_specification& spec,
                         const data::rewriter& rewriter,
                         bool use_enumeration_caching = false,
                         bool use_summand_pruning = false);

    /// \brief Returns an iterator for generating the successors of the given state.
    iterator begin(const state& state, enumerator_queue* enumeration_queue)
    {
      return iterator(this, state, &m_substitution, m_all_summands, enumeration_queue);
    }

    /// \brief Returns an iterator for generating the successors of the given state.
    iterator begin(const state& state, summand_subset& summand_subset, enumerator_queue* enumeration_queue)
    {
      return iterator(this, state, &m_substitution, summand_subset, enumeration_queue);
    }

    /// \brief Returns an iterator for generating the successors of the given state.
    /// Only the successors with respect to the summand with the given index are generated.
    iterator begin(const state& state, std::size_t summand_index, enumerator_queue* enumeration_queue)
    {
      return iterator(this, state, &m_substitution, summand_index, enumeration_queue);
    }

    /// \brief Returns an iterator pointing to the end of a next state list.
    iterator end()
    {
      return iterator();
    }

    /// \brief Gets the initial state.
    const transition::state_probability_list& initial_states() const
    {
      return m_initial_states;
    }

    /// \brief Returns the rewriter associated with this generator.
    data::rewriter& rewriter()
    {
      return m_rewriter;
    }

    /// \brief Returns a reference to the summand subset containing all summands.
    summand_subset& full_subset()
    {
      return m_all_summands;
    }

    // Calculate the set of states with associated probabilities from a symbolic state
    // and an associated stochastic distribution for the free variables in that state.
    // The result is a list of closed states with associated probabilities.
    const transition::state_probability_list calculate_distribution(
                         const stochastic_distribution& dist,
                         const data::data_expression_vector& state_args,
                         rewriter_substitution& sigma);
};

} // namespace lps

} // namespace mcrl2

#endif // MCRL2_LPS_NEXT_STATE_GENERATOR_H
