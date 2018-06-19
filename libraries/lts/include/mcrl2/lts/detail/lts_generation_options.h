// Author(s): Jeroen Keiren
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/lts/detail/lts_generation_options.h
/// \brief Options used during state space generation.

#ifndef MCRL2_LTS_DETAIL_LTS_GENERATION_OPTIONS_H
#define MCRL2_LTS_DETAIL_LTS_GENERATION_OPTIONS_H

#include "mcrl2/data/rewrite_strategy.h"
#include "mcrl2/lts/lts_io.h"
#include "mcrl2/lts/detail/exploration_strategy.h"
#include "mcrl2/process/action_parse.h"

namespace mcrl2
{
namespace lts
{


class lts_generation_options
{
  private:
    static const std::size_t default_max_states=ULONG_MAX;
    static const std::size_t default_bithashsize=209715200ULL; // ~25 MB
    static const std::size_t default_init_tsize=10000UL;

  public:
    static const std::size_t default_max_traces=ULONG_MAX;

    mcrl2::lps::stochastic_specification specification;
    bool usedummies = true;
    bool removeunused = true;

    mcrl2::data::rewriter::strategy strat = mcrl2::data::jitty;
    exploration_strategy expl_strat = es_breadth;
    std::string priority_action;
    std::size_t todo_max = (std::numeric_limits< std::size_t >::max)();
    std::size_t max_states = default_max_states;
    std::size_t initial_table_size = default_init_tsize;
    bool suppress_progress_messages = false;

    bool bithashing = false;
    std::size_t bithashsize = default_bithashsize;

    mcrl2::lts::lts_type outformat = mcrl2::lts::lts_none;
    bool outinfo = true;
    std::string lts;

    bool trace = false;
    std::size_t max_traces = default_max_traces;
    std::string trace_prefix;
    bool save_error_trace = false;
    bool detect_deadlock = false;
    bool detect_nondeterminism = false;
    bool detect_divergence = false;
    bool detect_action = false;
    std::set < mcrl2::core::identifier_string > trace_actions;
    std::set < std::string > trace_multiaction_strings;
    std::set < mcrl2::lps::multi_action > trace_multiactions;

    bool use_enumeration_caching = false;
    bool use_summand_pruning = false;
    std::set< mcrl2::core::identifier_string > actions_internal_for_divergencies;

    /// \brief Constructor
    lts_generation_options() = default;

    /// \brief Copy assignment operator.
    lts_generation_options& operator=(const lts_generation_options& )=default;

    void validate_actions()
    {
      for (const std::string& s: trace_multiaction_strings)
      {
        try
        {
          trace_multiactions.insert(mcrl2::lps::parse_multi_action(s, specification.action_labels(), specification.data()));
        }
        catch (mcrl2::runtime_error& e)
        {
          throw mcrl2::runtime_error(std::string("Multi-action ") + s + " does not exist: " + e.what());
        }
        mCRL2log(log::verbose) << "Checking for action \"" << s << "\"\n";
      }
      if (detect_action)
      {
        for (const mcrl2::core::identifier_string& ta: trace_actions)
        {
          bool found = (std::string(ta) == "tau");
          for(const process::action_label& al: specification.action_labels())
          {
            if (al.name() == ta)
            {
              found=true;
              break;
            }
          }
          if (!found)
          {
            throw mcrl2::runtime_error(std::string("Action label ") + core::pp(ta) + " is not declared.");
          }
          else
          {
            mCRL2log(log::verbose) << "Checking for action " << ta << "\n";
          }
        }
      }
      for (const mcrl2::core::identifier_string& ta: actions_internal_for_divergencies)
      {
        mcrl2::process::action_label_list::iterator it = specification.action_labels().begin();
        bool found = (std::string(ta) == "tau");
        while (!found && it != specification.action_labels().end())
        {
          found = (it++->name() == ta);
        }
        if (!found)
        {
          throw mcrl2::runtime_error(std::string("Action label ") + core::pp(ta) + " is not declared.");
        }
      }
    }

};

} // namespace lts
} // namespace mcrl2

#endif // MCRL2_LTS_DETAIL_LTS_GENERATION_OPTIONS_H
