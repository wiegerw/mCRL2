// Author(s): Muck van Weerdenburg
// Copyright: see the accompanying file COPYING or copy at
// https://github.com/mCRL2org/mCRL2/blob/master/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file lps2lts_lts.cpp

// NAME is defined in lps2lts.h
#define AUTHOR "Muck van Weerdenburg"

#include <string>
#include <cassert>
#include <csignal>

#include "mcrl2/utilities/logger.h"

#include "mcrl2/utilities/input_output_tool.h"
#include "mcrl2/data/rewriter_tool.h"

#include "mcrl2/lps/io.h"
#include "mcrl2/process/action_parse.h"

#include "mcrl2/lts/lts_io.h"
#include "mcrl2/lts/detail/exploration.h"

#ifdef MCRL2_DISPLAY_REWRITE_STATISTICS
#include "mcrl2/data/detail/rewrite_statistics.h"
#endif

using namespace std;
using namespace mcrl2::utilities::tools;
using namespace mcrl2::utilities;
using namespace mcrl2::core;
using namespace mcrl2::lts;
using namespace mcrl2::lps;
using namespace mcrl2::log;

using mcrl2::data::tools::rewriter_tool;

static
std::list<std::string> split_actions(const std::string& s)
{
  std::size_t pcount = 0;
  std::string a;
  std::list<std::string> result;
  for (std::string::const_iterator i = s.begin(); i != s.end(); ++i)
  {
    if (*i == ',' && pcount == 0)
    {
      result.push_back(a);
      a.clear();
    }
    else
    {
      if (*i == '(') ++pcount;
      else if (*i == ')') --pcount;
      a.push_back(*i);
    }
  }
  if (!a.empty())
    result.push_back(a);
  return result;
}

typedef  rewriter_tool< input_output_tool > lps2lts_base;
class lps2lts_tool : public lps2lts_base
{
  protected:
    mcrl2::lts::lps2lts_algorithm m_lps2lts;
    lts_generation_options m_options;
    std::string m_filename;

  public:
    
    lps2lts_tool() :
      lps2lts_base("lps2lts",AUTHOR,
                   "generate an LTS from an LPS",
                   "Generate an LTS from the LPS in INFILE and save the result to OUTFILE. "
                   "If INFILE is not supplied, stdin is used. "
                   "If OUTFILE is not supplied, the LTS is not stored.\n"
                   "\n"
                   "If the 'jittyc' rewriter is used, then the MCRL2_COMPILEREWRITER environment "
                   "variable (default value: 'mcrl2compilerewriter') determines the script that "
                   "compiles the rewriter, and MCRL2_COMPILEDIR (default value: '.') determines "
                   "where temporary files are stored.\n"
                   "\n"
                   "Note that lps2lts can deliver multiple transitions with the same label between"
                   "any pair of states. If this is not desired, such transitions can be removed by"
                   "applying a strong bisimulation reducton using for instance the tool ltsconvert.\n"
                   "\n"
                   "The format of OUTFILE is determined by its extension (unless it is specified "
                   "by an option). The supported formats are:\n"
                   "\n"
                   +mcrl2::lts::detail::supported_lts_formats_text()+"\n"
                   "If the jittyc rewriter is used, then the MCRL2_COMPILEREWRITER environment "
                   "variable (default value: mcrl2compilerewriter) determines the script that "
                   "compiles the rewriter, and MCRL2_COMPILEDIR (default value: '.') "
                   "determines where temporary files are stored."
                   "\n"
                   "Note that lps2lts can deliver multiple transitions with the same "
                   "label between any pair of states. If this is not desired, such "
                   "transitions can be removed by applying a strong bisimulation reducton "
                   "using for instance the tool ltsconvert."
                  )
    {
    }

    void abort()
    {
      m_lps2lts.abort();
    }

    bool run()
    {
      load_lps(m_options.specification, m_filename);
      m_options.trace_prefix = m_filename.substr(0, m_options.trace_prefix.find_last_of('.'));

      m_options.validate_actions(); // Throws an exception if actions are not properly declared.

      if (!m_lps2lts.initialise_lts_generation(&m_options))
      {
        return false;
      }

      try
      {
        m_lps2lts.generate_lts();
      }
      catch (mcrl2::runtime_error& e)
      {
        mCRL2log(error) << e.what() << std::endl;
        m_lps2lts.finalise_lts_generation();
        return false;
      }

      m_lps2lts.finalise_lts_generation();

      return true;
    }

  protected:
    void add_options(interface_description& desc)
    {
      lps2lts_base::add_options(desc);

      desc.
      add_option("cached",
                 "use enumeration caching techniques to speed up state space generation. ").
      add_option("dummy", make_mandatory_argument("BOOL"),
                 "replace free variables in the LPS with dummy values based on the value of BOOL: 'yes' (default) or 'no'. ", 'y').
      add_option("unused-data",
                 "do not remove unused parts of the data specification. ", 'u').
      add_option("max", make_mandatory_argument("NUM"),
                 "explore at most NUM states", 'l').
      add_option("todo-max", make_mandatory_argument("NUM"),
                 "keep at most NUM states in todo lists; this option is only relevant for "
                 "breadth-first search, where NUM is the maximum number of states per "
                 "level, and for depth first search, where NUM is the maximum depth. ").
      add_option("nondeterminism",
                 "detect nondeterministic states, i.e. states with outgoing transitions with the same label to different states. ", 'n').
      add_option("deadlock",
                 "detect deadlocks (i.e. for every deadlock a message is printed). ", 'D').
      add_option("action", make_mandatory_argument("NAMES"),
                 "report whether an action from NAMES occurs in the transitions system, "
                 "where NAMES is a comma-separated list. A message "
                 "is printed for every occurrence of one of these action names. "
                 "With the -t flag traces towards these actions are generated. "
                 "When using -tN only N traces are generated after which the generation of the state space stops. ", 'a').
      add_option("multiaction", make_mandatory_argument("NAMES"),
                 "detect and report multiactions in the transitions system "
                 "from NAMES, a comma-separated list. Works like -a, except that multi-actions "
                 "are matched exactly, including data parameters. ", 'm').
      add_option("trace", make_optional_argument("NUM", std::to_string(lts_generation_options::default_max_traces)),
                 "Write a shortest trace to each state that is reached with an action from NAMES "
                 "with the option --action, is a deadlock with the option --deadlock, is nondeterministic with the option --nondeterminism, or is a "
                 "divergence with the option --divergence to a file. "
                 "No more than NUM traces will be written. If NUM is not supplied the number of "
                 "traces is unbounded. "
                 "For each trace that is to be written a unique file with extension .trc (trace) "
                 "will be created containing a shortest trace from the initial state to the deadlock "
                 "state. The traces can be pretty printed and converted to other formats using tracepp. ", 't').
      add_option("error-trace",
                 "if an error occurs during exploration, save a trace to the state that could "
                 "not be explored. ").
      add_option("out", make_mandatory_argument("FORMAT"),
                 "save the output in the specified FORMAT. ", 'o').
      add_option("no-info", "do not add state information to OUTFILE. "
                 "Without this option lps2lts adds state vector to the LTS. This "
                 "option causes this information to be discarded and states are only "
                 "indicated by a sequence number. Explicit state information is useful "
                 "for visualisation purposes, for instance, but can cause the OUTFILE "
                 "to grow considerably. Note that this option is implicit when writing "
                 "in the AUT format. ").
      add_option("suppress","in verbose mode, do not print progress messages indicating the number of visited states and transitions. "
                 "For large state spaces the number of progress messages can be quite "
                 "horrendous. This feature helps to suppress those. Other verbose messages, "
                 "such as the total number of states explored, just remain visible. ").
      add_option("init-tsize", make_mandatory_argument("NUM"),
                 "set the initial size of the internally used hash tables (default is 10000). ").
      add_option("tau",make_mandatory_argument("ACTNAMES"),
                 "consider actions with a name in the comma separated list ACTNAMES to be internal. "
                 "This list is only used and allowed when searching for divergencies. ");
    }

    void parse_options(const command_line_parser& parser)
    {
      lps2lts_base::parse_options(parser);
      m_options.removeunused    = parser.options.count("unused-data") == 0;
      m_options.detect_deadlock = parser.options.count("deadlock") != 0;
      m_options.detect_nondeterminism = parser.options.count("nondeterminism") != 0;
      m_options.outinfo         = parser.options.count("no-info") == 0;
      m_options.suppress_progress_messages = parser.options.count("suppress") !=0;
      m_options.strat           = parser.option_argument_as< mcrl2::data::rewriter::strategy >("rewriter");

      m_options.use_enumeration_caching = parser.options.count("cached") > 0;

      if (parser.options.count("dummy"))
      {
        if (parser.options.count("dummy") > 1)
        {
          parser.error("Multiple use of option -y/--dummy; only one occurrence is allowed.");
        }
        std::string dummy_str(parser.option_argument("dummy"));
        if (dummy_str == "yes")
        {
          m_options.usedummies = true;
        }
        else if (dummy_str == "no")
        {
          m_options.usedummies = false;
        }
        else
        {
          parser.error("Option -y/--dummy has illegal argument '" + dummy_str + "'.");
        }
      }

      if (parser.options.count("max"))
      {
        m_options.max_states = parser.option_argument_as< unsigned long > ("max");
      }
      if (parser.options.count("action"))
      {
        m_options.detect_action = true;
        std::list<std::string> actions = split_actions(parser.option_argument("action"));
        for (const std::string& s: actions)
        {
          m_options.trace_actions.insert(mcrl2::core::identifier_string(s));
        }
      }
      if (parser.options.count("multiaction"))
      {
        std::list<std::string> actions = split_actions(parser.option_argument("multiaction"));
        m_options.trace_multiaction_strings.insert(actions.begin(), actions.end());
      }
      if (parser.options.count("tau")>0)
      {
        std::list<std::string> actions = split_actions(parser.option_argument("tau"));
      }
      if (parser.options.count("trace"))
      {
        m_options.trace      = true;
        m_options.max_traces = parser.option_argument_as< unsigned long > ("trace");
      }

      if (parser.options.count("out"))
      {
        m_options.outformat = mcrl2::lts::detail::parse_format(parser.option_argument("out"));

        if (m_options.outformat == lts_none)
        {
          parser.error("Format '" + parser.option_argument("out") + "' is not recognised.");
        }
      }
      if (parser.options.count("init-tsize"))
      {
        m_options.initial_table_size = parser.option_argument_as< unsigned long >("init-tsize");
      }
      if (parser.options.count("todo-max"))
      {
        m_options.todo_max = parser.option_argument_as< unsigned long >("todo-max");
      }
      if (parser.options.count("error-trace"))
      {
        m_options.save_error_trace = true;
      }

      if (parser.options.count("suppress") && !mCRL2logEnabled(verbose))
      {
        parser.error("Option --suppress requires --verbose (of -v).");
      }

      if (2 < parser.arguments.size())
      {
        parser.error("Too many file arguments.");
      }
      if (0 < parser.arguments.size())
      {
        m_filename = parser.arguments[0];
      }
      if (1 < parser.arguments.size())
      {
        m_options.lts = parser.arguments[1];
      }

      if (!m_options.lts.empty() && m_options.outformat == lts_none)
      {
        m_options.outformat = mcrl2::lts::detail::guess_format(m_options.lts);

        if (m_options.outformat == lts_none)
        {
          mCRL2log(warning) << "no output format set or detected; using default (mcrl2)" << std::endl;
          m_options.outformat = lts_lts;
        }
      }
    }

};

lps2lts_tool *tool_instance;

static
void premature_termination_handler(int)
{
  // Reset signal handlers.
  signal(SIGABRT,NULL);
  signal(SIGINT,NULL);
  tool_instance->abort();
}

int main(int argc, char** argv)
{
  int result;
  tool_instance = new lps2lts_tool();

  signal(SIGABRT,premature_termination_handler);
  signal(SIGINT,premature_termination_handler); // At ^C invoke the termination handler.

  try
  {
    result = tool_instance->execute(argc, argv);
#ifdef MCRL2_DISPLAY_REWRITE_STATISTICS
    mcrl2::data::detail::display_rewrite_statistics();
#endif
  }
  catch (...)
  {
    delete tool_instance;
    throw;
  }
  delete tool_instance;
  return result;
}
