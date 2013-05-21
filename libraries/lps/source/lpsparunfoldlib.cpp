#include <algorithm>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include "mcrl2/utilities/logger.h"
#include <iterator>

#include "mcrl2/data/function_symbol.h"
#include "mcrl2/data/data_specification.h"
#include "mcrl2/data/set_identifier_generator.h"

#include "mcrl2/lps/lpsparunfoldlib.h"
#include "mcrl2/lps/linear_process.h"
#include "mcrl2/lps/replace.h"

using namespace std;
using namespace mcrl2;
using namespace mcrl2::core;
using namespace mcrl2::data;
using namespace mcrl2::log;

/* Remarks
- replace on vectors does not work
- vector pretty print does not work
- alias::name() [basic_sort] results in a basic sort, differs form basic_sort::name() [string]
*/

lpsparunfold::lpsparunfold(mcrl2::lps::specification spec,
    std::map< mcrl2::data::sort_expression , lspparunfold::unfold_cache_element > *cache,
    bool add_distribution_laws
)
{
  m_cache = cache;

  m_add_distribution_laws = add_distribution_laws;
  mCRL2log(debug) << "Processing" << std::endl;
  m_data_specification = spec.data() ;
  m_lps = spec.process();
  m_init_process = spec.initial_process();
  m_glob_vars = spec.global_variables();
  m_action_label_list = spec.action_labels();

  for (std::vector<sort_expression>::const_iterator i =  m_data_specification.sorts().begin();
       i != m_data_specification.sorts().end();
       ++i)
  {
    if (is_basic_sort(*i))
    {
      sort_names.insert((basic_sort(*i)).name());
    }
  };

  {
    size_t size = mapping_and_constructor_names.size();
    function_symbol_vector fsl= m_data_specification.constructors();
    for (function_symbol_vector::const_iterator i = fsl.begin();
         i != fsl.end();
         ++i)
    {
      mapping_and_constructor_names.insert(i -> name());
    };
    mCRL2log(debug) << "- Specification has " <<   mapping_and_constructor_names.size() - size << " constructors" << std::endl;
  }

  {
    size_t size = mapping_and_constructor_names.size();
    function_symbol_vector fsl= m_data_specification.mappings();
    for (function_symbol_vector::const_iterator i = fsl.begin();
         i != fsl.end();
         ++i)
    {
      mapping_and_constructor_names.insert(i -> name());
    };
    mCRL2log(debug) << "- Specification has " <<  mapping_and_constructor_names.size() - size << " mappings " << std::endl;
  }
}

mcrl2::data::basic_sort lpsparunfold::generate_fresh_basic_sort(std::string str)
{
  //Generate a fresh Basic Sort
  mcrl2::data::set_identifier_generator generator;
  generator.add_identifiers(sort_names);
  mcrl2::core::identifier_string nstr = generator(str);
  mCRL2log(verbose) << "Generated fresh sort \"" <<  string(nstr) << "\" for \"" <<  str << "\"" << std::endl;
  sort_names.insert(nstr);
  return basic_sort(std::string(nstr));
}

mcrl2::core::identifier_string lpsparunfold::generate_fresh_constructor_and_mapping_name(std::string str)
{
  //Generate a fresh name for a constructor of mapping

  str.resize(std::remove_if(str.begin(), str.end(), &char_filter) - str.begin());

  mcrl2::data::set_identifier_generator generator;
  generator.add_identifiers(mapping_and_constructor_names);
  mcrl2::core::identifier_string nstr = generator(str);
  mCRL2log(debug) << "Generated a fresh mapping: " <<  string(nstr) << std::endl;
  mapping_and_constructor_names.insert(nstr);
  return nstr;
}

function_symbol_vector lpsparunfold::determine_affected_constructors()
{
  function_symbol_vector t = m_data_specification.constructors(m_unfold_process_parameter);
  function_symbol_vector k = function_symbol_vector(t.begin(), t.end());

  mCRL2log(debug) << "k:\t";
  mCRL2log(verbose) << "" <<  unfold_parameter_name  << " has " <<  k.size() << " constructor function(s)" << std::endl;

  for (function_symbol_vector::iterator i = k.begin(); i != k.end(); ++i)
  {
    mCRL2log(debug) << "\t" <<  *i << std::endl;
  }
  return k;
}


function_symbol_vector lpsparunfold::new_constructors(mcrl2::data::function_symbol_vector k)
{
  using namespace mcrl2::data;

  function_symbol_vector elements_of_new_sorts;

  for (function_symbol_vector::iterator i = k.begin(); i != k.end(); ++i)
  {

    std::string prefix = "c_";
    mcrl2::core::identifier_string fresh_name = generate_fresh_constructor_and_mapping_name(prefix.append(i -> name()));
    const data::function_symbol f(fresh_name , fresh_basic_sort);
    elements_of_new_sorts.push_back(f);
    m_data_specification.add_constructor(f);
    mCRL2log(debug) << "\t" << data::function_symbol(fresh_name , fresh_basic_sort) << std::endl;
    mapping_and_constructor_names.insert(fresh_name);

  }
  mCRL2log(debug) << "- Created " <<  elements_of_new_sorts.size() << " fresh \" c_ \" constructor(s)" << std::endl;
  return elements_of_new_sorts;
}

mcrl2::data::function_symbol lpsparunfold::create_case_function(size_t k)
{
  mcrl2::data::function_symbol fs;
  std::string str = "C_";
  str.append(fresh_basic_sort.name()).append("_");
  mcrl2::core::identifier_string idstr = generate_fresh_constructor_and_mapping_name(str);
  mcrl2::data::sort_expression_vector fsl;
  fsl.push_back(fresh_basic_sort);
  for (size_t i = 0; i < k; ++i)
  {
    fsl.push_back(m_unfold_process_parameter);
  }

  fs = data::function_symbol(idstr , mcrl2::data::function_sort(fsl, m_unfold_process_parameter));

  mCRL2log(debug) << "- Created C map: " << fs << std::endl;

  return fs;
}

mcrl2::data::function_symbol lpsparunfold::create_determine_function()
{
  mcrl2::data::function_symbol fs;
  std::string str = "Det_";
  str.append(string(fresh_basic_sort.name()).append("_"));
  mcrl2::core::identifier_string idstr = generate_fresh_constructor_and_mapping_name(str);
  mcrl2::data::sort_expression_list fsl;
  fs = data::function_symbol(idstr , mcrl2::data::make_function_sort(m_unfold_process_parameter , fresh_basic_sort));
  mCRL2log(debug) << "\t" <<  fs << std::endl;

  return fs;
}

mcrl2::data::function_symbol_vector lpsparunfold::create_projection_functions(function_symbol_vector k)
{
  mcrl2::data::function_symbol_vector sfs;
  std::string str = "pi_";
  str.append(string(fresh_basic_sort.name()).append("_"));

  std::set<mcrl2::data::sort_expression> processed;
  for (function_symbol_vector::iterator i = k.begin() ; i != k.end(); ++i)
  {
    if (is_function_sort(i->sort()))
    {
      function_sort fs = function_sort(i->sort());
      sort_expression_list sel  = fs.domain();
      for (sort_expression_list::const_iterator j = sel.begin(); j != sel.end(); j++)
      {
        mcrl2::core::identifier_string idstr = generate_fresh_constructor_and_mapping_name(str);
        data::function_symbol map(idstr , mcrl2::data::make_function_sort(m_unfold_process_parameter , *j));
        m_data_specification.add_mapping(map);
        sfs.push_back(map);
        processed.insert(*j);
      }
    }
  }
  for (function_symbol_vector::iterator i = sfs.begin(); i != sfs.end(); ++i)
  {
    mCRL2log(debug) << "\t" <<  *i << std::endl;
  }
  return sfs;
}

void lpsparunfold::create_data_equations(
                function_symbol_vector pi, 
                data::function_symbol case_function, 
                function_symbol_vector elements_of_new_sorts, 
                function_symbol_vector k, 
                data::function_symbol determine_function)
{
  variable_vector vars;        /* Equation variables  */
  // data_equation_vector del;    /* Generated equations */
  std::set<mcrl2::core::identifier_string> var_names; /* var_names */
  mcrl2::data::set_identifier_generator generator;
  variable v;


  std::string cstr = "c";
  std::string dstr = "d";

  /* Creating variable for detector function */
  mcrl2::core::identifier_string istr = generator(cstr);
  v = variable(istr, fresh_basic_sort);
  vars.push_back(v);

  /* Create Equations */
  if (m_add_distribution_laws)
  {
    mCRL2log(verbose) << "Adding equations with additional distribution laws..." << std::endl;
  }
  else
  {
    mCRL2log(verbose) << "Adding equations..." << std::endl;
  }

  {
    generate_case_functions(elements_of_new_sorts, case_function);
  }

  /* Creating Detector equations */
  std::map< sort_expression, variable_vector > sort_vars;//type_var_list;    /* Mapping for Sort |-> [Variable] */
  std::map< sort_expression, size_t             > sort_index;//type_var_count;  /* Mapping for counting the number of unique Sorts of an equation */

  size_t e = 0;
  for (function_symbol_vector::iterator i = k.begin(); i != k.end(); ++i)
  {
    sort_index.clear();

    if (is_basic_sort(i -> sort()))
    {
      data_expression lhs = make_application(determine_function, *i);
      mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, elements_of_new_sorts[e])) << std::endl;
      set< variable > svars = find_all_variables(lhs);
      set< variable > tmp_var = find_all_variables(elements_of_new_sorts[e]);
      svars.insert(tmp_var.begin(), tmp_var.end());
      m_data_specification.add_equation(data_equation(variable_list(svars.begin(), svars.end()), lhs, elements_of_new_sorts[e]));
    }

    if (is_function_sort(i -> sort()))
    {
      function_sort fs = function_sort(i -> sort());
      sort_expression_list sel = fs.domain();

      data_expression_vector dal;
      for (sort_expression_list::const_iterator j = sel.begin(); j != sel.end(); ++j)
      {
        if (sort_vars[*j].size() == sort_index[ *j])
        {
          istr = generator(dstr);
          data_expression v = variable(istr, *j);
          sort_vars[*j].push_back(v);
        }
        variable y = sort_vars[*j].at(sort_index[*j]);
        sort_index[*j] = sort_index[*j]+1;
        dal.push_back(y);
      }
      data_expression lhs = make_application(determine_function , application(*i, mcrl2::data::data_expression_list(dal.begin(), dal.end())));
      mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, elements_of_new_sorts[e])) << std::endl;
      set< variable > svars = find_all_variables(lhs);
      set< variable > tmp_var = find_all_variables(elements_of_new_sorts[e]);
      svars.insert(tmp_var.begin(), tmp_var.end());
      m_data_specification.add_equation(data_equation(variable_list(svars.begin(), svars.end()), lhs, elements_of_new_sorts[e]));

      /* Equations for projection functions */
      size_t f = 0;

      while (!pi.empty() && f < static_cast<size_t>(std::distance(dal.begin(), dal.end())))
      {
        data_expression lhs = make_application(pi.front(), application(*i, mcrl2::data::data_expression_list(dal.begin(), dal.end())));
        mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, dal[f])) << std::endl;
        set< variable > vars = find_all_variables(lhs);
        set< variable > tmp_var = find_all_variables(dal[f]);
        vars.insert(tmp_var.begin(), tmp_var.end());
        m_data_specification.add_equation(data_equation(variable_list(vars.begin(), vars.end()), lhs, dal[f]));

        if (m_add_distribution_laws)
        {
          /* Add additional distribution laws for pi over if

             pi(if(b,x,y))=if(b,pi(x),pi(y));
          */
          basic_sort bool_sort("Bool");
          sort_expression if_arg_sort(function_sort(pi.front().sort()).domain().front());
          data::function_symbol if_function_symbol("if", make_function_sort(bool_sort, if_arg_sort, if_arg_sort , if_arg_sort));
          m_data_specification.add_equation(create_distribution_law_over_case(pi.front(), if_function_symbol, false));
          /* Add additional distribution laws for pi over case

             pi(C(e,x1,x2,...))=C(e,pi(x1),pi(x2),...);
          */
          const data_equation e=create_distribution_law_over_case(pi.front(), case_function, true);
          m_data_specification.add_equation(e);

          /* Create additional case function */

          generate_case_functions(elements_of_new_sorts, atermpp::aterm_cast<function_symbol>(application(e.rhs()).head()));

        }
        ++f;
        pi.erase(pi.begin());
      }
    }

    if (is_structured_sort(i -> sort())) // This case seems to be identical to the basic sort.
    {

      data_expression lhs = make_application(determine_function , *i);
      mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, elements_of_new_sorts[e])) << std::endl;
      set< variable > vars = find_all_variables(lhs);
      set< variable > tmp_var = find_all_variables(elements_of_new_sorts[e]);
      vars.insert(tmp_var.begin(), tmp_var.end());
      m_data_specification.add_equation(data_equation(variable_list(vars.begin(), vars.end()), lhs, elements_of_new_sorts[e]));

    }

    if (is_container_sort(i -> sort()))
    {
      container_sort cs = container_sort(i -> sort());
      sort_expression element_sort = cs.element_sort();

      if (sort_vars[element_sort].size() == sort_index[ element_sort ])
      {
        istr = generator(dstr);
        data_expression v = variable(istr, element_sort);
        sort_vars[ element_sort ].push_back(v);
      }
      //variable y = sort_vars[ element_sort ].at(sort_index[ element_sort ]);
      sort_index[ element_sort ] = sort_index[ element_sort ]+1;

      data_expression lhs, rhs;
      if (cs.container_name() == list_container())
      {
        lhs = make_application(determine_function , sort_list::empty(element_sort));
      }
      else
      {
        mCRL2log(mcrl2::log::warning) << "ACCESSING UNIMPLEMENTED CONTAINER SORT" << endl;
      }

      mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, elements_of_new_sorts[e])) << std::endl;
      set< variable > vars = find_all_variables(lhs);
      set< variable > tmp_var = find_all_variables(elements_of_new_sorts[e]);
      vars.insert(tmp_var.begin(), tmp_var.end());
      m_data_specification.add_equation(data_equation(variable_list(vars.begin(), vars.end()), lhs, elements_of_new_sorts[e]));

    }
    e++;
  }

  if (m_add_distribution_laws)
  {
    /*  Add additional distribution laws for Det over if

               Det(if(b,x,y))=if(b,Det(x),Det(y));
    */
    basic_sort bool_sort("Bool");
    sort_expression if_arg_sort(function_sort(determine_function.sort()).domain().front());
    data::function_symbol if_function_symbol("if", make_function_sort(bool_sort, if_arg_sort, if_arg_sort , if_arg_sort));
    m_data_specification.add_equation(create_distribution_law_over_case(determine_function, if_function_symbol, false));

    /*  Add additional distribution laws for Det over case

        Det(C(e,x1,x2,...))=C(e,Det(x1),Det(x2),...);
    */
    m_data_specification.add_equation(create_distribution_law_over_case(determine_function, case_function, true));
  }
}

mcrl2::core::identifier_string lpsparunfold::generate_fresh_process_parameter_name(std::string str, std::set<mcrl2::core::identifier_string>& process_parameter_names)
{
  mcrl2::data::set_identifier_generator generator;
  generator.add_identifiers(process_parameter_names);
  mcrl2::core::identifier_string idstr = generator(str.append("_pp"));
  process_parameter_names.insert(idstr);
  return idstr;
}

void lpsparunfold::unfold_summands(mcrl2::lps::action_summand_vector& summands, const mcrl2::data::function_symbol& determine_function, const mcrl2::data::function_symbol_vector& pi)
{
  for (mcrl2::lps::action_summand_vector::iterator j = summands.begin(); j != summands.end(); ++j)
  {
    mcrl2::data::assignment_list ass = j->assignments();
    //Create new left-hand assignment_list & right-hand assignment_list
    mcrl2::data::data_expression_vector new_ass_left;
    mcrl2::data::data_expression_vector new_ass_right;
    for (mcrl2::data::assignment_list::iterator k = ass.begin(); k != ass.end(); ++k)
    {
      if (proc_par_to_proc_par_inj.find(k->lhs()) != proc_par_to_proc_par_inj.end())
      {
        for (mcrl2::data::variable_vector::iterator l =  proc_par_to_proc_par_inj[ k -> lhs() ].begin()
             ; l != proc_par_to_proc_par_inj[ k -> lhs() ].end()
             ; ++l)
        {
          new_ass_left.push_back(*l);
        }

        mcrl2::data::data_expression_vector ins = unfold_constructor(k -> rhs(), determine_function, pi);
        //Replace unfold parameters in affected assignments
        new_ass_right.insert(new_ass_right.end(), ins.begin(), ins.end());
      }
      else
      {
        new_ass_left.push_back(k-> lhs());
        new_ass_right.push_back(k-> rhs());
      }
    }

    //cout << new_ass_left.size()<< " " << new_ass_right.size() << endl;
    //cout << mcrl2::data::pp(*j) << endl;
    //cout << mcrl2::data::pp(new_ass_left)  << endl;
    //cout << mcrl2::data::pp(new_ass_right)  << endl;

    assert(new_ass_left.size() == new_ass_right.size());
    mcrl2::data::assignment_vector new_ass;
    while (!new_ass_left.empty())
    {
      new_ass.push_back(mcrl2::data::assignment(new_ass_left.front(), new_ass_right.front()));
      new_ass_left.erase(new_ass_left.begin());
      new_ass_right.erase(new_ass_right.begin());
    }
    j->assignments() = mcrl2::data::assignment_list(new_ass.begin(), new_ass.end());
  }
}

mcrl2::lps::linear_process lpsparunfold::update_linear_process(function_symbol case_function , function_symbol_vector affected_constructors, function_symbol determine_function, size_t parameter_at_index, function_symbol_vector pi)
{
  /* Get process parameters from lps */
  mcrl2::data::variable_list lps_proc_pars =  m_lps.process_parameters();

  /* Get process_parameters names from lps */
  std::set<mcrl2::core::identifier_string> process_parameter_names;
  for (mcrl2::data::variable_list::iterator i = lps_proc_pars.begin();
       i != lps_proc_pars.end();
       ++i)
  {
    process_parameter_names.insert(i -> name());
  }

  mCRL2log(verbose) << "Updating LPS..." << std::endl;
  /* Create new process parameters */
  mcrl2::data::variable_vector new_process_parameters;
  for (mcrl2::data::variable_list::iterator i = lps_proc_pars.begin();
       i != lps_proc_pars.end();
       ++i)
  {
    if (static_cast<size_t>(std::distance(lps_proc_pars.begin(), i)) == parameter_at_index)
    {
      mCRL2log(verbose) << "Unfolding parameter " << i->name() << " at index " << std::distance(lps_proc_pars.begin(), i) << "..." << std::endl;
      mcrl2::data::variable_vector process_parameters_injection;

      /* Generate fresh process parameter for new Sort */
      mcrl2::core::identifier_string idstr = generate_fresh_process_parameter_name(unfold_parameter_name, process_parameter_names);
      process_parameters_injection.push_back(mcrl2::data::variable(idstr , fresh_basic_sort));

      mCRL2log(verbose) << "- Created process parameter " <<  data::pp(process_parameters_injection.back()) << " of type " <<  data::pp(fresh_basic_sort) << "" << std::endl;

      for (mcrl2::data::function_symbol_vector::iterator j = affected_constructors.begin()
           ; j != affected_constructors.end()
           ; ++j)
      {
        //cout << *j << endl;

        bool processed = false;
        if (is_function_sort(j -> sort()))
        {
          sort_expression_list dom = function_sort(j -> sort()).domain();
          for (sort_expression_list::iterator k = dom.begin(); k != dom.end(); ++k)
          {
            mcrl2::core::identifier_string idstr = generate_fresh_process_parameter_name(unfold_parameter_name, process_parameter_names);
            process_parameters_injection.push_back(mcrl2::data::variable(idstr ,  *k));
            mCRL2log(verbose) << "- Injecting process parameter: " <<  data::pp(idstr) << "::" <<  data::pp(*k) << std::endl;
          }
          processed = true;
        }

        if (is_basic_sort(j -> sort()))
        {
          mCRL2log(debug) << "- No processed parameter are injected for basic sort: " <<  *j << std::endl;
          processed = true;
        }

        if (is_structured_sort(j -> sort()))
        {
          processed = true;
        }

        if (is_container_sort(j -> sort()))
        {
          processed = true;
        }
        if (!processed)
        {
          mCRL2log(mcrl2::log::debug) << data::pp(*j) << " is not processed" << endl;
          mCRL2log(mcrl2::log::debug) << *j << endl;
          abort();
        }
      }
      new_process_parameters.insert(new_process_parameters.end(), process_parameters_injection.begin(), process_parameters_injection.end());

      /* store mapping: process parameter -> process parameter injection:
         Required for process parameter replacement in summands
      */
      proc_par_to_proc_par_inj[*i] = process_parameters_injection;

    }
    else
    {
      new_process_parameters.push_back(*i);
    }
  }
  mCRL2log(debug) << "- New LPS process parameters: " <<  mcrl2::data::pp(new_process_parameters) << std::endl;

  //Prepare parameter substitution
  std::map<mcrl2::data::data_expression, mcrl2::data::data_expression> parsub = parameter_substitution(proc_par_to_proc_par_inj, affected_constructors, case_function);

  // TODO: avoid unnecessary copies of the LPS
  mcrl2::lps::linear_process new_lps;
  new_lps.action_summands() = m_lps.action_summands();
  new_lps.deadlock_summands() = m_lps.deadlock_summands();

  // update the summands in new_lps
  unfold_summands(new_lps.action_summands(), determine_function, pi);

  new_lps.process_parameters() = mcrl2::data::variable_list(new_process_parameters.begin(), new_process_parameters.end());

  for (std::map<mcrl2::data::data_expression, mcrl2::data::data_expression>::iterator i = parsub.begin()
       ; i != parsub.end()
       ; ++i)
  {
    mutable_map_substitution< std::map< mcrl2::data::variable , mcrl2::data::data_expression > > s;
    s[ i->first ] = i->second;
    mcrl2::lps::replace_variables( new_lps, s );
  }

  mCRL2log(debug) << "\nNew LPS:\n" <<  lps::pp(new_lps) << std::endl;

  assert(is_well_typed(new_lps));

  return new_lps;
}

mcrl2::lps::process_initializer lpsparunfold::update_linear_process_initialization(data::function_symbol determine_function, size_t parameter_at_index, function_symbol_vector pi)
{
  //
  //update inital process
  //
  mCRL2log(verbose) << "Updating initialization...\n" << std::endl;

  data::assignment_list ass = m_init_process.assignments();
  //Create new left-hand assignment_list
  mcrl2::data::data_expression_vector new_ass_left;
  for (mcrl2::data::assignment_list::iterator k = ass.begin()
       ; k != ass.end()
       ; ++k)
  {
    if (proc_par_to_proc_par_inj.find(k-> lhs()) != proc_par_to_proc_par_inj.end())
    {
      for (mcrl2::data::variable_vector::iterator l =  proc_par_to_proc_par_inj[ k -> lhs() ].begin()
           ; l != proc_par_to_proc_par_inj[ k -> lhs() ].end()
           ; ++l)
      {
        new_ass_left.push_back(*l);
      }
    }
    else
    {
      new_ass_left.push_back(k-> lhs());
    }
  }
  //Create new right-hand assignment_list
  //Unfold parameters
  mcrl2::data::data_expression_vector new_ass_right;
  for (mcrl2::data::assignment_list::iterator k = ass.begin()
       ; k != ass.end()
       ; ++k)
  {
    if (static_cast<size_t>(std::distance(ass.begin(), k)) == parameter_at_index)
    {

      mcrl2::data::data_expression_vector ins = unfold_constructor(k -> rhs(), determine_function, pi);
      //Replace unfold parameters in affected assignments
      new_ass_right.insert(new_ass_right.end(), ins.begin(), ins.end());
    }
    else
    {
      new_ass_right.push_back(k-> rhs());
    }
  }

  //cout << new_ass_left.size()<< " " << new_ass_right.size() << endl;
  //cout << mcrl2::data::pp(new_ass_left)  << endl;
  //cout << mcrl2::data::pp(new_ass_right)  << endl;

  assert(new_ass_left.size() == new_ass_right.size());
  mcrl2::data::assignment_vector new_ass;
  while (!new_ass_left.empty())
  {
    new_ass.push_back(mcrl2::data::assignment(new_ass_left.front(), new_ass_right.front()));
    new_ass_left.erase(new_ass_left.begin());
    new_ass_right.erase(new_ass_right.begin());
  }

  mcrl2::lps::process_initializer new_init = mcrl2::lps::process_initializer(mcrl2::data::assignment_list(new_ass.begin(), new_ass.end()));
  mCRL2log(debug) << lps::pp(new_init) << std::endl;

  return new_init;
}

std::map<mcrl2::data::data_expression, mcrl2::data::data_expression> lpsparunfold::parameter_substitution(std::map<mcrl2::data::variable, mcrl2::data::variable_vector > proc_par_to_proc_par_inj, mcrl2::data::function_symbol_vector k, mcrl2::data::function_symbol case_function)
{
  std::map<mcrl2::data::data_expression, mcrl2::data::data_expression> result;
  data_expression_vector dev;

  set<mcrl2::data::variable_vector::iterator> used_iters;

  mcrl2::data::variable prev;
  for (std::map<mcrl2::data::variable, mcrl2::data::variable_vector >::iterator i = proc_par_to_proc_par_inj.begin()
       ; i != proc_par_to_proc_par_inj.end()
       ; ++i)
  {
    if (prev != i->first)
    {
      dev.clear();
    }

    dev.push_back(data_expression(i->second.front()));

    for (mcrl2::data::function_symbol_vector::iterator m = k.begin()
         ; m != k.end()
         ; ++m)
    {
      if (is_basic_sort(m -> sort()))
      {
        dev.push_back(*m);
      }

      if (is_structured_sort(m -> sort()))
      {
        dev.push_back(*m);
      }

      if (is_function_sort(m -> sort()))
      {
        sort_expression_list dom = function_sort(m -> sort()). domain();
        data_expression_vector arg;

        for (sort_expression_list::iterator n = dom.begin(); n != dom.end(); ++n)
        {
          for (mcrl2::data::variable_vector::iterator o = i->second.begin()
               ; o != i->second.end()
               ; ++o)
          {
            if (o -> sort() == *n && used_iters.find(o) == used_iters.end())
            {
              used_iters.insert(o);
              arg.push_back(*o);
              break;
            }
          }
        }
        dev.push_back(mcrl2::data::application(*m, arg));
      }

      if (is_container_sort(m -> sort()))
      {
        dev.push_back(*m);
      }

    }

    mCRL2log(verbose) << "Parameter substitution:\t" << data::pp(i->first) << "\t->\t" <<  data::pp(mcrl2::data::application(case_function, dev)) << std::endl;
    result.insert(std::pair<mcrl2::data::data_expression, mcrl2::data::data_expression>(i -> first,  mcrl2::data::application(case_function, dev)));
  }
  return result ;
}

mcrl2::data::data_expression_vector lpsparunfold::unfold_constructor(data_expression de, data::function_symbol determine_function, function_symbol_vector pi)
{
  mcrl2::data::data_expression_vector result;
  {
    /* Unfold parameter if function symbol occurs  */
    /* size of unfold parameter must be equal to 1 */
    data_expression_vector new_ass;

    /* Det function */
    new_ass.push_back(make_application(determine_function, de)) ;

    for (function_symbol_vector::iterator i = pi.begin(); i != pi.end(); ++i)
    {
      new_ass.push_back(make_application(*i, de)) ;
    }

    result = new_ass;
  }
  return result;
}

mcrl2::data::sort_expression lpsparunfold::sort_at_process_parameter_index(size_t parameter_at_index)
{
  mcrl2::data::variable_list lps_proc_pars_list =  m_lps.process_parameters();
  mcrl2::data::variable_vector lps_proc_pars = mcrl2::data::variable_vector(lps_proc_pars_list.begin(), lps_proc_pars_list.end());
  mCRL2log(debug) << "- Number of parameters in LPS: " <<  lps_proc_pars.size() << "" << std::endl;
  mCRL2log(verbose) << "Unfolding process parameter at index: " <<  parameter_at_index << "" << std::endl;
  if (lps_proc_pars.size() <= parameter_at_index)
  {
    mCRL2log(mcrl2::log::error) << "Given index out of bounds. Index value needs to be in the range [0," << lps_proc_pars.size() <<")." << endl;
    abort();
  }

  if (is_basic_sort(lps_proc_pars[parameter_at_index].sort()))
  {
    unfold_parameter_name = basic_sort(lps_proc_pars[parameter_at_index].sort()).name();
  }

  if (is_structured_sort(lps_proc_pars[parameter_at_index].sort()))
  {
    mcrl2::data::set_identifier_generator generator;
    mcrl2::core::identifier_string nstr;
    nstr = generator("S");
    sort_names.insert(nstr);
    generator.add_identifiers(sort_names);
    unfold_parameter_name = nstr;
  }

  if (is_container_sort(lps_proc_pars[parameter_at_index].sort()))
  {
    mcrl2::data::set_identifier_generator generator;
    mcrl2::core::identifier_string nstr;
    nstr = generator("S");
    sort_names.insert(nstr);
    generator.add_identifiers(sort_names);
    unfold_parameter_name = nstr;
  }

  return lps_proc_pars[parameter_at_index].sort();
}

mcrl2::data::data_equation lpsparunfold::create_distribution_law_over_case(
  mcrl2::data::function_symbol function_for_distribution,
  mcrl2::data::function_symbol case_function,
  const bool add_case_function_to_data_type)
{
  assert(function_sort(case_function.sort()).codomain() == function_sort(function_for_distribution.sort()).domain().front());

  mcrl2::data::set_identifier_generator generator;
  variable_vector variables_used;
  mcrl2::core::identifier_string istr;

  sort_expression_list case_function_sort_arguments = function_sort(case_function.sort()).domain();
  for (sort_expression_list::iterator i = case_function_sort_arguments.begin();
       i != case_function_sort_arguments.end();
       ++i)
  {
    if (i == case_function_sort_arguments.begin())
    {
      istr = generator("b");
    }
    else
    {
      istr = generator("d");
    }
    variable v(istr, *i);
    variables_used.push_back(v);
  }

  data_expression lhs(make_application(function_for_distribution, application(case_function, variables_used)));
  data_expression_vector rw_data_expressions;
  sort_expression_vector rw_sort_expressions;
  for (variable_vector::iterator i = variables_used.begin();
       i != variables_used.end();
       ++i)
  {
    if (i == variables_used.begin())
    {
      rw_data_expressions.push_back(*i);
      rw_sort_expressions.push_back(i->sort());
    }
    else
    {
      rw_data_expressions.push_back(make_application(function_for_distribution, *i));
      rw_sort_expressions.push_back(function_sort(function_for_distribution.sort()).codomain());
    }
  }

  mcrl2::data::function_symbol new_case_function = data::function_symbol(case_function.name(), 
                         function_sort(rw_sort_expressions,function_sort(function_for_distribution.sort()).codomain()));
  if (add_case_function_to_data_type)
  {
    m_data_specification.add_mapping(new_case_function);
    generate_case_functions(m_data_specification.constructors(function_sort(new_case_function.sort()).domain().front()),new_case_function);   //add equations
  }

  /* Generate new case functions for laws */
  application rhs(new_case_function , rw_data_expressions);

  mCRL2log(verbose) << "- Added distribution law for \"" << data::pp(function_for_distribution) << "\" over \"" << data::pp(case_function) << "\": " << data::pp(data_equation(lhs,  rhs)) << std::endl;

  return data_equation(variables_used, lhs, rhs);
}

void lpsparunfold::generate_case_functions(function_symbol_vector elements_of_new_sorts, data::function_symbol case_function)
{
  mCRL2log(verbose) << "- Generating case function for:\t" <<  mcrl2::data::pp(case_function) << ": " <<  mcrl2::data::pp(case_function.sort()) << "" << std::endl;

  /* Generate variable identifier string for projection */
  std::string fstr = "y";
  mcrl2::data::set_identifier_generator generator;
  mcrl2::core::identifier_string istr = generator(fstr);

  variable_vector vars;
  sort_expression_list dom = function_sort(case_function.sort()).domain();
  for (sort_expression_list::const_iterator i = dom.begin(); i != dom.end(); ++i)
  {
    istr = generator(fstr);
    variable v(istr, *i);
    vars.push_back(v);
  }

  size_t e = 1;
  data_expression_vector sub_args(vars.begin(), vars.end());
  for (size_t i = 1 ; i < function_sort(case_function.sort()).domain().size() ; ++i)
  {
    sub_args[0] = data_expression(elements_of_new_sorts[i-1]);
    data_expression lhs = application(case_function , sub_args);
    mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, vars[e])) << std::endl;
    set< variable > svars = find_all_variables(lhs);
    set< variable > tmp_var = find_all_variables(vars[e]);
    svars.insert(tmp_var.begin(), tmp_var.end());
    m_data_specification.add_equation(data_equation(variable_list(svars.begin(), svars.end()), lhs, vars[e]));
    ++e;
  }

  {
    data_expression_vector eq_args;

    eq_args = data_expression_vector(function_sort(case_function.sort()).domain().size() -1 , vars.back());
    eq_args.insert(eq_args.begin(), vars.front());

    data_expression lhs = application(case_function , eq_args);
    mCRL2log(verbose) << "- Added equation " <<  data::pp(data_equation(lhs, vars.back())) << std::endl;
    set< variable > svars = find_all_variables(lhs);
    svars.insert(vars.back());
    m_data_specification.add_equation(data_equation(variable_list(svars.begin(), svars.end()), lhs, vars.back()));
  }
}

mcrl2::lps::specification lpsparunfold::algorithm(size_t parameter_at_index)
{
  m_unfold_process_parameter = sort_at_process_parameter_index(parameter_at_index);

  /* Var Dec */
  mcrl2::lps::linear_process new_lps;
  mcrl2::lps::process_initializer new_init;
  function_symbol_vector k;
  function_symbol_vector elements_of_new_sorts;
  function_symbol_vector projection_functions;
  data::function_symbol case_function;
  data::function_symbol determine_function;
  // data_equation_vector data_equations;

  /* Updating cache*/
  if( m_cache->find(m_unfold_process_parameter) == m_cache->end() )
  {
    /* Not using cache */

    /*   Alg */
    /*     1 */
    fresh_basic_sort = generate_fresh_basic_sort(unfold_parameter_name);
    /*     2 */
    k = determine_affected_constructors();
    if (k.empty())
    {
      mCRL2log(verbose) << "The selected process parameter " <<  unfold_parameter_name << " has no constructors." << std::endl;
      mCRL2log(verbose) << "No need to unfold." << std::endl;
      new_lps = m_lps;
      new_init = m_init_process;
    }
    else
    {
      /*     4 */
      elements_of_new_sorts = new_constructors(k);
      /*     6 */
      case_function = create_case_function(k.size());
      /*     7 */
      determine_function = create_determine_function();
      /*  8-12 */
      projection_functions = create_projection_functions(k);
      /* 13-xx */
      create_data_equations(projection_functions, case_function, elements_of_new_sorts, k, determine_function);

      //Reconstruct data specification, where already quite a number of mappings, constructors and functions have been added.
      m_data_specification.add_sort(fresh_basic_sort);
      m_data_specification.add_mapping(determine_function);
      m_data_specification.add_mapping(case_function);

      /*----------------*/
      new_lps = update_linear_process(case_function, k, determine_function, parameter_at_index, projection_functions);
      new_init = update_linear_process_initialization(determine_function, parameter_at_index, projection_functions);

      /* Updating cache*/
      lspparunfold::unfold_cache_element e;
      e.cached_case_function = case_function;
      e.cached_k = k;
      e.cached_determine_function = determine_function;
      e.cached_projection_functions = projection_functions;
      e.cached_fresh_basic_sort = fresh_basic_sort;

      m_cache->insert( pair<mcrl2::data::sort_expression , lspparunfold::unfold_cache_element>( m_unfold_process_parameter , e ));

    }
  }
  else
  {
    /* Using cache */
    mCRL2log(verbose) << "Update using cache for sort: \"" <<  mcrl2::data::pp(m_unfold_process_parameter)  << "\"..." << std::endl;

    std::map< mcrl2::data::sort_expression , lspparunfold::unfold_cache_element >::iterator ce = m_cache->find(m_unfold_process_parameter);

    fresh_basic_sort = ce->second.cached_fresh_basic_sort;
    k = ce->second.cached_k;
    if (k.empty())
    {
      mCRL2log(verbose) << "The selected process parameter " <<  unfold_parameter_name << " has no constructors." << std::endl;
      mCRL2log(verbose) << "No need to unfold." << std::endl;
      new_lps = m_lps;
      new_init = m_init_process;
    }

    case_function = ce->second.cached_case_function;
    determine_function = ce->second.cached_determine_function;
    projection_functions = ce->second.cached_projection_functions;

    new_lps = update_linear_process(case_function, k, determine_function, parameter_at_index, projection_functions);
    new_init = update_linear_process_initialization(determine_function, parameter_at_index, projection_functions);
  }

  mcrl2::lps::specification new_spec = mcrl2::lps::specification(m_data_specification, m_action_label_list, m_glob_vars, new_lps, new_init);

  assert(is_well_typed(new_spec));

  return new_spec;
}