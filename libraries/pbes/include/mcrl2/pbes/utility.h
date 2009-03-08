// Author(s): Jan Friso Groote
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/utility.h
/// \brief Utility functions for the pbes library.

#ifndef MCRL2_PBES_UTILITY_H
#define MCRL2_PBES_UTILITY_H

#include "mcrl2/pbes/pbes.h"
#include "mcrl2/modal_formula/mucalculus.h"
#include "mcrl2/lps/specification.h"
#include "mcrl2/new_data/data.h"
#include "mcrl2/new_data/utility.h"
#include "mcrl2/new_data/detail/sort_utility.h"
#include "mcrl2/new_data/sort_expression.h"
#include "mcrl2/atermpp/algorithm.h"

// JFG:
#include "mcrl2/new_data/rewriter.h"

namespace mcrl2 {

namespace pbes_system {

// Code below is added by JFG. At certain points an L is added to distinguish
// the function from the original version. The code needs to be cleaned up
// to conform to the standard use in the library.


static new_data::data_expression initialize_internal_true(new_data::data_expression &t,new_data::detail::Rewriter *r)
{
  t=(new_data::data_expression)r->toRewriteFormat(new_data::sort_bool_::true_());
  ATprotect((ATerm*)(&t));
  return t;
}

static bool is_true_in_internal_rewrite_format(new_data::data_expression d,new_data::detail::Rewriter *rewriter)
{ static new_data::data_expression internal_true=initialize_internal_true(internal_true,rewriter);
  return d==internal_true;
}


static new_data::data_expression initialize_internal_false(new_data::data_expression &t,new_data::detail::Rewriter *r)
{
  t=(new_data::data_expression)r->toRewriteFormat(new_data::sort_bool_::false_());
  ATprotect((ATerm*)(&t));
  return t;
}

static bool is_false_in_internal_rewrite_format(new_data::data_expression d,new_data::detail::Rewriter *rewriter)
{ static new_data::data_expression internal_false=initialize_internal_false(internal_false,rewriter);
  return d==internal_false;
}


struct compare_variableL
{
  atermpp::aterm v;

  compare_variableL(new_data::variable v_)
    : v(atermpp::aterm_appl(v_))
  {}

  bool operator()(atermpp::aterm t) const
  {
    return v == t;
  }
};

//  variable v occurs in l.
//
static bool occurs_in_varL(atermpp::aterm_appl l, new_data::variable v)
{
  return find_if(l, compare_variableL(v)) != atermpp::aterm();
}


///\brief returns a pbes expression which has been rewritten en where unnecessary quantifications have been removed. The result can optionally be translated to internal rewrite format.
/// This function simplifies all data expressions in p by applying the rewriter to it.
/// Data expressions that are true or false are translated to the pbes expressions true and false.
/// Quantified variables that do not occur in the body are removed.
/// Conjunctions and disjunctions of which one of the arguments is true or false are simplified.
/// If the boolean yield_internal_rewriter_format is set, the data expressions in the resulting
/// pbes expressions are translated into the internal format belonging to the rewriter. The
/// advantage of this is that the rewriter does not have to translate the data expressions
/// to internal format the next time the rewriter is applied to it. This is for instance useful
/// in the tool pbes2bool (or pbes2bes) where pbes expressions must iteratively be rewritten.

inline pbes_expression pbes_expression_rewrite_and_simplify(
                   pbes_expression p,
                   new_data::detail::Rewriter *rewriter,
                   const bool yield_internal_rewriter_format=false)
{
  using namespace pbes_system::pbes_expr;
  using namespace pbes_system::accessors;
  pbes_expression result;

  if (is_pbes_true(p))
  { // p is True
    result = p;
  }
  else if (is_pbes_false(p))
  { // p is False
    result = p;
  }
  else if (is_pbes_and(p))
  { // p = and(left, right)
    //Rewrite left and right as far as possible
    pbes_expression l = pbes_expression_rewrite_and_simplify(left(p),rewriter,yield_internal_rewriter_format);
    if (is_pbes_false(l))
    { result = false_();
    }
    else
    { pbes_expression r = pbes_expression_rewrite_and_simplify(right(p),rewriter,yield_internal_rewriter_format);
      //Options for left and right
      if (is_pbes_false(r))
      { result = false_();
      }
      else if (is_pbes_true(l))
      { result = r;
      }
      else if (is_pbes_true(r))
      { result = l;
      }
      else result = and_(l,r);
    }
  }
  else if (is_pbes_or(p))
  { // p = or(left, right)
    //Rewrite left and right as far as possible
    pbes_expression l = pbes_expression_rewrite_and_simplify(left(p),rewriter,yield_internal_rewriter_format);
    if (is_pbes_true(l))
    { result = true_();
    }
    else
    { pbes_expression r = pbes_expression_rewrite_and_simplify(right(p),rewriter,yield_internal_rewriter_format);
      if (is_pbes_true(r))
      { result = true_();
      }
      else if (is_pbes_false(l))
      { result = r;
      }
      else if (is_pbes_false(r))
      { result = l;
      }
      else result = or_(l,r);
    }
  }
  else if (is_pbes_forall(p))
  { // p = forall(new_data::data_expression_list, pbes_expression)
    new_data::variable_list data_vars = var(p);
    pbes_expression expr = pbes_expression_rewrite_and_simplify(arg(p),rewriter,yield_internal_rewriter_format);
    //Remove data_vars which do not occur in expr
    new_data::variable_list occurred_data_vars;
    for (new_data::variable_list::iterator i = data_vars.begin(); i != data_vars.end(); i++)
    {
      if (occurs_in_varL(expr, *i)) // The var occurs in expr
      {
        occurred_data_vars = push_front(occurred_data_vars, *i);
      }
    }

    // If no data_vars
    if (occurred_data_vars.empty())
    { result = expr;
    }
    else
    { result=pbes_expr::forall(occurred_data_vars,expr);
    }
  }
  else if (is_pbes_exists(p))
  { // p = exists(new_data::data_expression_list, pbes_expression)
    new_data::variable_list data_vars = var(p);
    pbes_expression expr = pbes_expression_rewrite_and_simplify(arg(p),rewriter,yield_internal_rewriter_format);
    //Remove data_vars which does not occur in expr
    new_data::variable_list occurred_data_vars;
    for (new_data::variable_list::iterator i = data_vars.begin(); i != data_vars.end(); i++)
    {
      if (occurs_in_varL(expr, *i)) // The var occurs in expr
      {
         occurred_data_vars = atermpp::push_front(occurred_data_vars, *i);
      }
    }

    //If no data_vars remaining
    if (occurred_data_vars.empty())
      result = expr;
    else
    { result=pbes_expr::exists(occurred_data_vars,expr);
    }
  }
  else if (is_propositional_variable_instantiation(p))
  { // p is a propositional variable
    propositional_variable_instantiation propvar = p;
    core::identifier_string name = propvar.name();
    new_data::data_expression_list current_parameters(propvar.parameters());
    new_data::data_expression_list parameters;
    if (yield_internal_rewriter_format)
    {
      for( new_data::data_expression_list::const_iterator l=current_parameters.begin();
           l != current_parameters.end(); ++l)
      {
        parameters = atermpp::push_front(parameters, new_data::data_expression(rewriter->rewriteInternal(
                                rewriter->toRewriteFormat(*l))));
      }
      parameters = atermpp::reverse(parameters);
    }
    else
    {
      atermpp::term_list< new_data::data_expression > expressions(rewriter->rewriteList(
        atermpp::term_list< new_data::data_expression >(current_parameters.begin(), current_parameters.end())));
      parameters=new_data::data_expression_list(expressions.begin(), expressions.end());
    }
    result = pbes_expression(propositional_variable_instantiation(name, parameters));
  }
  else
  { // p is a new_data::data_expression

    if (yield_internal_rewriter_format)
    {
    new_data::data_expression d = (new_data::data_expression)rewriter->rewriteInternal(rewriter->toRewriteFormat(p));
      if (is_true_in_internal_rewrite_format(d,rewriter))
      { result = true_();
      }
      else if (is_false_in_internal_rewrite_format(d,rewriter))
      { result = false_();
      }
      else
      { result = d;
      }
    }
    else
    {
      new_data::data_expression d(rewriter->rewrite(p));
      if (d == new_data::sort_bool_::true_())
      { result = true_();
      }
      else if (d == new_data::sort_bool_::false_())
      { result = false_();
      }
      else
      { result = d;
      }
    }
  }

  return result;
}

/// \brief gives the rank of a propositional_variable_instantiation. It is assumed that the variable
/// occurs in the pbes_specification.

template <typename Container>
inline unsigned long get_rank(const propositional_variable_instantiation current_variable_instantiation,
                              pbes<Container> pbes_spec)
{
  unsigned long rank=0;
  Container eqsys = pbes_spec.equations();
  fixpoint_symbol current_fixpoint_symbol=eqsys.begin()->symbol();
  for (typename Container::iterator eqi = eqsys.begin(); eqi != eqsys.end(); eqi++)
  {
    if (eqi->variable().name()==current_variable_instantiation.name())
    { return rank;
    }
    if (eqi->symbol()!=current_fixpoint_symbol)
    { current_fixpoint_symbol=eqi->symbol();
      rank=rank+1;
    }
  }
  assert(0); // It is assumed that the current_variable_instantiation occurs in the pbes_spec.
  return 0;
}


/// \brief gives the fixed point symbol of a propositional_variable_instantiation. It is assumed that the variable
/// occurs in the pbes_specification. Returns false if the symbol is mu, and true if the symbol in nu.

template <typename Container>
inline bool get_fixpoint_symbol(const propositional_variable_instantiation current_variable_instantiation,
                                pbes<Container> pbes_spec)
{
  Container eqsys = pbes_spec.equations();
  for (typename Container::iterator eqi = eqsys.begin(); eqi != eqsys.end(); eqi++)
  {
    if (eqi->variable().name()==current_variable_instantiation.name())
    { return (is_nu(eqi->symbol()));
    }
  }
  assert(0); // It is assumed that the variable instantiation occurs in the PBES.
  return false;
}


/// \brief Gives the instantiated right hand side for a propositional_variable_instantiation

template <typename Container>
inline pbes_expression give_the_instantiated_rhs(
                   const propositional_variable_instantiation current_variable_instantiation,
                   pbes<Container> pbes_spec,
                   new_data::detail::Rewriter *rewriter,
                   const bool use_internal_rewriter_format=false)
{
  Container eqsys = pbes_spec.equations();

  // Fill the pbes_equations table

  assert(eqsys.size()>0); // There should be at least one equation
  fixpoint_symbol current_fixpoint_symbol=eqsys.begin()->symbol();

  pbes_equation current_pbeq;
  for (typename Container::iterator eqi = eqsys.begin(); eqi != eqsys.end(); eqi++)
  {
    if (current_variable_instantiation.name()==eqi->variable().name())
    { current_pbeq=pbes_equation(eqi->symbol(),eqi->variable(),eqi->formula());
      break;
    }
  }

  new_data::data_expression_list::iterator elist=current_variable_instantiation.parameters().begin();

  for(new_data::variable_list::iterator vlist=current_pbeq.variable().parameters().begin() ;
               vlist!=current_pbeq.variable().parameters().end() ; vlist++)
  {
    assert(elist!=current_variable_instantiation.parameters().end());

    if (use_internal_rewriter_format)
    { rewriter->setSubstitutionInternal(*vlist,(atermpp::aterm)*elist);
    }
    else
    {
      rewriter->setSubstitution(*vlist,*elist);
    }
    elist++;
  }
  assert(elist==current_variable_instantiation.parameters().end());
  return  pbes_expression_substitute_and_rewrite(
                                current_pbeq.formula(),
                                pbes_spec.data(),
                                rewriter,
                                use_internal_rewriter_format);
}




/**************************************************************************************************/

// given a pbes expression of the form p1 && p2 && .... && pn, this will yield a
// set of the form { p1,p2, ..., pn}, assuming that pi does not have a && as main
// function symbol.

static void distribute_and(const pbes_expression &expr,atermpp::set < pbes_expression> &conjunction_set)
{ /* distribute the conjuncts of expr over the conjunction_set */
  if (pbes_expr::is_pbes_and(expr))
  { distribute_and(accessors::left(expr),conjunction_set);
    distribute_and(accessors::right(expr),conjunction_set);
  }
  else
  { conjunction_set.insert(expr);
  }
}

// given a set { p1, p2, ... , pn}, the function below yields an expression
// of the form p1 && ... && pn.

static pbes_expression make_conjunction(const atermpp::set < pbes_expression> &conjunction_set)
{ pbes_expression t=pbes_expr::true_();

  for(atermpp::set < pbes_expression>::const_iterator i=conjunction_set.begin();
          i!=conjunction_set.end() ; i++)
  { if (pbes_expr::is_pbes_true(t))
    { t=*i;
    }
    else
    { t=pbes_expr::and_(*i,t);
    }
  }
  return t;
}

// see distribute_and.

static void distribute_or(const pbes_expression &expr,atermpp::set < pbes_expression> &disjunction_set)
{ /* distribute the conjuncts of expr over the conjunction_set */
  if (pbes_expr::is_pbes_or(expr))
  { distribute_or(accessors::left(expr),disjunction_set);
    distribute_or(accessors::right(expr),disjunction_set);
  }
  else
  { disjunction_set.insert(expr);
  }
}

// see make_conjunction.

static pbes_expression make_disjunction(const atermpp::set < pbes_expression> &disjunction_set)
{ pbes_expression t=pbes_expr::false_();

  for(atermpp::set < pbes_expression>::const_iterator i=disjunction_set.begin();
          i!=disjunction_set.end() ; i++)
  {
    // std::cerr << "DISJUNCTION " << pp(*i) << "\n";
    if (pbes_expr::is_pbes_false(t))
    { t=*i;
    }
    else
    { t=pbes_expr::or_(*i,t);
    }
  }
  return t;
}

///\brief Substitute in p, remove quantifiers and return the result after rewriting.
/// Given a substitution, which is prepared within the rewriter, this function first
/// applies this substitution. Then it will attempt to eliminate all quantifiers.
/// I.e. if S is a constructorsort (which means that there exists at least one constructor
/// with S as target sort), then an expression of the form forall x:S.p(x) is replaced
/// by forall x1..xn.p(f1(x1..xn) &&  forall x1..xm.p(f2(x1..xm)) && .... for all
/// \brief Constructors fi with S as targetsort (this is done similarly for the exists).
/// if the constructors are constants (i.e. n=0, m=0, etc.), no new quantifications
/// will be generated. And if the expressions are not constant, expressions such as
/// p(f1(x1..xn)) are simplified, which can lead to the situation that certain
/// variables x1..xn do not occur anymore. In that case the quantors can be removed
/// also. The function pbes_expression_substitute_and_rewrite will continue substituting
/// \brief Constructors for quantified variables until there are no variables left, or
/// until there are only quantifications over non constructor sorts. In the last case,
/// the function will throw an exception. For every 100 new variables being used
/// in new quantifications, a message is printed.
///   The data_specification is needed to determine the constructors of a certain
/// sort. The rewriter is used to perform the rewriting steps. The rewriter can
/// either work on internal rewriting format, in which case all data expressions in
/// the pbes are already in internal rewriting format. In this case the data expressions
/// in the result are also in internal format. The use of internal format saves
/// a lot of internal compilation time.

inline pbes_expression pbes_expression_substitute_and_rewrite(
                   const pbes_expression &p,
                   const new_data::data_specification &data,
                   new_data::detail::Rewriter *rewriter,
                   const bool use_internal_rewrite_format)
{
  // std::cerr << "SUBSTANDREWR " << pp(p) << "\n";
  using namespace pbes_system::pbes_expr;
  using namespace pbes_system::accessors;
  pbes_expression result;

  if (is_pbes_and(p))
  { // p = and(left, right)
    //Rewrite left and right as far as possible
    pbes_expression l = pbes_expression_substitute_and_rewrite(left(p),
                               data, rewriter,use_internal_rewrite_format);
    if (is_pbes_false(l))
    { result = false_();
    }
    else
    { pbes_expression r = pbes_expression_substitute_and_rewrite(right(p),
                 data, rewriter,use_internal_rewrite_format);
      //Options for left and right
      if (is_pbes_false(r))
      { result = false_();
      }
      else if (is_pbes_true(l))
      { result = r;
      }
      else if (is_pbes_true(r))
      { result = l;
      }
      else result = and_(l,r);
    }
  }
  else if (is_pbes_or(p))
  { // p = or(left, right)
    //Rewrite left and right as far as possible

    // std::cerr << "SUB&REWR OR: " << pp(p) << "\n";
    pbes_expression l = pbes_expression_substitute_and_rewrite(left(p),
                 data, rewriter,use_internal_rewrite_format);
    // std::cerr << "SUB&REWR OR LEFT: " << pp(left) << "\n";
    if (is_pbes_true(l))
    { result = true_();
    }
    else
    { pbes_expression r = pbes_expression_substitute_and_rewrite(right(p),
                 data, rewriter,use_internal_rewrite_format);
      // std::cerr << "SUB&REWR OR RIGHT: " << pp(right) << "\n";
      if (is_pbes_true(r))
      { result = true_();
      }
      else if (is_pbes_false(l))
      { result = r;
      }
      else if (is_pbes_false(r))
      { result = l;
      }
      else result = or_(l,r);
    }
   // std::cerr << "SUB&REWR OR RESULT: " << pp(result) << "\n";
  }
  else if (is_pbes_true(p))
  { // p is True
    result = p;
  }
  else if (is_pbes_false(p))
  { // p is False
    result = p;
  }
  else if (is_pbes_forall(p))
  {

    new_data::variable_list data_vars = var(p);
    pbes_expression expr = pbes_expression_substitute_and_rewrite(arg(p), data, rewriter,use_internal_rewrite_format);

    // If no data_vars
    if (data_vars.empty())
    {
      result = expr;
    }
    else
    { /* Replace the quantified variables of constructor sorts by constructors.
         E.g. forall x:Nat.phi(x) is replaced by phi(0) && forall x':Nat.phi(successor(x')),
         assuming 0 and successor are the constructors of Nat  (which is btw. not the case
         in de data-implementation of mCRL2).  Simplify the resulting expressions. */

      new_data::fresh_variable_generator variable_generator;
      unsigned int no_variables=0;
      variable_generator.set_context(expr);
      variable_generator.set_hint("x");
      new_data::variable_list new_data_vars;
      atermpp::set < pbes_expression > conjunction_set;
      distribute_and(expr,conjunction_set);
      bool constructor_sorts_found=true;
      for( ; constructor_sorts_found ; )
      { constructor_sorts_found=false;
        for (new_data::variable_list::iterator i = data_vars.begin(); i != data_vars.end(); i++)
        {
          if (!new_data::detail::is_constructorsort(i->sort(),data))
          { /* The sort of variable i is not a constructor sort.  */
             new_data_vars = atermpp::push_front(new_data_vars, *i);
          }
          else
          {
            atermpp::set <pbes_expression> new_conjunction_set;
            for(atermpp::set <pbes_expression >::iterator t=conjunction_set.begin() ;
                         t!=conjunction_set.end() ; t++)
            {
              if (!occurs_in_varL(*t,*i))
              { new_conjunction_set.insert(*t);
              }
              else
              {
                for (new_data::data_specification::constructors_const_range f(data.constructors(i->sort())); !f.empty(); f.advance_begin(1))
                {
                  boost::iterator_range< new_data::sort_expression_list::const_iterator > dsorts;
                  if (f.front().sort().is_function_sort())
                  {
                    new_data::function_sort sa=f.front().sort();
                    assert(!sa.codomain().is_function_sort()); // In case the function f has a sort A->(B->C),
                                                           // then the function below does not work correctly.
                                                           // This code must be replaced by enumerator code,
                                                           // developed by Wieger.
                    dsorts=sa.domain();
                  }
                  // else dsorts is empty.

                  // XXXXXXXXXXXXXX argument_sorts(), result_sort()  =source(f->sort());
                  new_data::variable_list function_arguments;
                  for( new_data::sort_expression_list::const_iterator s=dsorts.begin() ;
                       s!=dsorts.end() ; ++s )
                  { variable_generator.set_sort(*s);
                    constructor_sorts_found=constructor_sorts_found || new_data::detail::is_constructorsort(*s,data);
                    new_data::variable new_variable=variable_generator();
                    ++no_variables;
                    if ((no_variables % 100)==0)
                    { std::cerr << "Used " << no_variables << " variables when eliminating universal quantifier\n";
                      if (!use_internal_rewrite_format)
                      { std::cerr << "Vars: " << mcrl2::new_data::pp(data_vars) << "\nExpression: " << mcrl2::core::pp(*t) << std::endl;
                      }
                    }
                    new_data_vars = atermpp::push_front(new_data_vars, new_variable);
                    function_arguments = atermpp::push_front(function_arguments, new_variable);
                  }
                  pbes_expression d(core::detail::gsMakeDataApplList(f.front(),
                     atermpp::term_list< new_data::variable >(function_arguments.begin(), function_arguments.end())));
                  rewriter->setSubstitution(*i,d);
                  pbes_expression r(pbes_expression_substitute_and_rewrite(*t,data,rewriter,use_internal_rewrite_format));
                  rewriter->clearSubstitution(*i);
                  if (pbes_expr::is_pbes_false(r)) /* the resulting expression is false, so we can terminate */
                  {
                    return pbes_expr::false_();
                  }
                  else
                  {
                    new_conjunction_set.insert(r);
                  }
                }
              }
            }
            conjunction_set=new_conjunction_set;
          }
        }
        data_vars=new_data_vars;
        new_data_vars=new_data::variable_list();
      }

      if (!new_data_vars.empty())
      {
        std::string message("Cannot eliminate universal quantifiers of variables ");

        message.append(new_data::pp(new_data_vars));

        if (!use_internal_rewrite_format)
        {
          message.append(" in ").append(mcrl2::core::pp(p));
        }

        throw mcrl2::runtime_error(message);
      }
      result=make_conjunction(conjunction_set);
    }
  }
  else if (is_pbes_exists(p))
  {
    // std::cerr << "EXISTS_: " << pp(p) << "\n";
    new_data::variable_list data_vars = var(p);
    pbes_expression expr = pbes_expression_substitute_and_rewrite(arg(p), data, rewriter,use_internal_rewrite_format);
    // std::cerr << "REWRITTEN EXPR " << pp(expr) << "\n";
    // If no data_vars
    if (data_vars.empty())
    {
      result = expr;
    }
    else
    { /* Replace the quantified variables of constructor sorts by constructors.
         E.g. forall x:Nat.phi(x) is replaced by phi(0) || forall x':Nat.phi(successor(x')),
         assuming 0 and successor are the constructors of Nat  (which is btw. not the case
         in de data-implementation of mCRL2).  Simplify the resulting expressions. */

      new_data::fresh_variable_generator variable_generator;
      unsigned int no_variables=0;
      variable_generator.set_context(expr);
      variable_generator.set_hint("x");
      new_data::variable_list new_data_vars;
      atermpp::set < pbes_expression > disjunction_set;
      distribute_or(expr,disjunction_set);
      bool constructor_sorts_found=true;
      for( ; constructor_sorts_found ; )
      { constructor_sorts_found=false;
        for (new_data::variable_list::iterator i = data_vars.begin(); i != data_vars.end(); i++)
        {
          if (!new_data::detail::is_constructorsort(i->sort(),data))
          { /* The sort of variable i is not a constructor sort.  */
             new_data_vars = atermpp::push_front(new_data_vars, *i);
          }
          else
          {
            atermpp::set <pbes_expression> new_disjunction_set;
            for(atermpp::set <pbes_expression >::iterator t=disjunction_set.begin() ;
                         t!=disjunction_set.end() ; t++)
            {
              if (!occurs_in_varL(*t,*i))
              { new_disjunction_set.insert(*t);
              }
              else
              {
                for (new_data::data_specification::constructors_const_range f(data.constructors(i->sort())); !f.empty(); f.advance_begin(1))
                {
                  boost::iterator_range< new_data::sort_expression_list::const_iterator > dsorts;
                  if (f.front().sort().is_function_sort())
                  {
                    new_data::function_sort sa=f.front().sort();
                    assert(!sa.codomain().is_function_sort()); // In case the function f has a sort A->(B->C),
                                                           // then the function below does not work correctly.
                                                           // This code must be replaced by enumerator code,
                                                           // developed by Wieger.
                    dsorts=sa.domain();
                  }
                  // else dsorts is empty.

                  // std::cerr << "Function " << f->name() << " Domain sorts " << dsorts << std::endl;
                  new_data::variable_list function_arguments;
                  for( new_data::sort_expression_list::const_iterator s=dsorts.begin() ;
                       s!=dsorts.end() ; ++s )
                  { variable_generator.set_sort(*s);
                    constructor_sorts_found=constructor_sorts_found || new_data::detail::is_constructorsort(*s,data);
                    new_data::variable new_variable=variable_generator();
                    ++no_variables;
                    if ((no_variables % 100)==0)
                    { std::cerr << "Used " << no_variables << " variables when eliminating existential quantifier\n";
                      if (!use_internal_rewrite_format)
                      { std::cerr << "Vars: " << mcrl2::new_data::pp(data_vars) << "\nExpression: " << mcrl2::core::pp(*t) << std::endl;
                      }
                    }
                    new_data_vars = atermpp::push_front(new_data_vars, new_variable);
                    function_arguments = atermpp::push_front(function_arguments, new_variable);
                  }
                  pbes_expression d(core::detail::gsMakeDataApplList(f.front(),atermpp::term_list< new_data::variable >(function_arguments.begin(), function_arguments.end())));
                  rewriter->setSubstitution(*i,d);
                  // std::cerr << "SETVARIABLE " << pp(*i) << ":=" << pp(d) << "\n";
                  pbes_expression r(pbes_expression_substitute_and_rewrite(*t,data,rewriter,use_internal_rewrite_format));
                  rewriter->clearSubstitution(*i);
                  if (pbes_expr::is_pbes_true(r)) /* the resulting expression is true, so we can terminate */
                  { // std::cerr << "Return true\n";
                    return pbes_expr::true_();
                  }
                  else
                  { new_disjunction_set.insert(r);
                  }
                }
              }
            }
            disjunction_set=new_disjunction_set;
          }
        }
        data_vars=new_data_vars;
        new_data_vars=new_data::variable_list();
      }

      if (!new_data_vars.empty())
      {
        std::string message("Cannot eliminate existential quantifiers of variables ");

        message.append(new_data::pp(new_data_vars));

        if (!use_internal_rewrite_format)
        {
          message.append(" in ").append(mcrl2::core::pp(p));
        }

        throw mcrl2::runtime_error(message);
      }
      result=make_disjunction(disjunction_set);
    }
    // std::cerr << "Return result " << pp(result) << "\n";
  }
  else if (is_propositional_variable_instantiation(p))
  { // p is a propositional variable
    propositional_variable_instantiation propvar = p;
    core::identifier_string name = propvar.name();
    new_data::data_expression_list current_parameters(propvar.parameters());
    new_data::data_expression_list parameters;
    if (use_internal_rewrite_format)
    {
      atermpp::term_list< new_data::data_expression > expressions(rewriter->rewriteInternalList(
        atermpp::term_list< new_data::data_expression >(current_parameters.begin(), current_parameters.end())));
      parameters=new_data::data_expression_list(expressions.begin(), expressions.end());
    }
    else
    {
      atermpp::term_list< new_data::data_expression > expressions(rewriter->rewriteList(
        atermpp::term_list< new_data::data_expression >(current_parameters.begin(), current_parameters.end())));
      parameters=new_data::data_expression_list(expressions.begin(), expressions.end());
    }
    result = pbes_expression(propositional_variable_instantiation(name, parameters));
  }
  else
  { // p is a new_data::data_expression
    if (use_internal_rewrite_format)
    {
      new_data::data_expression d = (new_data::data_expression)rewriter->rewriteInternal((atermpp::aterm)p);
      if (is_true_in_internal_rewrite_format(d,rewriter))
      { result = pbes_expr::true_();
      }
      else if (is_false_in_internal_rewrite_format(d,rewriter))
      { result = pbes_expr::false_();
      }
      else
      {
        result = d;
      }
    }
    else
    {
      new_data::data_expression d(rewriter->rewrite(p));
      // std::cerr << "REWRITE DATA EXPR: " << pp(p) << " ==> " << pp(d) << "\n";
      // ATfprintf(stderr,"FORMAT: %t\n",(ATermAppl)(d));
      if (d == new_data::sort_bool_::true_())
      { result = pbes_expr::true_();
      }
      else if (d == new_data::sort_bool_::false_())
      { result = pbes_expr::false_();
      }
      else
      {
        result = d;
      }
    }
  }

  return result;
}


} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_UTILITY_H
