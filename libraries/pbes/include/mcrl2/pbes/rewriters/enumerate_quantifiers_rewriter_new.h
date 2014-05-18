// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/rewriters/enumerate_quantifiers_rewriter.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_REWRITERS_ENUMERATE_QUANTIFIERS_REWRITER_NEW_H
#define MCRL2_PBES_REWRITERS_ENUMERATE_QUANTIFIERS_REWRITER_NEW_H

#include <numeric>
#include <set>
#include <utility>
#include <deque>
#include <sstream>
#include <vector>
#include "mcrl2/core/detail/print_utility.h"
#include "mcrl2/data/data_specification.h"
#include "mcrl2/data/detail/split_finite_variables.h"
#include "mcrl2/pbes/rewriters/simplify_rewriter.h"
#include "mcrl2/pbes/enumerator.h"
#include "mcrl2/utilities/optimized_boolean_operators.h"
#include "mcrl2/utilities/detail/join.h"
#include "mcrl2/utilities/optimized_boolean_operators.h"

namespace mcrl2 {

namespace pbes_system {

namespace detail {

// Simplifying PBES rewriter that eliminates quantifiers using enumeration.
/// \param MutableSubstitution This must be a MapSubstitution.
template <typename Derived, typename DataRewriter, typename MutableSubstitution>
struct enumerate_quantifiers_builder: public simplify_data_rewriter_builder<Derived, DataRewriter, MutableSubstitution>
{
  typedef simplify_data_rewriter_builder<Derived, DataRewriter, MutableSubstitution> super;
  typedef enumerate_quantifiers_builder<Derived, DataRewriter, MutableSubstitution> self;
  typedef core::term_traits<pbes_expression> tr;

  using super::enter;
  using super::leave;
  using super::operator();
  using super::sigma;
  using super::R;

  const data::data_specification& m_dataspec;

  /// If true, quantifier variables of infinite sort are enumerated.
  bool m_enumerate_infinite_sorts;

  /// The enumerator
  enumerator_algorithm<self, MutableSubstitution> E;

  /// \brief Constructor.
  /// \param r A data rewriter
  /// \param dataspec A data specification
  /// \param enumerate_infinite_sorts If true, quantifier variables of infinite sort are enumerated as well
  enumerate_quantifiers_builder(const data::rewriter& R, MutableSubstitution& sigma, const data::data_specification& dataspec, bool enumerate_infinite_sorts = true)
    : super(R, sigma), m_dataspec(dataspec), m_enumerate_infinite_sorts(enumerate_infinite_sorts), E(*this, sigma, m_dataspec)
  { }

  Derived& derived()
  {
    return static_cast<Derived&>(*this);
  }

  std::vector<data::data_expression> undo_substitution(const data::variable_list& v)
  {
    std::vector<data::data_expression> result;
    for (auto i = v.begin(); i != v.end(); ++i)
    {
      result.push_back(sigma(*i));
      sigma[*i] = *i;
    }
    return result;
  }

  void redo_substitution(const data::variable_list& v, const std::vector<data::data_expression>& undo)
  {
    assert(v.size() == undo.size());
    auto i = v.begin();
    auto j = undo.begin();
    while (i != v.end())
    {
      sigma[*i++] = *j++;
    }
  }

  pbes_expression enumerate_forall(const data::variable_list& v, const pbes_expression& phi)
  {
    auto undo = undo_substitution(v);
    pbes_expression result = tr::true_();
    enumerator_list P;
    P.push_back(enumerator_list_element(v, derived()(phi)));
    while (!P.empty())
    {
      pbes_expression e = E.next(P, is_not_true());
      if (e == data::undefined_data_expression())
      {
        continue;
      }
      result = utilities::optimized_and(result, e);
      if (tr::is_false(result))
      {
        break;
      }
    }
    redo_substitution(v, undo);
    return result;
  }

  pbes_expression enumerate_exists(const data::variable_list& v, const pbes_expression& phi)
  {
    auto undo = undo_substitution(v);
    pbes_expression result = tr::false_();
    enumerator_list P;
    P.push_back(enumerator_list_element(v, derived()(phi)));
    while (!P.empty())
    {
      pbes_expression e = E.next(P, is_not_false());
      if (e == data::undefined_data_expression())
      {
        continue;
      }
      result = utilities::optimized_or(result, e);
      if (tr::is_true(result))
      {
        break;
      }
    }
    redo_substitution(v, undo);
    return result;
  }

  pbes_expression operator()(const forall& x)
  {
    pbes_expression result;
    if (m_enumerate_infinite_sorts)
    {
      result = enumerate_forall(x.variables(), x.body());
    }
    else
    {
      data::variable_list finite;
      data::variable_list infinite;
      data::detail::split_finite_variables(x.variables(), m_dataspec, finite, infinite);
      if (finite.empty())
      {
        result = utilities::optimized_forall(infinite, derived()(x.body()));
      }
      else
      {
        result = enumerate_forall(finite, x.body());
        result = utilities::optimized_forall_no_empty_domain(infinite, result);
      }
    }
    return result;
  }

  pbes_expression operator()(const exists& x)
  {
    pbes_expression result;
    if (m_enumerate_infinite_sorts)
    {
      result = enumerate_exists(x.variables(), x.body());
    }
    else
    {
      data::variable_list finite;
      data::variable_list infinite;
      data::detail::split_finite_variables(x.variables(), m_dataspec, finite, infinite);
      if (finite.empty())
      {
        result = utilities::optimized_exists(infinite, derived()(x.body()));
      }
      else
      {
        result = enumerate_exists(finite, x.body());
        result = utilities::optimized_exists_no_empty_domain(infinite, result);
      }
    }
    return result;
  }

  // N.B. This function has been added to make this class operate well with the enumerator.
  pbes_expression operator()(const pbes_expression& x, MutableSubstitution&)
  {
    return derived()(x);
  }
};

/// \brief Adds special handling for conjunctions in the body of a forall and disjunctions
/// in the body of an exists. N.B. not finished yet! The enumeration of the subterms needs
/// to be done cyclically.
template <typename Derived, typename DataRewriter, typename MutableSubstitution>
struct enumerate_quantifiers_split_builder: public enumerate_quantifiers_builder<Derived, DataRewriter, MutableSubstitution>
{
  typedef enumerate_quantifiers_builder<Derived, DataRewriter, MutableSubstitution> super;
  typedef core::term_traits<pbes_expression> tr;
  using super::enter;
  using super::leave;
  using super::operator();
  using super::sigma;
  using super::enumerate_exists;
  using super::enumerate_forall;

  std::vector<pbes_expression> split_or(const pbes_expression& x) const
  {
    using namespace accessors;
    std::vector<pbes_expression> result;
    utilities::detail::split(x, std::back_insert_iterator<std::vector<pbes_expression> >(result), is_universal_or, data_left, data_right);
    return result;
  }

  std::vector<pbes_expression> split_and(const pbes_expression& x) const
  {
    using namespace accessors;
    std::vector<pbes_expression> result;
    utilities::detail::split(x, std::back_insert_iterator<std::vector<pbes_expression> >(result), is_universal_and, data_left, data_right);
    return result;
  }

  pbes_expression operator()(const forall& x)
  {
    if (is_universal_and(x.body()))
    {
      pbes_expression result = tr::true_();
      std::vector<pbes_expression> factors = split_and(x.body());
      for (auto i = factors.begin(); i != factors.end(); ++i)
      {
        result = utilities::optimized_and(result, enumerate_forall(x.variables(), *i));
        if (tr::is_false(result))
        {
          return result;
        }
      }
      return result;
    }
    return super::operator()(x);
  }

  pbes_expression operator()(const exists& x)
  {
    if (is_universal_or(x.body()))
    {
      pbes_expression result = tr::false_();
      std::vector<pbes_expression> factors = split_or(x.body());
      for (auto i = factors.begin(); i != factors.end(); ++i)
      {
        result = utilities::optimized_or(result, enumerate_exists(x.variables(), *i));
        if (tr::is_true(result))
        {
          return result;
        }
      }
      return result;
    }
    return super::operator()(x);
  }
};

template <template <class, class, class> class Builder, class DataRewriter, class MutableSubstitution>
struct apply_enumerate_builder: public Builder<apply_enumerate_builder<Builder, DataRewriter, MutableSubstitution>, DataRewriter, MutableSubstitution>
{
  typedef Builder<apply_enumerate_builder<Builder, DataRewriter, MutableSubstitution>, DataRewriter, MutableSubstitution> super;
  using super::enter;
  using super::leave;
  using super::operator();

  apply_enumerate_builder(const DataRewriter& R, MutableSubstitution& sigma, const data::data_specification& dataspec, bool enumerate_infinite_sorts)
    : super(R, sigma, dataspec, enumerate_infinite_sorts)
  {}

#ifdef BOOST_MSVC
#include "mcrl2/core/detail/builder_msvc.inc.h"
#endif
};

template <template <class, class, class> class Builder, class DataRewriter, class MutableSubstitution>
apply_enumerate_builder<Builder, DataRewriter, MutableSubstitution>
make_apply_enumerate_builder(const DataRewriter& R, MutableSubstitution& sigma, const data::data_specification& dataspec, bool enumerate_infinite_sorts)
{
  return apply_enumerate_builder<Builder, DataRewriter, MutableSubstitution>(R, sigma, dataspec, enumerate_infinite_sorts);
}

} // namespace detail

/// \brief An attempt for improving the efficiency.
struct enumerate_quantifiers_rewriter
{
  /// \brief A data rewriter
  data::rewriter m_rewriter;

  /// \brief A data specification
  data::data_specification m_dataspec;

  /// \brief If true, quantifier variables of infinite sort are enumerated.
  bool m_enumerate_infinite_sorts;

  typedef pbes_expression term_type;
  typedef data::variable variable_type;

  enumerate_quantifiers_rewriter(const data::rewriter& R, const data::data_specification& dataspec, bool enumerate_infinite_sorts = true)
    : m_rewriter(R), m_dataspec(dataspec), m_enumerate_infinite_sorts(enumerate_infinite_sorts)
  {}

  pbes_expression operator()(const pbes_expression& x) const
  {
    data::rewriter::substitution_type sigma;
    return detail::make_apply_enumerate_builder<detail::enumerate_quantifiers_builder>(m_rewriter, sigma, m_dataspec, m_enumerate_infinite_sorts)(x);
  }

  template <typename MutableSubstitution>
  pbes_expression operator()(const pbes_expression& x, MutableSubstitution& sigma) const
  {
    return detail::make_apply_enumerate_builder<detail::enumerate_quantifiers_builder>(m_rewriter, sigma, m_dataspec, m_enumerate_infinite_sorts)(x);
  }
};

} // namespace pbes_system

} // namespace mcrl2

#endif // MCRL2_PBES_REWRITERS_ENUMERATE_QUANTIFIERS_REWRITER_NEW_H