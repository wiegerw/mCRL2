// Author(s): Wieger Wesselink
// Copyright: see the accompanying file COPYING or copy at
// https://svn.win.tue.nl/trac/MCRL2/browser/trunk/COPYING
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
/// \file mcrl2/pbes/detail/term_traits_optimized.h
/// \brief add your file description here.

#ifndef MCRL2_PBES_DETAIL_TERM_TRAITS_OPTIMIZED_H
#define MCRL2_PBES_DETAIL_TERM_TRAITS_OPTIMIZED_H

#include "mcrl2/pbes/pbes_expression.h"
#include "mcrl2/utilities/optimized_boolean_operators.h"

namespace mcrl2 {

namespace core {

/// \brief Contains type information for terms.
template <typename T>
struct term_traits_optimized
{
};

/// \brief Contains type information for pbes expressions.
template <>
struct term_traits_optimized<pbes_system::pbes_expression>: public core::term_traits<pbes_system::pbes_expression>
{
  typedef core::term_traits<pbes_system::pbes_expression> super;

  static inline
  term_type not_(const term_type& x)
  {
    return utilities::optimized_not(x);
  }

  static inline
  term_type and_(const term_type& x, const term_type& y)
  {
    return utilities::optimized_and(x, y);
  }

  static inline
  term_type or_(const term_type& x, const term_type& y)
  {
    return utilities::optimized_or(x, y);
  }

  static inline
  term_type imp(const term_type& x, const term_type& y)
  {
    return utilities::optimized_imp(x, y);
  }

  static inline
  term_type forall(const variable_sequence_type& d, const term_type& x)
  {
    return utilities::optimized_forall(d, x);
  }

  static inline
  term_type exists(const variable_sequence_type& d, const term_type& x)
  {
    return utilities::optimized_exists(d, x);
  }
};

} // namespace core

} // namespace mcrl2

#endif // MCRL2_PBES_DETAIL_TERM_TRAITS_OPTIMIZED_H
