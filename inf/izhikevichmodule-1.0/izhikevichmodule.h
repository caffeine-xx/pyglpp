/*
 *  izhikevichmodule.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2008 by
 *  The NEST Initiative
 *
 *  See the file AUTHORS for details.
 *
 *  Permission is granted to compile and modify
 *  this file for non-commercial use.
 *  See the file LICENSE for details.
 *
 */

#ifndef IZHIKEVICHMODULE_H
#define IZHIKEVICHMODULE_H

#include "dynmodule.h"
#include "slifunction.h"

namespace nest
{
  class Network;
}

namespace icnest {

/**
 * Implementation of Izhikevich neuron model.
 *
 * Mark Wildie (mwild@doc.ic.ac.uk) Elena Phoka (e.phoka07@imperial.ac.uk)
 */

class IzhikevichModule : public DynModule
{
public:

  // Interface functions ------------------------------------------

  /**
   * @note The constructor registers the module with the dynamic loader.
   *       Initialization proper is performed by the init() method.
   */
  IzhikevichModule();

  /**
   * @note The destructor does not do much in modules. Proper "downrigging"
   *       is the responsibility of the unregister() method.
   */
  ~IzhikevichModule();

  /**
   * Initialize module by registering models with the network.
   * @param SLIInterpreter* SLI interpreter
   * @param nest::Network*  Network with which to register models
   * @note  Parameter Network is needed for historical compatibility
   *        only.
   */
  void init(SLIInterpreter*, nest::Network*);

  /**
   * Return the name of your model.
   */
  const std::string name(void) const;

  /**
   * Return the name of a sli file to execute when mymodule is loaded.
   * This mechanism can be used to define SLI commands associated with your
   * module, in particular, set up type tries for functions you have defined.
   */
  const std::string commandstring(void) const;

  };
} // namespace icnest

#endif
