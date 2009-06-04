/*
 *  izhikevichmodule.cpp
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

#include "config.h"
#include "network.h"
#include "model.h"
#include "dynamicloader.h"
#include "genericmodel.h"
#include "generic_connector.h"
#include "booldatum.h"
#include "integerdatum.h"
#include "tokenarray.h"
#include "exceptions.h"
#include "sliexceptions.h"
#include "nestmodule.h"

#include "izhikevichmodule.h"
#include "izhikevich_simple.h"
#include "izhikevich_large.h"

// -- Interface to dynamic module loader ---------------------------------------

icnest::IzhikevichModule izhikevichmodule_LTX_mod;

// -- DynModule functions ------------------------------------------------------

icnest::IzhikevichModule::IzhikevichModule()
  {
#ifdef LINKED_MODULE
     nest::DynamicLoaderModule::registerLinkedModule(this);
#endif
   }

icnest::IzhikevichModule::~IzhikevichModule()
   {
   }

const std::string icnest::IzhikevichModule::name(void) const
   {
     return std::string("Izhikevich Module"); // Return name of the module
   }

  //-------------------------------------------------------------------------------------

  void icnest::IzhikevichModule::init(SLIInterpreter *i, nest::Network*)
  {
    nest::register_model<nest::izhikevich_simple>(nest::NestModule::get_network(),
    		"izhikevich_simple");
    nest::register_model<nest::izhikevich_large>(nest::NestModule::get_network(),
    		"izhikevich_large");
  }

  const std::string icnest::IzhikevichModule::commandstring(void) const
  {
    return std::string(
      ""
      );
  }


