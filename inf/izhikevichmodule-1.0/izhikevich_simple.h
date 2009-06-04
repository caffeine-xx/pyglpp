/*
 *  izhikevich_simple.h
 *
 *  This file is part of NEST
 *
 *  Copyright (C) 2004-2008 by
 *  The NEST Initiative
 *
 *  See the file AUTHORS for details.
 *
 *  Permission is granted to compile and modify
 *  this file for non-commercial use.
 *  See the file LICENSE for details.
 *
 */

#ifndef IAF_IZHIKEVICH_SIMPLE_H
#define IAF_IZHIKEVICH_SIMPLE_H

#include "nest.h"
#include "event.h"
#include "archiving_node.h"
#include "ring_buffer.h"
#include "connection.h"
#include "analog_data_logger.h"

namespace nest
{
  class Network;

  /* BeginDocumentation
Name: izhikevich_simple - Izhikevich neuron model.

Description:

  izhikevich_simple is an implementation of the Izhikevich neuron model
  as described in [1], consisting of a 2D system of differential equations

    v = v + tau*(0.04*v^2+5*v+140-u+I);
    u = u + tau*a*(b*V-u);

Parameters:

  The following parameters can be set in the status dictionary.

  a        double - Time scale of membrane recovery variable u.
  b        double - Sensitivity of membrane recovery variable u.
  c        double - Fast-threshold after-spike reset value of u.
  d        double - Slow-threshold after-spike reset value of u.
  I_e      double - Constant external input current in pA.

  c1       double - Constant ODE param 1 (default 0.04).
  c2       double - Constant ODE param 2 (default 5.0).
  c3       double - Constant ODE param 3 (default 140.0).

References:
  [1] Izhikevich E. M. (2003) Simple Model of Spiking Neurons.
      IEEE Transactions on Neural Networks, 14:1569 - 1572
  [2] Izhikevich E. M. (2004) Which Model to Use for Cortical Spiking Neurons?
      IEEE Transactions on Neural Networks, 15:1063 - 1070

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, PotentialRequest

Author:  Mark Wildie (mwild@doc.ic.ac.uk)
*/


  /**
   * Izhikevich simple.
   */
  class izhikevich_simple: public Archiving_Node
  {

  public:

    typedef Node base;

    izhikevich_simple();
    izhikevich_simple(const izhikevich_simple&);

#ifndef IS_BLUEGENE
    using Node::check_connection;
#endif
    using Node::connect_sender;
    using Node::handle;

    port check_connection(Connection&, port);

    void handle(SpikeEvent &);
    void handle(CurrentEvent &);
    void handle(PotentialRequest &);

    port connect_sender(SpikeEvent&, port);
    port connect_sender(CurrentEvent&, port);
    port connect_sender(PotentialRequest&, port);

    void get_status(DictionaryDatum &) const;
    void set_status(const DictionaryDatum &);

  private:

    void init_node_(const Node& proto);
    void init_state_(const Node& proto);
    void init_buffers_();
    void calibrate();

    void update(Time const &, const long_t, const long_t);

    // ----------------------------------------------------------------

    /**
     * Independent parameters of the model.
     */
    struct Parameters_ {
      double_t a_; //!< Time scale of membrane recovery variable u.
      double_t b_; //!< Sensitivity of membrane recovery variable u.
      double_t c_; //!< Fast-threshold after-spike reset value of u.
      double_t d_; //!< Constant external input current in pA.

      double_t I_e_; //!< Constant external input current in pA.

      // allowing setting of ODE constants
      // required for some neuron behaviours e.g.
      // class 1 excitory, integrator
      double_t c1_; //!< ODE constant - default value 0.04
      double_t c2_; //!< ODE constant - default value 5.0
      double_t c3_; //!< ODE constant - default value 140.0

      Parameters_();  //!< Sets default parameter values

      void get(DictionaryDatum&) const;  //!< Store current values in dictionary
      void set(const DictionaryDatum&);  //!< Set values from dicitonary
    };

    // ----------------------------------------------------------------

    /**
     * State variables of the model.
     */
    struct State_ {

      double_t u_; //!< Membrane recovery variable
      double_t v_; //!< Membrane potential

      State_();  //!< Default initialization

      void get(DictionaryDatum&, const Parameters_&) const;
      void set(const DictionaryDatum&, const Parameters_&);
    };

    // ----------------------------------------------------------------

    /**
     * Buffers of the model.
     */
    struct Buffers_ {
      /** buffers and summs up incoming spikes/currents */
      RingBuffer spikes_;
      RingBuffer currents_;

      /**
       * Buffer for membrane potential.
       */
      AnalogDataLogger<PotentialRequest> potentials_;
    };

    // ----------------------------------------------------------------

    /**
     * Internal variables of the model.
     */
    struct Variables_ {
    };

    // ----------------------------------------------------------------

    /**
     * Instances of private data structures.
     */
    Parameters_ P_;
    State_      S_;
    Variables_  V_;
    Buffers_    B_;

};

inline
port izhikevich_simple::check_connection(Connection& c, port receptor_type)
{
  SpikeEvent e;
  e.set_sender(*this);
  c.check_event(e);
  return c.get_target()->connect_sender(e, receptor_type);
}

inline
port izhikevich_simple::connect_sender(SpikeEvent&, port receptor_type)
{
  if (receptor_type != 0)
    throw UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline
port izhikevich_simple::connect_sender(CurrentEvent&, port receptor_type)
{
  if (receptor_type != 0)
    throw UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline
port izhikevich_simple::connect_sender(PotentialRequest& pr, port receptor_type)
{
  if (receptor_type != 0)
      throw UnknownReceptorType(receptor_type, get_name());
  B_.potentials_.connect_logging_device(pr);
  return 0;
}

inline
void izhikevich_simple::get_status(DictionaryDatum &d) const
{
  P_.get(d);
  S_.get(d, P_);
  Archiving_Node::get_status(d);
}

inline
void izhikevich_simple::set_status(const DictionaryDatum &d)
{
  Parameters_ ptmp = P_;  // temporary copy in case of errors
  ptmp.set(d);                       // throws if BadProperty
  State_      stmp = S_;  // temporary copy in case of errors
  stmp.set(d, ptmp);                 // throws if BadProperty

  // We now know that (ptmp, stmp) are consistent. We do not
  // write them back to (P_, S_) before we are also sure that
  // the properties to be set in the parent class are internally
  // consistent.
  Archiving_Node::set_status(d);

  // if we get here, temporaries contain consistent set of properties
  P_ = ptmp;
  S_ = stmp;
}


} // namespace

#endif /* #ifndef IAF_IZHIKEVICH_SIMPLE_H */
