/*
 *  iaf_large.h
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

#ifndef IZHIKEVICH_LARGE_H
#define IZHIKEVICH_LARGE_H

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
Name: izhikevich_large - Izhikevich neuron model (Large-Scale Model
      of Mammalian Thalamocortical Systems).

Description:

  izhikevich_large is an implementation of the Izhikevich neuron model
  as described in the supplementary material of [1], consisting of a
  2D system of differential equations

    Cv = k(v-v_r)(v-v_t)-U+I;
    u = a*(b(v-v_r)-u);

Parameters:

  The following parameters can be set in the status dictionary.

  a        double - Time scale of membrane recovery variable u.
  b        double - Sensitivity of membrane recovery variable u.
  c        double - Fast-threshold after-spike reset value of u.
  d        double - Slow-threshold after-spike reset value of u.
  v_t      double - Instantaneous threshold potential.
  v_r      double - Resting potential.
  C        double - Membrane capacitance.
  k        double - Constant k.
  v_peak   double - Peak membrane potenial required for spiking.
  I_e      double - Constant external input current in pA.

References:
  [1] Izhikevich E. M. (2008) Large-Scale Model of Mammalian Thalamocortical Systems.
      PNAS, 105:3593-3598

Sends: SpikeEvent

Receives: SpikeEvent, CurrentEvent, PotentialRequest

Author:  Mark Wildie (mwild@doc.ic.ac.uk) Elena Phoka (e.phoka07@imperial.ac.uk)
*/

  /**
   * Izhikevich neuron.
   */
  class izhikevich_large: public Archiving_Node
  {

  public:

    typedef Node base;

    izhikevich_large();
    izhikevich_large(const izhikevich_large&);

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

      double_t vt_; //!< Instantaneous threshold potential
      double_t vr_; //!< Resting potential
      double_t C_;  //!< Membrane capacitance
      double_t k_;  //!< Constant k

      double_t v_peak_;  //!< Peak membrane potenial required for spiking

      double_t I_e_; //!< Constant external input current in pA.

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
port izhikevich_large::check_connection(Connection& c, port receptor_type)
{
  SpikeEvent e;
  e.set_sender(*this);
  c.check_event(e);
  return c.get_target()->connect_sender(e, receptor_type);
}

inline
port izhikevich_large::connect_sender(SpikeEvent&, port receptor_type)
{
  if (receptor_type != 0)
    throw UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline
port izhikevich_large::connect_sender(CurrentEvent&, port receptor_type)
{
  if (receptor_type != 0)
    throw UnknownReceptorType(receptor_type, get_name());
  return 0;
}

inline
port izhikevich_large::connect_sender(PotentialRequest& pr, port receptor_type)
{
  if (receptor_type != 0)
      throw UnknownReceptorType(receptor_type, get_name());
  B_.potentials_.connect_logging_device(pr);
  return 0;
}

inline
void izhikevich_large::get_status(DictionaryDatum &d) const
{
  P_.get(d);
  S_.get(d, P_);
  Archiving_Node::get_status(d);
}

inline
void izhikevich_large::set_status(const DictionaryDatum &d)
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

#endif /* #ifndef IZHIKEVICH_LARGE_H */
