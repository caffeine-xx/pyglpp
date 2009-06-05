#include "exceptions.h"
#include "izhikevich_large.h"
#include "network.h"
#include "dict.h"
#include "integerdatum.h"
#include "doubledatum.h"
#include "dictutils.h"
#include "numerics.h"
#include "analog_data_logger_impl.h"

#include <limits>

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::izhikevich_large::Parameters_::Parameters_()
  : a_(0.01), // default to RS
    b_(5.0),
    c_(-60.0),
    d_(400.0),
    vt_(-50.0),
    vr_(-60.0),
    C_(100.0),
    k_(3.0),
    v_peak_(50.0),
    I_e_(0.0)
{}

nest::izhikevich_large::State_::State_()
{
	  v_ = -65.0;
	  u_ = 5.0*v_; // b_*v_
}

/* ----------------------------------------------------------------
 * Paramater and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void nest::izhikevich_large::Parameters_::get(DictionaryDatum &d) const
{
  def<double>(d, names::a, a_);
  def<double>(d, names::b, b_);
  def<double>(d, names::c, c_);
  def<double>(d, names::d, d_);
  def<double>(d, "vt", vt_);
  def<double>(d, "vr", vr_);
  def<double>(d, "k", k_);
  def<double>(d, "C", C_);
  def<double>(d, "v_peak", v_peak_);
  def<double>(d, names::I_e, I_e_);
}

void nest::izhikevich_large::Parameters_::set(const DictionaryDatum& d)
{

  updateValue<double>(d, names::a, a_);
  updateValue<double>(d, names::b, b_);
  updateValue<double>(d, names::c, c_);
  updateValue<double>(d, names::d, d_);
  updateValue<double>(d, "vt", vt_);
  updateValue<double>(d, "vr", vr_);
  updateValue<double>(d, "k", k_);
  updateValue<double>(d, "C", C_);
  updateValue<double>(d, "v_peak", v_peak_);
  updateValue<double>(d, names::I_e, I_e_);
}

void nest::izhikevich_large::State_::get(DictionaryDatum &d, const Parameters_& p) const
{
  def<double>(d, names::V_m, v_); // Membrane potential
}

void nest::izhikevich_large::State_::set(const DictionaryDatum& d, const Parameters_& p)
{
  double utmp;
  if ( updateValue<double>(d, names::V_m, utmp) )
    v_ = utmp;

  u_ = p.b_*v_;
}

/* ----------------------------------------------------------------
 * Default and copy constructor for node
 * ---------------------------------------------------------------- */

nest::izhikevich_large::izhikevich_large()
  : Archiving_Node(),
    P_(),
    S_()
{}

nest::izhikevich_large::izhikevich_large(const izhikevich_large& n)
  : Archiving_Node(n),
    P_(n.P_),
    S_(n.S_)
{}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void nest::izhikevich_large::init_node_(const Node& proto)
{
  const izhikevich_large& pr = downcast<izhikevich_large>(proto);
  P_ = pr.P_;
  S_ = pr.S_;
}

void nest::izhikevich_large::init_state_(const Node& proto)
{
  const izhikevich_large& pr = downcast<izhikevich_large>(proto);
  S_ = pr.S_;
}

void nest::izhikevich_large::init_buffers_()
{
  B_.spikes_.clear();    // includes resize
  B_.currents_.clear();  // include resize
  B_.potentials_.clear_data(); // includes resize
  Archiving_Node::clear_history();
}

void nest::izhikevich_large::calibrate()
{

}

/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void nest::izhikevich_large::update(Time const & origin, const long_t from, const long_t to)
{
  assert(to >= 0 && (delay) from < Scheduler::get_min_delay());
  assert(from < to);

  const double h = Time::get_resolution().get_ms();

  for ( long_t lag = from ; lag < to ; ++lag )
  {
  	  double current = B_.currents_.get_value(lag);
	  double spikes = B_.spikes_.get_value(lag);
	  double I = P_.I_e_+current+spikes;

	  S_.v_ = S_.v_ + h * (1/P_.C_)*( P_.k_*(S_.v_-P_.vr_)*(S_.v_-P_.vt_) - S_.u_+ I );
	  S_.u_ = S_.u_ + h * (P_.a_*(P_.b_*(S_.v_-P_.vr_) - S_.u_));

	  // check if fired
	  if(S_.v_ > P_.v_peak_)
	  {
		  S_.v_ = P_.c_;
		  S_.u_ = S_.u_ + P_.d_;

		  // send spike
		  set_spiketime(Time::step(origin.get_steps()+lag+1));
		  SpikeEvent se;
		  network()->send(*this, se, lag);

		  // record spike event
		  B_.potentials_.record_data(origin.get_steps()+lag, P_.v_peak_);
	  } else {
		  B_.potentials_.record_data(origin.get_steps()+lag, S_.v_);
	  }
  }
}

void nest::izhikevich_large::handle(SpikeEvent & e)
{
  assert(e.get_delay() > 0);

  B_.spikes_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
		       e.get_weight() * e.get_multiplicity() );
}

void nest::izhikevich_large::handle(CurrentEvent& e)
{
  assert(e.get_delay() > 0);

  const double_t c=e.get_current();
  const double_t w=e.get_weight();

  // add weighted current; HEP 2002-10-04
  B_.currents_.add_value(e.get_rel_delivery_steps(network()->get_slice_origin()),
			 w * c);
}

void nest::izhikevich_large::handle(PotentialRequest& e)
{
  B_.potentials_.handle(*this, e);
}
