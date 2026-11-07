//! The right-hand side of msoaxon's two-compartment model, in plain Rust.
//!
//! A port of `msoaxon.two._rhs` (with `applied_current` and
//! `event_currents`), in the Python's floating-point order, operation for
//! operation, as [`crate::core`] is of `multi._rhs`: bit-identical where
//! numpy and Rust call the same libm (darwin-arm64), close elsewhere. The
//! gating kinetics are `core.rs`'s expressions on the same [`Gating`]
//! constants; the parameters are `two.get_params`'s, resolved there, so the
//! quirks `TwoCpt.m` settles in its parameters (the `'Active-sodium'`
//! spelling, KLT inactivation `z` frozen at rest) come in as values. The
//! one `TwoCptODE.m` keeps in its equations is kept here: every stimulus
//! but "Events" is driven by the soma voltage V1, also when it enters the
//! axon compartment.
//!
//! The state is `two.py`'s: V1, V2, w1, h1, w2, m1, m2, h2, p, a1, a2.

use std::collections::HashMap;

use crate::core::{finite_over, logistic, relax, CoreError, Epsg, EventTable, Gating, Table};

/// The number of states.
pub const NSTATES: usize = 11;

/// Each gate's state index and its compartment (0 the soma, 1 the axon).
const GATES: [(usize, usize); 9] = [
    (2, 0),  // w1
    (3, 0),  // h1
    (4, 1),  // w2
    (5, 0),  // m1
    (6, 1),  // m2
    (7, 1),  // h2
    (8, 0),  // p
    (9, 0),  // a1
    (10, 1), // a2
];

/// Declares `TwoParams`, one `f64` per field of `two.TwoCptParams` the
/// right-hand side reads, and its constructor from the dict that
/// `_fast._build_two` passes (`dataclasses.asdict`). Each field is
/// `name = "TwoCptParams name"`.
macro_rules! two_params {
    ($($(#[$doc:meta])* $field:ident = $name:literal),* $(,)?) => {
        /// `two.get_params`'s parameters, as `_rhs` reads them.
        #[derive(Clone, Debug)]
        pub struct TwoParams {
            $($(#[$doc])* pub $field: f64,)*
        }

        impl TwoParams {
            /// Reads every field from `map`; a missing name, or one that is
            /// neither a field nor in [`UNREAD`], is an error, so a new
            /// knob in `TwoCptParams` cannot be dropped unnoticed.
            pub fn from_map(map: &HashMap<String, f64>) -> Result<Self, CoreError> {
                const NAMES: &[&str] = &[$($name),*];
                if let Some(name) = map
                    .keys()
                    .find(|k| !NAMES.contains(&k.as_str()) && !UNREAD.contains(&k.as_str()))
                {
                    return Err(CoreError(format!("unknown two_cpt parameter {name:?}")));
                }
                let get = |name: &str| {
                    map.get(name).copied().ok_or_else(|| {
                        CoreError(format!("missing two_cpt parameter {name:?}"))
                    })
                };
                Ok(TwoParams {
                    $($field: get($name)?,)*
                })
            }
        }
    };
}

/// The fields of `TwoCptParams` that only set the others in `get_params`.
const UNREAD: &[&str] = &[
    "couple12", "couple21", "Vrest", "gTot1", "gTot2", "tau1", "tau2",
];

two_params! {
    /// Capacitances [pF] and the coupling conductance [nS].
    cap1 = "cap1", cap2 = "cap2", g_c = "gC",
    /// Leak conductances [nS] and reversal [mV].
    g_lk1 = "glk1", g_lk2 = "glk2", e_lk = "Elk",
    /// Na conductances [nS], reversal [mV], resting currents [pA] and the
    /// (negated) inactivation midpoints.
    g_na1 = "gNa1", g_na2 = "gNa2", e_na = "ENa", ina0_1 = "INa0_1", ina0_2 = "INa0_2",
    na_vhalf1 = "na_vhalf1", na_vhalf2 = "na_vhalf2",
    /// KLT conductances, reversal, resting currents and the frozen `z`.
    g_klt1 = "gKLT1", g_klt2 = "gKLT2", e_k = "EK", iklt0_1 = "IKLT0_1",
    iklt0_2 = "IKLT0_2", z_rest = "z_rest",
    /// h conductances, reversal and resting currents.
    g_h1 = "gh1", g_h2 = "gh2", v_h = "Vh", ih0_1 = "Ih0_1", ih0_2 = "Ih0_2",
    /// The soma's KHT conductance, reversal and resting current.
    g_kht = "gKHT", v_k = "VK", ikht0 = "IKHT0",
    /// The compartment the stimulus enters: 1 the soma, else the axon.
    iapp_loc = "IappLoc",
}

/// The stimulus, with the fields of `_common.stimulus` that
/// `two.applied_current` reads.
///
/// `epsg` is (1 / peak, decay, rise) (`s.inv_peak` and `s.epsg_tau`),
/// `ramp_rise` is `_solve.RAMP_RISE`, where ramp2 starts. `Default` is
/// "none".
#[derive(Clone, Debug)]
pub struct TwoStimSpec {
    pub stim_type: String,
    pub start: f64,
    pub stop: f64,
    pub amp: f64,
    pub t_end: f64,
    pub slope: f64,
    pub t_top: f64,
    pub two_pi_f: f64,
    pub ramp_rise: f64,
    pub epsg: (f64, f64, f64),
    pub t_syn: Option<Vec<f64>>,
    pub g_syn: Option<Vec<f64>>,
    pub vsyn: f64,
    pub diff: Option<f64>,
    /// The event table of "Events"; `amp` is the run's `I`.
    pub events: Option<EventTable>,
}

impl Default for TwoStimSpec {
    fn default() -> Self {
        TwoStimSpec {
            stim_type: "none".to_string(),
            start: 0.0,
            stop: 0.0,
            amp: 0.0,
            t_end: 0.0,
            slope: 0.0,
            t_top: 0.0,
            two_pi_f: 0.0,
            ramp_rise: 0.0,
            epsg: (1.0, 1.0, 1.0),
            t_syn: None,
            g_syn: None,
            vsyn: 0.0,
            diff: None,
            events: None,
        }
    }
}

/// One stimulus, as `applied_current` (and `event_currents`) apply it.
#[derive(Clone, Debug)]
enum Stim {
    None,
    Step {
        start: f64,
        stop: f64,
        amp: f64,
    },
    Ramp {
        start: f64,
        t_top: f64,
        amp: f64,
        slope: f64,
    },
    Ramp2 {
        rise: f64,
        amp: f64,
        slope: f64,
    },
    Sine {
        start: f64,
        stop: f64,
        amp: f64,
        two_pi_f: f64,
    },
    /// Synaptic, and SynapticPair with `pair` = (diff, t_end).
    Synaptic {
        start: f64,
        stop: f64,
        table: Table,
        vsyn: f64,
        pair: Option<(f64, f64)>,
    },
    Epsg {
        start: f64,
        amp: f64,
        w: Epsg,
    },
    EpsgPair {
        start: f64,
        td: f64,
        amp: f64,
        w: Epsg,
    },
    /// Each event at its own compartment: `soma` for node 1, else the axon.
    Events {
        onset: Vec<f64>,
        soma: Vec<bool>,
        /// `I * g`, as numpy multiplies them first.
        scale_g: Vec<f64>,
        e_rev: Vec<f64>,
        w: Vec<Epsg>,
    },
}

impl Stim {
    /// `applied_current` at (t, V1), for every stimulus but "Events".
    fn applied(&self, t: f64, v1: f64) -> f64 {
        let on = |c: bool| if c { 1.0 } else { 0.0 };
        match *self {
            Stim::None | Stim::Events { .. } => 0.0,
            Stim::Step { start, stop, amp } => {
                if start <= t && t < stop {
                    amp
                } else {
                    0.0
                }
            }
            Stim::Ramp {
                start,
                t_top,
                amp,
                slope,
            } => on(start <= t && t <= t_top) * amp * (t - start) * slope,
            Stim::Ramp2 { rise, amp, slope } => {
                let b = on(t >= rise) * amp * (t - rise) * slope;
                if b < amp {
                    b
                } else {
                    amp
                } // Python's min(amp, b)
            }
            Stim::Sine {
                start,
                stop,
                amp,
                two_pi_f,
            } => {
                let wave = (two_pi_f * (t - start) / 1000.0).sin();
                on(start <= t && t <= stop) * amp * wave * on(wave > 0.0)
            }
            Stim::Synaptic {
                start,
                stop,
                ref table,
                vsyn,
                pair,
            } => -on(start <= t && t <= stop) * self.synaptic_g(t, table, pair) * (v1 - vsyn),
            Stim::Epsg { start, amp, w } => {
                let te = t - start;
                amp * (0.0 - v1) * on(te >= 0.0) * w.unitary(te)
            }
            Stim::EpsgPair { start, td, amp, w } => {
                let te = t - start;
                amp * (0.0 - v1)
                    * (on(te >= 0.0) * w.unitary(te) + on(te >= td) * w.unitary(te - td))
            }
        }
    }

    /// The synaptic conductance: the table at t, plus at t + diff for a
    /// pair (0 once t + diff reaches t_end).
    fn synaptic_g(&self, t: f64, table: &Table, pair: Option<(f64, f64)>) -> f64 {
        match pair {
            None => table.interp(t),
            Some((diff, t_end)) => {
                let g2 = if t + diff < t_end {
                    table.interp(t + diff)
                } else {
                    0.0
                };
                table.interp(t) + g2
            }
        }
    }

    /// d `applied` / d V1, under the same on/off conditions.
    fn applied_slope(&self, t: f64) -> f64 {
        let on = |c: bool| if c { 1.0 } else { 0.0 };
        match *self {
            Stim::Synaptic {
                start,
                stop,
                ref table,
                pair,
                ..
            } => -on(start <= t && t <= stop) * self.synaptic_g(t, table, pair),
            Stim::Epsg { start, amp, w } => {
                let te = t - start;
                -amp * on(te >= 0.0) * w.unitary(te)
            }
            Stim::EpsgPair { start, td, amp, w } => {
                let te = t - start;
                -amp * (on(te >= 0.0) * w.unitary(te) + on(te >= td) * w.unitary(te - td))
            }
            _ => 0.0,
        }
    }
}

/// `np.sum` of a float64 array: numpy's pairwise summation (fewer than 8
/// values in a row from -0.0, 8 accumulators up to 128, halves above),
/// added to the reduction's identity 0.0.
fn np_sum(a: &[f64]) -> f64 {
    fn pairwise(a: &[f64]) -> f64 {
        let n = a.len();
        if n < 8 {
            let mut res = -0.0;
            for v in a {
                res += v;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - n % 8 {
                for (j, acc) in r.iter_mut().enumerate() {
                    *acc += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for v in &a[i..] {
                res += v;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    0.0 + pairwise(a)
}

/// The gating kinetics of `constants.py`, as `core.rs` evaluates them:
/// each gate's (x_inf(V) - x) / tau(V).
fn rate_m(c: &Gating, v: f64, x: f64) -> f64 {
    let inf = 1.0 / (1.0 + ((v + c.minf_v) / c.minf_k).exp());
    let tau = (c.taum_a + (c.taum_b / (1.0 + ((c.taum_v - v) / c.taum_k).exp()))) / c.taum_div;
    (inf - x) / tau
}

fn rate_h(c: &Gating, v: f64, vhalf: f64, x: f64) -> f64 {
    let inf = 1.0 / (1.0 + ((v + vhalf) / c.hinf_k).exp());
    let tau = (c.tauh_a + (c.tauh_b / (1.0 + ((c.tauh_v - v) / c.tauh_k).exp()))) / c.tauh_div;
    (inf - x) / tau
}

fn rate_p(c: &Gating, v: f64, x: f64) -> f64 {
    let inf = 1.0 / (1.0 + (-(v + c.pinf_v) / c.pinf_k).exp());
    let tau = c.taup_temp
        * (c.taup_num
            / (c.taup_a * ((v + c.taup_v) / c.taup_ka).exp()
                + c.taup_b * (-(v + c.taup_v) / c.taup_kb).exp())
            + c.taup_c);
    (inf - x) / tau
}

fn rate_w(c: &Gating, v: f64, x: f64) -> f64 {
    let inf = 1.0 / (1.0 + ((v + c.winf_v) / c.winf_k).exp());
    let tau = c.tauw_num
        / (c.tauw_a * ((v + c.tauw_v) / c.tauw_ka).exp()
            + c.tauw_b * (-(v + c.tauw_v) / c.tauw_kb).exp())
        + c.tauw_c;
    (inf - x) / tau
}

fn rate_a(c: &Gating, v: f64, x: f64) -> f64 {
    let inf = 1.0 / (1.0 + (c.ainf_k * (v + c.ainf_v)).exp());
    let s = v + c.taua_v;
    // `(V + 61.5) ** 2` on a numpy scalar is libm pow, not the s * s of
    // numpy's arrays (core.rs): the two differ in the last bit now and then.
    // LLVM folds powf(2.0) into s * s, so the exponent is hidden from it.
    let sq = s.powf(std::hint::black_box(2.0));
    let tau = c.taua_temp * (c.taua_c + c.taua_a * (-sq / c.taua_k).exp());
    (inf - x) / tau
}

/// d/dV and d/dx of each rate above, as `MultiRhs::jacobian` has them.
fn jac_m(c: &Gating, v: f64, x: f64) -> (f64, f64) {
    let (inf, ds) = logistic((v + c.minf_v) / c.minf_k);
    let (q, dq) = logistic((c.taum_v - v) / c.taum_k);
    let tau = (c.taum_a + c.taum_b * q) / c.taum_div;
    let d_tau = c.taum_b * (-dq / c.taum_k) / c.taum_div;
    relax(inf, ds / c.minf_k, x, tau, d_tau)
}

fn jac_h(c: &Gating, v: f64, vhalf: f64, x: f64) -> (f64, f64) {
    let (inf, ds) = logistic((v + vhalf) / c.hinf_k);
    let (q, dq) = logistic((c.tauh_v - v) / c.tauh_k);
    let tau = (c.tauh_a + c.tauh_b * q) / c.tauh_div;
    let d_tau = c.tauh_b * (-dq / c.tauh_k) / c.tauh_div;
    relax(inf, ds / c.hinf_k, x, tau, d_tau)
}

fn jac_p(c: &Gating, v: f64, x: f64) -> (f64, f64) {
    let (inf, ds) = logistic(-(v + c.pinf_v) / c.pinf_k);
    let s = v + c.taup_v;
    let (e1, e2) = ((s / c.taup_ka).exp(), (-s / c.taup_kb).exp());
    let d = c.taup_a * e1 + c.taup_b * e2;
    let d_d = c.taup_a * e1 / c.taup_ka - c.taup_b * e2 / c.taup_kb;
    let tau = c.taup_temp * (c.taup_num / d + c.taup_c);
    let d_tau = finite_over(-c.taup_temp * c.taup_num * d_d, d);
    relax(inf, -ds / c.pinf_k, x, tau, d_tau)
}

fn jac_w(c: &Gating, v: f64, x: f64) -> (f64, f64) {
    let (inf, ds) = logistic((v + c.winf_v) / c.winf_k);
    let s = v + c.tauw_v;
    let (e1, e2) = ((s / c.tauw_ka).exp(), (-s / c.tauw_kb).exp());
    let d = c.tauw_a * e1 + c.tauw_b * e2;
    let d_d = c.tauw_a * e1 / c.tauw_ka - c.tauw_b * e2 / c.tauw_kb;
    let tau = c.tauw_num / d + c.tauw_c;
    relax(
        inf,
        ds / c.winf_k,
        x,
        tau,
        finite_over(-c.tauw_num * d_d, d),
    )
}

fn jac_a(c: &Gating, v: f64, x: f64) -> (f64, f64) {
    let (inf, ds) = logistic(c.ainf_k * (v + c.ainf_v));
    let s = v + c.taua_v;
    let bump = c.taua_a * (-(s * s) / c.taua_k).exp();
    let tau = c.taua_temp * (c.taua_c + bump);
    let d_tau = c.taua_temp * bump * (-2.0 * s / c.taua_k);
    relax(inf, ds * c.ainf_k, x, tau, d_tau)
}

/// The right-hand side of one `two_cpt` run: f(t, x) -> dx/dt.
#[derive(Clone, Debug)]
pub struct TwoCptRhs {
    p: TwoParams,
    gating: Gating,
    stim: Stim,
    /// Whether the stimulus enters the soma (`IappLoc == 1`).
    soma_input: bool,
    /// [`jac_sparsity`], built once.
    rows: Vec<usize>,
    cols: Vec<usize>,
}

impl TwoCptRhs {
    /// Builds the right-hand side of one run.
    pub fn new(p: TwoParams, gating: Gating, spec: &TwoStimSpec) -> Result<Self, CoreError> {
        let TwoStimSpec {
            ref stim_type,
            start,
            stop,
            amp,
            t_end,
            slope,
            t_top,
            two_pi_f,
            ramp_rise,
            epsg,
            ref t_syn,
            ref g_syn,
            vsyn,
            diff,
            ref events,
        } = *spec;
        let w = Epsg {
            inv_peak: epsg.0,
            decay: epsg.1,
            rise: epsg.2,
        };
        let stim = match stim_type.as_str() {
            "none" => Stim::None,
            "step" => Stim::Step { start, stop, amp },
            "ramp" => Stim::Ramp {
                start,
                t_top,
                amp,
                slope,
            },
            "ramp2" => Stim::Ramp2 {
                rise: ramp_rise,
                amp,
                slope,
            },
            "sine" => Stim::Sine {
                start,
                stop,
                amp,
                two_pi_f,
            },
            "Synaptic" | "SynapticPair" => {
                let (Some(x), Some(y)) = (t_syn, g_syn) else {
                    return Err(CoreError(format!("{stim_type} needs t_syn and g_syn")));
                };
                if x.is_empty() || x.len() != y.len() {
                    return Err(CoreError(
                        "t_syn and g_syn must be non-empty and the same length".to_string(),
                    ));
                }
                let pair = if stim_type == "SynapticPair" {
                    let d = diff.ok_or_else(|| CoreError("SynapticPair needs diff".to_string()))?;
                    Some((d, t_end))
                } else {
                    None
                };
                Stim::Synaptic {
                    start,
                    stop,
                    table: Table {
                        x: x.clone(),
                        y: y.clone(),
                    },
                    vsyn,
                    pair,
                }
            }
            "EPSG" => Stim::Epsg { start, amp, w },
            "EPSGpair" => Stim::EpsgPair {
                start,
                td: stop - start,
                amp,
                w,
            },
            "Events" => {
                let Some(ev) = events else {
                    return Err(CoreError("Events needs an event table".to_string()));
                };
                let count = ev.t.len();
                if count == 0 {
                    return Err(CoreError("Events needs at least one event".to_string()));
                }
                for (name, len) in [
                    ("k", ev.k.len()),
                    ("g", ev.g.len()),
                    ("rise", ev.rise.len()),
                    ("decay", ev.decay.len()),
                    ("inv_peak", ev.inv_peak.len()),
                    ("e_rev", ev.e_rev.len()),
                ] {
                    if len != count {
                        return Err(CoreError(format!(
                            "events {name} has {len} entries, expected {count}"
                        )));
                    }
                }
                if let Some(k) = ev.k.iter().find(|&&k| k < 0) {
                    return Err(CoreError(format!("events k={k} is negative")));
                }
                Stim::Events {
                    onset: ev.t.clone(),
                    soma: ev.k.iter().map(|&k| k == 0).collect(),
                    scale_g: ev.g.iter().map(|g| amp * g).collect(),
                    e_rev: ev.e_rev.clone(),
                    w: (0..count)
                        .map(|e| Epsg {
                            inv_peak: ev.inv_peak[e],
                            decay: ev.decay[e],
                            rise: ev.rise[e],
                        })
                        .collect(),
                }
            }
            other => return Err(CoreError(format!("unknown stim_type {other:?}"))),
        };
        let soma_input = p.iapp_loc == 1.0;
        let (rows, cols) = jac_sparsity();
        Ok(TwoCptRhs {
            p,
            gating,
            stim,
            soma_input,
            rows,
            cols,
        })
    }

    /// (Iapp1, Iapp2): the input currents into the soma and the axon [pA],
    /// as `_rhs` routes them.
    fn currents(&self, t: f64, v1: f64, v2: f64) -> (f64, f64) {
        if let Stim::Events {
            ref onset,
            ref soma,
            ref scale_g,
            ref e_rev,
            ref w,
        } = self.stim
        {
            // synaptic.event_currents, then i[soma].sum(), i[~soma].sum()
            let (mut i1, mut i2) = (Vec::new(), Vec::new());
            for e in 0..onset.len() {
                let v = if soma[e] { v1 } else { v2 };
                let i = scale_g[e] * (e_rev[e] - v) * w[e].unitary(t - onset[e]);
                if soma[e] {
                    i1.push(i);
                } else {
                    i2.push(i);
                }
            }
            return (np_sum(&i1), np_sum(&i2));
        }
        let iapp = self.stim.applied(t, v1);
        if self.soma_input {
            (iapp, 0.0)
        } else {
            (0.0, iapp)
        }
    }

    /// Writes dx/dt for state `x` at time `t` into `out` (both 11 long).
    pub fn eval(&self, t: f64, x: &[f64], out: &mut [f64]) {
        let (p, c) = (&self.p, &self.gating);
        let [v1, v2, w1, h1, w2, m1, m2, h2, kp, a1, a2] =
            <[f64; NSTATES]>::try_from(&x[..NSTATES]).unwrap();
        let z = p.z_rest;

        let ikht = p.g_kht * kp * (v1 - p.v_k) - p.ikht0;
        let ina1 = p.g_na1 * m1.powf(4.0) * (c.na_open_h * h1 + c.na_open_floor) * (v1 - p.e_na)
            - p.ina0_1;
        let ilk1 = p.g_lk1 * (v1 - p.e_lk);
        let iklt1 = p.g_klt1 * w1.powf(4.0) * z * (v1 - p.e_k) - p.iklt0_1;
        let ih1 = p.g_h1 * a1 * (v1 - p.v_h) - p.ih0_1;

        let ilk2 = p.g_lk2 * (v2 - p.e_lk);
        let ina2 = p.g_na2 * m2.powf(4.0) * (c.na_open_h * h2 + c.na_open_floor) * (v2 - p.e_na)
            - p.ina0_2;
        let iklt2 = p.g_klt2 * w2.powf(4.0) * z * (v2 - p.e_k) - p.iklt0_2;
        let ih2 = p.g_h2 * a2 * (v2 - p.v_h) - p.ih0_2;

        let ic = p.g_c * (v1 - v2);
        let (iapp1, iapp2) = self.currents(t, v1, v2);

        out[0] = (-ilk1 - iklt1 - ic + iapp1 - ina1 - ih1 - ikht) / p.cap1;
        out[1] = (-ilk2 - iklt2 + ic + iapp2 - ina2 - ih2) / p.cap2;
        out[2] = rate_w(c, v1, w1);
        out[3] = rate_h(c, v1, p.na_vhalf1, h1);
        out[4] = rate_w(c, v2, w2);
        out[5] = rate_m(c, v1, m1);
        out[6] = rate_m(c, v2, m2);
        out[7] = rate_h(c, v2, p.na_vhalf2, h2);
        out[8] = rate_p(c, v1, kp);
        out[9] = rate_a(c, v1, a1);
        out[10] = rate_a(c, v2, a2);
    }

    /// Writes the exact Jacobian of [`eval`](Self::eval) at (t, x) into
    /// `out`: one value per entry of [`jac_sparsity`], in that order.
    ///
    /// The values are those of `eval`'s expressions, not its float order:
    /// only `eval` is bit-pinned.
    pub fn jacobian(&self, t: f64, x: &[f64], out: &mut [f64]) {
        let (p, c) = (&self.p, &self.gating);
        let [v1, v2, w1, h1, w2, m1, m2, h2, kp, a1, a2] =
            <[f64; NSTATES]>::try_from(&x[..NSTATES]).unwrap();
        let z = p.z_rest;
        let mut j = [[0.0f64; NSTATES]; NSTATES];

        // d Iapp_c / d V_v: every stimulus but "Events" is driven by V1,
        // into the compartment it enters; an event by its own compartment
        let mut d = [[0.0f64; 2]; 2];
        if let Stim::Events {
            ref onset,
            ref soma,
            ref scale_g,
            ref w,
            ..
        } = self.stim
        {
            for e in 0..onset.len() {
                let k = if soma[e] { 0 } else { 1 };
                d[k][k] -= scale_g[e] * w[e].unitary(t - onset[e]);
            }
        } else {
            d[if self.soma_input { 0 } else { 1 }][0] = self.stim.applied_slope(t);
        }

        // the voltage rows: dV = (... - I) / cap, so each conductance and
        // each open fraction's derivative times its driving force, negated
        let (e_na1, e_na2) = (v1 - p.e_na, v2 - p.e_na);
        let (e_k1, e_k2) = (v1 - p.e_k, v2 - p.e_k);
        let (m1_3, m2_3, w1_3, w2_3) = (m1 * m1 * m1, m2 * m2 * m2, w1 * w1 * w1, w2 * w2 * w2);
        let (nah1, nah2) = (
            c.na_open_h * h1 + c.na_open_floor,
            c.na_open_h * h2 + c.na_open_floor,
        );
        let g1 = p.g_lk1
            + p.g_klt1 * w1_3 * w1 * z
            + p.g_c
            + p.g_na1 * m1_3 * m1 * nah1
            + p.g_h1 * a1
            + p.g_kht * kp;
        j[0][0] = (d[0][0] - g1) / p.cap1;
        j[0][1] = (p.g_c + d[0][1]) / p.cap1;
        j[0][2] = -(p.g_klt1 * 4.0 * w1_3 * z * e_k1) / p.cap1;
        j[0][3] = -(p.g_na1 * m1_3 * m1 * c.na_open_h * e_na1) / p.cap1;
        j[0][5] = -(p.g_na1 * 4.0 * m1_3 * nah1 * e_na1) / p.cap1;
        j[0][8] = -(p.g_kht * (v1 - p.v_k)) / p.cap1;
        j[0][9] = -(p.g_h1 * (v1 - p.v_h)) / p.cap1;
        let g2 =
            p.g_lk2 + p.g_klt2 * w2_3 * w2 * z + p.g_c + p.g_na2 * m2_3 * m2 * nah2 + p.g_h2 * a2;
        j[1][1] = (d[1][1] - g2) / p.cap2;
        j[1][0] = (p.g_c + d[1][0]) / p.cap2;
        j[1][4] = -(p.g_klt2 * 4.0 * w2_3 * z * e_k2) / p.cap2;
        j[1][6] = -(p.g_na2 * 4.0 * m2_3 * nah2 * e_na2) / p.cap2;
        j[1][7] = -(p.g_na2 * m2_3 * m2 * c.na_open_h * e_na2) / p.cap2;
        j[1][10] = -(p.g_h2 * (v2 - p.v_h)) / p.cap2;

        // the gate rows, in state order
        let rows = [
            jac_w(c, v1, w1),
            jac_h(c, v1, p.na_vhalf1, h1),
            jac_w(c, v2, w2),
            jac_m(c, v1, m1),
            jac_m(c, v2, m2),
            jac_h(c, v2, p.na_vhalf2, h2),
            jac_p(c, v1, kp),
            jac_a(c, v1, a1),
            jac_a(c, v2, a2),
        ];
        for ((g, k), (dv, dx)) in GATES.into_iter().zip(rows) {
            j[g][k] = dv;
            j[g][g] = dx;
        }
        for (e, (&r, &col)) in self.rows.iter().zip(&self.cols).enumerate() {
            out[e] = j[r][col];
        }
    }

    /// The Jacobian pattern, [`jac_sparsity`].
    pub fn pattern(&self) -> (&[usize], &[usize]) {
        (&self.rows, &self.cols)
    }
}

/// The Jacobian pattern of the two-compartment model, `(rows, cols)` in
/// column-major (CSC) order, rows ascending within a column.
///
/// Each voltage couples to both voltages (the coupling current; an input
/// into the axon is driven by V1) and to its own compartment's gates; each
/// gate to its own voltage and itself.
pub fn jac_sparsity() -> (Vec<usize>, Vec<usize>) {
    let (mut rows, mut cols) = (Vec::new(), Vec::new());
    for v in 0..2 {
        let mut col = vec![0, 1];
        col.extend(GATES.iter().filter(|g| g.1 == v).map(|g| g.0));
        for r in col {
            rows.push(r);
            cols.push(v);
        }
    }
    for (g, k) in GATES {
        rows.extend([k, g]);
        cols.extend([g, g]);
    }
    (rows, cols)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::core::tests::gating;

    /// The parameters of `two.get_params(-68, 3, 1, "active-full")`,
    /// rounded: a fixed model, not the exact one.
    pub(crate) fn params(iapp_loc: f64) -> TwoParams {
        let map: HashMap<String, f64> = [
            ("cap1", 13.0),
            ("cap2", 0.6),
            ("gC", 60.0),
            ("glk1", 30.0),
            ("glk2", 3.0),
            ("Elk", -68.0),
            ("gNa1", 450.0),
            ("gNa2", 119.0),
            ("ENa", 69.0),
            ("INa0_1", -2.5),
            ("INa0_2", -0.6),
            ("na_vhalf1", 62.5),
            ("na_vhalf2", 62.5),
            ("gKLT1", 600.0),
            ("gKLT2", 40.0),
            ("EK", -90.0),
            ("IKLT0_1", 80.0),
            ("IKLT0_2", 5.0),
            ("z_rest", 0.65),
            ("gh1", 70.0),
            ("gh2", 2.0),
            ("Vh", -35.0),
            ("Ih0_1", -25.0),
            ("Ih0_2", -0.7),
            ("gKHT", 11.0),
            ("VK", -90.0),
            ("IKHT0", -3.0),
            ("IappLoc", iapp_loc),
            ("couple12", 0.8),
        ]
        .map(|(k, v)| (k.to_string(), v))
        .into_iter()
        .collect();
        TwoParams::from_map(&map).unwrap()
    }

    pub(crate) fn spec(stim_type: &str, start: f64, stop: f64, amp: f64) -> TwoStimSpec {
        TwoStimSpec {
            stim_type: stim_type.to_string(),
            start,
            stop,
            amp,
            t_end: 20.0,
            slope: 1.0 / (stop - start),
            t_top: stop,
            two_pi_f: 2.0 * std::f64::consts::PI * 500.0,
            ramp_rise: 5.0,
            epsg: (1.0 / 0.21317, 0.18, 0.1),
            ..TwoStimSpec::default()
        }
    }

    /// One model per stimulus, into the soma and into the axon, each with
    /// a time inside its stimulus.
    pub(crate) fn models() -> Vec<(String, TwoCptRhs, f64)> {
        let syn = |pair: bool| TwoStimSpec {
            t_syn: Some(vec![0.0, 5.0, 6.0, 20.0]),
            g_syn: Some(vec![0.0, 0.0, 30.0, 0.0]),
            vsyn: 0.0,
            diff: pair.then_some(0.3),
            ..spec(
                if pair { "SynapticPair" } else { "Synaptic" },
                5.0,
                15.0,
                0.0,
            )
        };
        let events = TwoStimSpec {
            events: Some(EventTable {
                t: vec![5.0, 4.8, 5.2],
                k: vec![0, 2, 0],
                g: vec![60.0, 30.0, 40.0],
                rise: vec![0.1, 0.28, 0.1],
                decay: vec![0.18, 1.5, 0.18],
                inv_peak: vec![1.0 / 0.21317, 2.5, 1.0 / 0.21317],
                e_rev: vec![0.0, -80.0, 0.0],
            }),
            ..spec("Events", 0.0, 0.0, 1.0)
        };
        let specs = [
            (spec("step", 5.0, 10.0, 3000.0), 6.0),
            (spec("ramp", 5.0, 5.5, 3000.0), 5.3),
            (spec("ramp2", 5.0, 5.5, 3000.0), 5.3),
            (spec("sine", 5.0, 7.5, 3000.0), 5.4),
            (syn(false), 5.5),
            (syn(true), 5.5),
            (spec("EPSG", 5.0, 10.0, 100.0), 5.2),
            (spec("EPSGpair", 5.0, 5.2, 100.0), 5.3),
            (events, 5.3),
        ];
        let mut out = Vec::new();
        for (s, t) in specs {
            for loc in [1.0, 2.0] {
                let rhs = TwoCptRhs::new(params(loc), gating(), &s).unwrap();
                out.push((format!("{} at {loc}", s.stim_type), rhs, t));
            }
        }
        out
    }

    fn states() -> Vec<(&'static str, Vec<f64>)> {
        let gates = |g: f64| [g; 9];
        let mut out = Vec::new();
        for (label, v1, v2, g) in [
            ("rest", -68.0, -68.0, 0.3),
            ("depolarised", -40.0, -20.0, 0.6),
            ("hyperpolarised", -85.0, -80.0, 0.1),
        ] {
            let mut x = vec![v1, v2];
            x.extend(
                gates(g)
                    .iter()
                    .enumerate()
                    .map(|(i, g)| g + 0.03 * i as f64),
            );
            out.push((label, x));
        }
        out
    }

    #[test]
    fn the_jacobian_is_its_pattern_and_central_differences_of_eval() {
        // as core.rs's test: off the pattern no row moves at all, on it the
        // analytic value agrees with a central difference
        let (rows, cols) = jac_sparsity();
        let mut slot = vec![None; NSTATES * NSTATES];
        for (e, (&r, &c)) in rows.iter().zip(&cols).enumerate() {
            assert!(
                slot[r * NSTATES + c].replace(e).is_none(),
                "({r}, {c}) twice"
            );
        }
        for (name, rhs, t) in models() {
            for (label, x) in states() {
                let mut jac = vec![f64::NAN; rows.len()];
                rhs.jacobian(t, &x, &mut jac);
                assert!(jac.iter().all(|v| v.is_finite()), "{name}, {label}");
                let mut scale = [0.0_f64; NSTATES];
                for (e, &r) in rows.iter().enumerate() {
                    scale[r] = scale[r].max(jac[e].abs());
                }
                let (mut fp, mut fm) = ([0.0; NSTATES], [0.0; NSTATES]);
                for c in 0..NSTATES {
                    let h = 1e-5 * (1.0 + x[c].abs());
                    let mut y = x.clone();
                    y[c] = x[c] + h;
                    rhs.eval(t, &y, &mut fp);
                    y[c] = x[c] - h;
                    rhs.eval(t, &y, &mut fm);
                    for r in 0..NSTATES {
                        match slot[r * NSTATES + c] {
                            None => assert_eq!(fp[r], fm[r], "{name}, {label}: ({r}, {c})"),
                            Some(e) => {
                                let fd = (fp[r] - fm[r]) / (2.0 * h);
                                let err = (jac[e] - fd).abs() / scale[r];
                                assert!(
                                    err < 1e-7,
                                    "{name}, {label}: ({r}, {c}) {} vs {fd} ({err:e})",
                                    jac[e]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_pattern_has_both_voltages_and_every_gate() {
        let (rows, cols) = jac_sparsity();
        // 2 x 2 voltages; each of the 9 gates in its voltage's column, and
        // its voltage's row and itself in its own
        assert_eq!(rows.len(), 4 + 9 * 3);
        assert!(cols.windows(2).all(|w| w[0] <= w[1]));
        for (r, c) in [(1, 0), (0, 1), (2, 0), (4, 1), (0, 2), (2, 2), (1, 10)] {
            assert!(rows.iter().zip(&cols).any(|e| e == (&r, &c)), "({r}, {c})");
        }
    }

    #[test]
    fn an_axonal_input_is_driven_by_the_soma_voltage() {
        // TwoCptODE.m: EPSG into the axon still has the soma's driving force
        let s = spec("EPSG", 5.0, 10.0, 100.0);
        let rhs = TwoCptRhs::new(params(2.0), gating(), &s).unwrap();
        let (v1, v2) = (-40.0, -20.0);
        let (i1, i2) = rhs.currents(5.2, v1, v2);
        assert_eq!(i1, 0.0);
        assert!(i2 > 0.0);
        assert_eq!(rhs.currents(5.2, v1, v2 + 30.0), (0.0, i2));
        assert_ne!(rhs.currents(5.2, v1 + 30.0, v2).1, i2);
        // an event in the axon is driven by the axon's own voltage
        let ev = TwoStimSpec {
            events: Some(EventTable {
                t: vec![5.0],
                k: vec![2],
                g: vec![60.0],
                rise: vec![0.1],
                decay: vec![0.18],
                inv_peak: vec![1.0 / 0.21317],
                e_rev: vec![0.0],
            }),
            ..spec("Events", 0.0, 0.0, 1.0)
        };
        let rhs = TwoCptRhs::new(params(1.0), gating(), &ev).unwrap();
        let (_, i2) = rhs.currents(5.2, v1, v2);
        assert_eq!(rhs.currents(5.2, v1 + 30.0, v2).1, i2);
        assert_ne!(rhs.currents(5.2, v1, v2 + 30.0).1, i2);
    }

    #[test]
    fn np_sum_follows_numpy() {
        // numpy: 0.0 + the pairwise sum, which starts from -0.0
        assert_eq!(np_sum(&[]).to_bits(), 0.0f64.to_bits());
        assert_eq!(np_sum(&[-0.0]).to_bits(), 0.0f64.to_bits());
        let a: Vec<f64> = (0..20).map(|i| 0.1 * i as f64).collect();
        let r: [f64; 8] = std::array::from_fn(|j| a[j] + a[j + 8]);
        let want = ((r[0] + r[1]) + (r[2] + r[3]))
            + ((r[4] + r[5]) + (r[6] + r[7]))
            + a[16]
            + a[17]
            + a[18]
            + a[19];
        assert_eq!(np_sum(&a), 0.0 + want);
    }

    #[test]
    fn unknown_and_missing_parameters_are_errors() {
        let mut map: HashMap<String, f64> = HashMap::new();
        map.insert("gC".to_string(), 1.0);
        let e = TwoParams::from_map(&map).unwrap_err();
        assert!(e.0.starts_with("missing two_cpt parameter"), "{e}");
        map.insert("gc".to_string(), 1.0);
        let e = TwoParams::from_map(&map).unwrap_err();
        assert_eq!(e.0, "unknown two_cpt parameter \"gc\"");
    }
}
