"""Simulator for the shared-angle problem.

Three independent channels, labelled A, B and C. Channel i yields a
four-component observation

    X_i = (x_i^+, y_i^+, x_i^-, y_i^-)  ~  Normal(mu_i, V_i)

with mean

    mu_i = r_i * ( cos(delta_i + gamma), sin(delta_i + gamma),
                   cos(delta_i - gamma), sin(delta_i - gamma) ).

The parameters are a single gamma, shared by all three channels, together with
an amplitude r_i >= 0 and an offset delta_i for each channel separately: seven
in all, against twelve measurements. gamma is the parameter of interest; the
r_i and delta_i are nuisance parameters.

Read geometrically, each channel's observation is a pair of plane vectors of
common length r_i whose bisector points along delta_i and which are separated
by an angle 2*gamma. So r_i is their length, delta_i their bisector, and gamma
half the angle between them.

The covariances V_i are KNOWN CONSTANTS. They do not depend on any parameter,
and in particular they do not shrink when r_i does.

This module gives you the data and the covariances, and nothing else: the
likelihood, the fits and all of the studies are yours to write.
"""

import numpy as np

CHANNELS = ['A', 'B', 'C']
CHANNEL_LABELS = ['Channel A', 'Channel B', 'Channel C']

# True values used by the simulator.
#
# You need these for the toy studies - you cannot measure coverage without
# knowing what the truth is - but of course a real analysis would not have
# them. Treat gamma = 65 degrees as "the answer" only when checking coverage.
TRUTH = {
    'gamma':   65.0,
    'r_A':     0.10,   'r_B':     0.15,   'r_C':     0.20,
    'delta_A': 130.0,  'delta_B': 270.0,  'delta_C': 130.0,
}

# Known standard deviations on (x+, y+, x-, y-), one row per channel, in the
# same (arbitrary) units as r. These are absolute and fixed: they are a
# property of the measurement so they do NOT change when the parameters 
# change.
SIGMA = [[0.025, 0.040, 0.025, 0.025],
         [0.045, 0.055, 0.045, 0.045],
         [0.055, 0.075, 0.055, 0.055]]

# Correlation between the x and y components, per channel: [the + pair, the - pair]
RHO = [[-0.10, -0.20],
       [ 0.10, -0.10],
       [ 0.05, -0.10]]


def covariance(channel):
    """The known 4x4 covariance of (x+, y+, x-, y-) for one channel.

    `channel` may be an index (0, 1, 2) or a name ('A', 'B', 'C').
    """
    i = CHANNELS.index(channel) if isinstance(channel, str) else channel
    s = np.asarray(SIGMA[i], dtype=float)
    rp, rm = RHO[i]
    corr = np.array([[1.0,  rp,  0.0, 0.0],
                     [rp,  1.0,  0.0, 0.0],
                     [0.0, 0.0,  1.0, rm],
                     [0.0, 0.0,  rm, 1.0]])
    return np.diag(s) @ corr @ np.diag(s)


def truth_with(**overrides):
    """A copy of TRUTH with some values replaced.

    Handy for the toy studies, e.g. to weaken every channel's amplitude:

        pars = truth_with(r_A=0.02, r_B=0.03, r_C=0.04)
    """
    pars = dict(TRUTH)
    pars.update(overrides)
    return pars


def _expected(gamma, r, delta):
    """Noise-free (x+, y+, x-, y-) for one channel."""
    ap = np.radians(delta + gamma)
    am = np.radians(delta - gamma)
    return r * np.array([np.cos(ap), np.sin(ap), np.cos(am), np.sin(am)])


def simulate(pars=None, rng=None):
    """Draw one realisation of the whole experiment.

    Returns an array of shape (3, 4): one row per channel, holding that
    channel's (x+, y+, x-, y-).

    `pars` is a dict like TRUTH; anything left out falls back to TRUTH, so
    simulate() draws at the nominal truth and simulate({'r_A': 0.01}) weakens
    channel A only. `rng` may be a numpy Generator for reproducibility.
    """
    p = dict(TRUTH)
    if pars:
        p.update(pars)
    rng = np.random.default_rng() if rng is None else rng

    out = np.empty((len(CHANNELS), 4))
    for i, ch in enumerate(CHANNELS):
        mu = _expected(p['gamma'], p[f'r_{ch}'], p[f'delta_{ch}'])
        L = np.linalg.cholesky(covariance(i))
        out[i] = mu + L @ rng.standard_normal(4)
    return out


def observed():
    """THE dataset for Part A - the single realisation everyone analyses.
    Fixed, so your Part A numbers are reproducible and comparable."""
    return simulate(rng=np.random.default_rng(20260101))
