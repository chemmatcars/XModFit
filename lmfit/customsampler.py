import warnings
from itertools import count
from typing import Dict, List, Optional, Union

import numpy as np

from emcee.backends import Backend
from emcee.model import Model
from emcee.moves import StretchMove
from emcee.pbar import get_progress_bar
from emcee.state import State
from emcee.utils import deprecated, deprecation_warning
from emcee import EnsembleSampler
from emcee.ensemble import walkers_independent

class CustomSampler(EnsembleSampler):

    def __init__(self,
        nwalkers,
        ndim,
        log_prob_fn,
        pool=None,
        moves=None,
        args=None,
        kwargs=None,
        backend=None,
        vectorize=False,
        blobs_dtype=None,
        parameter_names: Optional[Union[Dict[str, int], List[str]]] = None):
        super().__init__(nwalkers, ndim, log_prob_fn, pool, moves, args, kwargs, backend, vectorize, blobs_dtype,
                         parameter_names)

    def custom_sample(
            self,
            initial_state,
            log_prob0=None,  # Deprecated
            rstate0=None,  # Deprecated
            blobs0=None,  # Deprecated
            iterations=1,
            angleindex=[],
            tune=False,
            skip_initial_state_check=False,
            thin_by=1,
            thin=None,
            store=True,
            progress=False,
            progress_kwargs=None,
    ):
        """Advance the chain as a generator

        Args:
            initial_state (State or ndarray[nwalkers, ndim]): The initial
                :class:`State` or positions of the walkers in the
                parameter space.
            iterations (Optional[int or NoneType]): The number of steps to generate.
                ``None`` generates an infinite stream (requires ``store=False``).
            tune (Optional[bool]): If ``True``, the parameters of some moves
                will be automatically tuned.
            thin_by (Optional[int]): If you only want to store and yield every
                ``thin_by`` samples in the chain, set ``thin_by`` to an
                integer greater than 1. When this is set, ``iterations *
                thin_by`` proposals will be made.
            store (Optional[bool]): By default, the sampler stores (in memory)
                the positions and log-probabilities of the samples in the
                chain. If you are using another method to store the samples to
                a file or if you don't need to analyze the samples after the
                fact (for burn-in for example) set ``store`` to ``False``.
            progress (Optional[bool or str]): If ``True``, a progress bar will
                be shown as the sampler progresses. If a string, will select a
                specific ``tqdm`` progress bar - most notable is
                ``'notebook'``, which shows a progress bar suitable for
                Jupyter notebooks.  If ``False``, no progress bar will be
                shown.
            progress_kwargs (Optional[dict]): A ``dict`` of keyword arguments
                to be passed to the tqdm call.
            skip_initial_state_check (Optional[bool]): If ``True``, a check
                that the initial_state can fully explore the space will be
                skipped. (default: ``False``)


        Every ``thin_by`` steps, this generator yields the
        :class:`State` of the ensemble.

        """
        if iterations is None and store:
            raise ValueError("'store' must be False when 'iterations' is None")
        # Interpret the input as a walker state and check the dimensions.
        state = State(initial_state, copy=True)
        state_shape = np.shape(state.coords)
        if state_shape != (self.nwalkers, self.ndim):
            raise ValueError(f"incompatible input dimensions {state_shape}")
        if (not skip_initial_state_check) and (
                not walkers_independent(state.coords)
        ):
            raise ValueError(
                "Initial state has a large condition number. "
                "Make sure that your walkers are linearly independent for the "
                "best performance"
            )

        # Try to set the initial value of the random number generator. This
        # fails silently if it doesn't work but that's what we want because
        # we'll just interpret any garbage as letting the generator stay in
        # it's current state.
        if rstate0 is not None:
            deprecation_warning(
                "The 'rstate0' argument is deprecated, use a 'State' "
                "instead"
            )
            state.random_state = rstate0
        self.random_state = state.random_state

        # If the initial log-probabilities were not provided, calculate them
        # now.
        if log_prob0 is not None:
            deprecation_warning(
                "The 'log_prob0' argument is deprecated, use a 'State' "
                "instead"
            )
            state.log_prob = log_prob0
        if blobs0 is not None:
            deprecation_warning(
                "The 'blobs0' argument is deprecated, use a 'State' instead"
            )
            state.blobs = blobs0
        if state.log_prob is None:
            state.log_prob, state.blobs = self.compute_log_prob(state.coords)
        if np.shape(state.log_prob) != (self.nwalkers,):
            raise ValueError("incompatible input dimensions")

        # Check to make sure that the probability function didn't return
        # ``np.nan``.
        if np.any(np.isnan(state.log_prob)):
            raise ValueError("The initial log_prob was NaN")

        # Deal with deprecated thin argument
        if thin is not None:
            deprecation_warning(
                "The 'thin' argument is deprecated. " "Use 'thin_by' instead."
            )

            # Check that the thin keyword is reasonable.
            thin = int(thin)
            if thin <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = 1
            checkpoint_step = thin
            if store:
                nsaves = iterations // checkpoint_step
                self.backend.grow(nsaves, state.blobs)

        else:
            # Check that the thin keyword is reasonable.
            thin_by = int(thin_by)
            if thin_by <= 0:
                raise ValueError("Invalid thinning argument")

            yield_step = thin_by
            checkpoint_step = thin_by
            if store:
                self.backend.grow(iterations, state.blobs)

        # Set up a wrapper around the relevant model functions
        if self.pool is not None:
            map_fn = self.pool.map
        else:
            map_fn = map
        model = Model(
            self.log_prob_fn, self.compute_log_prob, map_fn, self._random
        )
        if progress_kwargs is None:
            progress_kwargs = {}

        # Determine presence of angle variables
        thetaind = np.where(angleindex == 1)[0]
        phiind = np.where(angleindex == 2)[0]

        # Inject the progress bar
        total = None if iterations is None else iterations * yield_step
        with get_progress_bar(progress, total, **progress_kwargs) as pbar:
            i = 0
            for _ in count() if iterations is None else range(iterations):
                for _ in range(yield_step):
                    # Choose a random move
                    move = self._random.choice(self._moves, p=self._weights)

                    # Propose
                    state, accepted = move.propose(model, state)
                    state.random_state = self.random_state

                    if tune:
                        move.tune(state, accepted)

                    # Wrap angle values
                    if len(thetaind) != 0 and len(phiind) != 0:
                        theta = state.coords[:, thetaind]
                        phi = state.coords[:, phiind]
                        state.coords[:, thetaind] = np.piecewise(theta,
                                                      [np.logical_and((theta % 360) >= 0, (theta % 360) <= 180),
                                                           np.logical_and((theta % 360) > 180, (theta % 360) < 360)],
                                                          [lambda x: x % 360, lambda x: 360 - (x % 360)])
                        state.coords[:, phiind] = np.piecewise(phi,
                                                          [np.logical_and((theta % 360) >= 0, (theta % 360) <= 180),
                                                           np.logical_and((theta % 360) > 180, (theta % 360) < 360)],
                                                          [lambda x: x % 360, lambda x: (x + 180) % 360])
                    elif len(thetaind) != 0 and len(phiind) == 0:
                        theta = state.coords[:, thetaind]
                        state.coords[:, thetaind] = np.piecewise(theta,
                                                      [np.logical_and((theta % 360) >= 0, (theta % 360) <= 180),
                                                           np.logical_and((theta % 360) > 180, (theta % 360) < 360)],
                                                          [lambda x: x % 360, lambda x: 360 - (x % 360)])
                    elif len(thetaind) == 0 and len(phiind) != 0:
                        phi = state.coords[:, phiind]
                        state.coords[:, phiind] = phi % 360


                    # Save the new step
                    if store and (i + 1) % checkpoint_step == 0:
                        self.backend.save_step(state, accepted)

                    pbar.update(1)
                    i += 1

                # Yield the result as an iterator so that the user can do all
                # sorts of fun stuff with the results so far.
                yield state


    def custom_run_mcmc(self, initial_state, nsteps, angleindex, **kwargs):

        """
               Iterate :func:`sample` for ``nsteps`` iterations and return the result

               Args:
                   initial_state: The initial state or position vector. Can also be
                       ``None`` to resume from where :func:``run_mcmc`` left off the
                       last time it executed.
                   nsteps: The number of steps to run.

               Other parameters are directly passed to :func:`sample`.

               This method returns the most recent result from :func:`sample`.

               """
        if initial_state is None:
            if self._previous_state is None:
                raise ValueError(
                    "Cannot have `initial_state=None` if run_mcmc has never "
                    "been called."
                )
            initial_state = self._previous_state

        results = None
        for results in self.custom_sample(initial_state, iterations=nsteps, angleindex=angleindex, **kwargs):
            pass

        # Store so that the ``initial_state=None`` case will work
        self._previous_state = results

        return results