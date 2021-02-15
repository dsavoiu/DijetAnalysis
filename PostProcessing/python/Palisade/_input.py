from __future__ import print_function

import ast
import functools
import ROOT
import numpy as np
import operator as op
import os
import pandas as pd
import six
import sys
import uuid

from array import array

from rootpy import asrootpy
from rootpy.io import root_open, DoesNotExist
from rootpy.plotting import Hist1D, Hist2D, Profile1D, Efficiency, Graph
from rootpy.plotting.hist import _Hist, _Hist2D
from rootpy.plotting.profile import _ProfileBase

if six.PY2:
    from collections import Mapping
elif six.PY3:
    from collections.abc import Mapping

try:
    # import text similarity metric for more helpful alternative suggestions
    if six.PY2:
        from textdistance import jaccard as text_similarity_metric
    elif six.PY3:
        from textdistance import ratcliff_obershelp as text_similarity_metric
except ImportError:
    # use dummy similarity metric (will not sort suggestions by likelihood)
    def text_similarity_metric(s1, s2):
        return 1

import scipy.stats as stats


__all__ = ['InputROOTFile', 'InputROOT']


class HashableMap(Mapping):
    """A hashable, immutable mapping type.

    The arguments to ``HashableMap`` are processed the same way as those to ``dict``.
    Both the keys and the values contained in the map must be hashable.

    An exception are `lists`, which are converted to `tuple` in order  to make them
    hashable.

    Two `HashableMap` instances with identical keys and values are guaranteed to be
    identical. "Identical" in this context means having the same hash.
    """

    def __init__(self, *args, **kwargs):

        self._d = dict(*args, **kwargs)

        # validate values
        for key, value in six.iteritems(self._d):
            # check hashability
            try:
                hash(value)
            except TypeError:
                # attempt to wrap non-hashable types
                if isinstance(value, list) or isinstance(value, tuple):
                    value = tuple(enumerate(value))
                elif isinstance(value, dict):
                    value = tuple(six.iteritems(value))
                else:
                    raise

                self._d[key] = HashableMap(value)

        self._hash = None  # computed on request

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # compute hash if not available
        if self._hash is None:
            # hash computed from hashes of keys and values
            self._hash = 0
            for key, value in self.iteritems():
                self._hash ^= hash(key)
                self._hash ^= hash(value)

        return self._hash

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self._d)


class _ROOTObjectFunctions(object):

    @staticmethod
    def get_all():
        """return all methods not beginning with a '_'"""
        return {
            _method : getattr(_ROOTObjectFunctions, _method)
            for _method in dir(_ROOTObjectFunctions)
            if not _method.startswith('_') and callable(getattr(_ROOTObjectFunctions, _method))
        }

    @staticmethod
    def _project_or_clone(tobject, projection_options=None):

        if isinstance(tobject, _ProfileBase):
            # create an "x-projection" with a unique suffix
            if projection_options is None:
                return asrootpy(tobject.ProjectionX(uuid.uuid4().get_hex()))
            else:
                return asrootpy(tobject.ProjectionX(uuid.uuid4().get_hex(), projection_options))
        else:
            return tobject.Clone()

    @staticmethod
    def hist(tobject):
        """Turn a profile in to a histogram."""

        if not isinstance(tobject, _ProfileBase):
            return tobject.Clone()  # return direct copy

        _h = tobject.empty_clone(type='D')  # TODO: avoid hardcoding

        for _b_in, _b_out in zip(tobject, _h):
            _b_out.value = _b_in.value
            _b_out.error = _b_in.error

        return _h

    @staticmethod
    def histdivide(tobject_1, tobject_2, option=""):
        """divide two histograms, taking error calculation option into account"""

        _new_tobject_1 = _ROOTObjectFunctions._project_or_clone(tobject_1)
        _new_tobject_2 = _ROOTObjectFunctions._project_or_clone(tobject_2)

        _new_tobject_1.Divide(_new_tobject_1, _new_tobject_2, 1, 1, option)

        return _new_tobject_1

    @staticmethod
    def max_yield_index(yields, efficiencies, eff_threshold):
        """for each bin, return index of object in `yields` which is maximizes yield, subject to the efficiency remaining above threshold"""

        # `yields` and `efficiencies` must have the same length
        assert len(yields) == len(efficiencies)
        # all `yields` and `efficiencies` must have the same number of bins
        assert all([len(_tobj_yld) == len(yields[0]) for _tobj_yld in yields[1:]])

        _new_tobject = _ROOTObjectFunctions._project_or_clone(yields[0])

        for _bin_idx in range(len(yields[0])):
            _max_yield_for_bin = 0
            _max_yield_obj_idx = -1

            for _obj_index, (_tobj_yld, _tobj_eff) in enumerate(zip(yields, efficiencies)):
                # skip bins with efficiency below threshold
                if _tobj_eff[_bin_idx].value < eff_threshold:
                    continue
                # keep index of object with maximum yield
                if _tobj_yld[_bin_idx].value > _max_yield_for_bin:
                    _max_yield_for_bin = _tobj_yld[_bin_idx].value
                    _max_yield_obj_idx = _obj_index

            _new_tobject[_bin_idx].value = _max_yield_obj_idx
            _new_tobject[_bin_idx].error = 0

        return _new_tobject

    @staticmethod
    def max_value_index(tobjects):
        """for each bin *i*, return index of object in `tobjects` which contains the largest value for bin *i*"""

        # all `tobjects` must have the same number of bins
        assert all([len(_tobj) == len(tobjects[0]) for _tobj in tobjects[1:]])

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobjects[0])

        for _bin_idx in range(len(tobjects[0])):
            _max_yield_for_bin = 0
            _max_yield_obj_idx = -1

            for _obj_index, _tobj in enumerate(tobjects):
                # keep index of object with maximum yield
                if _tobj[_bin_idx].value > _max_yield_for_bin:
                    _max_yield_for_bin = _tobj[_bin_idx].value
                    _max_yield_obj_idx = _obj_index

            _new_tobject[_bin_idx].value = _max_yield_obj_idx
            _new_tobject[_bin_idx].error = 0

        return _new_tobject

    @staticmethod
    def select(tobjects, indices):
        """the content of each bin *i* in the return object is taken from the object whose index in `tobjects` is given by bin *i* in `indices`"""
        # if indices are outside the range of `tobjects`, the bins are set to zero
        # all `yields` and `efficiencies` must have the same binning
        # `yields` and `efficiencies` must have the same length
        assert len(tobjects) > 0
        assert len(tobjects[0]) == len(indices)
        # all `tobjects` must have the same number of bins
        assert all([len(_tobj) == len(tobjects[0]) for _tobj in tobjects[1:]])

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobjects[0])

        for _i_bin, (_bin_proxy, _obj_idx) in enumerate(zip(_new_tobject, indices)):
            # range check
            if _obj_idx.value >= 0 and _obj_idx.value < len(tobjects):
                _bin_proxy.value = tobjects[int(_obj_idx.value)][_i_bin].value
                _bin_proxy.error = tobjects[int(_obj_idx.value)][_i_bin].error
            else:
                _bin_proxy.value = 0
                _bin_proxy.error = 0

        return _new_tobject

    @staticmethod
    def mask_lookup_value(tobject, tobject_lookup, lookup_value):
        """bin *i* in return object is bin *i* in `tobject` if bin *i* in `tobject_lookup` is equal to `lookup_value`"""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)

        for _bin_proxy, _bin_proxy_lookup in zip(_new_tobject, tobject_lookup):
            # mask bins where looked up value is different than the reference
            if _bin_proxy_lookup.value != lookup_value:
                _bin_proxy.value = 0
                _bin_proxy.error = 0

        return _new_tobject

    @staticmethod
    def apply_efficiency_correction(tobject, efficiency, threshold=None):
        """Divide each bin in `tobject` by the corresponding bin in `efficiency`. If `efficiency` is lower than `threshold`, the number of events is set to zero."""
        assert len(efficiency) == len(tobject)

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)

        for _eff, _bin_proxy in zip(efficiency.efficiencies(), _new_tobject):
            if (_eff <= 0) or (threshold is not None and _eff < threshold):
                _bin_proxy.value = 0
                _bin_proxy.error = 0
            else:
                _bin_proxy.value /= _eff
                _bin_proxy.error /= _eff  # should this be done?

        return _new_tobject

    @staticmethod
    def efficiency(tobject_numerator, tobject_denominator):
        """Compute TEfficiency"""

        return Efficiency(tobject_numerator, tobject_denominator)

    @staticmethod
    def efficiency_graph(tobject_numerator, tobject_denominator):
        """Compute TEfficiency with proper clopper-pearson intervals"""

        _eff = Efficiency(tobject_numerator, tobject_denominator)
        return asrootpy(_eff.CreateGraph())

    @staticmethod
    def project_x(tobject):
        """Apply ProjectionX() operation."""

        if hasattr(tobject, 'ProjectionX'):
            _new_tobject = asrootpy(tobject.ProjectionX(uuid.uuid4().get_hex()))
        else:
            print("[INFO] `project_x` not available for object with type {}".format(type(tobject)))
            return tobject

        return _new_tobject

    @staticmethod
    def project_y(tobject):
        """Apply ProjectionY() operation."""

        if hasattr(tobject, 'ProjectionY'):
            _new_tobject = asrootpy(tobject.ProjectionY(uuid.uuid4().get_hex()))
        else:
            raise ValueError("`project_y` not available for object with type {}".format(type(tobject)))

        return _new_tobject

    @staticmethod
    def diagonal(th2d):
        """Return a TH1D containing the main diagonal of an input TH2D."""

        if hasattr(th2d, 'ProjectionX'):
            _new_tobject = asrootpy(th2d.ProjectionX(uuid.uuid4().get_hex()))
        else:
            raise ValueError("`diagonal` not available for object with type {}".format(type(tobject)))

        for _bp in th2d:
            if _bp.xyz[0] == _bp.xyz[1]:
                _new_tobject[_bp.xyz[0]].value = _bp.value
                _new_tobject[_bp.xyz[0]].error = _bp.error

        return _new_tobject

    @staticmethod
    def yerr(tobject):
        """replace bin value with bin error and set bin error to zero"""

        # project preserving errors
        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject, "e")

        for _bin_proxy in _new_tobject:
            _bin_proxy.value, _bin_proxy.error = _bin_proxy.error, 0

        return _new_tobject

    @staticmethod
    def atleast(tobject, min_value):
        """mask all values below threshold"""

        # project preserving errors
        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject, "e")

        for _bin_proxy in _new_tobject:
            if hasattr(_bin_proxy, 'graph_'):
                # for TGraph etc.
                if _bin_proxy.y.value < min_value:
                    _bin_proxy.y.value = 0
                    _bin_proxy.y.error_hi = 0
                    # 'low' error setter has a bug in rootpy. workaround:
                    _bin_proxy.graph_.SetPointEYlow(_bin_proxy.idx_, 0)
            else:
                # for TH1D etc.
                if _bin_proxy.value < min_value:
                    _bin_proxy.value, _bin_proxy.error = 0, 0


        return _new_tobject

    @staticmethod
    def threshold(tobject, min_value):
        """returns a histogram like tobject with bins set to zero if they fall
        below the miminum value and to one if not. Errors are always set to zero"""

        # project preserving errors
        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)

        for _bin_proxy in _new_tobject:
            if _bin_proxy.value < min_value:
                _bin_proxy.value, _bin_proxy.error = 0, 0
            else:
                _bin_proxy.value, _bin_proxy.error = 1, 0

        return _new_tobject

    @staticmethod
    def discard_errors(tobject):
        """set all bin errors to zero"""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)

        for _bin_proxy in _new_tobject:
            if hasattr(_bin_proxy, 'graph_'):
                # for TGraph etc.
                _bin_proxy.y.error_hi = 0
                # 'low' error setter has a bug in rootpy. workaround:
                _bin_proxy.graph_.SetPointEYlow(_bin_proxy.idx_, 0)
            else:
                # for TH1D etc.
                _bin_proxy.error = 0

        return _new_tobject

    @staticmethod
    def bin_width(tobject):
        """replace bin value with width of bin and set bin error to zero"""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)

        for _bin_proxy in _new_tobject:
            _bin_proxy.value, _bin_proxy.error = _bin_proxy.x.width, 0

        return _new_tobject

    @staticmethod
    def max(*tobjects):
        """binwise `max` for a collection of histograms with identical binning"""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobjects[0], "e")

        _tobj_clones = []
        for _tobj in tobjects:
            _tobj_clones.append(_ROOTObjectFunctions._project_or_clone(_tobj, "e"))

        for _bin_proxies in zip(_new_tobject, *_tobj_clones):
            _argmax = max(range(1, len(_bin_proxies)), key=lambda idx: _bin_proxies[idx].value)
            _bin_proxies[0].value = _bin_proxies[_argmax].value
            _bin_proxies[0].error = _bin_proxies[_argmax].error

        # cleanup
        for _tobj_clone in _tobj_clones:
            _tobj_clone.Delete()

        return _new_tobject

    @staticmethod
    def max_val_min_err(*tobjects):
        """binwise 'max' on value followed by a binwise 'min' on error."""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobjects[0], "e")

        _tobj_clones = []
        for _tobj in tobjects:
            _tobj_clones.append(_ROOTObjectFunctions._project_or_clone(_tobj, "e"))

        for _bin_proxies in zip(_new_tobject, *_tobj_clones):
            _maxval = None
            for _bin_proxy in _bin_proxies[1:]:
                if _maxval is None or _bin_proxy.value > _maxval:
                    _maxval = _bin_proxy.value
                    _minerr = _bin_proxy.error
                elif _maxval == _bin_proxy.value and _bin_proxy.error < _minerr:
                    _minerr = _bin_proxy.error
            _bin_proxies[0].value = _maxval
            _bin_proxies[0].error = _minerr

        # cleanup
        for _tobj_clone in _tobj_clones:
            _tobj_clone.Delete()

        return _new_tobject

    @staticmethod
    def mask_if_less(tobject, tobject_ref):
        """set `tobject` bins and their errors to zero if their content is less than the value in `tobject_ref`"""

        assert len(tobject) == len(tobject_ref)

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject, "e")
        _new_tobject_ref = _ROOTObjectFunctions._project_or_clone(tobject_ref, "e")

        for _bin_proxy, _bin_proxy_ref in zip(_new_tobject, _new_tobject_ref):
            if hasattr(_bin_proxy, 'graph_'):
                # for TGraph etc.
                if _bin_proxy.y < _bin_proxy_ref.y:
                    _bin_proxy.y.value = 0
                    _bin_proxy.y.error_hi = 0
                    # 'low' error setter has a bug in rootpy. workaround:
                    _bin_proxy.graph_.SetPointEYlow(_bin_proxy.idx_, 0)
            else:
                # for TH1D etc.
                if _bin_proxy.value < _bin_proxy_ref.value:
                    _bin_proxy.value, _bin_proxy.error = 0, 0

        # cleanup
        _new_tobject_ref.Delete()

        return _new_tobject

    @staticmethod
    def double_profile(tprofile_x, tprofile_y):
        """creates a graph with points whose x and y values and errors are taken from the bins of two profiles with identical binning"""
        ## Note: underflow and overflow bins are discarded

        if len(tprofile_x) != len(tprofile_y):
            raise ValueError("Cannot build double profile: x and y profiles "
                             "have different number or bins ({} and {})".format(
                                len(tprofile_x)-2, len(tprofile_y)-2))

        _dp_graph = Graph(len(tprofile_x)-2, type='errors')  # symmetric errors

        _i_point = 0
        for _i_bin, (_bin_proxy_x, _bin_proxy_y) in enumerate(zip(tprofile_x, tprofile_y)):
            # disregard overflow/underflow bins
            if _i_bin == 0 or _i_bin == len(tprofile_x) - 1:
                continue

            if _bin_proxy_y.value:
                _dp_graph.SetPoint(_i_point, _bin_proxy_x.value, _bin_proxy_y.value)
                _dp_graph.SetPointError(_i_point, _bin_proxy_x.error, _bin_proxy_y.error)
                _i_point += 1

        # remove "unfilled" points
        while (_dp_graph.GetN() > _i_point):
            _dp_graph.RemovePoint(_dp_graph.GetN()-1)

        return _dp_graph

    @staticmethod
    def threshold_by_ref(tobject, tobject_ref):
        """set `tobject` bins to zero if their content is less than the value in `tobject_ref`, and to 1 otherwise.
        Result bin errors are always set to zero."""

        assert len(tobject) == len(tobject_ref)

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject)
        _new_tobject_ref = _ROOTObjectFunctions._project_or_clone(tobject_ref)

        for _bin_proxy, _bin_proxy_ref in zip(_new_tobject, _new_tobject_ref):
            if _bin_proxy.value < _bin_proxy_ref.value:
                _bin_proxy.value, _bin_proxy.error = 0, 0
            else:
                _bin_proxy.value, _bin_proxy.error = 1, 0

        # cleanup
        _new_tobject_ref.Delete()

        return _new_tobject

    @staticmethod
    def normalize_x(tobject):
        """Normalize bin contents of each x slice of a TH2D by dividing by the y integral over each x slice."""

        if not isinstance(tobject, _Hist2D):
            raise ValueError("Cannot apply function `normalize_x` to object of type '{}': must be Hist2D [TH2D]!".format(type(tobject)))

        _new_tobject = asrootpy(tobject.Clone())
        #_projection = asrootpy(tobject.ProjectionX(uuid.uuid4().get_hex(), 1, len(list(tobject.y()))-1))
        _projection = asrootpy(tobject.ProjectionX(uuid.uuid4().get_hex()))

        # divide 2D bin contents by integral over x slice (= result of ProjectionX())
        for _bin_proxy in _new_tobject:
            if _projection[_bin_proxy.xyz[0]].value:
                _bin_proxy.value /= _projection[_bin_proxy.xyz[0]].value
                _bin_proxy.error /= _projection[_bin_proxy.xyz[0]].value
            else:
                _bin_proxy.value, _bin_proxy.error = 0, 0

        _projection.Delete()  # cleanup

        return _new_tobject

    @staticmethod
    def unfold(th1d_input, th2d_response, th1d_marginal_gen, th1d_marginal_reco):
        """
        Use TUnfold to unfold a reconstructed spectrum.

        Parameters
        ----------
            th1d_input : `ROOT.TH1D`
                measured distribution to unfold

            th2d_response :`ROOT.TH2D`
                2D response histogram. Contains event numbers
                per (gen, reco) bin **after** rejecting spurious reconstructions
                and accounting for losses due to the reco acceptance.
                Gen bins should be on the `x` axis.
                Overflow/underflow should not be present and will be ignored!
                Acceptance losses and spurious reconstructions ("fakes") are
                inferred from the difference between the projections of the response
                and the full marginal distributions, which are given separately.

            th1d_marginal_gen : `ROOT.TH1D`
                marginal distribution on gen-level
                Contains event numbers per gen bin, **without** accounting for
                losses due to detector acceptance. The losses are inferred
                by comparing to the projection of the 2D response histogram,
                where these losses are accounted for.

            th1d_marginal_reco : `ROOT.TH1D`
                marginal distribution on reco-level
                Contains event numbers per reco bin, **without** subtracting
                spurious reconstructions ("fakes"). The fakes are inferred
                by comparing to the projection of the 2D response histogram,
                where these fakes are not present.
        """

        # input sanity checks
        _nbins_gen = th1d_marginal_gen.GetNbinsX()
        _nbins_reco = th1d_marginal_reco.GetNbinsX()
        assert(th2d_response.GetNbinsX() == _nbins_gen)
        assert(th2d_response.GetNbinsY() == _nbins_reco)
        assert(th2d_response.GetNbinsY() == _nbins_reco)

        _th2d_response_clone = asrootpy(th2d_response.Clone())
        _th1d_input_clone = asrootpy(th1d_input.Clone())

        # determine relative fake rate per reco. bin
        _th1d_true_reco = asrootpy(th2d_response.ProjectionY(uuid.uuid4().get_hex()))
        _th1d_true_fraction_reco = _th1d_true_reco / th1d_marginal_reco

        # determine absolute number of lost events per gen. bin
        _th1d_accepted_gen = asrootpy(th2d_response.ProjectionX(uuid.uuid4().get_hex()))
        _th1d_rejected_gen = th1d_marginal_gen - _th1d_accepted_gen

        # correct reco. distribution for fakes using inferred true fraction
        _th1d_input_clone = _th1d_true_fraction_reco * _th1d_input_clone

        # get rid of gen-level underflow/overflow:
        for _bin_proxy in _th2d_response_clone:
            # get rid of gen-level underflow/overflow:
            if _bin_proxy.xyz[0] == 0 or _bin_proxy.xyz[0] == _nbins_gen + 1:
                _bin_proxy.value = 0
                _bin_proxy.error = 0
            # fill losses into y-underflow bins
            elif _bin_proxy.xyz[1] == 0:
                _bin_proxy.value = _th1d_rejected_gen[_bin_proxy.xyz[0]].value
                _bin_proxy.error = _th1d_rejected_gen[_bin_proxy.xyz[0]].error

        # construct TUnfold instance and perform unfolding
        _tunfold = ROOT.TUnfold(
            _th2d_response_clone,
            ROOT.TUnfold.kHistMapOutputHoriz,   # gen-level on x axis
            ROOT.TUnfold.kRegModeNone,          # no regularization
            ROOT.TUnfold.kEConstraintNone,      # no constraints
        )
        # ensure maximal numerical precision on matrix inversion
        _tunfold.SetEpsMatrix(sys.float_info.epsilon)
        # set the unfolding input
        _tunfold.SetInput(
            _th1d_input_clone
        )
        _tunfold.DoUnfold(0)

        _th1d_output = th2d_response.ProjectionX(uuid.uuid4().get_hex())
        _tunfold.GetOutput(_th1d_output)

        _th2d_response_clone.Delete()
        _th1d_input_clone.Delete()
        _th1d_accepted_gen.Delete()
        _th1d_rejected_gen.Delete()
        _th1d_true_reco.Delete()
        _th1d_true_fraction_reco.Delete()
        _tunfold.Delete()

        return asrootpy(_th1d_output)

    @staticmethod
    def normalize_to_ref(tobject, tobject_ref):
        """Normalize `tobject` to the integral over `tobject_ref`."""

        _new_tobject = asrootpy(tobject.Clone())
        if tobject.integral():
            _factor = float(tobject_ref.integral()) / float(tobject.integral())

            return _new_tobject * _factor
        else:
            return _new_tobject

    @staticmethod
    def cumulate(tobject):
        """Make value of n-th bin equal to the sum of all bins up to and including n (but excluding underflow bins)."""
        #                                     forward  suffix
        return asrootpy(tobject.GetCumulative(True,    uuid.uuid4().get_hex()))

    @staticmethod
    def cumulate_reverse(tobject):
        """Make value of n-th bin equal to the sum of all bins from n up to and inclufing the last bin (but excluding overflow bins)."""
        #                                     forward  suffix
        return asrootpy(tobject.GetCumulative(False,   uuid.uuid4().get_hex()))

    @staticmethod
    def bin_differences(tobject):
        """Make value of n-th bin equal to the difference between the n-th and (n-1)-th bins."""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject, "e")

        for _bin_proxy, _bin_proxy_1, _bin_proxy_2 in zip(_new_tobject[1:-2], tobject[2:-1], tobject[1:-2]):
            if _bin_proxy_2.value:
                _bin_proxy.value = _bin_proxy_1.value - _bin_proxy_2.value
            else:
                _bin_proxy.error = 0
            _bin_proxy.error = 0

        return _new_tobject

    @staticmethod
    def bin_ratios(tobject):
        """Make value of n-th bin equal to the ratio between the n-th and (n-1)-th bins."""

        _new_tobject = _ROOTObjectFunctions._project_or_clone(tobject, "e")

        for _bin_proxy, _bin_proxy_num, _bin_proxy_denom in zip(_new_tobject[1:-2], tobject[2:-1], tobject[1:-2]):
            if _bin_proxy_denom.value:
                _bin_proxy.value = _bin_proxy_num.value / _bin_proxy_denom.value
            else:
                _bin_proxy.error = 0
            _bin_proxy.error = 0

        return _new_tobject


class InputROOTFile(object):
    """An input module for accessing objects from a single ROOT file.

    Multiple objects can be requested. They will be all be retrieved
    simultaneously and cached on the first subsequent call to `get()`.
    The file will thus only be opened once.

    Usage example:

    .. code:: python

       m = InputROOTFile('/path/to/rootfile.root')

       m.request(dict(object_path='MyDirectory/myObject'))
       my_object = m.get('MyDirectory/myObject')
    """

    def __init__(self, filename):
        self._filename = filename
        self._outstanding_requests = dict()
        self._plot_data_cache = dict()

    def _process_outstanding_requests(self):
        # if no requests, return immediately
        if not self._outstanding_requests:
            return

        # process outstanding requests
        with root_open(self._filename) as _tfile:
            for tobj_path, request_spec in six.iteritems(self._outstanding_requests):
                _rebin_factor = request_spec.pop('rebin_factor', None)
                _profile_error_option = request_spec.pop('profile_error_option', None)

                try:
                    _tobj = _tfile.Get(tobj_path)
                except DoesNotExist as e:
                    _available_paths = ['{}/{}'.format(_path, _obj_name) for _path, _, _obj_names in _tfile.walk() for _obj_name in _obj_names]
                    if not _available_paths:
                        six.raise_from(DoesNotExist(e.args[0] + " (file does not seem to contain any objects)."), e)

                    # sort available paths by similarity to original query
                    _sim_lambda = lambda s: text_similarity_metric(tobj_path, s)
                    _available_paths.sort(key=lambda s: text_similarity_metric(tobj_path, s), reverse=True)

                    # raise more informative exception
                    six.raise_from(DoesNotExist(e.args[0] + ". Did you mean '{}'?".format(_available_paths[0].strip('/'))), e)

                # for histograms: move to global directory
                try:
                    _tobj.SetDirectory(0)
                except AttributeError:
                    # call not needed to other objects
                    pass
                #print(tobj_path, _tobj)

                # aply rebinning (if requested)
                if _rebin_factor is not None:
                    _tobj.Rebin(_rebin_factor)

                # set TProfile error option (if requested)
                if _profile_error_option is not None:
                    # TOOD: check if profile?
                    _tobj.SetErrorOption(_profile_error_option)

                self._plot_data_cache[tobj_path] = _tobj

        self._outstanding_requests = dict()


    def get(self, object_path):
        """
        Get an object.

        Parameters
        ----------
            object_path : string, path to resource in ROOT file (e.g. "directory/object")
        """
        # request object if not present
        if object_path not in self._outstanding_requests:
            if object_path not in self._plot_data_cache:
                self.request([dict(object_path=object_path)])

        # process request if object is waiting
        if object_path in self._outstanding_requests:
            self._process_outstanding_requests()

        # return object
        return self._plot_data_cache[object_path]

    def request(self, request_specs):
        """
        Request an object. Requested objects are all retrieved in one
        go when one of them is retrived via 'get()'

        Parameters
        ----------
            request_specs : e.g. dict(object_path="directory/object")
        """
        for request_spec in request_specs:
            _object_path = request_spec.pop('object_path')
            _force_rerequest = request_spec.pop('force_rerequest', True)

            # override earlier request iff 'force_rerequest' is True
            if (not (_object_path in self._outstanding_requests or _object_path in self._plot_data_cache)) or _force_rerequest:
                self._outstanding_requests[_object_path] = request_spec

                if _object_path in self._plot_data_cache:
                    del self._plot_data_cache[_object_path]

    def clear(self):
        """
        Remove all cached data and outstanding requests.
        """
        self._plot_data_cache = {}
        self._outstanding_requests = {}


class InputROOT(object):
    """An input module for accessing objects from multiple ROOT files.

    A nickname can be registered for each file, which then allows
    object retrieval by prefixing it to the object path
    (i.e. ``<file_nickname>:<object_path_in_file>``).

    Single-file functionality is delegated to child
    :py:class:`~DijetAnalysis.PostProcessing.Palisade.InputROOTFile` objects.
    """


    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.BitXor: op.xor,
        ast.USub: op.neg
    }

    # input functions (meant to be applied to ROOT objects in files)
    functions = dict(
        _ROOTObjectFunctions.get_all(),
        # add some useful aliases
        **{
            'h':                                   _ROOTObjectFunctions.hist,  # alias
        }
    )

    # class-level cache for storing memoized function results
    _cache = {}

    def __init__(self, files_spec=None):
        """
        Parameters
        ----------
            files_spec : `dict`, optional
                specification of file nicknames (keys) and paths pointed to (values).
                Can be omitted (files can be added later via
                :py:class:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.add_file`)

        Usage example:

        .. code:: python

            m = InputROOT()

            # add a file and register a nickname for it
            m.add_file('/path/to/rootfile.root', nickname='file0')

            # optional: request object first (retrieves several objects at once)
            m.request(dict(file_nickname='file0', object_path='MyDirectory/myObject'))

            # retrieve an object from a file
            my_object = m.get('file0:MyDirectory/myObject')

            # apply simple arithmetical expressions to objects
            my_sum_object = m.get_expr('"file0:MyDirectory1/myObject1" + "file0:MyDirectory2/myObject2"')

            # use basic functions in expressions
            my_object_noerrors = m.get_expr('discard_errors("file0:MyDirectory1/myObject1")')

            # register user-defined input functions
            InputRoot.add_function(my_custom_function)  # `my_custom_function` defined beforehand

            # use function in expression:
            my_function_result = m.get_expr('my_custom_function("file0:MyDirectory1/myObject1")')
        """
        self._input_controllers = {}
        self._file_nick_to_realpath = {}
        self._locals = {}
        if files_spec is not None:
            for _nickname, _file_path in six.iteritems(files_spec):
                self.add_file(_file_path, nickname=_nickname)

    def _get_input_controller_for_file(self, file_spec):
        '''get input controller for file specification. handle nickname resolution'''
        _file_realpath = self._file_nick_to_realpath.get(file_spec, file_spec)
        if _file_realpath not in self._input_controllers:
            raise ValueError("Cannot get input controller for file specification '{}': have you added a file with this nickname or path?".format(file_spec))
        return self._input_controllers[_file_realpath]


    @staticmethod
    def _get_file_nickname_and_obj_path(object_spec):
        _file_nickname, _object_path_in_file = object_spec.split(':', 1)
        return _file_nickname, _object_path_in_file

    @classmethod
    def add_function(cls, function=None, name=None, override=False, memoize=False):
        '''Register a user-defined input function. Can also be used as a decorator.

        .. note::

            Functions are registered **globally** in the ``InputROOT`` **class** and are
            immediately available to all ``InputROOT`` instances.

        Parameters
        ----------
            function : `function`, optional
                function or method to add. Can be omitted when used as a decorator.
            name : `str`, optional
                function name. If not given, taken from ``function.__name__``
            override : `bool`, optional
                if ``True``, allow existing functions to be overridden (*default*: ``False``)
            memoize : `bool`, optional
                if ``True``, store function result in a cache on first call. For every
                subsequent call with identical arguments, the result will be retrieved
                from the cache instead of evaluating the function again.
                (*default*: ``False``)

        Usage examples:


        * as a simple decorator:

          .. code:: python

             @InputROOT.add_function
             def my_function(rootpy_object):
                 ...

        * to override a function that has already been registered:

          .. code:: python

             @InputROOT.add_function(override=True)
             def my_function(rootpy_object):
                 ...

        * to register a function under a different name:

          .. code:: python

             @InputROOT.add_function(name='short_name')
             def very_long_fuction_name_we_do_not_want_to_use_in_expressions(rootpy_object):
                     ...

        * as a method:

          .. code:: python

             InputROOT.add_function(my_function)

        .. note::

            All *Palisade* processors (and especially the
            :py:class:`~DijetAnalysis.PostProcessing.Lumberjack.PlotProcessor`)
            expect the objects returned by functions to
            be valid *rootpy* objects. When implementing user-defined functions,
            make sure to convert "naked" ROOT (PyROOT) objects by wrapping
            them in :py:func:`rootpy.asrootpy` before returning them.

        '''
        if name is None and function is not None:
            name = function.__name__

        if name in cls.special_functions:
            raise ValueError("Cannot add user-defined ROOT function with name "
                "'{}': name is reserved for internal use only!".format(name))

        if not override and name in cls.functions:
            raise ValueError("Cannot add user-defined ROOT function with name "
                "'{}': it already exists and `override` not explicitly allowed!".format(name))


        if memoize:
            def memoize(f):
                @functools.wraps(f)
                def _memoized_function(*args, **kwargs):
                    # compute unique hash key for the argument structure
                    key = HashableMap(func=f, args=args, kwargs=kwargs)

                    # look up in cache
                    if key in cls._cache:
                        # return if found
                        return cls._cache[key]

                    # compute and store if not found
                    _result = f(*args, **kwargs)
                    cls._cache[key] = _result

                    return _result

                return _memoized_function

            def _decorator(f):
                # replace 'f' with a version enabling memoization of results
                f = memoize(f)

                # add memoized function to mapping
                cls.functions[name or f.__name__] = f

                return f
        else:
            def _decorator(f):
                # add user-specified function to mapping
                cls.functions[name or f.__name__] = f

                return f

        if function is not None:
            # default decorator call, i.e. '@InputROOT.add_function'
            return _decorator(function)
        else:
            # decorator call with parameters , e.g. '@InputROOT.add_function(override=True)'
            return _decorator

    @classmethod
    def get_function(cls, name):
        '''Retrieve a defined input function by name. Returns `None` if no such function exists.'''
        return cls.functions.get(name, None)

    def add_file(self, file_path, nickname=None):
        """
        Add a ROOT file.

        Parameters
        ----------
            file_path : `str`
                path to ROOT file
            nickname : `str`, optional
                file nickname. If not given, ``file_path`` will be used
        """

        # determine real (absolute) path for file
        if '://' in file_path:
            # keep URL-like paths as they are
            _file_realpath = file_path
        else:
            _file_realpath = os.path.realpath(file_path)

        # register file name (as given) as file nickname
        self._file_nick_to_realpath[file_path] = _file_realpath

        # register nickname (if given)
        if nickname is not None:
            if nickname in self._file_nick_to_realpath:
                raise ValueError("Cannot add file for nickname '{}': nickname already registered for file '{}'".format(nickname, file_path))
            self._file_nick_to_realpath[nickname] = _file_realpath

        # create controller for file
        if _file_realpath not in self._input_controllers:
            self._input_controllers[_file_realpath] = InputROOTFile(_file_realpath)


    def get(self, object_spec):
        """
        Get an object from one of the registered files.

        .. tip::

            If calling :py:meth:`get` on multiple objects (e.g. in a loop), consider
            issuing a
            :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.request`
            call for all objects beforehand. The first call to :py:meth:`get` will
            then retrieve all requested objects in one go, opening and closing
            the file only once.

        Parameters
        ----------
            object_spec : `str`
                file nickname and path to object in ROOT file, separated by
                a colon, e.g. :py:data:`"file_nickname:directory/object"`
        """
        _file_nickname, _object_path_in_file = self._get_file_nickname_and_obj_path(object_spec)
        _ic = self._get_input_controller_for_file(_file_nickname)
        return _ic.get(_object_path_in_file)


    def request(self, request_specs):
        """
        Request objects from the registered files. Requests for objects
        are stored until
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.get`
        is called for one of the objects.
        All requested objects are then be retrieved in one go and
        cached.

        Parameters
        ----------
            request_specs : `list` of `dict`
                each `dict` represents a request for *one* object from
                *one* file.

        A request dict must have either a key :py:const:`object_spec`, which
        contains both the file nickname and the path to the object
        within the file (separated by a colon, ``:``), or two keys
        :py:const:`file_nickname` and :py:const:`object_path` specifying these
        separately.

        The following requests behave identically:

        * :py:data:`dict(file_nickname='file0', object_path="directory/object")`
        * :py:data:`dict(object_spec="file0:directory/object")`

        """
        _delegations = {}
        for request_spec in request_specs:
            _file_nickname = request_spec.pop('file_nickname', None)
            _object_path_in_file = request_spec.pop('object_path', None)
            _object_spec = request_spec.pop('object_spec', None)

            if not ((_object_spec is not None) == ((_file_nickname is None) and (_object_path_in_file is None))):
                raise ValueError("Invalid request: must either contain both 'file_nickname' and 'object_path' keys or an 'object_spec' key, but contains: {}".format(request_spec.keys()))

            if _object_spec is not None:
                _file_nickname, _object_path_in_file = self._get_file_nickname_and_obj_path(_object_spec)

            if _file_nickname not in _delegations:
                _delegations[_file_nickname] = []
            _delegations[_file_nickname].append(dict(object_path=_object_path_in_file, **request_spec))

        for _file_nickname, _requests in six.iteritems(_delegations):
            _ic = self._get_input_controller_for_file(_file_nickname)
            _ic.request(_requests)


    def get_expr(self, expr, locals={}):
        """
        Evaluate an expression involving objects retrieved from file(s).

        The string given must be a valid Python expression.

        Strings contained in the expression are interpreted as
        specifications of objects in files (see
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.get`
        for the object specification syntax).
        Before the expression is evaluated, all strings are replaced
        by the objects they refer to.

        To interpret a string as a *literal string* (i.e. not referring
        to an object in a file), it must be wrapped inside the special
        function ``str``.

        Any *functions* called in the expression must have been defined
        beforehand using
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.add_function`.
        There are a number of special functions, which behave as follows:

        * ``str``: interpret a string as a *literal string*
        * ``no_input``: interpret *all* strings encountered anywhere
          inside the function call as literal strings
        * ``input``: interpret *all* strings encountered anywhere
          inside the function call as specifications of objects in files

        All *Python identifiers* used in the expression are interpreted as local
        variables. A map specifying the values of local variables for this call
        to `get_expr` can be given via the keyword argument `locals`.

        Alternatively, local variables can be registered for use by all calls to
        :py:meth:`get_expr` by calling
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.register_local`
        beforehand. Variables given in the `locals` dictionary will take precedence
        over those defined via :py:meth:`register_local`.

        Local variable lookup can be disabled completely by passing ``locals=None``.

        Parameters
        ----------
            expr : `str`
                valid Python expression
            locals : `dict` or ``None`` (default: ``{}``)
                mapping of local variable names that may appear in `expr` to values.

                These will override local variable values specified beforehand
                using :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.register_local`
                before calling this method.

                If :py:const:`None`, local variable lookup is disabled and a
                :py:exc:`NameError` will be raised if an identifier is encountered
                in the expression.

        Usage examples:

        .. code:: python

            my_result = my_input_root.get_expr(
                'my_function('                      # this function gets called
                    '"my_file:path/to/my_object",'  # this gets replaced by ROOT object
                    '42'                            # this argument is passed literally
                ')'
            )

        .. code:: python

            # register a local variable and assign it a value
            my_input_root.register_local('local_variable', 42)

            my_result = my_input_root.get_expr(
                'my_function('                      # this function gets called
                    '"my_file:path/to/my_object",'  # this gets replaced by ROOT object
                    'local_variable,'               # this gets replaced by its assigned value
                ')'
            )

        .. tip::

            Writing expressions inside a single string can get very convoluted.
            To maintain legiblity, the expression string can be spread out on
            several lines, by taking advantage of Python's automatic string concatenation
            inside parentheses (see above). Alternatively, triple-quoted strings
            can be used.

        """
        _locals = {}
        if locals is not None:
            _locals = dict(self._locals, **locals)

        expr = expr.strip()   # extraneous spaces otherwise interpreted as indentation

        self._request_all_objects_in_expression(expr)

        _result = self._eval(node=ast.parse(expr, mode='eval').body,
                             ctx=dict(operators=self.operators,
                                      functions=self.functions,
                                      locals=_locals,
                                      input=True))

        # raise exceptions unable to be raised during `_eval` for technical reasons
        # (e.g. due to expressions with self-referencing local variables that would
        # cause infinite recursion)
        if isinstance(_result, Exception):
            raise _result

        return _result

    def _request_all_objects_in_expression(self, expr, **other_request_params):
        """Walk through the expression AST and request an object for each string or identifier"""
        _ast = ast.parse(expr, mode='eval')
        _reqs = []
        for _node in ast.walk(_ast):
            if isinstance(_node, ast.Name):
                _obj_spec = _node.id
            elif isinstance(_node, ast.Str):
                _obj_spec = _node.s
            else:
                continue

            if ':' in _obj_spec:
                _reqs.append(dict(object_spec=_obj_spec, force_rerequest=False, **other_request_params))
        self.request(_reqs)

    def register_local(self, name, value, override=False):
        """
        Register a local variable to be used when evaluating an expression using
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.get_expr`

        Parameters
        ----------
            name : `str`
                valid Python identifier. Must not have been registered before.
            value :
                any Python object to be made accessible in expressions under :py:const:`name`.
        """
        if not override:
            try:
                assert name not in self._locals
            except AssertionError as e:
                print("[ERROR] Cannot register local '{}' with value {}! It already exists and is: {}".format(name, value, self._locals[name]))
                raise e
        self._locals[name] = value

    def clear_locals(self):
        """
        Clear all locals defined via
        :py:meth:`~DijetAnalysis.PostProcessing.Palisade.InputROOT.register_local`.
        """
        self._locals = dict()

    def _eval(self, node, ctx):
        """Evaluate an AST node"""
        if node is None:
            return None
        elif isinstance(node, ast.Name):  # <identifier>
            # lookup identifiers in local namespace
            if node.id in ctx['locals']:
                _local = ctx['locals'][node.id]

                # if local variable contains a list, evaluate each element by threading 'get_expr' over it
                if isinstance(_local, list):
                    _retlist = []
                    for _local_el in _local:
                        # non-string elements are simply passed through
                        if not isinstance(_local_el, str):
                            _retlist.append(_local_el)
                            continue

                        # string-valued elements are evaluated
                        try:
                            # NOTE: local variable lookup is disabled when threading
                            # over lists that were stored in local variables themselves.
                            # This is done to prevent infinite recursion errors for
                            # expressions which may reference themselves
                            _ret_el = self.get_expr(_local_el, locals=None)
                        except NameError as e:
                            # one element of the list references a local variable
                            # -> stop evaluation and return dummy
                            # use NameError object instead of None to identifiy
                            # dummy elements unambiguously later
                            _retlist.append(e)
                        else:
                            # evaluation succeeded
                            _retlist.append(_ret_el)
                    return _retlist
                # local variables containing strings are parsed
                elif isinstance(_local, str):
                    return self.get_expr(_local, locals=None)
                # all other types are simply passed through
                else:
                    return _local

            # if no local is found, try a few builtin Python literals
            elif node.id in ('True', 'False', 'None'):  # restrict subset of supported literals
                return ast.literal_eval(node.id)  # returns corresponding Python literal from string

            # if nothing above matched, assume mistyped identifier and give up
            # NOTE: do *not* assume identifier is a ROOT file path. ROOT file paths
            # must be given explicitly as strings.
            else:
                raise NameError("Cannot resolve identifier '{}': not a valid Python literal or a registered local variable!".format(node.id))
        elif isinstance(node, ast.Str): # <string> : array column
            if ctx['input']:
                # lookup in ROOT file
                return self.get(node.s)
            else:
                # return string as-is
                return node.s
        elif isinstance(node, ast.Num): # <number>
            return node.n
        elif isinstance(node, ast.Call): # node names containing parentheses (interpreted as 'Call' objects)
            # -- determine function to call

            # function handle is a simple identifier
            if isinstance(node.func, ast.Name):

                # handle special functions
                if node.func.id in self.special_functions:
                    _spec_func_spec = self.special_functions[node.func.id]
                    # callable for special function (default to no-op)
                    _callable = _spec_func_spec.get('func', lambda x: x)
                    # modify avaluation context for special function
                    ctx = dict(ctx, **_spec_func_spec.get('ctx', {}))

                # call a registered input function
                else:
                    try:
                        _callable = ctx['functions'][node.func.id]
                    except KeyError as e:
                        raise KeyError(
                            "Cannot call input function '{}': no such "
                            "function!".format(node.func.id))

            # function handle is an expression
            else:
                # evaluate 'func' as any other node
                _callable = self._eval(node.func, ctx)

            # evaluate unpacked positional arguments, if any
            _starargs_values = []
            if node.starargs is not None:
                _starargs_values = self._eval(node.starargs, ctx)

            # starred kwargs (**) not supported for the moment
            if node.kwargs:
                raise NotImplementedError(
                    "Unpacking keyword arguments in expressions via "
                    "** is not supported. Expression was: '{}'".format(
                        ast.dump(node, annotate_fields=False)))

            # evaluate arguments
            _args = map(lambda _arg: self._eval(_arg, ctx), node.args) + _starargs_values
            _kwargs = {
                _keyword.arg : self._eval(_keyword.value, ctx)
                for _keyword in node.keywords
            }

            # call function
            return _callable(*_args, **_kwargs)
        elif isinstance(node, ast.BinOp): # <left> <operator> <right>
            return ctx['operators'][type(node.op)](self._eval(node.left, ctx), self._eval(node.right, ctx))
        elif isinstance(node, ast.UnaryOp): # <operator> <operand> e.g., -1
            return ctx['operators'][type(node.op)](self._eval(node.operand, ctx))
        elif isinstance(node, ast.Subscript): # <operator> <operand> e.g., -1
            if isinstance(node.slice, ast.Index): # support subscripting via simple index
                return self._eval(node.value, ctx)[self._eval(node.slice.value, ctx)]
            elif isinstance(node.slice, ast.Slice): # support subscripting via slice
                return self._eval(node.value, ctx)[self._eval(node.slice.lower, ctx):self._eval(node.slice.upper, ctx):self._eval(node.slice.step, ctx)]
            else:
                raise TypeError(node)
        elif isinstance(node, ast.Attribute): # <value>.<attr>
            return getattr(self._eval(node.value, ctx), node.attr)
        elif isinstance(node, ast.List): # list of node names
            return [self._eval(_el, ctx) for _el in node.elts]
        elif isinstance(node, ast.Tuple): # tuple of node names
            return tuple(self._eval(_el, ctx) for _el in node.elts)
        else:
            raise TypeError(node)

    @classmethod
    def clear_cache(cls):
        cls._cache = {}

    # functions with special meanings/side effects
    # when encountered in expressions, these functions can change the
    # behavior of the evaluation of descendant nodes or access functionality
    # of the InputROOT object itself
    special_functions = {
        # enable ROOT input when evaluating descendant nodes
        'input':    dict(ctx=dict(input=True)),
        # disable ROOT input when evaluating descendant nodes
        'no_input': dict(ctx=dict(input=False)),
        # get argument as string (i.e. without ROOT input)
        'str':      dict(func=str, ctx=dict(input=False)),
    }
