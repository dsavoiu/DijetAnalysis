from __future__ import print_function

import colorsys  # for rgb_to_hls
import math
import os
import six
import warnings
import yaml

from copy import deepcopy
from functools import partial
from itertools import cycle

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation, LogFormatterMathtext
from matplotlib.colors import SymLogNorm, LogNorm, Normalize, colorConverter
from matplotlib.text import Text
from matplotlib.legend_handler import (
    HandlerBase as LegendHandlerBase,
    HandlerTuple as LegendHandlerTuple
)

from rootpy.plotting import Hist1D, Hist2D, Profile1D, Efficiency, F1
from rootpy.plotting.hist import _Hist, _Hist2D
from rootpy.plotting.profile import _ProfileBase

from ..._util import make_directory

from .._input import InputROOT
from .._colormaps import viridis

from ._base import ContextValue, LiteralString, _ProcessorBase

__all__ = [
    'LogFormatterSciNotationForceSublabels', 'PlotProcessor',
    'LegendHandlerTuple', 'LegendHandlerString', 'LegendHandlerText'
]


plt.register_cmap(name='viridis', cmap=viridis)

def _mplrc():
    mpl.rcParams.update({'font.size': 11})
    if int(mpl.__version__.split('.')[0]) >= 2:
        mpl.rc('xtick', direction='in', bottom=True, top=True)
        mpl.rc('ytick', direction='in', left=True, right=True)
    else:
        mpl.rc('xtick', direction='in')
        mpl.rc('ytick', direction='in')
    mpl.rc('mathtext', fontset='stixsans', fallback_to_cm=False, rm='sans')
    mpl.rc('axes', labelsize=16)
    mpl.rc('legend', labelspacing=.1, fontsize=8)

def _mathdefault(s):
    return '\\mathdefault{%s}' % s

def is_close_to_int(x):
    if not np.isfinite(x):
        return False
    return abs(x - round(x)) < 1e-10


class LogFormatterSciNotationForceSublabels(LogFormatterSciNotation):
    """Variant of LogFormatterSciNotation that always displays labels at
    certain non-decade positions. Needed because parent class may hide these
    labels based on axis spacing."""

    def __init__(self, *args, **kwargs):
        self._sci_min_exp = kwargs.pop('sci_min_exp', None)  # sci notation above, regular below
        self._sublabels_max_exp = kwargs.pop('sublabels_max_exp', None)  # no sublabels above exp
        super(LogFormatterSciNotationForceSublabels, self).__init__(*args, **kwargs)

    def set_locs(self, *args, **kwargs):
        '''override sublabels'''
        _ret = super(LogFormatterSciNotationForceSublabels, self).set_locs(*args, **kwargs)

        _locs = kwargs.pop("locs", None)

        # override locations
        _locs = kwargs.pop("locs", None)
        if _locs is not None:
            self._sublabels = _locs
        else:
            self._sublabels = {1.0, 2.0, 5.0, 10.0}

        return _ret

    def _non_decade_format(self, sign_string, base, fx, usetex):
        'Return string for non-decade locations'
        b = float(base)
        exponent = math.floor(fx)
        coeff = b ** fx / b ** exponent
        if is_close_to_int(coeff):
            coeff = np.round(coeff)
        if usetex:
            return (r'$%s%g\times%s^{%d}$') % \
                                        (sign_string, coeff, base, exponent)
        else:
            return ('$%s$' % _mathdefault(r'%s%g\times%s^{%d}' %
                                        (sign_string, coeff, base, exponent)))

    def __call__(self, x, pos=None):
        """
        Return the format for tick value *x*.

        The position *pos* is ignored.
        """
        usetex = mpl.rcParams['text.usetex']
        assert not usetex, "LogFormatterSciNotationForceSublabels does not (yet) support `text.usetex`"
        sci_min_exp = self._sci_min_exp #rcParams['axes.formatter.min_exponent']

        if x == 0:
            return '$0$'

        sign_string = '-' if x < 0 else ''
        x = abs(x)
        b = self._base

        # only label the decades
        fx = math.log(x) / math.log(b)
        is_x_decade = is_close_to_int(fx)
        exponent = np.round(fx) if is_x_decade else np.floor(fx)
        coeff = np.round(x / b ** exponent)
        if is_x_decade:
            fx = np.round(fx)

        if self.labelOnlyBase and not is_x_decade:
            return ''
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''

        # use string formatting of the base if it is not an integer
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b

        # TEMP: suppress minor ticks above threshold
        if not is_x_decade and self._sublabels_max_exp is not None and np.abs(fx) > self._sublabels_max_exp + 1:
            return ''

        if np.abs(fx) < sci_min_exp:
            return '${0}$'.format(_mathdefault(
                '{0}{1:g}'.format(sign_string, x)))
        elif not is_x_decade:
            return self._non_decade_format(sign_string, base, fx, usetex)
        else:
            return '$%s$' % _mathdefault('%s%s^{%d}' % (sign_string, base, fx))


# -- custom legend handlers

class LegendHandlerString(LegendHandlerBase):
    """
    Handler for bare Python strings. Will render a `Text` artist
    for strings supplied as legend handles.

    Parameters
    ----------
    fontweight : str, optional
        The font weight to use. Default: 'bold'
    """

    def __init__(self, fontweight='bold', **kwargs):

        self._fontweight = fontweight
        LegendHandlerBase.__init__(self, **kwargs)

    def create_artists(self, legend, text, xdescent, ydescent, width, height, fontsize, trans):
        return [Text(
            (width-xdescent)/2, (height-ydescent)/2, text,
            fontsize=fontsize, ha="center", va="center",
            fontweight=self._fontweight
        )]


class LegendHandlerText(LegendHandlerString):
    """
    Handler for using `Text` artists as legend handles.

    Parameters
    ----------
    fontweight : str, optional
        The font weight to use. Default: 'bold'
    """
    def create_artists(self, legend, artist, xdescent, ydescent, width, height, fontsize, trans):
        return super(LegendHandlerText, self).create_artists(
            legend, artist.get_text(), xdescent, ydescent, width, height, fontsize, trans
        )


# check if LegendHandlerTuple API supports `ndivide` and `pad`
try:
    LegendHandlerTuple(ndivide=None, pad=5)
except TypeError:  # old MPL version -> reimplement with new API
    class LegendHandlerTuple(LegendHandlerBase):
        """
        Handler for Tuple. (reimplemented from matplotlib 2.0)

        Additional kwargs are passed through to `HandlerBase`.

        Parameters
        ----------
        ndivide : int, optional
            The number of sections to divide the legend area into. If None,
            use the length of the input tuple. Default is 1.


        pad : float, optional
            If None, fall back to ``legend.borderpad`` as the default.
        In units of fraction of font size. Default is None.
        """

        def __init__(self, ndivide=1, pad=None, **kwargs):

            self._ndivide = ndivide
            self._pad = pad
            LegendHandlerBase.__init__(self, **kwargs)

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize,
                           trans):

            handler_map = legend.get_legend_handler_map()

            if self._ndivide is None:
                ndivide = len(orig_handle)
            else:
                ndivide = self._ndivide

            if self._pad is None:
                pad = legend.borderpad * fontsize
            else:
                pad = self._pad * fontsize

            if ndivide > 1:
                width = (width - pad * (ndivide - 1)) / ndivide

            xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))

            a_list = []
            for handle1 in orig_handle:
                handler = legend.get_legend_handler(handler_map, handle1)
                _a_list = handler.create_artists(
                    legend, handle1,
                    next(xds_cycle), ydescent, width, height, fontsize, trans)
                a_list.extend(_a_list)

            return a_list


def _plot_with_error_band(ax, *args, **kwargs):
    """display data as line. If `yerr` is given, an `y` +/- `yerr` error band is also drawn.
    You can use custom `_band_kwargs` to format the error band."""

    kwargs.pop('xerr', None)
    _yerr = kwargs.pop('yerr', None)
    _x = np.asarray(args[0])
    _y = np.asarray(args[1])

    # kwargs delegated to mpl fill_between
    _band_kwargs = kwargs.pop('band_kwargs', None) or dict()
    if _yerr is None:
        return ax.plot(*args, **kwargs)
    else:
        return (ax.plot(*args, **kwargs),
                ax.fill_between(_x, _y - _yerr[0], _y + _yerr[1], **dict(dict(kwargs, alpha=0.5, linewidth=0),
                                                                         **_band_kwargs)))


def _plot_as_step(ax, *args, **kwargs):
    """display data as horizontal bars with given by `x` +/- `xerr`. `y` error bars are also drawn."""
    assert len(args) == 2
    _x = np.ma.asarray(args[0])
    _y = np.ma.asarray(args[1])
    _zeros = np.zeros_like(_x)

    # kwarg `yerr_as_band` to display
    _show_yerr_as = kwargs.pop('show_yerr_as', None)
    if _show_yerr_as is not None and _show_yerr_as not in ('errorbar', 'band', 'hatch'):
        raise ValueError("Invalid value '{}' for 'show_yerr_as'. Available: {}".format(_show_yerr_as, ('errorbar', 'band', 'hatch')))

    assert 'xerr' in kwargs
    if len(kwargs['xerr']) == 1:
        _xerr_dn = _xerr_up = kwargs.pop('xerr')[0]
    else:
        _xerr_dn, _xerr_up = kwargs.pop('xerr')

    _yerr = kwargs.pop('yerr', None)
    if _yerr is not None:
        if len(_yerr) == 1:
            _yerr_dn = _yerr_up = _yerr[0]
        else:
            _yerr_dn, _yerr_up = _yerr
        _yerr_dn = np.asarray(_yerr_dn)
        _yerr_up = np.asarray(_yerr_up)

    _xerr_dn = np.asarray(_xerr_dn)
    _xerr_up = np.asarray(_xerr_up)

    # replicate each point five times -> bin anchors
    #  1 +       + 5
    #    |       |
    #    +---+---+
    #  2     3     4
    _x = np.ma.vstack([_x, _x, _x, _x, _x]).T.flatten()
    _y = np.ma.vstack([_y, _y, _y, _y, _y]).T.flatten()

    # stop processing y errors if they are zero
    if np.allclose(_yerr, 0):
        _yerr = None

    # attach y errors (if any) to "bin" center
    if _yerr is not None:
        if _show_yerr_as in ('band', 'hatch'):
            # error band: shade across entire bin width
            _yerr_dn = np.vstack([_zeros, _yerr_dn, _yerr_dn, _yerr_dn, _zeros]).T.flatten()
            _yerr_up = np.vstack([_zeros, _yerr_up, _yerr_up, _yerr_up, _zeros]).T.flatten()
        else:
            # errorbars: only show on central point
            _yerr_dn = np.vstack([_zeros, _zeros, _yerr_dn, _zeros, _zeros]).T.flatten()
            _yerr_up = np.vstack([_zeros, _zeros, _yerr_up, _zeros, _zeros]).T.flatten()
        _yerr = [_yerr_dn, _yerr_up]

    # shift left and right replicas in x by xerr
    _x += np.vstack([-_xerr_dn, -_xerr_dn, _zeros, _xerr_up, _xerr_up]).T.flatten()

    # obtain indices of points with a binning discontinuity
    _bin_edge_discontinuous_at = (np.flatnonzero(_x[0::5][1:] != _x[4::5][:-1]) + 1)*5

    # prevent diagonal connections across bin discontinuities
    if len(_bin_edge_discontinuous_at):
        _x = np.insert(_x, _bin_edge_discontinuous_at, [np.nan])
        _y = np.insert(_y, _bin_edge_discontinuous_at, [np.nan])
        if _yerr is not None:
            _yerr = np.insert(_yerr, _bin_edge_discontinuous_at, [np.nan], axis=1)

    # do actual plotting
    if _show_yerr_as == 'errorbar' or _show_yerr_as is None:
        return ax.errorbar(_x, _y, yerr=_yerr if _show_yerr_as else None, **kwargs)
    elif _show_yerr_as == 'band':
        _band_alpha = kwargs.pop('band_alpha', 0.5)
        _hatch = kwargs.pop('band_hatch', None)
        _boundary = kwargs.pop('band_boundary', False)
        _capsize = kwargs.pop('capsize', None)
        _markeredgecolor = kwargs.pop('markeredgecolor', None)
        _color = kwargs.pop('color', None)
        _alpha = kwargs.pop('alpha', None)
        _linestyle = kwargs.pop('linestyle', None)

        _return_artists = []

        # compute boundary step
        _y_shifted = (_y.copy(), _y.copy())
        for _ys, _ye, _fac in zip(_y_shifted, _yerr, (-1.0, 1.0)):
            _ye = _ye.copy()
            # set shift size at bin anchors 0,4 to y error (only at bin anchors 1,2,3)
            _ye[0::5] = _ye[2::5]
            _ye[4::5] = _ye[2::5]
            _ys += _fac * _ye

        if _boundary:
            _return_artists.extend([
                ax.errorbar(_x, _y_shifted[0], yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, color=_color, linestyle=_linestyle, **kwargs),
                ax.errorbar(_x, _y_shifted[1], yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, color=_color, linestyle=_linestyle, **kwargs),
            ])

        if _yerr is None:
            _yerr = 0, 0

        _return_artists.extend([
            ax.errorbar(_x, _y, yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, alpha=_alpha, color=_color, **kwargs),
            ax.fill_between(_x, _y_shifted[0], _y_shifted[1], **dict(kwargs, hatch=_hatch, alpha=_band_alpha, linewidth=0, color=_color))
            #ax.fill_between(_x, _y-_yerr[0], _y+_yerr[1], **dict(kwargs, alpha=_band_alpha, linewidth=0, hatch=_hatch, facecolor='none', color=None, edgecolor=kwargs.get('color')))
        ])

        return tuple(_return_artists)

    elif _show_yerr_as == 'hatch':
        raise NotImplementedError
        _band_alpha = kwargs.pop('band_alpha', 0.5)
        _hatch = kwargs.pop('hatch', '////')
        _capsize = kwargs.pop('capsize', None)
        _color = kwargs.pop('color', None)
        _alpha = kwargs.pop('alpha', None)
        _markeredgecolor = kwargs.pop('markeredgecolor', None)
        _linestyle = kwargs.pop('linestyle', None)  # invalid for hatch
        if _yerr is None:
            _yerr = 0, 0

        # compute boundary step
        _y_shifted = (_y.copy(), _y.copy())
        for _ys, _ye, _fac in zip(_y_shifted, _yerr, (-1.0, 1.0)):
            _ye = _ye.copy()
            # set shift size at bin anchors 0,4 to y error (only at bin anchors 1,2,3)
            _ye[0::5] = _ye[2::5]
            _ye[4::5] = _ye[2::5]
            _ys += _fac * _ye
        return (
            # central value
            ax.errorbar(_x, _y, yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, color=_color, **kwargs),
            # limiting upper/lower edges of hatch
            ax.errorbar(_x, _y_shifted[0], yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, color=_color, **kwargs),
            ax.errorbar(_x, _y_shifted[1], yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, color=_color, **kwargs),
            # hatch
            ax.fill_between(_x, _y-_yerr[0], _y+_yerr[1], **dict(kwargs, hatch=_hatch, alpha=0.3, facecolor='r', edgecolor='yellow', linewidth=0, color=_color)))
            #ax.fill_between(_x, _y-_yerr[0], _y+_yerr[1], **dict(linestyle='solid', hatch=_hatch, facecolor='none', linewidth=0)))


class PlotProcessor(_ProcessorBase):
    """Processor for plotting objects from ROOT files.

    .. todo::
        API documentation.
    """

    CONFIG_KEY_FOR_TEMPLATES = "figures"
    SUBKEYS_FOR_CONTEXT_REPLACING = ["subplots", "pads", "texts"]
    CONFIG_KEY_FOR_CONTEXTS = "expansions"

    _EXTERNAL_PLOT_METHODS = dict(
        step = _plot_as_step,
        plot = _plot_with_error_band
    )

    _PC_KEYS_MPL_AXES_METHODS = dict(
        x_label = dict(
            method='set_xlabel',
            kwargs=dict(x=1.0, ha='right'),
        ),
        y_label = dict(
            method='set_ylabel',
            kwargs=dict(y=1.0, ha='right'),
        ),
        x_range = dict(
            method='set_xlim',
        ),
        y_range = dict(
            method='set_ylim',
        ),
        x_scale = dict(
            method='set_xscale',
        ),
        y_scale = dict(
            method='set_yscale',
        ),
        x_ticklabels = dict(
            method='set_xticklabels',
        ),
        y_ticklabels = dict(
            method='set_yticklabels',
        ),
        x_ticks = dict(
            method='set_xticks',
        ),
        y_ticks = dict(
            method='set_yticks',
        ),
    )

    _DEFAULT_LEGEND_KWARGS = dict(
        ncol=1, numpoints=1, fontsize=12, frameon=False,
        loc='upper right',
        handler_map = {
            tuple: LegendHandlerTuple(ndivide=None, pad=4),
            Text: LegendHandlerText(fontweight='bold'),
            str: LegendHandlerString(fontweight='bold'),
        }
    )
    _DEFAULT_LINE_KWARGS = dict(
        linestyle='--', color='gray', linewidth=1, zorder=-99
    )


    def __init__(self, config, output_folder):
        super(PlotProcessor, self).__init__(config, output_folder)

        self._input_controller = InputROOT(
            files_spec=self._config['input_files']
        )
        self._figures = {}
        self._global_request_params = self._config.get("global_request_params", {})

        # introduce pseudo-context for accessing input file content
        self._config[self.CONFIG_KEY_FOR_CONTEXTS].update(
            _input_controller=[self._input_controller]
        )

    # -- helper methods

    def _get_figure(self, figure_name, figsize=None):
        if figure_name not in self._figures:
            self._figures[figure_name] = plt.figure(figsize=figsize)
        return self._figures[figure_name]

    @staticmethod
    def _merge_legend_handles_labels(handles, labels):
        '''merge handles for identical labels'''
        _seen_labels = []
        _seen_label_handles = []
        _new_label_indices = []
        for _ihl, (_h, _l) in enumerate(zip(handles, labels)):
            if _l not in _seen_labels:
                _seen_labels.append(_l)
                _seen_label_handles.append([_h])
            else:
                _idx = _seen_labels.index(_l)
                _seen_label_handles[_idx].append(_h)

        for _i, (_sh, _sl) in enumerate(zip(_seen_label_handles, _seen_labels)):
            _seen_label_handles[_i] = tuple(_seen_label_handles[_i])

        return _seen_label_handles, _seen_labels

    @staticmethod
    def _sort_legend_handles_labels(handles, labels, stack_labels=None):
        '''sort handles and labels, reversing the order of those that are part of a stack'''
        # if no stacks or a stack with a single label, don't sort
        if stack_labels is None or len(stack_labels) <= 1:
            return handles, labels

        # temporarily cast to array to use numpy indexing
        _hs, _ls = np.asarray(handles), np.asarray(labels)
        _criterion = np.vectorize(lambda label: label in stack_labels)

        # reverse sublist selected by criterion
        _ls[_criterion(_ls)] = _ls[_criterion(_ls)][::-1]
        _hs[_criterion(_ls)] = _hs[_criterion(_ls)][::-1]

        # cast back to artist container (artist container)
        _hs = map(tuple, _hs)

        # return as lists
        return list(_hs), list(_ls)

    # -- actions

    def _request(self, config):
        '''request all objects encountered in all subplot expressions'''
        for _subplot_cfg in config['subplots']:
            request_params = dict(self._global_request_params, **_subplot_cfg.get('request_params', {}))
            self._input_controller._request_all_objects_in_expression(_subplot_cfg['expression'], **request_params)
            #print('REQ', _subplot_cfg['expression'])


    def _plot(self, config):
        '''plot all figures'''
        _mplrc()

        # register expressions as locals for lookup by the input controller's `get` call
        self._input_controller.register_local('expressions', [_subplot_cfg['expression'] for _subplot_cfg in config['subplots']], override=True)

        _filename = os.path.join(self._output_folder, config['filename'])

        # prepare dict for YAML dump, if requested
        _dump_yaml = config.pop('dump_yaml', False)
        if _dump_yaml:
            _yaml_filename = '.'.join(_filename.split('.')[:-1]) + '.yml'
            # need to create directory first
            make_directory(os.path.dirname(_yaml_filename), exist_ok=True)
            # add input files to dump
            _config_for_dump = dict(deepcopy(config), input_files=self._config['input_files'])
        else:
            _config_for_dump = config  # dummy link to original config

        # step 1: create figure and pads

        _figsize = config.pop('figsize', None)
        _fig = self._get_figure(_filename, figsize=_figsize)

        # obtain configuration of pads
        _pad_configs = config.get('pads', None)
        if _pad_configs is None:
            # default pad configuration
            _pad_configs = [dict()]

        # get share
        _height_ratios = [_pc.pop('height_share', 1) for _pc in _pad_configs]

        # construct GridSpec from `pad_spec` or make default
        _gridspec_kwargs = config.get('pad_spec', dict())
        _gridspec_kwargs.pop('height_ratios', None)   # ignore explicit user-provided `height_ratios`
        _gs = GridSpec(nrows=len(_pad_configs), ncols=1, height_ratios=_height_ratios, **_gridspec_kwargs)

        # store `Axes` objects in pad configuration
        for _i_pad, _pad_config in enumerate(_pad_configs):
            _pad_config['_axes'] = _fig.add_subplot(_gs[_i_pad])

        _stack_bottoms = _pad_config.setdefault('_stack_bottoms', {})
        _bin_labels = _pad_config.setdefault('bin_labels', {})
        _bin_label_anchors = _pad_config.setdefault('bin_label_anchors', {})

        # enable text output, if requested
        if config.pop("text_output", False):
            _text_filename = '.'.join(_filename.split('.')[:-1]) + '.txt'
            # need to create directory first
            make_directory(os.path.dirname(_text_filename), exist_ok=True)
            _text_file = open(_text_filename, 'w')
        else:
            _text_file = None


        # step 2: retrieve data and plot
        assert len(config['subplots']) == len(_config_for_dump['subplots'])
        for _pc, _pc_for_dump in zip(config['subplots'], _config_for_dump['subplots']):
            _kwargs = deepcopy(_pc)

            # obtain and validate pad ID
            _pad_id = _kwargs.pop('pad', 0)
            if _pad_id >= len(_pad_configs):
                raise ValueError("Cannot plot to pad {}: only pads up to {} have been configured!".format(_pad_id, len(_pad_configs)-1))

            # select pad axes and configuration
            _pad_config = _pad_configs[_pad_id]
            _ax = _pad_config['_axes']
            _stack_bottoms = _pad_config.setdefault('_stack_bottoms', {})
            _stack_labels = _pad_config.setdefault('stack_labels', [])
            _bin_labels = _pad_config.setdefault('bin_labels', {})
            _bin_label_anchors = _pad_config.setdefault('bin_label_anchors', {})

            _expression = _kwargs.pop('expression')
            #("PLT {}".format(_expression))
            _plot_object = self._input_controller.get_expr(_expression)

            # extract arrays for keys which could be masked by 'mask_zero_errors'
            _plot_data = {
                _property_name : np.array(list(getattr(_plot_object, _property_name)()))
                for _property_name in ('x', 'xerr', 'y', 'yerr', 'xwidth', 'efficiencies', 'errors')
                if hasattr(_plot_object, _property_name)
            }

            # extract individual bin labels (if they exist)
            for _i_axis, _axis in enumerate("xyz"):
                try:
                    _root_obj_axis = _plot_object.axis(_i_axis)
                except AttributeError:
                    _root_obj_axis = None
                if _root_obj_axis is not None and bool(_root_obj_axis.GetLabels()):
                    _axis_nbins_method = getattr(_plot_object, "GetNbins{}".format(_axis.upper()))
                    _plot_data['{}binlabels'.format(_axis)] = [_root_obj_axis.GetBinLabel(_i_bin) for _i_bin in range(1, _axis_nbins_method() + 1)]

            # map fields for TEfficiency objects
            if isinstance(_plot_object, Efficiency):
                _total_hist = _plot_object.total
                _plot_data['x'] = np.array(list(_total_hist.x()))
                _plot_data['xerr'] = np.array(list(_total_hist.xerr()))
                _plot_data['y'] = _plot_data.pop('efficiencies', None)
                _plot_data['yerr'] = _plot_data.pop('errors', None)

            # map fields for TF1 objects
            elif isinstance(_plot_object, F1):
                _xmin, _xmax = _plot_object.xaxis.get_xmin(), _plot_object.xaxis.get_xmax()
                # compute support points (evenly-spaced)
                _plot_data['x'] = np.linspace(_xmin, _xmax, 100)  # TODO: make configurable
                _plot_data['xerr'] = np.zeros_like(_plot_data['x'])
                # evaluate TF1 at every point
                _plot_data['y'] = np.asarray(list(map(_plot_object, _plot_data['x'])))
                _plot_data['yerr'] = np.zeros_like(_plot_data['y'])  # TODO: function errors (?)

            # mask all points with erorrs set to zero
            _mze = _kwargs.pop('mask_zero_errors', False)
            if _mze and len(_plot_data['yerr']) != 0:
                _mask = np.all((_plot_data['yerr'] != 0), axis=1)
                _plot_data = {
                    _key : np.compress(_mask, _value, axis=0)
                    for _key, _value in six.iteritems(_plot_data)
                }

            # extract arrays for keys which cannot be masked by `mask_zero_errors`
            _plot_data.update({
                _property_name : np.array(list(getattr(_plot_object, _property_name)()))
                for _property_name in ('xedges', 'yedges', 'z')
                if hasattr(_plot_object, _property_name)
            })

            # additional mask using user-supplied function
            _mask = False  # used below, depending on plotting method
            _mask_func = _kwargs.pop('mask_func', None)
            if _mask_func:
                _mask = _mask_func(_plot_data)

            # -- draw

            _plot_method_name = _kwargs.pop('plot_method', 'errorbar')

            # -- obtain plot method
            if callable(_plot_method_name):
                _plot_method = partial(_plot_method_name, _ax)
            else:
                try:
                    # use external method (if available) and curry in the axes object
                    _plot_method = partial(self._EXTERNAL_PLOT_METHODS[_plot_method_name], _ax)
                except KeyError:
                    #
                    _plot_method = getattr(_ax, _plot_method_name)

            if _plot_method_name in ['errorbar', 'step']:
                _kwargs.setdefault('capsize', 0)
                if 'color' in _pc:
                    _kwargs.setdefault('markeredgecolor', _kwargs['color'])

            # remove connecting lines for 'errorbar' plots only
            if _plot_method_name == 'errorbar':
                _kwargs.setdefault('linestyle', '')

            # marker styles
            _marker_style = _kwargs.pop('marker_style', None)
            if _marker_style is not None:
                if _marker_style == 'full':
                    _kwargs.update(
                        markerfacecolor=_kwargs['color'],
                        markeredgewidth=0,
                    )
                elif _marker_style == 'white':
                    _kwargs.update(
                        markerfacecolor='w',
                        markeredgewidth=1,
                    )
                elif _marker_style == 'empty':
                    _kwargs.update(
                        markerfacecolor='none',
                        markeredgewidth=1,
                    )
                else:
                    raise ValueError(
                        "Invalid value '{}' for 'marker_style'. Expected one of: full, white, empty".format(_marker_style)
                    )

            # handle stacking
            _stack_name = _kwargs.pop('stack', None)
            _y_bottom = 0
            if _stack_name is not None:
                _y_bottom = _stack_bottoms.setdefault(_stack_name, 0.0)  # actually 'get' with default
                # keep track of stack labels in order to reverse the legend order later
                _stack_label = _kwargs.get('label', None)
                if _stack_label is not None:
                    _stack_labels.append(_stack_label)

            # different methods handle information differently
            if _plot_method_name == 'bar':
                _kwargs['width'] = _plot_data['xwidth']
                _kwargs.setdefault('align', 'center')
                _kwargs.setdefault('edgecolor', '')
                _kwargs.setdefault('linewidth', 0)
                if 'color' in _kwargs:
                    # make error bar color match fill color
                    _kwargs.setdefault('ecolor', _kwargs['color'])
                _kwargs['y'] = _plot_data['y']
                _kwargs['bottom'] = _y_bottom
            else:
                _kwargs['y'] = _plot_data['y'] + _y_bottom
                _kwargs['xerr'] = _plot_data['xerr'].T

            _show_yerr = _kwargs.pop('show_yerr', True)
            if _show_yerr:
                _kwargs['yerr'] = _plot_data['yerr'].T

            _y_data = _kwargs.pop('y')
            _normflag = _kwargs.pop('normalize_to_width', False)
            if _normflag:
                _y_data /= _plot_data['xwidth']
                if 'yerr' in _kwargs and _kwargs['yerr'] is not None:
                    _kwargs['yerr'] /= _plot_data['xwidth']

            # -- sort out positional arguments to plot method

            if _plot_method_name == 'pcolormesh':
                # apply user-supplied mask (if any) to 'z' values
                _z_masked = np.ma.masked_where(_mask, _plot_data['z'])

                # determine data range in z
                _z_range = _pad_config.get('z_range', None)
                if _z_range is not None:
                    # use specified values as range
                    _z_min, _z_max = _z_range
                else:
                    # use data values
                    _z_min, _z_max = _z_masked.min(), _z_masked.max()

                # determine colormap normalization (if not explicitly given)
                if 'norm' not in _kwargs:
                    _z_scale = _pad_config.pop('z_scale', "linear")
                    if _z_scale == 'linear':
                        _norm = Normalize(vmin=_z_min, vmax=_z_max)
                    elif _z_scale == 'log':
                        _norm = LogNorm(vmin=_z_min, vmax=_z_max)
                    elif _z_scale == 'symlog':
                        _norm = SymLogNorm(linthresh=0.1, vmin=_z_min, vmax=_z_max)
                    else:
                        raise ValueError("Unknown value '{}' for keyword 'z_scale': known are {{'linear', 'log'}}".format(_z_scale))
                    _kwargs['norm'] = _norm

                # Z array needs to be transposed because 'X' refers to columns and 'Y' to rows...
                _args = [_plot_data['xedges'], _plot_data['yedges'], _z_masked.T]
                _kwargs.pop('color', None)
                _kwargs.pop('xerr', None)
                _kwargs.pop('yerr', None)
                # some kwargs must be popped and stored for later use
                _label_bins_with_content = _kwargs.pop('label_bins_with_content', False)
                _bin_label_format = _kwargs.pop('bin_label_format', "{:f}")
                if not callable(_bin_label_format):
                    _bin_label_format = lambda v: _bin_label_format.format(v)
                _bin_label_color = _kwargs.pop('bin_label_color', 'auto')
                _bin_label_fontsize = _kwargs.pop('bin_label_fontsize', 16)
            else:
                _y_masked = np.ma.masked_where(_mask, _y_data)
                _args = [_plot_data['x'], _y_masked]

            # skip empty arguments
            if len(_args[0]) == 0:
                continue

            # run the plot method
            _plot_handle = _plot_method(
                *_args,
                **_kwargs
            )

            # store 2D plots for displaying color bars
            if _plot_method_name == 'pcolormesh':
                _pad_config.setdefault('2d_plots', []).append(_plot_handle)
                # add 2D bin annotations, if requested
                if _label_bins_with_content:
                    _bin_center_x = 0.5 * (_plot_data['xedges'][1:] + _plot_data['xedges'][:-1])
                    _bin_center_y = 0.5 * (_plot_data['yedges'][1:] + _plot_data['yedges'][:-1])
                    _bin_center_xx, _bin_center_yy = np.meshgrid(_bin_center_x, _bin_center_y)
                    _bin_content = _args[2]
                    for _row_x_y_content in zip(_bin_center_xx, _bin_center_yy, _bin_content):
                        for _x, _y, _content in zip(*_row_x_y_content):
                            # skip masked and invalid bin contents
                            if not isinstance(_content, np.ma.core.MaskedConstant) and not np.isnan(_content):
                                if _bin_label_color == 'auto':
                                    _patch_color_lightness = colorsys.rgb_to_hls(*(_plot_handle.to_rgba(_content)[:3]))[1]
                                    _text_color = 'w' if _patch_color_lightness < 0.5 else 'k'
                                else:
                                    _text_color = _bin_label_color
                                _ax.annotate(_bin_label_format(_content), (_x, _y),
                                         ha='center', va='center',
                                         fontsize=_bin_label_fontsize,
                                         color=_text_color,
                                         transform=_ax.transData,
                                         annotation_clip=True,
                                 )

            # write results to config dict that will be dumped
            if _dump_yaml:
                _pc_for_dump['plot_args'] = dict(
                    # prevent dumping numpy arrays as binary
                    args=[_a.tolist() if isinstance(_a, np.ndarray) else _a for _a in _args],
                    **{_kw : _val.tolist() if isinstance(_val, np.ndarray) else _val for _kw, _val in six.iteritems(_kwargs)}
                )

            if _text_file is not None:
                np.set_printoptions(threshold=np.inf)
                _text_file.write("- {}(\n\t{},\n\t{}\n)\n".format(
                    _plot_method_name,
                    ',\n\t'.join(["{}".format(repr(_arg)) for _arg in _args]),
                    ',\n\t'.join(["{} = {}".format(_k, repr(_v)) for _k, _v in six.iteritems(_kwargs)]),
                ))
                np.set_printoptions(threshold=1000)

            # update stack bottoms
            if _stack_name is not None:
                _stack_bottoms[_stack_name] += _plot_data['y']

            # keep track of the bin labels of each object in a pad
            for _i_axis, _axis in enumerate("xyz"):
                _bl_key = '{}binlabels'.format(_axis)
                _bl = _plot_data.get(_bl_key, None)
                if _bl is not None:
                    _bin_labels.setdefault(_axis, []).append(_bl)
                    _bin_label_anchors.setdefault(_axis, []).append(_plot_data.get(_axis, None))

        # close text output
        if _text_file is not None:
            _text_file.close()

        # step 3: pad adjustments
        for _pad_config in _pad_configs:
            _ax = _pad_config['_axes']

            # simple axes adjustments
            _pad_config['_applied_modifiers'] = dict()  # keep track
            for _prop_name, _meth_dict in six.iteritems(self._PC_KEYS_MPL_AXES_METHODS):
                _prop_val = _pad_config.pop(_prop_name, None)
                if _prop_val is not None:
                    getattr(_ax, _meth_dict['method'])(_prop_val, **_meth_dict.get('kwargs', {}))
                    _pad_config['_applied_modifiers'][_prop_name] = _prop_val

            # draw colorbar if there was a 2D plot involved and a colorbar should be drawn
            _2d_plots = _pad_config.pop('2d_plots', [])
            _z_label = _pad_config.pop('z_label', None)
            _z_labelpad = _pad_config.pop('z_labelpad', None)
            if _pad_config.pop('draw_colorbar', True):
                for _2d_plot in _2d_plots:
                    _cbar = _fig.colorbar(_2d_plot, ax=_ax)
                    if _z_label is not None:
                        _cbar.ax.set_ylabel(_z_label, rotation=90, va="bottom", ha='right', y=1.0, labelpad=_z_labelpad)

            # handle sets of horizontal and vertical lines
            for _axlines_key in ('axhlines', 'axvlines'):
                _ax_method_name = _axlines_key[:-1]
                assert hasattr(_ax, _ax_method_name)
                _axlines = _pad_config.pop(_axlines_key, [])
                # wrap in list if not already list
                if not isinstance(_axlines, list):
                    _axlines = [_axlines]
                for _axlines_set in _axlines:
                    if not isinstance(_axlines_set, dict):
                        # wrap inner 'values' in list if not already list
                        if not isinstance(_axlines_set, list):
                            _axlines_set = [_axlines_set]
                        _axlines_set = dict(values=_axlines_set)
                    _vals = _axlines_set.pop('values', [])
                    # draw the line
                    for _val in _vals:
                        getattr(_ax, _ax_method_name)(_val, **dict(self._DEFAULT_LINE_KWARGS, **_axlines_set))

            # -- handle plot legend

            # obtain legend handles and labels
            _hs, _ls = _ax.get_legend_handles_labels()

            # re-sort, reversing the order of labels that are part of a stack
            _stack_labels = _pad_config.pop("stack_labels", None)
            if _pad_config.pop("legend_reverse_stack_order", True):
                _hs, _ls = self._sort_legend_handles_labels(_hs, _ls, stack_labels=_stack_labels)

            # merge legend entries with identical labels
            _hs, _ls = self._merge_legend_handles_labels(_hs, _ls)

            # handle user-supplied legend entries
            _add_entries = _pad_config.pop('legend_additional_entries', [])
            for _i_add_entry, _add_entry in enumerate(_add_entries):
                _l, _pos = '', -1  # default label/insert position
                try:
                    _h = _add_entry['handle']
                except TypeError:
                    # non-dict supplied, use as handle
                    _h = _add_entry
                except KeyError as e:
                    # mandatory keyword not supplied, raise
                    raise ValueError(
                        "Missing mandatory key 'handle' in 'legend_additional_entries' at position {}: {}".format(
                            _i_add_entry, _add_entry))
                else:
                    # value is dict-like and 'handle' was supplied -> get optional kwargs
                    _l = _add_entry.get('label', '')
                    _pos = _add_entry.get('position', -1)

                # insert additional handles/labels at desired position
                _hs.insert(_pos, _h)
                _ls.insert(_pos, _l)

            # draw legend with user-specified kwargs
            _legend_kwargs = self._DEFAULT_LEGEND_KWARGS.copy()
            _legend_kwargs.update(_pad_config.pop('legend_kwargs', {}))
            _ax.legend(_hs, _ls, **_legend_kwargs)

            # handle log x-axis formatting (only if 'x_ticklabels' is not given as [])
            _user_suppressed_x_ticklabels = (
                'x_ticklabels' in _pad_config['_applied_modifiers']
                and not _pad_config['_applied_modifiers']['x_ticklabels']
            )
            if _ax.get_xscale() == 'log' and not _user_suppressed_x_ticklabels:
                _log_decade_ticklabels = _pad_config.get('x_log_decade_ticklabels', {1.0, 2.0, 5.0, 10.0})
                _minor_formatter = LogFormatterSciNotationForceSublabels(base=10.0, labelOnlyBase=False, sci_min_exp=5, sublabels_max_exp=3)
                _major_formatter = LogFormatterSciNotationForceSublabels(base=10.0, labelOnlyBase=True, sci_min_exp=5)
                _ax.xaxis.set_minor_formatter(_minor_formatter)
                _ax.xaxis.set_major_formatter(_major_formatter)

            # NOTE: do not force labeling of minor ticks in log-scaled y axes
            ## handle log y-axis formatting (only if 'y_ticklabels' is not given as [])
            #if _pad_config.get('y_scale', None) == 'log' and _pad_config.get('y_ticklabels', True):
            #    _log_decade_ticklabels = _pad_config.get('y_log_decade_ticklabels', {1.0, 5.0})
            #    _formatter = LogFormatterSciNotationForceSublabels(base=10.0, labelOnlyBase=False)
            #    _ax.yaxis.set_minor_formatter(_formatter)
            #    _formatter.set_locs(locs=_log_decade_ticklabels)

            # draw bin labels instead of numeric labels at ticks
            _bl_sets_by_axis = _pad_config.pop("bin_labels", {})
            _ba_sets_by_axis = _pad_config.pop("bin_label_anchors", {})
            for _axis in "xyz":
                _bl_sets = _bl_sets_by_axis.get(_axis, None)
                _ba_sets = _ba_sets_by_axis.get(_axis, None)

                # skip for axes without bin labels
                if not _bl_sets:
                    continue

                # check if bin labels are identical for all objects in the pad
                if len(_bl_sets) > 1:
                    if False in [_bl_sets[_i_set] == _bl_sets[0] for _i_set in range(1, len(_bl_sets))]:
                        raise ValueError("Bin labels for axis '{}' differ across objects for the same pad! Got the following sets: {}".format(_axis, _bl_sets))
                    elif False in [np.all(_ba_sets[_i_set] == _ba_sets[0]) for _i_set in range(1, len(_ba_sets))]:
                        raise ValueError("Bin label anchors for axis '{}' differ across objects for the same pad! Got the following sets: {}".format(_axis, _ba_sets))

                # draw bin labels
                if _axis == 'x':
                    for _bl, _ba in zip(_bl_sets[0], _ba_sets[0]):
                        _ax.annotate(_bl, xy=(_ba, 0), xycoords=('data', 'axes fraction'), xytext=(0, -6), textcoords='offset points', va='top', ha='right', rotation=30)
                    _ax.xaxis.set_ticks(_ba_sets[0]) # reset tick marks
                    _ax.xaxis.set_ticklabels([])     # hide numeric tick labels
                elif _axis == 'y':
                    for _bl, _ba in zip(_bl_sets[0], _ba_sets[0]):
                        _ax.annotate(_bl, xy=(0, _ba), xycoords=('axes fraction', 'data'), xytext=(-6, 0), textcoords='offset points', va='center', ha='right')
                    _ax.yaxis.set_ticks(_ba_sets[0]) # reset tick marks
                    _ax.yaxis.set_ticklabels([])     # hide numeric tick labels
                else:
                    print("WARNING: Bin labels found for axis '{}', but this is not supported. Ignoring...".format(_axis))

            # run user-defined code on axes
            _pad_config.pop('axes_epilog', lambda ax: None)(_ax)

            # warn about unknown keywords
            _unknown_kws = sorted([_kw for _kw in _pad_config if not _kw.startswith('_')])
            if _unknown_kws:
                print("[WARNING] Unknown or unused keywords supplied to pad config: {}".format(_unknown_kws))

        # step 4: text and annotations

        # draw text/annotations
        _text_configs = config.pop('texts', [])
        for _text_config in _text_configs:
            # retrieve target pad
            _pad_id = _text_config.pop('pad', 0)
            _ax = _pad_configs[_pad_id]['_axes']

            # handle deprecated keyword 'transform'
            if 'transform' in _text_config:
                raise ValueError(
                    "Keyword 'transform' is deprecated. Use keywords "
                    "'xycoords' and 'textcoords' to specify a coordinate "
                    "system.")

            # retrieve coordinates and text
            _xy = _text_config.pop('xy')
            _s = _text_config.pop('text')

            # draw annotation
            _text_config.setdefault('ha', 'left')
            _ax.annotate(_s, _xy,
                **_text_config
            )


        # step 5: figure adjustments

        # handle figure label ("upper_label")
        _upper_label = config.pop('upper_label', None)
        if _upper_label is not None:

            # keyword is deprecated
            warnings.warn(
                "Keyword `upper_label` is deprecated and will be removed in "
                "the future. Replace it with the following equivalent annotation under "
                "`texts`: dict(text='...', xy=(1, 1), xycoords='axes fraction', xytext=(0, 5), "
                "textcoords='offset points', ha='right', pad=0).", DeprecationWarning)

            # place above topmost `Axes`
            _pad_configs[0]['_axes'].annotate(_upper_label, xy=(1, 1),
                xycoords='axes fraction',
                xytext=(0, 5),
                textcoords='offset points',
                ha='right',
            )

        # run user-defined code on figure (and/or axes)
        config.get('epilog', lambda fig, axes: None)(_fig, [_pc['_axes'] for _pc in _pad_configs])

        # step 6: save figures
        make_directory(os.path.dirname(_filename), exist_ok=True)
        _fig.savefig('{}'.format(_filename))
        #plt.close(_fig)  # close figure to save memory

        # dump YAML to file, if requested
        if _dump_yaml:
            with open(_yaml_filename, 'w') as _yaml_file:
                yaml.dump(_config_for_dump, _yaml_file)

        # de-register all the locals after a plot is done
        # self._input_controller.clear_locals()


    # -- register action slots
    _ACTIONS = [_request, _plot]

    # -- additional public API

    def clear_figures(self):
        """Close all figures created while running this processor."""

        for _fign, _fig in six.iteritems(self._figures):
            plt.close(_fig)
        self._figures = {}
