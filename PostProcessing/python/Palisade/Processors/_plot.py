from __future__ import print_function

import colorsys  # for rgb_to_hls
import math
import os
import six
import yaml

from copy import deepcopy
from functools import partial

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import numpy as np

from matplotlib.gridspec import GridSpec
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.colors import LogNorm, Normalize, colorConverter

from rootpy.plotting import Hist1D, Hist2D, Profile1D, Efficiency, F1
from rootpy.plotting.hist import _Hist, _Hist2D
from rootpy.plotting.profile import _ProfileBase

from .._input import InputROOT
from .._colormaps import viridis

from ._base import ContextValue, LiteralString, _ProcessorBase, _make_directory

__all__ = ['LogFormatterSciNotationForceSublabels', 'PlotProcessor']


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

    def set_locs(self, *args, **kwargs):
        '''override sublabels'''
        _ret = super(LogFormatterSciNotationForceSublabels, self).set_locs(*args, **kwargs)

        # override locations
        _locs = kwargs.pop("locs", None)
        if _locs is not None:
            self._sublabels = _locs
        else:
            self._sublabels = {1.0, 2.0, 5.0, 10.0}

        return _ret


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
    _x = np.asarray(args[0])
    _y = np.asarray(args[1])
    _zeros = np.zeros_like(_x)

    # kwarg `yerr_as_band` to display
    _show_yerr_as = kwargs.pop('show_yerr_as', None)
    if _show_yerr_as is not None and _show_yerr_as not in ('errorbar', 'band'):
        raise ValueError("Invalid value '{}' for 'show_yerr_as'. Available: {}".format(_show_yerr_as, ('errorbar', 'band')))

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
    _x = np.vstack([_x, _x, _x, _x, _x]).T.flatten()
    _y = np.vstack([_y, _y, _y, _y, _y]).T.flatten()

    # stop processing y errors if they are zero
    if np.allclose(_yerr, 0):
        _yerr = None

    # attach y errors (if any) to "bin" center
    if _yerr is not None:
        if _show_yerr_as == 'band':
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
        _capsize = kwargs.pop('capsize', None)
        _markeredgecolor = kwargs.pop('markeredgecolor', None)
        if _yerr is None:
            _yerr = 0, 0
        return (
            ax.errorbar(_x, _y, yerr=None, capsize=_capsize, markeredgecolor=_markeredgecolor, **kwargs),
            ax.fill_between(_x, _y-_yerr[0], _y+_yerr[1], **dict(kwargs, alpha=_band_alpha, linewidth=0)))


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
        loc='upper right'
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
        self._input_controller.register_local('expressions', [_subplot_cfg['expression'] for _subplot_cfg in config['subplots']])

        _filename = os.path.join(self._output_folder, config['filename'])

        # prepare dict for YAML dump, if requested
        _dump_yaml = config.pop('dump_yaml', False)
        if _dump_yaml:
            _yaml_filename = '.'.join(_filename.split('.')[:-1]) + '.yml'
            # need to create directory first
            _make_directory(os.path.dirname(_yaml_filename))
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
        _height_ratios = [_pc.get('height_share', 1) for _pc in _pad_configs]

        # construct GridSpec from `pad_spec` or make default
        _gridspec_kwargs = config.get('pad_spec', dict())
        _gridspec_kwargs.pop('height_ratios', None)   # ignore explicit user-provided `height_ratios`
        _gs = GridSpec(nrows=len(_pad_configs), ncols=1, height_ratios=_height_ratios, **_gridspec_kwargs)

        # store `Axes` objects in pad configuration
        for _i_pad, _pad_config in enumerate(_pad_configs):
            _pad_config['axes'] = _fig.add_subplot(_gs[_i_pad])

        _stack_bottoms = _pad_config.setdefault('stack_bottoms', {})
        _bin_labels = _pad_config.setdefault('bin_labels', {})
        _bin_label_anchors = _pad_config.setdefault('bin_label_anchors', {})

        # enable text output, if requested
        if config.pop("text_output", False):
            _text_filename = '.'.join(_filename.split('.')[:-1]) + '.txt'
            # need to create directory first
            _make_directory(os.path.dirname(_text_filename))
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
            _ax = _pad_config['axes']
            _stack_bottoms = _pad_config.setdefault('stack_bottoms', {})
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

            # extract arrays for keys which cannot be masked
            _plot_data.update({
                _property_name : np.array(list(getattr(_plot_object, _property_name)()))
                for _property_name in ('xedges', 'yedges', 'z')
                if hasattr(_plot_object, _property_name)
            })

            # -- draw

            _plot_method_name = _kwargs.pop('plot_method', 'errorbar')

            # -- obtain plot method
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
                elif _marker_style == 'empty':
                    _kwargs.update(
                        markerfacecolor='w',
                        markeredgewidth=1,
                    )
                else:
                    raise ValueError("Unkown value for 'marker_style': {}".format(_marker_style))

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
                # mask zeros
                _z_masked = np.ma.array(_plot_data['z'], mask=_plot_data['z']==0)

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
                    _z_scale = _pad_config.get('z_scale', "linear")
                    if _z_scale == 'linear':
                        _norm = Normalize(vmin=_z_min, vmax=_z_max)
                    elif _z_scale == 'log':
                        _norm = LogNorm(vmin=_z_min, vmax=_z_max)
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
                _bin_label_color = _kwargs.pop('bin_label_color', 'k')
            else:
                _args = [_plot_data['x'], _y_data]

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
                                _ax.text(_x, _y, _bin_label_format.format(_content),
                                         ha='center', va='center',
                                         fontsize=16,
                                         color=_text_color,
                                         transform=_ax.transData
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
            _ax = _pad_config['axes']

            # simple axes adjustments
            for _prop_name, _meth_dict in six.iteritems(self._PC_KEYS_MPL_AXES_METHODS):
                _prop_val = _pad_config.get(_prop_name, None)
                if _prop_val is not None:
                    #print(_prop_name, _prop_val)
                    getattr(_ax, _meth_dict['method'])(_prop_val, **_meth_dict.get('kwargs', {}))

            # draw colorbar if there was a 2D plot involved
            if _pad_config.get('2d_plots', None):
                for _2d_plot in _pad_config['2d_plots']:
                    _cbar = _fig.colorbar(_2d_plot, ax=_ax)
                    _z_label = _pad_config.get('z_label', None)
                    _z_labelpad = _pad_config.get('z_labelpad', None)
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
                    _vals = _axlines_set.pop('values')
                    # draw the line
                    for _val in _vals:
                        getattr(_ax, _ax_method_name)(_val, **dict(self._DEFAULT_LINE_KWARGS, **_axlines_set))

            # -- handle plot legend

            # obtain legend handles and labels
            _hs, _ls = _ax.get_legend_handles_labels()

            # re-sort, reversing the order of labels that are part of a stack
            _hs, _ls = self._sort_legend_handles_labels(_hs, _ls, stack_labels=_pad_config.get("stack_labels", None))

            # merge legend entries with identical labels
            _hs, _ls = self._merge_legend_handles_labels(_hs, _ls)

            # draw legend with user-specified kwargs
            _legend_kwargs = self._DEFAULT_LEGEND_KWARGS.copy()
            _legend_kwargs.update(_pad_config.pop('legend_kwargs', {}))
            _ax.legend(_hs, _ls, **_legend_kwargs)

            # handle log x-axis formatting (only if 'x_ticklabels' is not given as [])
            if _pad_config.get('x_scale', None) == 'log' and _pad_config.get('x_ticklabels', True):
                _log_decade_ticklabels = _pad_config.get('x_log_decade_ticklabels', {1.0, 2.0, 5.0, 10.0})
                _formatter = LogFormatterSciNotationForceSublabels(base=10.0, labelOnlyBase=False)
                _ax.xaxis.set_minor_formatter(_formatter)
                _formatter.set_locs(locs=_log_decade_ticklabels)

            # NOTE: do not force labeling of minor ticks in log-scaled y axes
            ## handle log y-axis formatting (only if 'y_ticklabels' is not given as [])
            #if _pad_config.get('y_scale', None) == 'log' and _pad_config.get('y_ticklabels', True):
            #    _log_decade_ticklabels = _pad_config.get('y_log_decade_ticklabels', {1.0, 5.0})
            #    _formatter = LogFormatterSciNotationForceSublabels(base=10.0, labelOnlyBase=False)
            #    _ax.yaxis.set_minor_formatter(_formatter)
            #    _formatter.set_locs(locs=_log_decade_ticklabels)

            # draw bin labels instead of numeric labels at ticks
            for _axis in "xyz":
                _bl_sets = _pad_config["bin_labels"].get(_axis, None)
                _ba_sets = _pad_config["bin_label_anchors"].get(_axis, None)

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


        # step 4: text and annotations

        # draw text/annotations
        _text_configs = config.pop('texts', [])
        for _text_config in _text_configs:
            # retrieve target pad
            _pad_id = _text_config.pop('pad', 0)
            _ax = _pad_configs[_pad_id]['axes']

            # retrieve coordinates and text
            _x, _y = _text_config.pop('xy')
            _s = _text_config.pop('text')

            # lookup transform by string
            _transform = _text_config.pop('transform', None)
            if _transform is None or _transform == 'axes':
                _transform = _ax.transAxes
            elif _transform == 'data':
                _transform = _ax.transData
            elif _transform == 'figure':
                _transform = _ax.get_figure().transFigure
            elif _transform == 'display':
                _transform = None  # == identity transformation
            elif callable(_transform):
                _transform =  _transform(_ax)
            else:
                raise ValueError("Unknown coordinate transform specification '{}': expected e.g. 'axes' or 'data'".format(_transform))

            # draw text
            _text_config.setdefault('ha', 'left')
            _ax.text(_x, _y, _s,
                transform=_transform,
                **_text_config
            )


        # step 5: figure adjustments

        # handle figure label ("upper_label")
        _upper_label = config.pop('upper_label', None)
        if _upper_label is not None:
            # place above topmost `Axes`
            _ax_top = _pad_configs[0]['axes']
            _ax_top.text(1.0, 1.015,
                _upper_label,
                ha='right',
                transform=_ax_top.transAxes
            )


        # step 6: save figures
        _make_directory(os.path.dirname(_filename))
        _fig.savefig('{}'.format(_filename))
        #plt.close(_fig)  # close figure to save memory

        # dump YAML to file, if requested
        if _dump_yaml:
            with open(_yaml_filename, 'w') as _yaml_file:
                yaml.dump(_config_for_dump, _yaml_file)

        # de-register all the locals after a plot is done
        self._input_controller.clear_locals()


    # -- register action slots
    _ACTIONS = [_request, _plot]

    # -- additional public API

    def clear_figures(self):
        """Close all figures created while running this processor."""

        for _fign, _fig in six.iteritems(self._figures):
            plt.close(_fig)
        self._figures = {}
