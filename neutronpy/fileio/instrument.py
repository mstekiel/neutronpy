# -*- coding: utf-8 -*-
# TOML library builtin python>=3.10
try: import tomllib
except ModuleNotFoundError: import pip._vendor.tomli as tomllib
from enum import Enum
import pathlib
from typing import Callable

import numpy as np

from ..crystal import Sample
from ..neutron import Neutron
# from ..instrument import Instrument
# from ..instrument.tas_instrument import TripleAxisInstrument
# from ..instrument.tof_instrument import TimeOfFlightInstrument

taz_keys = {'algo': 'method',
            'ana_d': 'ana.d',
            'ana_effic': '',
            'ana_mosaic': 'ana.mosaic',
            'ana_scatter_sense': 'ana.direct',
            'h_coll_after_sample': 'hcol[-2]',
            'h_coll_ana': 'hcol[-1]',
            'h_coll_before_sample': 'hcol[-3]',
            'h_coll_mono': 'hcol[-4]',
            'mono_d': 'mono.d',
            'mono_mosaic': 'mono.mosaic',
            'mono_refl': '',
            'mono_scatter_sense': 'mono.direct',
            'pop_ana_curvh': 'ana.rh',
            'pop_ana_curvv': 'ana.rv',
            'pop_ana_h': 'ana.height',
            'pop_ana_thick': 'ana.depth',
            'pop_ana_use_curvh': '',
            'pop_ana_use_curvv': '',
            'pop_ana_w': 'ana.width',
            'pop_det_h': 'detector.height',
            'pop_det_rect': '',
            'pop_det_w': 'detector.width',
            'pop_dist_ana_det': 'arms[-2]',
            'pop_dist_mono_sample': 'arms[-4]',
            'pop_dist_sample_ana': 'arms[-3]',
            'pop_dist_src_mono': 'arms[-5]',
            'pop_guide_divh': 'arms[-1]',
            'pop_guide_divv': '',
            'pop_mono_curvh': 'mono.rh',
            'pop_mono_curvv': 'mono.rv',
            'pop_mono_h': 'mono.height',
            'pop_mono_thick': 'mono.depth',
            'pop_mono_use_curvh': '',
            'pop_mono_use_curvv': '',
            'pop_mono_w': 'mono.width',
            'pop_sample_cuboid': 'sample.depth',
            'pop_sample_h': 'sample.height',
            'pop_sample_wperpq': '',
            'pop_sample_wq': 'sample.width',
            'pop_src_h': 'guide.height',
            'pop_src_rect': '',
            'pop_src_w': 'guide.width',
            'sample_mosaic': 'sample.mosaic',
            'sample_scatter_sense': 'sample.direct',
            'use_guide': '',
            'v_coll_after_sample': 'vcol[-2]',
            'v_coll_ana': 'vcol[-1]',
            'v_coll_before_sample': 'vcol[-3]',
            'v_coll_mono': 'vcol[-4]'}

# Format of the neutronpy config dictionary is stored as a template
# Ideally each dictionary corresponds to a class, that class should have a constructor allowing to make instance from the dictionary.
with open(pathlib.Path(__file__).parent.joinpath(r'templates\instrument.toml'), "rb") as ff:
    NPY_CONFIG_INSTRUMENT_DEFAULT = tomllib.load(ff)

class TAS_loader():
    """Factory loading various datatypes to neutronpy internal format"""

    def __init__():
        pass

    @classmethod
    def default(cls):
        return NPY_CONFIG_INSTRUMENT_DEFAULT

    @classmethod
    def from_taz(cls, filename):
        """Convert Takin taz file to neutronpy config dictionary.
        
        Parameters
        ----------
        filename: str
            Path and filename of the taz file.

        Notes
        -----
        XML format, I think
        >>> <?xml version=X>
        >>> <ElementRoot_tag>
        >>>     <ElementChild1_tag attribute="attribute_value">element1_text</ElementChild1_tag>
        >>>     <ElementChild2_tag>element2_text</ElementChild2_tag>
        >>> </ElementRoot_tag>

        >>> root = ElementTree(xml_file_above).get_root()
        >>> root.tag
        ... ElementRoot_tag
        >>> for child_element in root:
        >>>     print(childe_element.tag, child_element.attrib, child_element.text)
        ... elementChild1_tag {'attribute': attribute_value} element1_text
        ... elementChild2_tag {} element2_text

        Takin XML
            1. Ignoring fields:
                - simple: for simple resolution calcualtion with deviatins in and out of plane
                - Violini: anything related to resolution of the TOF spectrometer

        """
        import xml.etree.ElementTree as et

        taz_root = et.ElementTree(file=filename).getroot()
        taz_setting = taz_root.find('reso')

        def taz(setting: str, casting_func: Callable, default_value=None):
            setting_element = taz_setting.find(setting)
            if setting_element is None:
                Warning(f'{setting!r} not found in TAZ configuration file.')
                return default_value
            
            return casting_func(setting_element.text)

        tas_config = dict()
        
        ###### Neutronpy specific fields
        # As per convention, assume constant kf
        tas_config['fixed_kf'] = True
        tas_config['fixed_wavevector'] = taz('kf', float)
        tas_config['name'] = 'from_taz'     # TODO replace with filename
        tas_config['a3_offset'] = 90        # seems like takin convention 
        tas_config['kf_vert'] = False       

        ###### Rest of the fields
        # senses are 01 no -1 1
        tas_config['scat_senses'] = [-1+2*taz(comp+"_scatter_sense", int) for comp in ['mono','sample','ana']]
        tas_config['arms'] = [taz("pop_dist_"+v, float) for v in 
                              ["src_mono","mono_sample","sample_ana","ana_det","mono_monitor"]]
        tas_config['hcol'] = [taz("h_coll_"+v, float) for v in
                              ["mono","before_sample","after_sample","ana"]]
        tas_config['vcol'] = [taz("v_coll_"+v, float) for v in
                              ["mono","before_sample","after_sample","ana"]]

        ### Source
        source_config = dict(
            name='Source',
            use_guide = bool(taz("use_guide", int)),
            shape = {0: "circular", 1:"rectangular"}[taz("pop_source_rect", int)],
            width = taz("pop_src_w", float),
            height = taz("pop_src_h", float),
            div_v = taz("pop_guide_divv", float),
            div_h = taz("pop_guide_divh", float)
        )
        tas_config['source'] = source_config

        ### Detector
        detector_config = dict(
            name = 'Detector',
            shape = {0: "circular", 1:"rectangular"}[taz("pop_det_rect", int)],
            width = taz("pop_det_w", float), 
            height = taz("pop_det_h", float)
        )
        tas_config['detector'] = detector_config

        ### Sample
        sample_config = dict(
            a=5, b=5, c=5, alpha=90, beta=90, gamma=90,
            orient1=[1,0,0], orient2=[0,1,0], 
            mosaic = taz("sample_mosaic", float),
            mosaic_v = taz("eck_sample_mosaic_v", float),
            width1 = taz("pop_sample_wq", float),
            width2 = taz("pop_sample_wperpq", float),
            height = taz("pop_sample_h", float), 
            shape = {0: "cylindrical", 1:"cuboid"}[taz("pop_sample_cuboid", int)],
        )
        tas_config['sample'] = sample_config

        ### Monochromator
        # reflectivity omitted
        mono_config = dict(
            name = '_mono',
            dspacing = taz("mono_d", float),
            mosaic = taz("mono_mosaic", float), 
            mosaic_v = taz("eck_mono_mosaic_v", float),
            width = taz("pop_mono_w", float), 
            height = taz("pop_mono_h", float),
            depth = taz("pop_mono_thick", float),
            curvr_h = taz("pop_mono_curvh", float), 
            curvr_v = taz("pop_mono_curvv", float)
        )
        tas_config["mono"] = mono_config

        ### Analyzer
        # reflectivity omitted
        ana_config = dict(
            name = '_ana',
            dspacing = taz("ana_d", float),
            mosaic = taz("ana_mosaic", float), 
            mosaic_v = taz("eck_ana_mosaic_v", float),
            width = taz("pop_ana_w", float), 
            height = taz("pop_ana_h", float),
            depth = taz("pop_ana_thick", float),
            curvr_h = taz("pop_ana_curvh", float), 
            curvr_v = taz("pop_ana_curvv", float)
        )
        tas_config["ana"] = ana_config

        # Takin uses flags here
        MONO_STATE = {0:'flat', 1:'optimal', 2:'curved'}
        tas_config['focussing'] = dict(
            mono_h = MONO_STATE[taz("pop_mono_use_curvh", int)], 
            mono_v = MONO_STATE[taz("pop_mono_use_curvv", int)], 
            ana_h  = MONO_STATE[taz("pop_ana_use_curvh", int)], 
            ana_v  = MONO_STATE[taz("pop_ana_use_curvv", int)]
        )
        # So if any flags were `0` the curvature needs to be set to inf

        return tas_config
    

        # for reso_setting in reso:
        #     key, value = reso_setting.tag, reso_setting.text

        # setup = _Dummy()


        # for key, value in taz_keys.items():
        #     npy_config_dict[value] = reso.find(key).text

        # hcol = [float(npy_config_dict[key]) for key in ['hcol[-4]', 'hcol[-3]', 'hcol[-2]', 'hcol[-1]'] if len(npy_config_dict[key]) > 0]
        # vcol = [float(npy_config_dict[key]) for key in ['vcol[-4]', 'vcol[-3]', 'vcol[-2]', 'vcol[-1]'] if len(npy_config_dict[key]) > 0]
        # arms = [float(npy_config_dict[key]) for key in ['arms[-5]', 'arms[-4]', 'arms[-3]', 'arms[-2]', 'arms[-1]'] if
        #         len(npy_config_dict[key]) > 0]

        # for key, value in zip(['hcol', 'vcol', 'arms'], [hcol, vcol, arms]):
        #     setattr(setup, key, value)

        # for key, value in npy_config_dict.items():
        #     if '[' not in key:
        #         if '.' in key:
        #             subobj = getattr(setup, key.split('.')[0])
        #             setattr(subobj, key.split('.')[1], value)
        #             setattr(setup, key.split('.')[0], subobj)
        #         else:
        #             setattr(setup, key, value)

        return npy_config

def load_instrument(filename, filetype='ascii'):
    r"""Creates TripleAxisInstrument class using input par and cfg files.

    Parameters
    ----------
    filename : str or tuple
        Path to the instrument file. If filetype is 'parcfg' then two
        files should be provided in order of '.par', '.cfg' as a tuple.

    filetype : str, optional
        Default: 'ascii'. Instrument file formats 'ascii', 'parcfg', 'hdf5',
        or 'taz' (xml) are available. 'ascii', 'hdf5' and 'taz' in format
        generated by :py:func:`.load_instrument`. 'taz' format compatible
        with Takin software. See Notes for information about 'parcfg' format
        details.

    Returns
    -------
    setup : obj
        Returns Instrument class object based on the information in the input
        files.

    Notes
    -----
    The format of the ``parfile`` consists of two tab-separated columns, the first
    column containing the values and the second column containing the value
    names preceded by a '%' character:

    +-------+---------+---------------------------------------------------------------------------------+
    | Type  | Name    | Description                                                                     |
    +=======+=========+=================================================================================+
    | float | %DM     | Monochromater d-spacing (Ang^-1)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %DA     | Analyzer d-spacing (Ang^-1)                                                     |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAM   | Monochromator mosaic (arc min)                                                  |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAA   | Analyzer mosaic (arc min)                                                       |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ETAS   | Sample mosaic (arc min)                                                         |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SM     | Scattering direction of monochromator (+1 clockwise, -1 counterclockwise)       |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SS     | Scattering direction of sample (+1 clockwise, -1 counterclockwise)              |
    +-------+---------+---------------------------------------------------------------------------------+
    | int   | %SA     | Scattering direction of analyzer (+1 clockwise, -1 counterclockwise)            |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %K      | Fixed wavevector (incident or final) of neutrons                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA1 | Horizontal collimation of in-pile collimator (arc min)                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA2 | Horizontal collimation of collimator between monochromator and sample (arc min) |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA3 | Horizontal collimation of collimator between sample and analyzer (arc min)      |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %ALPHA4 | Horizontal collimation of collimator between analyzer and detector (arc min)    |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA1  | Vertical collimation of in-pile collimator (arc min)                            |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA2  | Vertical collimation of collimator between monochromator and sample (arc min)   |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA3  | Vertical collimation of collimator between sample and analyzer (arc min)        |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BETA4  | Vertical collimation of collimator between analyzer and detector (arc min)      |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AS     | Sample lattice constant a (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BS     | Sample lattice constant b (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %CS     | Sample lattice constant c (Ang)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AA     | Sample lattice angle alpha (deg)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BB     | Sample lattice angle beta (deg)                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %CC     | Sample lattice angle gamma (deg)                                                |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AX     | Sample orientation vector u_x (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AY     | Sample orientation vector u_y (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %AZ     | Sample orientation vector u_z (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BX     | Sample orientation vector v_x (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BY     | Sample orientation vector v_y (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %BZ     | Sample orientation vector v_z (r.l.u.)                                          |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QX     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QY     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %QZ     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %EN     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqx    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqy    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %dqz    |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %de     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gh     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gk     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gl     |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+
    | float | %gmod   |                                                                                 |
    +-------+---------+---------------------------------------------------------------------------------+

    The format of the ``cfgfile`` (containing values necessary for Popovici type
    calculations) can consists of a single column of values, or two
    tab-separated columns, the first column containing the values and the
    second column containing the value descriptions preceded by a '%' character.
    The values MUST be in the following order:

    +-------+-------------------------------------------------------+
    | Type  | Description                                           |
    +=======+=======================================================+
    | float | =0 for circular source, =1 for rectangular source     |
    +-------+-------------------------------------------------------+
    | float | width/diameter of the source (cm)                     |
    +-------+-------------------------------------------------------+
    | float | height/diameter of the source (cm)                    |
    +-------+-------------------------------------------------------+
    | float | =0 No Guide, =1 for Guide                             |
    +-------+-------------------------------------------------------+
    | float | horizontal guide divergence (minutes/Angs)            |
    +-------+-------------------------------------------------------+
    | float | vertical guide divergence (minutes/Angs)              |
    +-------+-------------------------------------------------------+
    | float | =0 for cylindrical sample, =1 for cuboid sample       |
    +-------+-------------------------------------------------------+
    | float | sample width/diameter perp. to Q (cm)                 |
    +-------+-------------------------------------------------------+
    | float | sample width/diameter along Q (cm)                    |
    +-------+-------------------------------------------------------+
    | float | sample height (cm)                                    |
    +-------+-------------------------------------------------------+
    | float | =0 for circular detector, =1 for rectangular detector |
    +-------+-------------------------------------------------------+
    | float | width/diameter of the detector (cm)                   |
    +-------+-------------------------------------------------------+
    | float | height/diameter of the detector (cm)                  |
    +-------+-------------------------------------------------------+
    | float | thickness of monochromator (cm)                       |
    +-------+-------------------------------------------------------+
    | float | width of monochromator (cm)                           |
    +-------+-------------------------------------------------------+
    | float | height of monochromator (cm)                          |
    +-------+-------------------------------------------------------+
    | float | thickness of analyser (cm)                            |
    +-------+-------------------------------------------------------+
    | float | width of analyser (cm)                                |
    +-------+-------------------------------------------------------+
    | float | height of analyser (cm)                               |
    +-------+-------------------------------------------------------+
    | float | distance between source and monochromator (cm)        |
    +-------+-------------------------------------------------------+
    | float | distance between monochromator and sample (cm)        |
    +-------+-------------------------------------------------------+
    | float | distance between sample and analyser (cm)             |
    +-------+-------------------------------------------------------+
    | float | distance between analyser and detector (cm)           |
    +-------+-------------------------------------------------------+
    | float | horizontal curvature of monochromator 1/radius (cm-1) |
    +-------+-------------------------------------------------------+
    | float | vertical curvature of monochromator (cm-1) was 0.013  |
    +-------+-------------------------------------------------------+
    | float | horizontal curvature of analyser (cm-1) was 0.078     |
    +-------+-------------------------------------------------------+
    | float | vertical curvature of analyser (cm-1)                 |
    +-------+-------------------------------------------------------+
    | float | distance monochromator-monitor                        |
    +-------+-------------------------------------------------------+
    | float | width monitor (cm)                                    |
    +-------+-------------------------------------------------------+
    | float | height monitor (cm)                                   |
    +-------+-------------------------------------------------------+

    """
    if filetype == 'parcfg':
        parfile, cfgfile = filename
        with open(parfile, "r") as f:
            lines = f.readlines()
            par = {}
            for line in lines:
                rows = line.split()
                par[rows[1][1:].lower()] = float(rows[0])

        with open(cfgfile, "r") as f:
            lines = f.readlines()
            cfg = []
            for line in lines:
                rows = line.split()
                cfg.append(float(rows[0]))

        if par['sm'] == par['ss']:
            dir1 = -1
        else:
            dir1 = 1

        if par['ss'] == par['sa']:
            dir2 = -1
        else:
            dir2 = 1

        if par['kfix'] == 2:
            infin = -1
        else:
            infin = par['kfix']

        hcol = [par['alpha1'], par['alpha2'], par['alpha3'], par['alpha4']]
        vcol = [par['beta1'], par['beta2'], par['beta3'], par['beta4']]

        nsou = cfg[0]  # =0 for circular source, =1 for rectangular source.
        if nsou == 0:
            ysrc = cfg[1] / 4  # width/diameter of the source [cm].
            zsrc = cfg[2] / 4  # height/diameter of the source [cm].
        else:
            ysrc = cfg[1] / np.sqrt(12)  # width/diameter of the source [cm].
            zsrc = cfg[2] / np.sqrt(12)  # height/diameter of the source [cm].

        flag_guide = cfg[3]  # =0 for no guide, =1 for guide.
        guide_h = cfg[4]  # horizontal guide divergence [mins/Angs]
        guide_v = cfg[5]  # vertical guide divergence [mins/Angs]
        if flag_guide == 1:
            alpha_guide = np.pi / 60. / 180. * 2 * np.pi * guide_h / par['k']
            alpha0 = hcol[0] * np.pi / 60. / 180.
            if alpha_guide <= alpha0:
                hcol[0] = 2. * np.pi / par['k'] * guide_h
            beta_guide = np.pi / 60. / 180. * 2 * np.pi * guide_v / par['k']
            beta0 = vcol[0] * np.pi / 60. / 180.
            if beta_guide <= beta0:
                vcol[0] = 2. * np.pi / par['k'] * guide_v

        nsam = cfg[6]  # =0 for cylindrical sample, =1 for cuboid sample.
        if nsam == 0:
            xsam = cfg[7] / 4  # sample width/diameter perp. to Q [cm].
            ysam = cfg[8] / 4  # sample width/diameter along Q [cm].
            zsam = cfg[9] / 4  # sample height [cm].
        else:
            xsam = cfg[7] / np.sqrt(12)  # sample width/diameter perp. to Q [cm].
            ysam = cfg[8] / np.sqrt(12)  # sample width/diameter along Q [cm].
            zsam = cfg[9] / np.sqrt(12)  # sample height [cm].

        ndet = cfg[10]  # =0 for circular detector, =1 for rectangular detector.
        if ndet == 0:
            ydet = cfg[11] / 4  # width/diameter of the detector [cm].
            zdet = cfg[12] / 4  # height/diameter of the detector [cm].
        else:
            ydet = cfg[11] / np.sqrt(12)  # width/diameter of the detector [cm].
            zdet = cfg[12] / np.sqrt(12)  # height/diameter of the detector [cm].

        xmon = cfg[13]  # thickness of monochromator [cm].
        ymon = cfg[14]  # width of monochromator [cm].
        zmon = cfg[15]  # height of monochromator [cm].

        xana = cfg[16]  # thickness of analyser [cm].
        yana = cfg[17]  # width of analyser [cm].
        zana = cfg[18]  # height of analyser [cm].

        L0 = cfg[19]  # distance between source and monochromator [cm].
        L1 = cfg[20]  # distance between monochromator and sample [cm].
        L2 = cfg[21]  # distance between sample and analyser [cm].
        L3 = cfg[22]  # distance between analyser and detector [cm].

        romh = par['sm'] * cfg[23]  # horizontal curvature of monochromator 1/radius [cm-1].
        romv = par['sm'] * cfg[24]  # vertical curvature of monochromator [cm-1].
        roah = par['sa'] * cfg[25]  # horizontal curvature of analyser [cm-1].
        roav = par['sa'] * cfg[26]  # vertical curvature of analyser [cm-1].
        inv_rads = [romh, romv, roah, roav]
        for n, inv_rad in enumerate(inv_rads):
            if inv_rad == 0:
                inv_rads[n] = 1.e6
            else:
                inv_rads[n] = 1. / inv_rad
        [romh, romv, roah, roav] = inv_rads

        L1mon = cfg[27]  # distance monochromator monitor [cm]
        monitorw = cfg[28] / np.sqrt(12)  # monitor width [cm]
        monitorh = cfg[29] / np.sqrt(12)  # monitor height [cm]

        # -------------------------------------------------------------------------

        energy = Neutron(wavevector=par['k'])

        sample = Sample(par['as'], par['bs'], par['cs'],
                        par['aa'], par['bb'], par['cc'],
                        par['etas'])
        sample.u = [par['ax'], par['ay'], par['az']]
        sample.v = [par['bx'], par['by'], par['bz']]
        sample.shape = np.diag([xsam, ysam, zsam])

        setup = TripleAxisInstrument(energy.energy, sample, hcol, vcol,
                           2 * np.pi / par['dm'], par['etam'],
                           2 * np.pi / par['da'], par['etaa'])

        setup.method = 1
        setup.dir1 = dir1
        setup.dir2 = dir2
        setup.mondir = par['sm']
        setup.infin = infin
        setup.arms = [L0, L1, L2, L3, L1mon]
        setup.guide.width = ysrc
        setup.guide.height = zsrc

        setup.detector.width = ydet
        setup.detector.height = zdet

        setup.mono.depth = xmon
        setup.mono.width = ymon
        setup.mono.height = zmon
        setup.mono.rv = romv
        setup.mono.rh = romh

        setup.ana.depth = xana
        setup.ana.width = yana
        setup.ana.height = zana
        setup.ana.rv = roav
        setup.ana.rh = roah

        setup.monitor.width = monitorw
        setup.monitor.height = monitorh

        return setup

    elif filetype == 'ascii':
        with open(filename, 'r') as f:
            lines = []
            for line in f:
                value = line.replace('\n', '').replace('instrument.', '').split('=')
                left = value[0].split('.')
                for n, el in enumerate(left):
                    left[n] = el.strip()

                right = value[1].strip()
                lines.append([left, right])

        setup = TripleAxisInstrument()

        for line in lines:
            attr = line[0]
            value = line[1]
            if '[' in value:
                value = value.replace(']', '').replace('[', '').split(' ')
                value = [float(item) for item in value]
            else:
                try:
                    value = float(line[1])
                except ValueError:
                    pass

            if len(attr) == 1:
                setattr(setup, attr[0], value)
            else:
                subobj = getattr(setup, attr[0])
                setattr(subobj, attr[1], value)
                setattr(setup, attr[0], subobj)

        return setup

    elif filetype == 'hdf5':
        import h5py

        setup = TripleAxisInstrument()

        with h5py.File(filename, mode='r') as f:
            instrument = f['instrument']
            groups = list(instrument.keys())

            for key, value in instrument.attrs.items():
                setattr(setup, key, value)

            for grp in groups:
                subobj = getattr(setup, grp)
                for key, value in instrument[grp].attrs.items():
                    if isinstance(value, np.bytes_):
                        value = value.decode('UTF-8')
                    setattr(subobj, key, value)
                setattr(setup, grp, subobj)

        return setup

    elif filetype == 'taz':
        import xml.etree.ElementTree as et

        tree = et.ElementTree(file=filename)
        taz = tree.getroot()
        reso = taz.find('reso')

        setup = TripleAxisInstrument()

        values = dict()

        for key, value in taz_keys.items():
            values[value] = reso.find(key).text

        hcol = [float(values[key]) for key in ['hcol[-4]', 'hcol[-3]', 'hcol[-2]', 'hcol[-1]'] if len(values[key]) > 0]
        vcol = [float(values[key]) for key in ['vcol[-4]', 'vcol[-3]', 'vcol[-2]', 'vcol[-1]'] if len(values[key]) > 0]
        arms = [float(values[key]) for key in ['arms[-5]', 'arms[-4]', 'arms[-3]', 'arms[-2]', 'arms[-1]'] if
                len(values[key]) > 0]

        for key, value in zip(['hcol', 'vcol', 'arms'], [hcol, vcol, arms]):
            setattr(setup, key, value)

        for key, value in values.items():
            if '[' not in key:
                if '.' in key:
                    subobj = getattr(setup, key.split('.')[0])
                    setattr(subobj, key.split('.')[1], value)
                    setattr(setup, key.split('.')[0], subobj)
                else:
                    setattr(setup, key, value)

        return setup

    else:
        raise ValueError("Format not supported. Please use 'ascii', 'hdf5', or 'taz'")


def save_instrument(obj, filename, filetype='ascii', overwrite=False):
    r"""Saves a TAS or TOF instrument configuration into files for loading
    with :py:func:`.load_instrument`.

    Parameters
    ----------
    obj : object
        Instrument object

    filename : str
        Path to file (extension determined by filetype parameter).

    filetype : str, optional
        Default: `'ascii'`. Support for `'ascii'`, `'hdf5'`, or `'taz'`.

    overwrite : bool, optional
        Default: False. If True, overwrites the file, otherwise appends or
        creates new files.

    """
    if isinstance(obj, TripleAxisInstrument):
        save_tas_instrument(obj, filename, filetype, overwrite)
    elif isinstance(obj, TimeOfFlightInstrument):
        save_tof_instrument(obj, filename, filetype, overwrite)


def save_tas_instrument(obj, filename, filetype='ascii', overwrite=False):
    r"""Saves a TAS instrument configuration into files for loading with
    :py:func:`.load_instrument`.

    Parameters
    ----------
    obj : object
        Instrument object

    filename : str
        Path to file (extension determined by filetype parameter).

    filetype : str, optional
        Default: `'ascii'`. Support for `'ascii'`, `'hdf5'`, or `'taz'`.

    overwrite : bool, optional
        Default: False. If True, overwrites the file, otherwise appends or
        creates new files.

    """
    instr_attrs = ['efixed', 'arms', 'hcol', 'vcol', 'method', 'moncor', 'infin']
    mono_attrs = ['tau', 'height', 'width', 'depth', 'direct', 'mosaic', 'vmosaic', 'rh', 'rv']
    ana_attrs = ['tau', 'height', 'width', 'depth', 'direct', 'mosaic', 'vmosaic', 'rh', 'rv', 'horifoc', 'thickness',
                 'Q']
    det_attrs = ['height', 'width', 'depth']
    guide_attrs = ['height', 'width']
    sample_attrs = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'u', 'v', 'mosaic', 'vmosaic', 'height', 'width', 'depth',
                    'direct']
    Smooth_attrs = ['X', 'Y', 'Z', 'E']

    if filetype == 'ascii':
        if overwrite:
            mode = 'w+'
        else:
            mode = 'r+'

        lines = []
        for grp_name, attrs in zip(['', 'mono', 'ana', 'detector', 'guide', 'sample', 'Smooth'],
                                   [instr_attrs, mono_attrs, ana_attrs, det_attrs, guide_attrs, sample_attrs,
                                    Smooth_attrs]):
            for attr in attrs:
                value = ''
                if len(grp_name) == 0:
                    try:
                        value = getattr(obj, attr)
                        value = 'instrument.' + str(attr) + ' = ' + str(value) + '\n'
                    except AttributeError:
                        pass
                else:
                    try:
                        value = getattr(getattr(obj, grp_name), attr)
                        value = 'instrument.' + grp_name + '.' + str(attr) + ' = ' + str(value) + '\n'
                    except AttributeError:
                        pass
                if value:
                    lines.append(value)

        with open(filename + '.instr', mode) as f:
            f.writelines(lines)

    elif filetype == 'hdf5':
        import h5py

        if overwrite:
            mode = 'w'
        else:
            mode = 'a'

        with h5py.File(filename + '.h5', mode) as f:
            instrument = f.create_group('instrument')
            mono = instrument.create_group('mono')
            ana = instrument.create_group('ana')
            detector = instrument.create_group('detector')
            guide = instrument.create_group('guide')
            sample = instrument.create_group('sample')
            Smooth = instrument.create_group('Smooth')

            for grp, grp_name, attrs in zip([instrument, mono, ana, detector, guide, sample, Smooth],
                                            ['', 'mono', 'ana', 'detector', 'guide', 'sample', 'Smooth'],
                                            [instr_attrs, mono_attrs, ana_attrs, det_attrs, guide_attrs, sample_attrs,
                                             Smooth_attrs]):
                for attr in attrs:
                    try:
                        if len(grp_name) == 0:
                            value = getattr(obj, attr)
                            if isinstance(value, str):
                                value = value.encode('utf8')
                            grp.attrs.create(attr, value)
                        else:
                            value = getattr(getattr(obj, grp_name), attr)
                            if isinstance(value, str):
                                value = value.encode('utf8')
                            grp.attrs.create(attr, value)
                    except AttributeError:
                        pass

                if len(list(grp.attrs.keys())) == 0:
                    del instrument[grp_name]

    elif filetype == 'taz':
        import xml.etree.ElementTree as et
        from xml.dom import minidom

        def prettify(elem):
            """Return a pretty-printed XML string for the Element.
            """
            rough_string = et.tostring(elem, encoding='utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")

        if overwrite:
            mode = 'w+'
        else:
            mode = 'r+'

        taz = et.Element('taz')
        reso = et.SubElement(taz, 'reso')

        defaults = [1, 0, 0, 1, 0, 3.355, 45, 1, 5, 3.355, 45, 1, 30, 1e4, 30, 1e4, 1e4, 1e4, 1e4, 1e4, 12, 6, 1,
                    0, 200, 8, .15, 0, 1, 12, 1.5, 0, 3, 1.5, 0, 0, 8, 0.3, 0, 0, 12, 2.5, 5, 1, 10, 200, 115, 85, 15,
                    15]

        for (ele, attr), dflt in zip(taz_keys.items(), defaults):
            subel = et.SubElement(reso, ele)
            value = str(dflt)
            if '.' not in attr:
                if '[' not in attr:
                    try:
                        value = str(getattr(obj, attr))
                    except AttributeError:
                        pass
                else:
                    ind = int(attr[-3:-1])
                    attr = attr[:-4]
                    try:
                        value = str(getattr(obj, attr)[ind])
                    except AttributeError:
                        pass
            else:
                prnt, chld = attr.split('.')
                try:
                    value = str(getattr(getattr(obj, prnt), chld))
                except AttributeError:
                    pass

            subel.text = value

        taz_pretty = prettify(taz)

        with open(filename + '.taz', mode) as f:
            f.write(taz_pretty)

    else:
        raise ValueError("""Format not supported. Please use 'ascii', 'hdf5', or 'taz'""")


def save_tof_instrument(obj, filename, filetype='ascii', overwrite=True):
    r"""Saves a TOF instrument configuration into files for loading with
    :py:func:`.load_instrument`.

    Parameters
    ----------
    obj : object
        Instrument object

    filename : str
        Path to file (extension determined by filetype parameter).

    filetype : str, optional
        Default: `'ascii'`. Support for `'ascii'`, `'hdf5'`, or `'taz'`.

    overwrite : bool, optional
        Default: False. If True, overwrites the file, otherwise appends or
        creates new files.

    """
    instr_attrs = []

    if filetype == 'ascii':
        pass

    elif filetype == 'hdf5':
        pass

    elif filetype == 'taz':

        keys = {'viol_dist_pulse_mono': ['', 1000],
                'viol_dist_mono_sample': ['', 100],
                'viol_dist_sample_det': ['', 500],
                'viol_dist_pulse_mono_sig': ['', 1],
                'viol_dist_mono_sample_sig': ['', 1],
                'viol_dist_sample_det_sig': ['', 1],
                'viol_angle_ph_f': ['', 0],
                'viol_angle_ph_f_sig': ['', 0.4],
                'viol_angle_ph_i': ['', 0],
                'viol_angle_ph_i_sig': ['', 0.4],
                'viol_angle_tt_f_sig': ['', 0.4],
                'viol_angle_tt_i': ['', 0],
                'viol_angle_tt_i_sig': ['', 0.4],
                'viol_time_det_sig': ['', 5],
                'viol_time_mono_sig': ['', 5],
                'viol_time_pulse_sig': ['', 50],
                'viol_det_sph': ['', 1]}

    else:
        raise ValueError("""Format not supported. Please use 'ascii', 'hdf5', or 'taz'""")
