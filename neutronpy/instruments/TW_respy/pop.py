#
# implementation of the popovici algo
#
# @author Tobias Weber <tweber@ill.fr>
# @date feb-2015, oct-2019, apr-2022
# @license GPLv2
#
# @desc for algorithm: [pop75] M. Popovici, Acta Cryst. A 31, 507 (1975), doi: 10.1107/S0567739475001088
# @desc for alternate R0 normalisation: [mit84] P. W. Mitchell, R. A. Cowley and S. A. Higgins, Acta Cryst. Sec A, 40(2), 152-160 (1984), doi: 10.1107/S0108767384000325
# @desc The initial version of this calculation was based on the "rescal5" software by Zinkin, McMorrow, Tennant, Farhi, and Wildes (ca. 1995-2007).
#
# ----------------------------------------------------------------------------
# Takin (inelastic neutron scattering software package)
# Copyright (C) 2017-2024  Tobias WEBER (Institut Laue-Langevin (ILL),
#                          Grenoble, France).
# Copyright (C) 2013-2017  Tobias WEBER (Technische Universitaet Muenchen
#                          (TUM), Garching, Germany).
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 2 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
# ----------------------------------------------------------------------------
#

import numpy as np
import numpy.linalg as la

from . import reso, helpers

# --------------------------------------------------------------------
# matrix element indices
IDX_SRC_Y      = 0;  IDX_SRC_Z      = 1
IDX_MONO_X     = 2;  IDX_MONO_Y     = 3;  IDX_MONO_Z     = 4
IDX_SAMPLE_X   = 5;  IDX_SAMPLE_Y   = 6;  IDX_SAMPLE_Z   = 7
IDX_DET_Y      = 8;  IDX_DET_Z      = 9
IDX_ANA_X      = 10; IDX_ANA_Y      = 11; IDX_ANA_Z      = 12
NUM_MONO_POS   = 8;  NUM_POS        = 13

IDX_SRC_H      = 0;  IDX_SRC_V      = 1
IDX_MONO_H     = 2;  IDX_MONO_V     = 3
IDX_ANA_H      = 4;  IDX_ANA_V      = 5
IDX_SAMPLE_H   = 6;  IDX_SAMPLE_V   = 7
NUM_MONO_COMPS = 4;  NUM_COMPS      = NUM_MONO_COMPS*2

IDX_KI_H       = 0;  IDX_KI_V       = 1
IDX_KF_H       = 2;  IDX_KF_V       = 3
NUM_KI         = 2;  NUM_KIKF       = NUM_KI*2

IDX_KI_X       = 0;  IDX_KI_Y       = 1;  IDX_KI_Z       = 2
NUM_KI_COMPS   = 3
# --------------------------------------------------------------------



def get_mono_trafos(dist_vsrc_mono, dist_hsrc_mono,
    dist_mono_sample,
    thetam, thetas, inv_curvh, inv_curvv,
    ki, Q_ki, sense = 1., pointlike = False):
    """get matrices for ki axis"""

    sign_z = -1.  # to compare with the calculations by F. Bourdarot

    s_th_m = np.sin(thetam)
    c_th_m = sense * np.cos(thetam)
    cot_th_m = c_th_m / s_th_m
    s_th_s = np.sin(thetas)
    c_th_s = sense * np.cos(thetas)
    cot_th_s = c_th_s / s_th_s


    if pointlike:
        # C matrix, [pop75], Appendix 1
        C = np.zeros([NUM_KI, NUM_MONO_COMPS])
        C[IDX_KI_H, IDX_SRC_H] = 0.5
        C[IDX_KI_H, IDX_MONO_H] = 0.5
        C[IDX_KI_V, IDX_SRC_V] = sense * 0.5 / s_th_m
        C[IDX_KI_V, IDX_MONO_V] = -sense * 0.5 / s_th_m

    else:
        # D matrix, [pop75], Appendix 2
        D = np.zeros([NUM_MONO_COMPS, NUM_MONO_POS])

        D[IDX_SRC_H, IDX_SRC_Y] = -sense / dist_hsrc_mono
        D[IDX_SRC_H, IDX_MONO_X] = -c_th_m / dist_hsrc_mono
        D[IDX_SRC_H, IDX_MONO_Y] = s_th_m / dist_hsrc_mono

        D[IDX_SRC_V, IDX_SRC_Z] = -sign_z*sense / dist_vsrc_mono
        D[IDX_SRC_V, IDX_MONO_Z] = sign_z*sense / dist_vsrc_mono

        D[IDX_MONO_H, IDX_MONO_X] = c_th_m / dist_mono_sample
        D[IDX_MONO_H, IDX_MONO_Y] = s_th_m / dist_mono_sample
        D[IDX_MONO_H, IDX_SAMPLE_X] = s_th_s / dist_mono_sample
        D[IDX_MONO_H, IDX_SAMPLE_Y] = c_th_s / dist_mono_sample

        D[IDX_MONO_V, IDX_MONO_Z] = -sign_z*sense / dist_mono_sample
        D[IDX_MONO_V, IDX_SAMPLE_Z] = sign_z*sense / dist_mono_sample


        # T matrix, [pop75], Appendix 2
        T = np.zeros([NUM_KI, NUM_MONO_POS])

        T[IDX_KI_H, IDX_SRC_Y] = 0.5 * D[IDX_SRC_H, IDX_SRC_Y]
        T[IDX_KI_H, IDX_MONO_X] = 0.5 * (D[IDX_SRC_H, IDX_MONO_X] + D[IDX_MONO_H, IDX_MONO_X])
        T[IDX_KI_H, IDX_MONO_Y] = 0.5 * (D[IDX_SRC_H, IDX_MONO_Y] + D[IDX_MONO_H, IDX_MONO_Y]) - inv_curvh
        T[IDX_KI_H, IDX_SAMPLE_X] = 0.5 * D[IDX_MONO_H, IDX_SAMPLE_X]
        T[IDX_KI_H, IDX_SAMPLE_Y] = 0.5 * D[IDX_MONO_H, IDX_SAMPLE_Y]

        # signs for kf
        T[IDX_KI_V, IDX_SRC_Z] = 0.5 * sense * D[IDX_SRC_V, IDX_SRC_Z] / s_th_m
        T[IDX_KI_V, IDX_MONO_Z] = 0.5 * sense * (D[IDX_SRC_V, IDX_MONO_Z] - D[IDX_MONO_V, IDX_MONO_Z]) / s_th_m - sign_z*inv_curvv
        T[IDX_KI_V, IDX_SAMPLE_Z] = -0.5 * sense * D[IDX_MONO_V, IDX_SAMPLE_Z] / s_th_m


    # A matrix, [pop75], Appendix 1
    A = np.zeros([NUM_KI_COMPS, NUM_MONO_COMPS])
    A[IDX_KI_X, IDX_SRC_H] = 0.5 * ki * cot_th_m
    A[IDX_KI_X, IDX_MONO_H] = -0.5 * ki * cot_th_m
    A[IDX_KI_Y, IDX_MONO_H] = ki
    A[IDX_KI_Z, IDX_MONO_V] = sign_z * ki


    # B matrix, [mit84], equ. A.15 and [pop75] Appendix 1
    B = np.zeros([4, NUM_KI_COMPS])
    B[0:2, IDX_KI_X:IDX_KI_X+2] = sense * helpers.rotation_matrix_2d(Q_ki)
    B[2, IDX_KI_Z] = sense
    B[3, IDX_KI_X] = 2. * sense * ki * helpers.ksq2E


    if pointlike:
        return [C, A, B]
    else:
        return [D, T, A, B]



def combine_mono_ana_trafos(Dm, Tm, Am, Bm, Da, Ta, Aa, Ba, pointlike = False):
    """unite trafo matrices"""
    if pointlike:
        Cm = Dm
        Ca = Da

        N = Cm.shape[0]
        M = Cm.shape[1]
        C = np.zeros([2*N, 2*M])

        C[0:N, 0:M] = Cm
        C[N:2*N, M:2*M] = Ca

    else:
        N = Dm.shape[0]
        M = Dm.shape[1]
        D = np.zeros([2*N, 2*M - 3])

        D[0:N, 0:M] = Dm
        # add ana matrix (without sample columns) in lower right block
        D[N:2*N, M:2*M-3] = Da[:, 0:IDX_SAMPLE_X]
        # unite mono and ana sample columns
        D[N:2*N, IDX_SAMPLE_X:IDX_SAMPLE_Z+1] = Da[:, IDX_SAMPLE_X:IDX_SAMPLE_Z+1]


        N = Tm.shape[0]
        M = Tm.shape[1]
        T = np.zeros([2*N, 2*M - 3])

        T[0:N, 0:M] = Tm
        # add ana matrix (without sample columns) in lower right block
        T[N:2*N, M:2*M-3] = Ta[:, 0:IDX_SAMPLE_X]
        # unite mono and ana sample columns
        T[N:2*N, IDX_SAMPLE_X:IDX_SAMPLE_Z+1] = Ta[:, IDX_SAMPLE_X:IDX_SAMPLE_Z+1]


    N = Am.shape[0]
    M = Am.shape[1]
    A = np.zeros([2*N, 2*M])

    A[0:N, 0:M] = Am
    A[N:2*N, M:2*M] = Aa


    N = Bm.shape[0]
    M = Bm.shape[1]
    B = np.zeros([N, 2*M])

    B[0:N, 0:M] = Bm
    B[0:N, M:2*M] = Ba


    BA = np.dot(B, A)


    if pointlike:
        return [C, BA]
    else:
        return [D, T, BA]



#
# resolution algorithm
#
def calc(param: dict, pointlike: bool=False):
    '''Determine resolution of the measurement.
    
    Parameters
    ----------
    param:
        Dictionary with vey specific keys and values describing the measurement.

    pointlike:
        I think this is about the source, either pointlike or square cross-section.

    Returns
    -------
    reso: dict[str, Any]
        Dictionary with calculation results.

        Most significant entries:
        - r0: Resolution prefactor
        - reso: Resolution ellipsoid
        - res_vol: Volume of the resolution ellipsoid.
    '''

    # instrument position
    ki = param["ki"]
    kf = param["kf"]
    E = param["E"]
    Q = param["Q"]

    lam = helpers.k2lam(ki)

    # angles
    twotheta = helpers.get_scattering_angle(ki, kf, Q) * param["sample_sense"]
    thetas = twotheta * 0.5
    thetam = helpers.get_mono_angle(ki, param["mono_xtal_d"]) * param["mono_sense"]
    thetaa = helpers.get_mono_angle(kf, param["ana_xtal_d"]) * param["ana_sense"]
    Q_ki = helpers.get_angle_Q_ki(ki, kf, Q) * param["sample_sense"]
    Q_kf = helpers.get_angle_Q_kf(ki, kf, Q) * param["sample_sense"]

    if param["verbose"]:
        print("2theta = %g deg, thetam = %g deg, thetaa = %g deg, Q_ki = %g deg, Q_kf = %g deg.\n" %
            (twotheta*helpers.rad2deg, thetam*helpers.rad2deg, thetaa*helpers.rad2deg,
            Q_ki*helpers.rad2deg, Q_kf*helpers.rad2deg))


    # --------------------------------------------------------------------
    # mono/ana focus
    # use fixed values
    mono_curv_h = param["mono_curv_h"]
    mono_curv_v = param["mono_curv_v"]
    ana_curv_h = param["ana_curv_h"]
    ana_curv_v = param["ana_curv_v"]

    # use a user-defined curvature formula, if given
    if param["mono_curv_h_formula"] != None:
        mono_curv_h = param["mono_curv_h_formula"](param) * helpers.cm2A
    if param["mono_curv_v_formula"] != None:
        mono_curv_v = param["mono_curv_v_formula"](param) * helpers.cm2A
    if param["ana_curv_h_formula"] != None:
        ana_curv_h = param["ana_curv_h_formula"](param) * helpers.cm2A
    if param["ana_curv_v_formula"] != None:
        ana_curv_v = param["ana_curv_v_formula"](param) * helpers.cm2A

    if param["mono_is_optimally_curved_h"]:
        mono_curv_h = helpers.foc_curv(param["dist_hsrc_mono"], \
            param["dist_mono_sample"], np.abs(2.*thetam), False)
    if param["mono_is_optimally_curved_v"]:
        mono_curv_v = helpers.foc_curv(param["dist_vsrc_mono"], \
            param["dist_mono_sample"], np.abs(2.*thetam), True)
    if param["ana_is_optimally_curved_h"]:
        ana_curv_h = helpers.foc_curv(param["dist_sample_ana"], \
            param["dist_ana_det"], np.abs(2.*thetaa), False)
    if param["ana_is_optimally_curved_v"]:
        ana_curv_v = helpers.foc_curv(param["dist_sample_ana"], \
            param["dist_ana_det"], np.abs(2.*thetaa), True)

    if param["verbose"]:
        print("Mono curvature radius: vertical: %g cm, horizontal: %g cm." %
               (mono_curv_v/helpers.cm2A, mono_curv_h/helpers.cm2A))
        print("Ana curvature radius: vertical: %g cm, horizontal: %g cm.\n" %
               (ana_curv_v/helpers.cm2A, ana_curv_h/helpers.cm2A))

    inv_mono_curv_h = 0.
    inv_mono_curv_v = 0.
    inv_ana_curv_h = 0.
    inv_ana_curv_v = 0.

    if param["mono_is_curved_h"]:
        inv_mono_curv_h = 1./mono_curv_h * param["mono_sense"]
    if param["mono_is_curved_v"]:
        inv_mono_curv_v = 1./mono_curv_v * param["mono_sense"]
    if param["ana_is_curved_h"]:
        inv_ana_curv_h = 1./ana_curv_h * param["ana_sense"]
    if param["ana_is_curved_v"]:
        inv_ana_curv_v = 1./ana_curv_v * param["ana_sense"]
    # --------------------------------------------------------------------


    # dict with results
    res = {}

    res["Q_avg"] = np.array([ Q, 0., 0., E ])
    res["ki"] = ki
    res["kf"] = kf
    res["Q_ki"] = Q_ki
    res["Q_kf"] = Q_kf
    res["twotheta"] = twotheta
    res["theta_m"] = thetam
    res["theta_a"] = thetaa


    # -------------------------------------------------------------------------
    # - if the instruments works in kf=const mode and the scans are counted for
    #   or normalised to monitor counts no ki^3 or kf^3 factor is needed.
    # - if the instrument works in ki=const mode the kf^3 factor is needed.

    # TODO
    #tupScFact = get_scatter_factors(param.flags, param.thetam, param.ki, param.thetaa, param.kf)
    tupScFact = [1., 1., 1.]

    dmono_refl = param["dmono_refl"] * tupScFact[0]
    dana_effic = param["dana_effic"] * tupScFact[1]
    dxsec = tupScFact[2]
    #if param.mono_refl_curve:
    #    dmono_refl *= (*param.mono_refl_curve)(param.ki)
    #if param.ana_effic_curve:
    #    dana_effic *= (*param.ana_effic_curve)(param.kf)
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    # collimators, crystal mosaics, and instrument geometry
    coll_h_pre_mono = param["coll_h_pre_mono"]
    coll_v_pre_mono = param["coll_v_pre_mono"]

    if param["use_guide"]:
        coll_h_pre_mono = lam*param["guide_div_h"]
        coll_v_pre_mono = lam*param["guide_div_v"]

    # G matrix, [pop75], Appendix 1
    G = np.zeros([NUM_COMPS, NUM_COMPS])
    G[IDX_SRC_H, IDX_SRC_H] = 1. / coll_h_pre_mono**2.
    G[IDX_SRC_V, IDX_SRC_V] = 1. / coll_v_pre_mono**2.
    G[IDX_MONO_H, IDX_MONO_H] = 1. / param["coll_h_pre_sample"]**2.
    G[IDX_MONO_V, IDX_MONO_V] = 1. / param["coll_v_pre_sample"]**2.
    G[IDX_SAMPLE_H, IDX_SAMPLE_H] = 1. / param["coll_h_post_sample"]**2.
    G[IDX_SAMPLE_V, IDX_SAMPLE_V] = 1. / param["coll_v_post_sample"]**2.
    G[IDX_ANA_H, IDX_ANA_H] = 1. / param["coll_h_post_ana"]**2.
    G[IDX_ANA_V, IDX_ANA_V] = 1. / param["coll_v_post_ana"]**2.

    # F matrix, [pop75], Appendix 1
    F = np.zeros([NUM_KIKF, NUM_KIKF])
    F[IDX_KI_H, IDX_KI_H] = 1. / param["mono_mosaic"]**2.
    F[IDX_KI_V, IDX_KI_V] = 1. / param["mono_mosaic_v"]**2.
    F[IDX_KF_H, IDX_KF_H] = 1. / param["ana_mosaic"]**2.
    F[IDX_KF_V, IDX_KF_V] = 1. / param["ana_mosaic_v"]**2.

    # S matrix, [pop75], Appendices 2 and 3
    S = np.zeros([NUM_POS, NUM_POS])
    if param["src_shape"] == "rectangular":
        src_factor = 12.
    elif param["src_shape"] == "circular":
        src_factor = 16.
    else:
        raise "ResPy: No valid source shape given."
    if param["sample_shape"] == "cuboid":
        sample_factor = 12.
    elif param["sample_shape"] == "cylindrical":
        sample_factor = 16.
    else:
        raise "ResPy: No valid sample shape given."
    if param["det_shape"] == "rectangular":
        det_factor = 12.
    elif param["det_shape"] == "circular":
        det_factor = 16.
    else:
        raise "ResPy: No valid detector shape given."

    if not pointlike:
        S[IDX_SRC_Y, IDX_SRC_Y] = src_factor / param["src_w"]**2.
        S[IDX_SRC_Z, IDX_SRC_Z] = src_factor / param["src_h"]**2.
        S[IDX_MONO_X, IDX_MONO_X] = 12. / param["mono_d"]**2.
        S[IDX_MONO_Y, IDX_MONO_Y] = 12. / param["mono_w"]**2.
        S[IDX_MONO_Z, IDX_MONO_Z] = 12. / param["mono_h"]**2.
        S[IDX_SAMPLE_X, IDX_SAMPLE_X] = sample_factor / param["sample_d"]**2.
        S[IDX_SAMPLE_Y, IDX_SAMPLE_Y] = sample_factor / param["sample_w"]**2.
        S[IDX_SAMPLE_Z, IDX_SAMPLE_Z] = 12. / param["sample_h"]**2.
        S[IDX_ANA_X, IDX_ANA_X] = 12. / param["ana_d"]**2.
        S[IDX_ANA_Y, IDX_ANA_Y] = 12. / param["ana_w"]**2.
        S[IDX_ANA_Z, IDX_ANA_Z] = 12. / param["ana_h"]**2.
        S[IDX_DET_Y, IDX_DET_Y] = det_factor / param["det_w"]**2.
        S[IDX_DET_Z, IDX_DET_Z] = det_factor / param["det_h"]**2.
        S /= helpers.sig2fwhm**2.
        Sinv = la.inv(S)
    # -------------------------------------------------------------------------


    # -------------------------------------------------------------------------
    if pointlike:
        # ki and kf trafo matrices
        [Cm, Am, Bm] = get_mono_trafos(param["dist_vsrc_mono"], param["dist_hsrc_mono"], \
            param["dist_mono_sample"], \
            thetam, thetas, inv_mono_curv_h, inv_mono_curv_v, ki, Q_ki, 1., pointlike)
        [Ca, Aa, Ba] = get_mono_trafos(param["dist_ana_det"], param["dist_ana_det"], \
            param["dist_sample_ana"], \
            thetaa, thetas, inv_ana_curv_h, inv_ana_curv_v, kf, Q_kf, -1., pointlike)

        [C, BA] = combine_mono_ana_trafos(Cm, None, Am, Bm, Ca, None, Aa, Ba, pointlike)

        # [pop75], equ. 8
        H = np.dot(np.dot(np.transpose(C), F), C)

    else:
        # ki and kf trafo matrices
        [Dm, Tm, Am, Bm] = get_mono_trafos(param["dist_vsrc_mono"], param["dist_hsrc_mono"], \
            param["dist_mono_sample"], \
            thetam, thetas, inv_mono_curv_h, inv_mono_curv_v, ki, Q_ki, 1., pointlike)
        [Da, Ta, Aa, Ba] = get_mono_trafos(param["dist_ana_det"], param["dist_ana_det"], \
            param["dist_sample_ana"], \
            thetaa, thetas, inv_ana_curv_h, inv_ana_curv_v, kf, Q_kf, -1., pointlike)

        [D, T, BA] = combine_mono_ana_trafos(Dm, Tm, Am, Bm, Da, Ta, Aa, Ba, pointlike)

        # [pop75], equ. 20
        K = S + np.dot(np.dot(np.transpose(T), F), T)
        Kinv = la.inv(K)

        # [pop75], equ. 17
        Hinv = np.dot(np.dot(D, Kinv), np.transpose(D))
        H = la.inv(Hinv)
    # -------------------------------------------------------------------------

    # [pop75], equ. 18
    HG = H + G
    HGinv = la.inv(HG)

    # [pop75], equ. 20
    cov = np.dot(np.dot(BA, HGinv), np.transpose(BA))


    cov[1, 1] += Q**2. * param["sample_mosaic"]**2.
    cov[2, 2] += Q**2. * param["sample_mosaic_v"]**2.
    R = la.inv(cov) * helpers.sig2fwhm**2.

    if param["mirror_Qperp"] and param["sample_sense"] < 0.:
        # mirror Q_perp
        matMirror = helpers.mirror_matrix(len(R), 1)
        R = np.dot(np.dot(np.transpose(matMirror), R), matMirror)

    R0 = 0.
    if param["calc_R0"]:
        R0 = dmono_refl*dana_effic * dxsec * (0.5*np.pi)**2. \
            / (np.sin(thetam) * np.sin(thetaa))

        # [pop75], equ. 5 & 9
        if pointlike:
            R0 *= np.sqrt(la.det(F) / la.det(H))

        # [pop75], equ. 13a & 16
        else:
            DS = np.dot(np.dot(D, Sinv), np.transpose(D))
            DSinv = la.inv(DS) + G
            R0 *= np.sqrt(la.det(S)*la.det(F) / (la.det(K)*la.det(DSinv)))

    res["reso"] = R
    res["reso_v"] = np.array([0., 0., 0., 0.])
    res["reso_s"] = 0.
    res["r0"] = R0
    res["res_vol"] = reso.ellipsoid_volume(R)

    res["ok"] = True
    if np.isnan(res["r0"]) or np.isinf(res["r0"]) or np.isnan(res["reso"].any()) or np.isinf(res["reso"].any()):
        res["ok"] = False

    return res
