import numpy as np

from pytest import approx
from pytest import raises
from pytest import mark
from pytest import fixture

from hypothesis                import       assume
from hypothesis                import        given
from hypothesis                import     settings
from hypothesis.strategies     import     integers
from hypothesis.strategies     import       floats
from hypothesis.strategies     import sampled_from
from hypothesis.strategies     import    composite
from hypothesis.extra.numpy    import       arrays

from .. core.core_functions    import           weighted_mean_and_std
from .. core.random_sampling   import                    NoiseSampler
from .. core.system_of_units_c import                           units
from .. core.testing_utils     import                         exactly
from .. core.testing_utils     import assert_SensorResponses_equality
from .. core.testing_utils     import            assert_Peak_equality
from .. core.testing_utils     import                  previous_float

from invisible_cities.database import load_db as DB

from .  pmaps import  PMTResponses
from .  pmaps import SiPMResponses
from .  pmaps import            S1
from .  pmaps import            S2
from .  pmaps import          PMap
from .  pmaps import    SiPMCharge


wf_min =   0
wf_max = 100


@composite
def sensor_responses(draw, n_samples=None, subtype=None, ids=None):
    n_sensors   = draw(integers(1,  5)) if       ids is None else len(ids)
    n_samples   = draw(integers(1, 50)) if n_samples is None else n_samples
    shape       = n_sensors, n_samples
    all_wfs     = draw(arrays(float,     shape, floats  (wf_min, wf_max)))
    if     ids is None:
        ids     = draw(arrays(  int, n_sensors, integers(0, 1e3), unique=True))
    if subtype is None:
        subtype = draw(sampled_from((PMTResponses, SiPMResponses)))
    args        = np.sort(ids), all_wfs
    return args, subtype(*args)


@composite
def peaks(draw, subtype=None, pmt_ids=None, with_sipms=True):
    nsamples      = draw(integers(1, 20))
    _, pmt_r      = draw(sensor_responses(nsamples,  PMTResponses, pmt_ids))
    sipm_r        = SiPMResponses.build_empty_instance()
    assume(pmt_r.sum_over_sensors[ 0] != 0)
    assume(pmt_r.sum_over_sensors[-1] != 0)

    if subtype is None:
        subtype   = draw(sampled_from((S1, S2)))
    if with_sipms:
        _, sipm_r = draw(sensor_responses(nsamples, SiPMResponses))

    times      = draw(arrays(float, nsamples,
                             floats(min_value=0, max_value=1e3),
                             unique = True).map(sorted))

    bin_widths = np.array([1])
    if len(times) > 1:
        time_differences = np.diff(times)
        bin_widths = np.append(time_differences, max(time_differences))

    args       = times, bin_widths, pmt_r, sipm_r
    return args, subtype(*args)


@composite
def pmaps(draw, pmt_ids=None):
    n_s1 = draw(integers(0, 3))
    n_s2 = draw(integers(0, 3))
    assume(n_s1 + n_s2 > 0)

    s1s  = tuple(draw(peaks(S1, pmt_ids, False))[1] for i in range(n_s1))
    s2s  = tuple(draw(peaks(S2, pmt_ids, True ))[1] for i in range(n_s2))
    args = s1s, s2s
    return args, PMap(*args)


@given(sensor_responses())
def test_SensorResponses_all_waveforms(srs):
    (_, all_waveforms), sr = srs
    assert all_waveforms == approx(sr.all_waveforms)


@given(sensor_responses())
def test_SensorResponses_ids(srs):
    (ids, _), sr = srs
    assert ids == exactly(sr.ids)


@given(sensor_responses())
def test_SensorResponses_waveform(srs):
    (ids, all_waveforms), sr = srs
    for sensor_id, waveform in zip(ids, all_waveforms):
        assert waveform == approx(sr.waveform(sensor_id))


@given(sensor_responses())
def test_SensorResponses_time_slice(srs):
    (_, all_waveforms), sr = srs
    for i, time_slice in enumerate(all_waveforms.T):
        assert time_slice == approx(sr.time_slice(i))


@given(sensor_responses())
def test_SensorResponses_sum_over_times(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=1) == approx(sr.sum_over_times)


@given(sensor_responses())
def test_SensorResponses_sum_over_sensors(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=0) == approx(sr.sum_over_sensors)


@mark.parametrize("SR", (PMTResponses, SiPMResponses))
@given(size=integers(1, 10))
def test_SensorResponses_raises_exception_when_shapes_dont_match(SR, size):
    with raises(ValueError):
        sr = SR(np.empty(size),
                np.empty((size + 1, 1)))


@given(peaks())
def test_Peak_sipms(pks):
    (_, _, _, sipm_r), peak = pks
    assert_SensorResponses_equality(sipm_r, peak.sipms)


@given(peaks())
def test_Peak_pmts(pks):
    (_, _, pmt_r, _), peak = pks
    assert_SensorResponses_equality(pmt_r, peak.pmts)


@given(peaks())
def test_Peak_times(pks):
    (times, _, _, _), peak = pks
    assert times == approx(peak.times)


@given(peaks())
def test_Peak_time_at_max_energy(pks):
    _, peak = pks
    index_at_max_energy = np.argmax(peak.pmts.sum_over_sensors)
    assert peak.time_at_max_energy == peak.times[index_at_max_energy]


@given(peaks())
def test_Peak_total_energy(pks):
    _, peak = pks
    assert peak.total_energy == approx(peak.pmts.all_waveforms.sum())


@given(peaks())
def test_Peak_total_charge(pks):
    _, peak = pks
    assert peak.total_charge == approx(peak.sipms.all_waveforms.sum())


@given(peaks())
def test_Peak_height(pks):
    _, peak = pks
    assert peak.height == approx(peak.pmts.sum_over_sensors.max())


#@given(peaks())
#def test_Peak_width(pks):
#    _, peak = pks
#    assert peak.width == approx(peak.times[-1] - peak.times[0])

def test_Peak_width_correct():
    nsamples = 3
    times  = np.arange(nsamples)
    widths = np.full(nsamples, 1)
    pmts   = PMTResponses(np.arange(12), np.full((12, nsamples), 1))

    peak = S1(times, widths, pmts, SiPMResponses.build_empty_instance())
    assert peak.width == nsamples


def _get_indices_above_thr(sr, thr):
    return np.where(sr.sum_over_sensors > thr)[0]


@given(peaks())
def test_Peak_energy_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    assert peak.energy_above_threshold(sum_wf_min) == approx(peak.total_energy)


@given(peaks())
def test_Peak_energy_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    assert peak.energy_above_threshold(peak.height) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_energy_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = _get_indices_above_thr(peak.pmts, thr)
    assert (peak.pmts.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.energy_above_threshold(thr)))


@given(peaks())
def test_Peak_charge_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.sipms.sum_over_sensors.min())
    assert peak.charge_above_threshold(sum_wf_min) == approx(peak.total_charge)


@given(peaks())
def test_Peak_charge_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    sipms_max = peak.sipms.sum_over_sensors.max()
    assert peak.charge_above_threshold(sipms_max) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_charge_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = _get_indices_above_thr(peak.sipms, thr)
    assert (peak.sipms.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.charge_above_threshold(thr)))


@given(peaks())
#def test_Peak_width_above_threshold_less_than_wf_min(pks):
def test_Peak_width_above_threshold_with_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    full_width = peak.width_above_threshold(sum_wf_min)
    assert full_width == approx(np.sum(peak.bin_widths))


@given(peaks())
#def test_Peak_width_above_threshold_greater_than_equal_to_wf_max(pks):
def test_Peak_width_above_threshold_max_zero(pks):
    _, peak = pks
    assert peak.width_above_threshold(peak.height) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_width_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = _get_indices_above_thr(peak.pmts, thr)
    expected        = (np.sum(peak.bin_widths[i_above_thr])
                       if np.size(i_above_thr) > 0
                       else 0)
    assert peak.width_above_threshold(thr) == approx(expected)


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_rms_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = _get_indices_above_thr(peak.pmts, thr)
    times_above_thr = peak.times[i_above_thr]
    wf_above_thr    = peak.pmts.sum_over_sensors[i_above_thr]
    expected        = (weighted_mean_and_std(times_above_thr, wf_above_thr)[1]
                       if np.size(i_above_thr) > 1 and np.sum(wf_above_thr) > 0
                       else 0)
    assert peak.rms_above_threshold(thr) == approx(expected)


@given(peaks())
def test_Peak_rms_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    assert peak.rms == approx(peak.rms_above_threshold(sum_wf_min))


@given(peaks())
def test_Peak_rms_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    assert peak.rms_above_threshold(peak.height) == 0


@mark.parametrize("PK", (S1, S2))
@given(sr1=sensor_responses(), sr2=sensor_responses())
def test_Peak_raises_exception_when_shapes_dont_match(PK, sr1, sr2):
    with raises(ValueError):
        (ids, wfs), sr1 = sr1
        _         , sr2 = sr2
        n_samples       = wfs.shape[1]
        pk = PK(np.empty(n_samples + 1),
                np.empty(n_samples + 1), sr1, sr2)


@given(pmaps())
def test_PMap_s1s(pmps):
    (s1s, _), pmp = pmps
    assert len(pmp.s1s) == len(s1s)
    for kept_s1, true_s1 in zip(pmp.s1s, s1s):
        assert_Peak_equality(kept_s1, true_s1)


@given(pmaps())
def test_PMap_s2s(pmps):
    (_, s2s), pmp = pmps
    assert len(pmp.s2s) == len(s2s)
    for kept_s2, true_s2 in zip(pmp.s2s, s2s):
        assert_Peak_equality(kept_s2, true_s2)



@fixture(scope='module')
def signal_to_noise_6400():
    return NoiseSampler('new', 6400).signal_to_noise


@fixture(scope='module')
def s2_peak():
    times      = np.arange(20) * units.mus
    bin_widths = np.full_like(times, units.mus)

    pmt_ids  = DB.DataPMT ('new', 6400).SensorID.values
    sipm_ids = DB.DataSiPM('new', 6400).index.values

    pmts  = PMTResponses(pmt_ids,
                         np.random.uniform(0, 100,
                                           (len(pmt_ids), len(times))))

    sipms = SiPMResponses(sipm_ids,
                          np.random.uniform(0, 10,
                                            (len(sipm_ids), len(times))))

    return S2(times, bin_widths, pmts, sipms)


@mark.parametrize("charge_type", SiPMCharge)
def test_sipm_charge_array(charge_type         ,
                           s2_peak             ,
                           signal_to_noise_6400):
    charge_arr = s2_peak.sipm_charge_array(signal_to_noise_6400,
                                           charge_type         ,
                                           single_point = False)

    all_wf = s2_peak.sipms.all_waveforms
    assert np.array(charge_arr).shape   == all_wf.T.shape
    assert np.count_nonzero(charge_arr) == np.count_nonzero(all_wf)


@mark.parametrize("charge_type", SiPMCharge)
def test_sipm_charge_array_single(charge_type         ,
                                  s2_peak             ,
                                  signal_to_noise_6400):
    charge_arr = s2_peak.sipm_charge_array(signal_to_noise_6400,
                                           charge_type         ,
                                           single_point =  True)

    assert charge_arr.shape == s2_peak.sipms.ids.shape
    orig_zeros = np.count_nonzero(s2_peak.sipms.sum_over_times)
    assert np.count_nonzero(charge_arr) == orig_zeros
