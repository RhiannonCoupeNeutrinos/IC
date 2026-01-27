import os

import warnings

import numpy  as np
import tables as tb

from numpy.testing import assert_array_equal
from numpy.testing import assert_approx_equal
from pytest        import mark
from pytest        import raises
from scipy.signal  import find_peaks_cwt

from .                       import calib_functions as cf
from .. core                 import   tbl_functions as tbl
from .. core                 import   fit_functions as fitf
from .. core                 import system_of_units as units
from .. core.stat_functions  import   poisson_sigma
from .. evm.nh5              import     SensorTable
from .. types.symbols        import      SensorType
from .. cities.components    import  get_run_number

## Looked at this one, its simple enough and makes sense. It tests the bin waveform function
def test_bin_waveforms():
    bins=np.arange(0, 20) #20 bins that the waveforms are binned into. Can imagine as 20 energy bins each one unit wide.
    
    fake_wf_1=np.random.uniform(0, 20, size=100) #Fake waveform 100 microseconds long, containing values between 0 and 20 in each microsecond chunk.
    fake_wf_2=np.random.uniform(0, 30, size=100) #Another fake waveform where some of the values can fall outside the binning region (to test this)
    fake_file=[fake_wf_1, fake_wf_2] #Puts the two fake waveforms into a similar format to an incoming RWF file.
    
    test_value=np.stack((np.histogram(fake_wf_1, bins)[0], np.histogram(fake_wf_2, bins)[0])) #Value produced by test, what it should be.
    function_value=cf.bin_waveforms(fake_file, bins) #Value produced by the function being tested.
    assert_array_equal(function_value, test_value)

def test_spaced_integrals():
    limits = np.array([2, 4, 6])
    data   = np.arange(20).reshape(2, 10)

    expected = np.array([[5, 9, 30], [25, 29, 70]])
    actual   = cf.spaced_integrals(data, limits)
    assert_array_equal(actual, expected)


@mark.parametrize("limits",
                  ([-1, 0,  2],
                   [-1, 1, 10],
                   [ 0, 2, 10],
                   [ 0, 3, 11]))
def test_spaced_integrals_raises_ValueError_with_wrong_limits(limits):
    limits = np.array(limits)
    data   = np.arange(20).reshape(2, 10)
    with raises(ValueError):
        cf.spaced_integrals(data, limits)


def test_integral_limits():
    sampling    =  1 * units.mus
    n_integrals = 10
    start_int   =  5
    width_int   =  1
    period      = 50

    expected_llim = np.array([  5,   6,  55,  56, 105, 106, 155, 156, 205, 206, 255, 256, 305, 306, 355, 356, 405, 406, 455, 456])
    expected_dlim = np.array([  2,   3,  52,  53, 102, 103, 152, 153, 202, 203, 252, 253, 302, 303, 352, 353, 402, 403, 452, 453])

    (actual_llimits,
     actual_dlimits) = cf.integral_limits(sampling, n_integrals, start_int, width_int, period)

    assert_array_equal(actual_llimits, expected_llim)
    assert_array_equal(actual_dlimits, expected_dlim)

def test_filter_limits():
    sampling         =   1 * units.mus
    n_integrals      =  10
    start_int        =   5
    width_int        =   1
    period           =  50
    fake_data_length = 400
    n_fit=int(fake_data_length/period)  #Number of integrals that would fit inside the fake data length.
    
    (unfiltered_llimits,
     unfiltered_dlimits) = cf.integral_limits(sampling, n_integrals, start_int, width_int, period)

    filtered_llimits = cf.filter_limits(unfiltered_llimits, fake_data_length)
    filtered_dlimits = cf.filter_limits(unfiltered_dlimits, fake_data_length)
    
    expected_llimts=unfiltered_llimits[0:(2*n_fit)]
    expected_dlimts=unfiltered_dlimits[0:(2*n_fit)]

    assert len(filtered_llimits) < len(unfiltered_llimits)  #Tests filtered limits are a smaller set than the calculated integral limits.
    assert len(filtered_dlimits) < len(unfiltered_dlimits)
    assert len(filtered_llimits) % 2 == 0   #Tests that the filter limits are divisible by two, so no open ended integrals.
    assert len(filtered_dlimits) % 2 == 0
    assert_array_equal(filtered_llimits, expected_llimts)  #Tests that the filtered limits are the values of, and stop where we expect them to.
    assert_array_equal(filtered_dlimits, expected_dlimts)

@mark.parametrize("sensor_type        sensors".split(),
                  ((None,              None),
                   ('DataPMT',      (0, 11)),
                   ('DataSiPM', (1013, 1000))))
def test_copy_sensor_table(config_tmpdir, sensor_type, sensors):

    ## Create an input file
    in_name = os.path.join(config_tmpdir, 'test_copy_in.h5')
    with tb.open_file(in_name, 'w') as input_file:
        if sensor_type:
            sens_group = input_file.create_group(input_file.root,
                                                       'Sensors')
            sens_table = input_file.create_table(sens_group ,
                                                 sensor_type,
                                                 SensorTable,
                                                          "",
                                                 tbl.filters("NOCOMPR"))
            row = sens_table.row
            row["channel"]  = sensors[0]
            row["sensorID"] = sensors[1]
            row.append()
            sens_table.flush

    out_name = os.path.join(config_tmpdir, 'test_copy_out.h5')
    with tb.open_file(out_name, 'w') as out_file:

        cf.copy_sensor_table(in_name, out_file)

        if sensor_type:
            assert   'Sensors' in out_file.root
            assert sensor_type in out_file.root.Sensors

            sensor_info = getattr(out_file.root.Sensors,
                                            sensor_type)
            assert sensor_info[0][0] == sensors[0]
            assert sensor_info[0][1] == sensors[1]


@mark.parametrize('sensor_type, Detector, n_channel, gain_seed, gain_sigma_seed',
                  ((SensorType.SIPM, "new",        1,   16.5622,         2.5),
                   (SensorType.PMT , "new",      5,   24.9557,         9.55162),
                   (SensorType.SIPM, "next100", 1,  17.0108,           1.95237),
                   (SensorType.PMT, "next100",  5, 32.5141,            10.59980 ))) 


#Edited to also test a PMT and SiPM from the NEXT100 database, to ensure this function works for both NEW and NEXT100.
def test_seeds_db(sensor_type, n_channel, gain_seed, gain_sigma_seed, Detector):
    if Detector=='new':
        run_number_new = 6217
        detector_new='new'
        result = cf.seeds_db(sensor_type, detector_new, run_number_new, n_channel)
        assert result == (gain_seed, gain_sigma_seed)
    elif Detector=='next100':
        run_number_n100 = 15539
        detector_n100='next100'
        result = cf.seeds_db(sensor_type, detector_n100, run_number_n100, n_channel)
        assert result == (gain_seed, gain_sigma_seed)

_dark_scaler_sipm = cf.dark_scaler(np.array([612, 1142, 2054, 3037, 3593, 3769, 3777, 3319, 2321, 1298, 690]))
_dark_scaler_pmt  = cf.dark_scaler(np.array([ 30,  107,  258,  612, 1142, 2054, 3037, 3593                 ]))

@mark.parametrize('     sensor_type,            scaler,        mu',
                  ((SensorType.SIPM, _dark_scaler_sipm, 0.0698154),
                   (SensorType.PMT , _dark_scaler_pmt , 0.0950066)))
def test_poisson_mu_seed(sensor_type, scaler, mu):
    bins     = np.array([-8,  -7,  -6,  -5,  -4,    -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec     = np.array([28,  98,  28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])

    result   = cf.poisson_mu_seed(sensor_type, scaler, bins, spec, ped_vals)
    assert_approx_equal(result, mu)


@mark.parametrize('     sensor_type,            scaler,    expected_range, min_b, max_b, half_width, p1pe_seed, lim_p',
                  ((SensorType.SIPM, _dark_scaler_sipm,  np.arange(4 ,20),    10,    22,          5,         3, 10000),
                   (SensorType.PMT ,  _dark_scaler_pmt,  np.arange(10,20),    15,    50,         10,         7, 10000)))
def test_sensor_values(sensor_type, scaler, expected_range, min_b, max_b, half_width, p1pe_seed, lim_p):
    bins        = np.array([ -6,  -5,   -4,   -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec        = np.array([ 28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals    = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])
    sens_values = cf.sensor_values(sensor_type, scaler, bins, spec, ped_vals)

    assert_array_equal(sens_values.peak_range, expected_range)
    assert len(sens_values.spectra)        == len(spec)
    assert     sens_values.min_bin_peak    == min_b
    assert     sens_values.max_bin_peak    == max_b
    assert     sens_values.half_peak_width == half_width
    assert     sens_values.p1pe_seed       == p1pe_seed
    assert     sens_values.lim_ped         == lim_p


@mark.parametrize('sensor_type, run_number, n_chann,            scaler',
                  ((      None,       6217,    1023, _dark_scaler_sipm),
                   (      None,       6217,       0, _dark_scaler_pmt)))
def test_incorrect_sensor_type_raises_ValueError(sensor_type, dbnew, run_number, n_chann, scaler):
    bins     = np.array([ -6,  -5,   -4,   -3,   -2,   -1,    0,    1,    2,    3,    4,   5,   6,   7])
    spec     = np.array([ 28, 539, 1072, 1845, 2805, 3251, 3626, 3532, 3097, 2172, 1299, 665, 371, 174])
    ped_vals = np.array([2.65181178e+04, 1.23743445e-01, 2.63794236e+00])

    with raises(ValueError):
        cf.       seeds_db(sensor_type, dbnew, run_number, n_chann)
        cf.poisson_mu_seed(sensor_type, scaler, bins, spec, ped_vals)
        cf.  sensor_values(sensor_type, scaler, bins, spec, ped_vals)


def test_pedestal_values():
    ped_vals   = np.array([6.14871401e+04, -1.46181517e-01, 5.27614635e+00])
    ped_errs   = np.array([9.88752708e+02,  5.38541961e-02, 1.07169703e-01])
    ped_values = cf.pedestal_values(ped_vals, 10000, ped_errs)

    assert_approx_equal(ped_values.gain     ,   -0.14618, 5)
    assert_approx_equal(ped_values.sigma    ,    5.27614, 5)
    assert_approx_equal(ped_values.gain_min , -538.68814, 5)
    assert_approx_equal(ped_values.gain_max ,  538.39577, 5)
    assert_approx_equal(ped_values.sigma_max, 1076.97317, 5)
    assert_approx_equal(ped_values.sigma_min,      0.001)

#Edited this test to use an example spectrum with known gain and gain_sigma rather than use an input file. 
def test_compute_seeds_from_spectrum(ICDATADIR):
    #A mock SiPM noise spectrum which has the following properties:
    #Gain = 17 ADC
    #Sigma_gain = 2 ADC
    #Total number of events = 10,000.
    #SiPM noise parameter (mu) = 0.05 counts/mus.

    fake_x_bins=np.arange(35,135)

    fake_y_vals=np.array([1.15779866e-09, 4.34460742e-08, 1.26968040e-06, 2.88977927e-05,
       5.12225632e-04, 7.07105674e-03, 7.60210243e-02, 6.36516243e-01,
       4.15060729e+00, 2.10785231e+01, 8.33671760e+01, 2.56788980e+02,
       6.16004740e+02, 1.15084837e+03, 1.67447449e+03, 1.89742818e+03,
       1.67447450e+03, 1.15084842e+03, 6.16005061e+02, 2.56790715e+02,
       8.33754548e+01, 2.11133783e+01, 4.28011029e+00, 1.06114024e+00,
       1.30471131e+00, 3.14464162e+00, 7.07113664e+00, 1.40616446e+01,
       2.46789054e+01, 3.82234346e+01, 5.22452408e+01, 6.30197913e+01,
       6.70842247e+01, 6.30198213e+01, 5.22453566e+01, 3.82238233e+01,
       2.46801018e+01, 1.40650100e+01, 7.07947448e+00, 3.15880078e+00,
       1.27554690e+00, 5.19771223e-01, 3.07261417e-01, 3.40398733e-01,
       4.91476319e-01, 7.04783524e-01, 9.41461229e-01, 1.15918311e+00,
       1.31347523e+00, 1.36935416e+00, 1.31347362e+00, 1.15914794e+00,
       9.41183386e-01, 7.03148687e-01, 4.83417032e-01, 3.05993996e-01,
       1.78626821e-01, 9.67197060e-02, 4.95314942e-02, 2.55046694e-02,
       1.52667867e-02, 1.24433026e-02, 1.31857342e-02, 1.53081668e-02,
       1.75586151e-02, 1.91887609e-02, 1.97731080e-02, 1.91590200e-02,
       1.74436414e-02, 1.49210364e-02, 1.19912500e-02, 9.05506035e-03,
       6.42744662e-03, 4.29259343e-03, 2.70405296e-03, 1.61709860e-03,
       9.33322097e-04, 5.40367178e-04, 3.37849055e-04, 2.48651749e-04,
       2.19690527e-04, 2.17419011e-04, 2.22154813e-04, 2.23347072e-04,
       2.16323699e-04, 2.00216956e-04, 1.76562200e-04, 1.48206124e-04,
       1.18386790e-04, 9.00064461e-05, 6.51654784e-05, 4.49879149e-05,
       2.96994942e-05, 1.88657874e-05, 1.16830926e-05, 7.23623968e-06,
       4.67730244e-06, 3.31789791e-06, 2.65295029e-06, 2.34286859e-06])

    #Fit to determine pedestal parameters.
    gb0     = [(0, 1, 0), (np.inf, 100, 10000)]
    sd0     = (fake_y_vals.sum(), 50, 2)
    errs    = fitf.poisson_sigma(fake_y_vals, default=0.1)
    gfitRes = fitf.fit(fitf.gauss, fake_x_bins, fake_y_vals, sd0, errs, bounds=gb0)    
    ped_vals =np.array([gfitRes.values[0], gfitRes.values[1], gfitRes.values[2]])

    #Isolate bins and values around n=1 spectrum peak.
    first_peak_bins=[]
    first_peak_values=[]

    for i in range(25, 40):
        first_peak_bins.append(fake_x_bins[i])
        first_peak_values.append(fake_y_vals[i])
    
    first_peak_bins=np.array(first_peak_bins)

    #Make into format that the computer_seeds_from_spectrum function reads.
    scaler_func = cf.dark_scaler(first_peak_values)
    first_peak_sesnsor_vals=cf.sensor_values(SensorType.SIPM, scaler_func, first_peak_bins, first_peak_values, ped_vals)
    first_peak_BINS= (first_peak_bins>= 60) & (first_peak_bins <= 74)

    #Applies function that is being tested.
    gain_seed, gain_sigma_seed = cf.compute_seeds_from_spectrum(first_peak_sesnsor_vals, first_peak_bins, ped_vals)

    #Testing lines. Ensures calculated gains are within 1% of the ones known to be true for this spectrum.
    assert (0.99*17)<gain_seed<(1.01*17)
    assert (0.99*2)<gain_sigma_seed<(1.01*2)



def test_seeds_without_using_db(ICDATADIR, dbnew):
    PATH_IN = os.path.join(ICDATADIR, 'sipmcalspectra_R6358.h5')
    # Suppress warnings from division by zero in some bins.
    with warnings.catch_warnings(), tb.open_file(PATH_IN) as h5in:
        warnings.simplefilter("ignore", category=RuntimeWarning)
        run_no  = get_run_number(h5in)

        specsL   = np.array(h5in.root.HIST.sipm_spe).sum(axis=0)
        specsD   = np.array(h5in.root.HIST.sipm_dark).sum(axis=0)
        bins     = np.array(h5in.root.HIST.sipm_spe_bins)
        min_stat = 10

        for ich, (led, dar) in enumerate(zip(specsL, specsD)):
            b1 = 0
            b2 = len(dar)
            try:
                valid_bins = np.argwhere(led>=min_stat)
                b1 = valid_bins[ 0][0]
                b2 = valid_bins[-1][0]
            except IndexError:
                continue

            peaks_dark = find_peaks_cwt(dar, np.arange(2, 20), min_snr=2)
            if len(peaks_dark) == 0:
                continue

            gb0     = [(0, -100, 0), (np.inf, 100, 10000)]
            sd0     = (dar.sum(), 0, 2)
            sel     = np.arange(peaks_dark[0]-5, peaks_dark[0]+5)
            errs    = poisson_sigma(dar[sel], default=0.1)
            gfitRes = fitf.fit(fitf.gauss, bins[sel], dar[sel], sd0,
                               sigma=errs, bounds=gb0)

            ped_vals      = np.array([gfitRes.values[0], gfitRes.values[1],
                                      gfitRes.values[2]])
            p_range       = slice(b1, b2)
            p_bins        = (bins[p_range] >= -5) & (bins[p_range] <= 5)
            scaler_func   = cf.dark_scaler(dar[p_range][p_bins])
            seeds, bounds = cf.seeds_and_bounds(SensorType.SIPM, run_no, ich,
                                                scaler_func, bins[p_range],
                                                led[p_range], ped_vals, dbnew,
                                                gfitRes.errors,
                                                use_db_gain_seeds=False)
            assert all(seeds)
            assert bounds == ((0, 0, 0, 0.001), (np.inf, 10000, 10000, 10000))
