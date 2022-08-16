import math, os
import numpy as np
from scipy import signal
import resampy

##predefined filter coefficients, as found by Jan Brond
A_coeff = np.array(
    [1, -4.1637, 7.5712, -7.9805, 5.385, -2.4636, 0.89238, 0.06361, -1.3481, 2.4734, -2.9257, 2.9298, -2.7816, 2.4777,
     -1.6847, 0.46483, 0.46565, -0.67312, 0.4162, -0.13832, 0.019852])
B_coeff = np.array(
    [0.049109, -0.12284, 0.14356, -0.11269, 0.053804, -0.02023, 0.0063778, 0.018513, -0.038154, 0.048727, -0.052577,
     0.047847, -0.046015, 0.036283, -0.012977, -0.0046262, 0.012835, -0.0093762, 0.0034485, -0.00080972, -0.00019623])


def pptrunc(data, max_value):
    '''
    Saturate a vector such that no element's absolute value exceeds max_abs_value.
    Current name: absolute_saturate().
      :param data: a vector of any dimension containing numerical data
      :param max_value: a float value of the absolute value to not exceed
      :return: the saturated vector
    '''
    outd = np.where(data > max_value, max_value, data)
    return np.where(outd < -max_value, -max_value, outd)


def trunc(data, min_value):
    '''
    Truncate a vector such that any value lower than min_value is set to 0.
    Current name zero_truncate().
    :param data: a vector of any dimension containing numerical data
    :param min_value: a float value the elements of data should not fall below
    :return: the truncated vector
    '''

    return np.where(data < min_value, 0, data)


def runsum(data, length, threshold):
    '''
    Compute the running sum of values in a vector exceeding some threshold within a range of indices.
    Divides the data into len(data)/length chunks and sums the values in excess of the threshold for each chunk.
    Current name run_sum().
    :param data: a 1D numerical vector to calculate the sum of
    :param len: the length of each chunk to compute a sum along, as a positive integer
    :param threshold: a numerical value used to find values exceeding some threshold
    :return: a vector of length len(data)/length containing the excess value sum for each chunk of data
    '''

    N = len(data)
    cnt = int(math.ceil(N / length))

    rs = np.zeros(cnt)

    for n in range(cnt):
        for p in range(length * n, length * (n + 1)):
            if p < N and data[p] >= threshold:
                rs[n] = rs[n] + data[p] - threshold

    return rs


def counts(data, filesf, B=B_coeff, A=A_coeff):
    '''
    Get activity counts for a set of accelerometer observations.
    First resamples the data frequency to 30Hz, then applies a Butterworth filter to the signal, then filters by the
    coefficient matrices, saturates and truncates the result, and applies a running sum to get the final counts.
    Current name get_actigraph_counts()
    :param data: the vertical axis of accelerometer readings, as a vector
    :param filesf: the number of observations per second in the file
    :param a: coefficient matrix for filtering the signal, as found by Jan Brond
    :param b: coefficient matrix for filtering the signal, as found by Jan Brond
    :return: a vector containing the final counts
    '''

    deadband = 0.068
    sf = 30
    peakThreshold = 2.13
    adcResolution = 0.0164
    integN = 10
    gain = 0.965

    # if filesf>sf:
    data = resampy.resample(np.asarray(data), filesf, sf)

    B2, A2 = signal.butter(4, np.array([0.01, 7]) / (sf / 2), btype='bandpass')
    dataf = signal.filtfilt(B2, A2, data)

    B = B * gain

    # NB: no need for a loop here as we only have one axis in array
    fx8up = signal.lfilter(B, A, dataf)

    fx8 = pptrunc(fx8up[::3], peakThreshold)  # downsampling is replaced by slicing with step parameter

    return runsum(np.floor(trunc(np.abs(fx8), deadband) / adcResolution), integN, 0)


def POI(sample):
    """
    Calculate the percentage of time spent immobile in a window
    """

    def calc_mob_per_min(countx, county, countz):
        mob_per_min = []
        for i in range(0, len(countx), 60):
            countx_1m = np.mean(countx[i:i + 60])
            county_1m = np.mean(county[i:i + 60])
            countz_1m = np.mean(countz[i:i + 60])
            mob_per_min.append(np.mean([countx_1m, county_1m, countz_1m]))
        return mob_per_min

    def percentagem_of_immobility(mob_per_min):
        mob_per_min = np.asarray(mob_per_min)
        inactivity_counts = (mob_per_min <= 4).sum()
        return inactivity_counts / len(mob_per_min)

    # calculate counts per axis
    sample = sample.squeeze()
    c1_1s = counts(sample[0], 10)
    c2_1s = counts(sample[1], 10)
    c3_1s = counts(sample[2], 10)
    mob_per_min = calc_mob_per_min(c1_1s, c2_1s, c3_1s)
    POI = percentagem_of_immobility(mob_per_min)
    return POI
