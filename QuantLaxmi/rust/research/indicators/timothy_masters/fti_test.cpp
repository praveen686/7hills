// FTI Parity Test - C++ Reference
// Compile: g++ -o fti_test fti_test.cpp -lm -std=c++11
// Run: ./fti_test

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Minimal FTI implementation extracted from FTI.CPP
class FTI {
public:
    int use_log;
    int min_period;
    int max_period;
    int half_length;
    int lookback;
    double beta;
    double noise_cut;

    double *y;
    double *coefs;
    double *filtered;
    double *width;
    double *fti;
    int *sorted;
    double *diff_work;
    double *leg_work;
    double *sort_work;
    int n_periods;

    FTI(int use_log_, int min_p, int max_p, int half_len, int look, double beta_, double noise) {
        use_log = use_log_;
        min_period = min_p;
        max_period = max_p;
        half_length = half_len;
        lookback = look;
        beta = beta_;
        noise_cut = noise;
        n_periods = max_period - min_period + 1;

        y = new double[lookback + half_length];
        coefs = new double[n_periods * (half_length + 1)];
        filtered = new double[n_periods];
        width = new double[n_periods];
        fti = new double[n_periods];
        sorted = new int[n_periods];
        diff_work = new double[lookback];
        leg_work = new double[lookback];
        sort_work = new double[n_periods];

        // Compute filter coefficients
        for (int period = min_period; period <= max_period; period++) {
            find_coefs(period);
        }
    }

    ~FTI() {
        delete[] y;
        delete[] coefs;
        delete[] filtered;
        delete[] width;
        delete[] fti;
        delete[] sorted;
        delete[] diff_work;
        delete[] leg_work;
        delete[] sort_work;
    }

    void find_coefs(int period) {
        int idx_base = (period - min_period) * (half_length + 1);
        double d[4] = {0.35577019, 0.2436983, 0.07211497, 0.00630165};

        double fact = 2.0 / (double)period;
        coefs[idx_base] = fact;

        fact *= 3.14159265358979323846;
        for (int i = 1; i <= half_length; i++) {
            coefs[idx_base + i] = sin((double)i * fact) / ((double)i * 3.14159265358979323846);
        }

        coefs[idx_base + half_length] *= 0.5;

        double sumg = coefs[idx_base];
        for (int i = 1; i <= half_length; i++) {
            double sum = d[0];
            double fact2 = (double)i * 3.14159265358979323846 / (double)half_length;
            for (int j = 1; j <= 3; j++) {
                sum += 2.0 * d[j] * cos((double)j * fact2);
            }
            coefs[idx_base + i] *= sum;
            sumg += 2.0 * coefs[idx_base + i];
        }

        for (int i = 0; i <= half_length; i++) {
            coefs[idx_base + i] /= sumg;
        }
    }

    void process(double *data, int chronological) {
        // Copy data to y in chronological order
        for (int i = 0; i < lookback; i++) {
            int idx = chronological ? (lookback - 1 - i) : i;
            double price = data[idx];
            y[lookback - 1 - i] = use_log ? log(price) : price;
        }

        // Extend with least-squares line
        double xmean = -0.5 * (double)half_length;
        double ymean = 0.0;
        for (int i = 0; i <= half_length; i++) {
            ymean += y[lookback - 1 - i];
        }
        ymean /= (double)(half_length + 1);

        double xsq = 0.0, xy = 0.0;
        for (int i = 0; i <= half_length; i++) {
            double xdiff = -(double)i - xmean;
            double ydiff = y[lookback - 1 - i] - ymean;
            xsq += xdiff * xdiff;
            xy += xdiff * ydiff;
        }
        double slope = xy / xsq;

        for (int i = 0; i < half_length; i++) {
            y[lookback + i] = ((double)i + 1.0 - xmean) * slope + ymean;
        }

        // Process each period
        for (int iperiod = min_period; iperiod <= max_period; iperiod++) {
            int period_idx = iperiod - min_period;
            int coef_base = period_idx * (half_length + 1);

            int extreme_type = 0;
            double extreme_value = 0.0;
            int n_legs = 0;
            double longest_leg = 0.0;
            double prior = 0.0;

            for (int iy = half_length; iy < lookback; iy++) {
                double sum = coefs[coef_base] * y[iy];
                for (int i = 1; i <= half_length; i++) {
                    sum += coefs[coef_base + i] * (y[iy + i] + y[iy - i]);
                }

                if (iy == lookback - 1) {
                    filtered[period_idx] = sum;
                }

                diff_work[iy - half_length] = fabs(y[iy] - sum);

                if (iy == half_length) {
                    extreme_type = 0;
                    extreme_value = sum;
                    n_legs = 0;
                    longest_leg = 0.0;
                } else if (extreme_type == 0) {
                    if (sum > extreme_value) {
                        extreme_type = -1;
                    } else if (sum < extreme_value) {
                        extreme_type = 1;
                    }
                } else if (iy == lookback - 1) {
                    if (extreme_type != 0) {
                        leg_work[n_legs] = fabs(extreme_value - sum);
                        if (leg_work[n_legs] > longest_leg) {
                            longest_leg = leg_work[n_legs];
                        }
                        n_legs++;
                    }
                } else {
                    if (extreme_type == 1 && sum > prior) {
                        leg_work[n_legs] = extreme_value - prior;
                        if (leg_work[n_legs] > longest_leg) {
                            longest_leg = leg_work[n_legs];
                        }
                        n_legs++;
                        extreme_type = -1;
                        extreme_value = prior;
                    } else if (extreme_type == -1 && sum < prior) {
                        leg_work[n_legs] = prior - extreme_value;
                        if (leg_work[n_legs] > longest_leg) {
                            longest_leg = leg_work[n_legs];
                        }
                        n_legs++;
                        extreme_type = 1;
                        extreme_value = prior;
                    }
                }

                prior = sum;
            }

            // Sort diff_work for fractile
            int diff_len = lookback - half_length;
            for (int i = 0; i < diff_len - 1; i++) {
                for (int j = i + 1; j < diff_len; j++) {
                    if (diff_work[j] < diff_work[i]) {
                        double tmp = diff_work[i];
                        diff_work[i] = diff_work[j];
                        diff_work[j] = tmp;
                    }
                }
            }
            int width_idx = (int)(beta * (double)(diff_len + 1)) - 1;
            if (width_idx < 0) width_idx = 0;
            if (width_idx >= diff_len) width_idx = diff_len - 1;
            width[period_idx] = diff_work[width_idx];

            // Mean of legs above noise
            double noise_level = noise_cut * longest_leg;
            double leg_sum = 0.0;
            int n = 0;
            for (int i = 0; i < n_legs; i++) {
                if (leg_work[i] > noise_level) {
                    leg_sum += leg_work[i];
                    n++;
                }
            }

            double mean_leg = (n > 0) ? (leg_sum / (double)n) : 0.0;
            fti[period_idx] = mean_leg / (width[period_idx] + 1e-5);
        }

        // Sort FTI peaks
        sort_fti_peaks();
    }

    void sort_fti_peaks() {
        int n = 0;
        for (int i = 0; i < n_periods; i++) {
            int is_peak = (i == 0) || (i == n_periods - 1) ||
                         (fti[i] >= fti[i-1] && fti[i] >= fti[i+1]);
            if (is_peak) {
                sort_work[n] = -fti[i];
                sorted[n] = i;
                n++;
            }
        }

        // Insertion sort
        for (int i = 1; i < n; i++) {
            double key = sort_work[i];
            int key_idx = sorted[i];
            int j = i;
            while (j > 0 && sort_work[j-1] > key) {
                sort_work[j] = sort_work[j-1];
                sorted[j] = sorted[j-1];
                j--;
            }
            sort_work[j] = key;
            sorted[j] = key_idx;
        }
    }

    double get_best_fti() {
        return fti[sorted[0]];
    }

    int get_best_period() {
        return sorted[0] + min_period;
    }

    double get_filtered(int period) {
        return filtered[period - min_period];
    }

    double get_width(int period) {
        return width[period - min_period];
    }
};

int main() {
    // Same parameters as Rust test
    FTI fti(0, 5, 65, 32, 100, 0.9, 0.2);

    // Generate same test data: prices[i] = 100.0 + sin(i * 0.1) * 5.0
    double prices[100];
    for (int i = 0; i < 100; i++) {
        prices[i] = 100.0 + sin((double)i * 0.1) * 5.0;
    }

    fti.process(prices, 1);  // chronological = true

    printf("=== FTI Parity Test Output (C++) ===\n");
    printf("best_fti: %.10f\n", fti.get_best_fti());
    printf("best_period: %d\n", fti.get_best_period());
    printf("filtered[best]: %.10f\n", fti.get_filtered(fti.get_best_period()));
    printf("width[best]: %.10f\n", fti.get_width(fti.get_best_period()));

    return 0;
}
