"""Reproduce the synthetic material presets using passive Debye responses."""
from pathlib import Path
import csv


# Each tuple is (high-frequency value, relaxation strength, relaxation GHz).
MODELS = {
    'demo_low_loss': ((2.1, .1, 20.), (1., 0., 1.)),
    'demo_high_permittivity': ((8., 2., 30.), (1., 0., 1.)),
    'demo_lossy_dielectric': ((3., 7., 8.), (1., 0., 1.)),
    'demo_magnetic_loss': ((2.5, 1.5, 12.), (1., 2., 6.)),
}


def response(frequency_ghz, parameters):
    high, strength, relaxation = parameters
    return high + strength / (1 + 1j * frequency_ghz / relaxation)


def main():
    destination = Path(__file__).resolve().parents[1] / 'materials'
    # 80 intervals per decade, from 1 MHz through 1 THz, including endpoints.
    frequencies = [10 ** (6 + index / 80) for index in range(481)]
    for name, (electric, magnetic) in MODELS.items():
        with (destination / (name + '.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('frequency_hz', 'eps_real', 'eps_imag', 'mu_real', 'mu_imag'))
            for frequency in frequencies:
                eps = response(frequency / 1e9, electric)
                mu = response(frequency / 1e9, magnetic)
                writer.writerow(format(v, '.12g') for v in
                                (frequency, eps.real, eps.imag, mu.real, mu.imag))


if __name__ == '__main__':
    main()
