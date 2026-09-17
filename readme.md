# Betting for Sim-to-Real Performance Certificates

This repository provides the reproducible implementation of the paper
*Betting for Sim-to-Real Performance Certificates*. Every figure in the paper can
be regenerated from the code and data here.

## Reproducibility

Each folder below has its own README with run instructions, data description,
and configuration.

| folder | reproduces | contents |
| --- | --- | --- |
| [`synthetic`](synthetic/README.md) | **Fig. 1**, **Fig. 2**, **Fig. 3a**, **Fig. 4a-c** | the synthetic evaluation, the Section I toy example, and the wealth-regret demo |
| [`gr00t_command_tracking`](gr00t_command_tracking/README.md) | **Fig. 5** | command-tracking accuracy of a Unitree G1 running GR00T, with a matched MuJoCo simulator bank |
| [`nist_continuous_manipulation`](nist_continuous_manipulation/README.md) | **Fig. 6a** | replay of the NIST continuous mobile-manipulator peg-in-hole dataset |
| [`astm_wk86916_go2`](astm_wk86916_go2/README.md) | **Fig. 6b** | replay of an ASTM WK86916 push-over test on a Unitree Go2 |

The certificate implementations live in [`synthetic/method`](synthetic/README.md#method);
the three robot studies import them from there.

## Supplement

Proofs of the propositions, lemmas, and theorems, together with the remaining
appendices, are in [`supplement.pdf`](supplement.pdf). 

## Disclaimer

The development of this codebase involves, in part, a mixture of generative AI models for debugging, figure plotting, and comment editting. The authors take full responsibility for the code.
