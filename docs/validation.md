# Validation

The results now live on the website, at
[sohan-bhat.github.io/wobbl/validation](https://sohan-bhat.github.io/wobbl/validation/), with
the setup of each case on
[its own page](https://sohan-bhat.github.io/wobbl/validation/cases/).

Twenty five cases with known answers run through the same pipeline you use, covering textbook
formulas, the NAFEMS LE1 and LE10 benchmarks, five real parts against converged reference
models, three heat cases, three circuits and six motion cases against exact closed forms. Every case passes its tolerance, and every case whose
reference is exact for the modeled setup lands within 2%. `wobbl validate` runs them all in about
30 minutes. The raw numbers are in [validation-results.json](validation-results.json).
