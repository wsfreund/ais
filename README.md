# Anneled Importance Sampling (AIS)

This repository contains a tensorflow 2 re-implementation of the AIS algorithm
as employed in paper [On the Quantitative Analysis of Decoder-Based Generative
Models](https://arxiv.org/abs/1611.04273). It is based on the original paper
implementation [available here](https://github.com/tonywu95/eval_gen) (theano
based) and another tensorflow re-implementation [available
here](https://github.com/jiamings/ais.git).

Current version lacks the implementation of the backwards AIS algorithm,
neither the IWAE bounds or the VAE-based prior. The implementation of such
missing components should be trivial, however, by extending current code with
the original theano implementation.

*The code was validated (same results = up to 5 digits precision) with respect to
the original implementation.* 


# License and warranty

© 2020 WTFPL - Werner Spolidoro Freund

This work is free. It comes without any warranty, to the
extent permitted by applicable law. You can redistribute it 
and/or modify it under the terms of the WTFPL, Version 2, 
as published by Sam Hocevar. See http://www.wtfpl.net/ 
for more details.
