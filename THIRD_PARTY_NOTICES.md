# Third-party methods and software

## WMFB and FHRMA

The standalone Python WMFB port derives from the FHRMA MATLAB toolbox, which
declares GPL-3.0 at https://github.com/data-coeur/FHRMA . Its source includes
`aamwmfb.m`, `butterfilt.m` and related numerical helpers. The port in
`third_party/wmfb/wmfb.py` retains those GPL terms; see its `COPYING` and notice.
It is not relicensed under the first-party research-only license. No FHRMA
dataset or original MATLAB software bundle is redistributed here.

## Neural comparator adaptations

The seven network implementations come from local research adaptations of
published methods, not a wholesale download of author repositories. Their
scientific attribution and implementation scope are in `README.md`.
WaveFHR's hard-vote interpretation was checked against the public repository
https://github.com/Apple-cake12/WaveFHR-VNet . Public visibility is not itself
a grant to copy or relicense that repository. The local adaptation does not
establish a clean-room or complete rights-clearance claim.

## Dependencies

PyTorch, NumPy, pandas, SciPy and PyWavelets are separate dependencies under
their respective licenses. They are not vendored or relicensed by this project.

The first-party license applies only to material the authors control. Resolve
unclear copied-code origins before distributing an archive publicly.

License references checked on 18 November 2026:
- https://github.com/data-coeur/FHRMA/blob/master/LICENSE
- https://opensource.org/osd (commercial-use restrictions are not open source)
