# Research Question and Hypotheses

## Primary research question

Can a Pix2Pix-style LV endocardium segmentation model become more geometrically reliable and functionally consistent by adding explicit boundary supervision and paired ED/ES functional-consistency supervision?

## Hypotheses

**H1:** Boundary-aware supervision reduces HD95/ASD without materially degrading Dice.

**H2:** Paired ED/ES functional-consistency supervision reduces errors in phase-to-phase LV size change and improves downstream clinical-index stability.

**H3:** The combined method is more robust than the baseline under echo-relevant corruptions such as speckle noise, blur, contrast loss, and shadow-like attenuation.

**H4:** Improvements remain statistically consistent across multiple random seeds and are not caused by test-set model selection.

## Primary endpoint

Choose one before final experiments and do not change it after seeing the test results. Recommended:

- primary segmentation endpoint: LVendo Dice on held-out CAMUS test patients;
- key secondary endpoint: HD95;
- clinical endpoint after validation: EF absolute error / correlation.

## Null interpretation

If boundary or functional loss does not improve the relevant endpoint across seeds, report that honestly. A carefully executed negative result can still improve the project and prevents building a paper around optimizer folklore.
