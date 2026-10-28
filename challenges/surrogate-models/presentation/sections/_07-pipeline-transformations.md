## 2 · Transformations that travel with the data {#pipe-transformations}

::: {.columns}
::: {.column width="60%"}

<div class="viz-card">
<div class="viz-switch" data-for="split-hist"><button class="active" data-view="raw">Physical units</button><button data-view="scaled">As the model sees it</button></div>
<div id="split-hist" class="viz viz-md" data-viz="split-hist" data-view="raw"></div>
</div>

::: {.caption}
Trajectory configurations by gas pressure, split as the Hard objective asks: train below the cut, score in-range and above-cut separately. Toggle the legend to isolate a group.
:::

:::
::: {.column width="37%"}

### Stored in the feature view, not in a notebook

<div class="chips" data-viz="normalisation-chips"></div>

::: {.compact}
- Statistics come from **training dataset v<span data-kpi="td_version"></span>**, over the inputs only, so the holdout's targets cannot leak into them.
- A view version is **immutable**: other normalisations mean a new `FV_VERSION`.
- Splits take **whole configurations**: <span data-kpi="split_train"></span> train, <span data-kpi="split_in_range"></span> in-range and <span data-kpi="split_above_cut"></span> above-cut holdout. No simulation sits on both sides.
- Targets are scaled inside the model, so predictions return in physical units.
:::

:::
:::
