## One pipeline, six steps {#pipe-overview}

Every model in the team runs through the same path, whoever wrote it and wherever it trains: the notebook `surrogate_pipeline.ipynb`, the headless `train.py`, or a job on DALIA's GPUs.

<div class="pipe-steps">
<a class="pipe-step" href="#/pipe-feature-store"><span class="pipe-num">1</span><b>Feature store</b><span>Read the PALLAS campaigns from Hopsworks</span><em><span data-kpi="fg_rows"></span> rows</em></a>
<a class="pipe-step" href="#/pipe-transformations"><span class="pipe-num">2</span><b>Traced transformations</b><span>Normalisations and statistics stored in the feature view</span><em><span data-kpi="n_inputs"></span> inputs · TD v<span data-kpi="td_version"></span></em></a>
<a class="pipe-step" href="#/pipe-training"><span class="pipe-num">3</span><b>Training</b><span>Any <code>models/*.py</code>, on a laptop or on DALIA</span><em><span data-kpi="hpo_trials"></span> HPO trials</em></a>
<a class="pipe-step" href="#/pipe-registry"><span class="pipe-num">4</span><b>Model registry</b><span>Model, code, pins and metadata, linked to its view</span><em><span data-kpi="models"></span> versions</em></a>
<a class="pipe-step" href="#/pipe-benchmark"><span class="pipe-num">5</span><b>Benchmark &amp; status</b><span>Every registered model against the bar, live</span><em><span data-kpi="tasks"></span> cases</em></a>
<a class="pipe-step highlight" href="#/pipe-handin"><span class="pipe-num">6</span><b>Hand-in</b><span>Best model per case → CSV → organisers → W&amp;B</span><em><span data-kpi="official"></span> official scores</em></a>
</div>

::: {.pipe-legend}
<span class="tool tool-hops">Hopsworks</span> steps 1, 2, 4 and 5 &nbsp;·&nbsp; <span class="tool tool-dalia">DALIA GB200</span> step 3 &nbsp;·&nbsp; <span class="tool tool-wandb">W&amp;B</span> step 6
:::

::: {.caption}
Click a step to jump to it. The figures in this section come from Hopsworks and W&B, snapshot taken <span data-kpi="built"></span>.
:::
