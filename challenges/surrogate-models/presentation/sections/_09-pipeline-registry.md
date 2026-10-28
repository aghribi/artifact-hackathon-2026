## 4 · A self-contained model registry {#pipe-registry}

::: {.columns}
::: {.column width="38%"}

<ul class="file-tree">
<li class="dir">sj_trajectory_bayesian_nn <em>v5</em></li>
<li title="The fitted SurrogateModel, target scaling included"><code>model.joblib</code><span>fitted model</span></li>
<li title="The models/*.py file it was built from"><code>bayesian_nn.py</code><span>model code</span></li>
<li title="Code needed to unpickle and predict"><code>pallas_pipeline.py</code><span>pipeline code</span></li>
<li title="The organisers' scorer, as used for the holdout"><code>pallas_score.py</code><span>scorer</span></li>
<li title="Versions installed when it was trained"><code>requirements.txt</code><span>pinned versions</span></li>
<li title="Case, inputs, normalisations, FV and TD versions, split, holdout configurations and scores"><code>model_meta.json</code><span>provenance</span></li>
</ul>

![](index_files/images/Screenshot 2026-10-15 at 10.02.13.png){.lightbox .lineage fig-alt="Hopsworks provenance graph linking a feature group, feature view, training dataset, and registered model"}

::: {.caption}
Each version is linked to its feature view and training dataset, so anyone can load it and reproduce the exact inputs.
:::

:::
::: {.column width="59%"}

<div class="kpi-row">
<div><b data-kpi="models"></b><span>versions</span></div>
<div><b data-kpi="model_names"></b><span>models</span></div>
<div><b data-kpi="authors"></b><span>authors</span></div>
<div><b data-kpi="tasks"></b><span>cases</span></div>
</div>

<div id="registry-chart" class="viz viz-md" data-viz="registry-growth"></div>

::: {.caption}
Uploads to the team registry, hour by hour: per day, stacked by author, and per author. It plays on arrival; drag the slider or use Play/Pause, and hover a bar for its models.
:::

:::
:::
