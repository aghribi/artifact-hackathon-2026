## 6 · Hand-in: export, score, publish {#pipe-handin}

::: {.columns}
::: {.column width="36%"}

<ol class="flow">
<li><b><code>make_submission.py</code></b><span>loads the chosen version per case back from the registry and predicts the test file</span></li>
<li><b>CSV + <code>manifest.json</code></b><span>checked against the test ids, fingerprinted with md5</span></li>
<li><b>DALIA</b><span>the CSVs are copied to the organisers' folder</span></li>
<li><b>Organisers</b><span>run <code>pallas_score.py</code> on the hidden test</span></li>
<li><b><code>log_official_score.py</code></b><span>transcribes their score to W&amp;B, <code>--dry-run</code> first</span></li>
<li><b>Status page</b><span>matches the run to its registry version and updates itself</span></li>
</ol>

:::
::: {.column width="61%"}

<div class="viz-card">
<div class="viz-switch" data-for="handin-chart"><button class="active" data-view="time">Official scores</button><button data-view="calibration">Holdout vs official</button><button data-view="manifest">Hand-in manifest</button></div>
<div id="handin-chart" class="viz viz-md" data-viz="handin" data-view="time"></div>
</div>

::: {.caption}
<span data-kpi="official"></span> official scores on the hidden test, published to W&B. Only the organisers compute them; our holdout only decides what to send.
:::

:::
:::
