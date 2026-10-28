## 5 · Benchmark straight from the registry {#pipe-benchmark}

<div class="viz-card">
<div class="viz-switch" data-for="benchmark-chart"><button class="active" data-view="trajectory">Trajectory</button><button data-view="ood">OOD in density</button><button data-view="direct_B">Direct B</button><button data-view="inverse_A">Inverse A</button><button data-view="inverse_B">Inverse B</button><button data-view="moments">Moments</button></div>
<div id="benchmark-chart" class="viz viz-lg" data-viz="benchmark" data-view="trajectory"></div>
</div>

::: {.caption}
`pp.model_benchmark(project, CASE)`: best version of each registered model against the bar; ★ = handed in (hover for per-target and official R²). Each author's own holdout split: a guide, not a ranking.
:::
