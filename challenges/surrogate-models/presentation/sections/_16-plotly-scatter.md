## An interactive Plotly scatter plot

::: {.columns}
::: {.column width="68%"}

<div id="dummy-scatter-plot" class="plotly-demo" aria-label="Interactive Plotly scatter plot of dummy PALLAS data"></div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script>
  const dummyScatter = [
    {
      x: [5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000, 6100, 6200, 6300, 6400, 6500, 6600, 6700, 6800, 6900],
      y: [61.2, 64.7, 68.1, 71.5, 75.0, 78.4, 81.9, 85.3, 88.8, 92.1, 95.7, 99.0, 102.6, 106.0, 109.4, 112.9, 116.2, 119.8, 123.1, 126.7],
      text: ['1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014','1015','1016','1017','1018','1019','1020'],
      name: 'Nominal', mode: 'markers', type: 'scatter', marker: {size: 10, color: '#22d3ee'}
    },
    {
      x: [5150, 5750, 6350, 6850], y: [84.0, 143.0, 72.0, 171.0],
      text: ['2001','2002','2003','2004'],
      name: 'Candidate outlier', mode: 'markers', type: 'scatter', marker: {size: 13, color: '#e879f9', symbol: 'diamond'}
    }
  ];
  Plotly.newPlot('dummy-scatter-plot', dummyScatter, {
    title: 'Dummy trajectory population: E_MeV vs P_max',
    xaxis: {title: 'P_max'}, yaxis: {title: 'E_MeV'},
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(13,27,42,0.75)',
    font: {color: '#e6edf3'}, legend: {orientation: 'h'},
    margin: {l: 60, r: 20, t: 55, b: 55}, height: 420
  }, {responsive: true, displaylogo: false});
</script>

:::
::: {.column width="28%"}

### What this demonstrates

- Hover over a point to inspect its configuration ID.
- Toggle populations from the legend.
- Compare a nominal trend with deliberately injected candidate outliers.
- Replace the inline arrays with data loaded from the real explorer output.

Dummy source data: [data/dummy_scatter.csv](../data/dummy_scatter.csv)

::: {.notes}
This example uses Plotly’s browser JavaScript bundle. The slide therefore remains interactive in HTML, but needs network access to load the CDN script unless Plotly is bundled locally.
:::

:::
:::
