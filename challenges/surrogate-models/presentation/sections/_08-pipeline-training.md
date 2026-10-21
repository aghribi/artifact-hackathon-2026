## 3 · Train any model, anywhere {#pipe-training}

::: {.columns}
::: {.column width="44%"}

::: {.panel-tabset}

### Model file

```python
# models/my_model.py: the whole contract
PARAMS = {"hidden": (128, 128, 128)}

def build_model(**params):
    return Regressor(**params)  # fit / predict

model = pp.SurrogateModel(
    module.build_model(**params),
    case.targets, TARGET_SCALING)
model.fit(X_train, train_rows)
```

### Headless

```bash
python train.py --case trajectory \
  --split ood_density \
  --model-file models/MO_DeepOnet.py \
  --params '{"ensemble": 4}' --register
```

Same steps as the notebook: view → split → fit → holdout → registry → CSV.

### DALIA

```bash
dalia/dalia.sh submit --split ood_density \
  --model-file models/MO_DeepOnet.py
dalia/dalia.sh sweep dalia/sweeps/deeponet_ood.txt
dalia/dalia.sh hpo --study deeponet_ood_v1
dalia/dalia.sh fetch
```

One job per GB200 GPU, in an Apptainer image; `--dry-run` prints it first.

:::

:::
::: {.column width="53%"}

<div class="viz-card">
<div class="viz-switch" data-for="hpo-chart"><button class="active" data-view="trials">Optuna trials</button><button data-view="importance">What mattered</button></div>
<div id="hpo-chart" class="viz viz-md" data-viz="hpo" data-view="trials"></div>
</div>

::: {.caption}
`hpo.py` on DALIA: <span data-kpi="hpo_trials"></span> TPE trials of MO_DeepOnet on 2 GPUs, scored on a validation cut of the training configurations (best <span data-kpi="hpo_best"></span>). The above-cut set is never seen; only the winner is retrained, scored once and registered.
:::

:::
:::
