[//]: # (Project: Accera)
[//]: # (Version: 1.2.0)

# Accera 1.2.0 Reference

## `accera.Nest.create_action_plan([target])`
Create an action plan using the default schedule for the nest.

## Arguments

argument | description | type/default
--- | --- | ---
`target` | The target platform. Defaults to `acc.Target.HOST` | `Target`

## Returns
`ActionPlan`

## Examples

Create an action plan for the host computer, using the default schedule for a nest:

```python
plan = nest.create_action_plan()
```

Create an action plan for an Intel Core 7th Generation, using the default schedule for a nest:

```python
corei9 = acc.Target(model=acc.Target.Model.INTEL_CORE_GENERATION_7, num_threads=44)
plan = nest.create_action_plan(corei9)
```


<div style="page-break-after: always;"></div>
