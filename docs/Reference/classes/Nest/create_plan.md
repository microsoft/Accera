[//]: # (Project: Accera)
[//]: # (Version: v1.2.1)

# Accera v1.2.1 Reference

## `accera.Nest.create_plan([target])`
Create a plan using the default schedule for the nest.

## Arguments

argument | description | type/default
--- | --- | ---
`target` | The target platform. Defaults to `acc.Target.HOST` | `Target`

## Returns
`Plan`

## Examples

Create a plan for the host computer, using the default schedule for a nest:

```python
plan = nest.create_plan()
```

Create a plan for an Intel Core 7th Generation, using the default schedule for a nest:

```python
corei9 = acc.Target("Intel 7900X", num_threads=44)
plan = nest.create_plan(corei9)
```


<div style="page-break-after: always;"></div>
