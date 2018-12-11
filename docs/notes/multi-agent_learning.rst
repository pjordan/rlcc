:github_url: https://github.com/pjordan/rlcc

Multi-Agent Learning
====================

The `rlcc` packages supports multi-agent learning through stacked classes: 

- `rlcc.act.StackedActor`
- `rlcc.observe.StackedObserver`
- `rlcc.learn.StackedLearner`

Each of these classes makes an underlying assumption that the input will be a list,
one per agent.  Learning, observing, and acting can then be composed of independent
instances per agent.

