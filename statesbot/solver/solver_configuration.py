import attr


@attr.s
class SolverConfiguration:
    initial_sampler_direction_choices = attr.ib(default=5)
    initial_sampler_equality_ratio_limit = attr.ib(default=2)
    equality_ratio_limit = attr.ib(default=1.33)
    improvement_fraction_edges = attr.ib(default=0.33)
    improvement_retries_per_edge = attr.ib(default=3)
