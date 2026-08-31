import unittest

from experiments.plot_pareto import pareto_frontier


class ParetoFrontierTests(unittest.TestCase):
    """Which operating points a rational deployer would ever choose.

    A point is on the frontier if nothing else is at least as accurate for no
    more cloud cost. Cheaper and more accurate always wins.
    """

    def test_a_dominated_point_is_dropped(self):
        # Same cost, less accurate.
        points = [("cheap", 0.0, 0.59), ("worse", 0.0, 0.55)]
        self.assertEqual([p[0] for p in pareto_frontier(points)], ["cheap"])

    def test_more_expensive_and_less_accurate_is_dropped(self):
        points = [("free", 0.0, 0.59), ("pricey", 300.0, 0.57)]
        self.assertEqual([p[0] for p in pareto_frontier(points)], ["free"])

    def test_more_expensive_but_more_accurate_is_kept(self):
        points = [("free", 0.0, 0.59), ("pricey", 300.0, 0.63)]
        self.assertEqual(
            [p[0] for p in pareto_frontier(points)], ["free", "pricey"]
        )

    def test_frontier_is_sorted_by_cost(self):
        points = [("mid", 300.0, 0.63), ("free", 0.0, 0.59), ("top", 369.0, 0.64)]
        self.assertEqual(
            [p[0] for p in pareto_frontier(points)], ["free", "mid", "top"]
        )

    def test_ties_on_both_axes_keep_only_one(self):
        points = [("a", 100.0, 0.6), ("b", 100.0, 0.6)]
        self.assertEqual(len(pareto_frontier(points)), 1)

    def test_empty_input(self):
        self.assertEqual(pareto_frontier([]), [])
