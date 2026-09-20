import unittest

from scripts.render_diagrams import find_figures


class FindFiguresTests(unittest.TestCase):
    """Pulling each diagram, and its output name, out of docs/DIAGRAMS.md.

    The headings are deliberately plain English so non-technical readers can use
    the file, but plain English makes long, unstable filenames -- and the paper
    cites those filenames. So a figure may pin its own short name with an HTML
    comment, and only falls back to the slugified heading when it does not.
    """

    def figure_doc(self, body: str) -> str:
        return "# Diagrams\n\nIntro text.\n\n" + body

    def test_it_finds_a_figure_and_its_mermaid_source(self):
        doc = self.figure_doc(
            "## Figure 1 — System architecture\n\nCaption.\n\n"
            "```mermaid\nflowchart LR\n  A --> B\n```\n"
        )
        figures = find_figures(doc)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0][1], "flowchart LR\n  A --> B")

    def test_the_name_comes_from_the_heading_by_default(self):
        doc = self.figure_doc(
            "## Figure 2 — Per question data flow\n\n```mermaid\nflowchart LR\n```\n"
        )
        self.assertEqual(find_figures(doc)[0][0], "fig2_per_question_data_flow")

    def test_an_explicit_name_overrides_the_heading(self):
        # So rewording a heading for clarity does not rename the file the paper
        # already cites.
        doc = self.figure_doc(
            "## Figure 2 — What happens to one question, in plain words\n"
            "<!-- figure: fig2_per_question_data_flow -->\n\n"
            "```mermaid\nflowchart LR\n```\n"
        )
        self.assertEqual(find_figures(doc)[0][0], "fig2_per_question_data_flow")

    def test_figures_come_back_in_document_order(self):
        doc = self.figure_doc(
            "## Figure 1 — First\n\n```mermaid\nflowchart LR\n  A\n```\n\n"
            "## Figure 2 — Second\n\n```mermaid\nflowchart LR\n  B\n```\n"
        )
        self.assertEqual([f[0] for f in find_figures(doc)], ["fig1_first", "fig2_second"])

    def test_a_heading_with_no_diagram_is_skipped(self):
        # A prose section between figures must not swallow the next figure's
        # source, which is what a greedy match would do.
        doc = self.figure_doc(
            "## Figure 1 — Real\n\n```mermaid\nflowchart LR\n  A\n```\n\n"
            "## What should NOT be a diagram\n\nJust prose, no mermaid here.\n"
        )
        figures = find_figures(doc)
        self.assertEqual(len(figures), 1)
        self.assertEqual(figures[0][0], "fig1_real")

    def test_punctuation_in_a_heading_does_not_reach_the_filename(self):
        doc = self.figure_doc(
            "## Figure 6 — How good does the marker need to be?\n\n"
            "```mermaid\nflowchart LR\n```\n"
        )
        name = find_figures(doc)[0][0]
        self.assertEqual(name, "fig6_how_good_does_the_marker_need_to_be")
        self.assertNotIn("?", name)


if __name__ == "__main__":
    unittest.main()
