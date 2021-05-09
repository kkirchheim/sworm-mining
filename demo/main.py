"""
Bokeh Demo Application
"""
from bokeh.plotting import figure
from bokeh.palettes import Turbo256, linear_palette
from bokeh.transform import factor_cmap
from bokeh.models import HoverTool
from datetime import date
from bokeh.models import ColumnDataSource
from bokeh.models import TextInput, DateRangeSlider, MultiChoice, Div, CustomJS, TapTool, RangeSlider
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models.widgets import Tabs, Panel
import time

import logging

import callbacks as cb
from .data import df, topic_list, journal_list, country_list

log = logging.getLogger(__name__)


def configure_logging():
    fmt = "[%(levelname)s] %(asctime)s - %(name)s: %(message)s"
    # formatter = logging.Formatter(fmt=fmt)
    logging.basicConfig(level=logging.INFO, format=fmt)

    logging.getLogger().setLevel(logging.DEBUG)

    # ch = logging.StreamHandler()
    # ch.setLevel(logging.INFO)
    # ch.setFormatter(formatter)
    # root.addHandler(ch)
    # root.setLevel(logging.DEBUG)
    # root.debug("Logging configured")


def create_color_mapper(topics):
    color_mapper = factor_cmap(
        field_name="cluster",
        palette=linear_palette(Turbo256, len(topics)),
        factors=topics,
    )
    return color_mapper


def create_hover_tool():
    hover = HoverTool(tooltips=[
        ("Title", '<div style="width:400px;">@title{safe}</div>'),
        ("Date", "@date{safe}"),
        ("Author(s)", "@author{safe}"),
        ("LDA Topics(s)", "@topics{safe}"),
        ("Journal", "@journal{safe}"),
        ("Citations", "@citations"),
        ("Country", "@country"),
        ("Abstract", '<div style="width:400px;">@abstract{safe}</div>'),  # wrap abstracts
        ("DOI", "@doi"),
    ], point_policy="follow_mouse")
    return hover


def main():
    configure_logging()

    log.info(f"Loading data...")
    t1 = time.perf_counter()
    d = df.copy()
    d["abstract"] = d["abstract"].apply(lambda x: "")

    source = ColumnDataSource(d)
    log.info(f"Loading took: {time.perf_counter() - t1} s")
    t2 = time.perf_counter()

    color_mapper = create_color_mapper(topic_list)
    hover = create_hover_tool()

    plot = figure(plot_width=1800, plot_height=1000,
                  tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap'],
                  title=None, toolbar_location="above")

    plot.scatter(source=source, x="x1", y="x2", size=5, fill_color=color_mapper, line_alpha=0.3,
                 line_color="black", legend_field="cluster")

    div_info = Div(text="Click on an article for details.", height=150)
    callback_selected = CustomJS(args=dict(source=source, current_selection=div_info), code=cb.selected_code())
    taptool = plot.select(type=TapTool)
    taptool.callback = callback_selected

    # other interactive components

    input_callback = cb.input_callback(source)

    text_search = TextInput(title="Search:")
    text_search.js_on_change("value", input_callback)

    text_cout_label = Div(text="Displayed Documents:", height=25)
    text_count = Div(text=f"{len(df)}", height=25)

    date_range_slider = DateRangeSlider(
        title="Publication Date",
        value=(date(1960, 1, 1), date.today()),
        start=date(1960, 1, 1),
        end=date.today(),
        step=1)

    date_range_slider.js_on_change("value", input_callback)

    citation_count_slider = RangeSlider(
        title="Citation Count",
        value=(0, df["citations"].max()),
        start=0,
        end=df["citations"].max(),
        step=1
    )
    citation_count_slider.js_on_change("value", input_callback)

    journal_choice = MultiChoice(value=journal_list, options=journal_list)
    journal_choice.js_on_change("value", input_callback)

    topic_choice = MultiChoice(value=topic_list, options=topic_list)
    topic_choice.js_on_change("value", input_callback)

    country_choice = MultiChoice(value=country_list, options=country_list)
    country_choice.js_on_change("value", input_callback)

    # pass call back arguments
    input_callback.args["text_search"] = text_search
    input_callback.args["date_range_slider"] = date_range_slider
    input_callback.args["journal_choice"] = journal_choice
    input_callback.args["text_count"] = text_count
    input_callback.args["topic_choice"] = topic_choice
    input_callback.args["citation_count_slider"] = citation_count_slider
    input_callback.args["country_choice"] = country_choice

    # non interactive components
    title = Div(text="<h1>SWORM - Social Work Research Map</h1>")
    filter_title = Div(text="<h2>Filter</h2>")
    selection_title = Div(text="<h2>Selection</h2>")

    # styling
    plot.sizing_mode = "scale_both"
    plot.margin = 5
    plot.legend.visible = True
    plot.legend.label_text_font_size = "10px"
    # plot.legend.click_policy="hide"
    plot.add_layout(plot.legend[0], 'right')
    plot.toolbar.logo = None

    # layout
    info_column = column([
        selection_title,
        div_info
    ])

    journal_pane = Panel(child=journal_choice, title="Journals")
    topic_pane = Panel(child=topic_choice, title="Topics")
    country_pane = Panel(child=country_choice, title="Countries")
    tab = Tabs(tabs=[journal_pane, topic_pane, country_pane])

    filter_column = column([
        filter_title,
        row([text_cout_label, text_count]),
        text_search,
        date_range_slider,
        citation_count_slider,
        tab
    ])

    content = row([filter_column, plot, info_column])
    layout = column([title, content], name="main")
    layout.sizing_mode = "scale_both"

    log.info(f"Creating document took {time.perf_counter() - t2} s")
    t3 = time.perf_counter()

    curdoc().add_root(layout)
    curdoc().title = "SWORM"
    log.info(f"Finished:  {time.perf_counter() - t3} s")


main()
