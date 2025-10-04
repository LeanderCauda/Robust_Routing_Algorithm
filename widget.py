import ipywidgets as widgets
import pyspark.sql.functions as F
from IPython.display import display
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from transportation_network.network import build_timetable, build_network
from graph.algorithms import dijkstra, yens_k_shortest_paths
from graph.graph import MultiDiGraph
from graph.path import Path
from predictive_modeling.predictive_model import PredictiveModel
from utilities import discretize_temperature_scalar

from config import (
    REGION_UUIDS,
    MIN_DEPARTURE_HOUR,
    MAX_ARRIVAL_HOUR,
    VALID_YEARS,
    WALKING_SPEED,
    DEFAULT_TRANSFER_TIME
)


class JourneyPlanningWidget:
    def __init__(
        self,
        spark: SparkSession,
        predictive_model: PredictiveModel,
        default_arrival_datetime: datetime,
        default_maximum_walking_distance: int,
        default_confidence_threshold: float,
        default_path_count: int,
        default_start_stop_id: str = None,
        default_end_stop_id: str = None
    ):
        self.spark = spark
        self.default_start_stop_id = default_start_stop_id
        self.default_end_stop_id = default_end_stop_id
        self.predictive_model = predictive_model

        layout_dict = dict(width='500px')
        style_dict = dict(description_width='175px')

        self.arrival_datetime_widget = widgets.DatetimePicker(
            description='Arrival time',
            style=style_dict,
            layout=layout_dict,
            value=default_arrival_datetime
        )

        self.maximum_walking_distance_widget = widgets.IntSlider(
            value=default_maximum_walking_distance,
            min=50,
            max=5000,
            step=10,
            description='Max walking (m)',
            style=style_dict,
            layout=layout_dict,
        )

        self.build_network_button = widgets.Button(
            description='✓ Build network', 
            button_style='success',
            layout=layout_dict,
            style=dict(font_weight='bold')
        )

        self.view_network = widgets.VBox([
            widgets.Label(
                "Praxivia - Robust Journey Planning", 
                style=dict(font_weight='bold', font_size='20px'),
                layout=dict(margin='20px 20px')
            ),
            self.arrival_datetime_widget,
            self.maximum_walking_distance_widget,
            self.build_network_button,
        ], layout=dict(width='510px'))

        self.start_stop_widget = widgets.Dropdown(options=[], description='Start stop', style=style_dict, layout=layout_dict)
        self.end_stop_widget = widgets.Dropdown(options=[], description='End stop', style=style_dict, layout=layout_dict)

        self.confidence_threshold_widget = widgets.FloatSlider(
            value=default_confidence_threshold,
            min=0.0,
            max=1.0,
            step=0.05,
            description='Confidence level',
            style=style_dict,
            layout=layout_dict
        )

        self.path_count_widget = widgets.IntSlider(
            value=default_path_count,
            min=1,
            max=10,
            step=1,
            description='Number of paths',
            style=style_dict,
            layout=layout_dict
        )

        self.rain_widget = widgets.Checkbox(
            value=False,
            description='Raining',
            style=style_dict,
            layout=layout_dict
        )

        self.temperature_widget = widgets.IntSlider(
            value=12,
            min=-10,
            max=30,
            step=1,
            description='Temperature (°C)',
            style=style_dict,
            layout=layout_dict
        )

        self.go_back_button = widgets.Button(
            description='← Go back', 
            button_style='primary',
            layout=dict(width='125px'),
            style=dict(font_weight='bold')
        )
            
        self.compute_path_button = widgets.Button(
            description='✓ Compute path', 
            button_style='success',
            layout=dict(width='375px'),
            style=dict(font_weight='bold')
        )

        self.output = widgets.Output(layout=dict(width='800px'))
    
        self.view_path = widgets.VBox([
            widgets.Label(
                "Praxivia - Robust Journey Planning", 
                style=dict(font_weight='bold', font_size='20px'),
                layout=dict(margin='20px 20px')
            ),
            self.start_stop_widget,
            self.end_stop_widget,
            self.confidence_threshold_widget,
            self.path_count_widget,
            self.rain_widget,
            self.temperature_widget,
            widgets.HBox([
                self.go_back_button,
                self.compute_path_button
            ]),
            self.output
        ], layout=dict(width='810px'))

        self.view_output = widgets.Output(layout=dict(width='1000px'))
                
        self.build_network_button.on_click(self.build_network)
        self.compute_path_button.on_click(self.compute_path)
        self.go_back_button.on_click(self.go_back)

        with self.view_output:
            display(self.view_network)

        display(self.view_output)

    def build_network(self, event):
        self.build_network_button.disabled = True

        with self.view_output:
            print("Building transportation network...")
        
        self.arrival_datetime = self.arrival_datetime_widget.value
        self.maximum_walking_distance = self.maximum_walking_distance_widget.value
        
        (
            self.stop_times_df, 
            self.stops_df, 
            self.transfer_df 
        ) = build_timetable(
            self.spark, 
            arrival_datetime=self.arrival_datetime,
            region_uuids=REGION_UUIDS,
            min_departure_hour=MIN_DEPARTURE_HOUR,
            max_arrival_hour=MAX_ARRIVAL_HOUR
        )
        
        self.network_df = build_network(
            self.stops_df,
            self.stop_times_df,
            self.transfer_df,
            maximum_walking_distance=self.maximum_walking_distance,
            walking_speed=WALKING_SPEED
        )

        self.network_df = self.network_df.cache()
        self.graph = MultiDiGraph.from_spark_dataframe(self.network_df, arrival_datetime=self.arrival_datetime)

        self.stop_options = sorted([
            (stop.stop_name, stop.stop_id) 
            for stop in self.graph.nodes.values()
        ], key=lambda x: x[0])

        self.start_stop_widget.options = self.stop_options
        self.end_stop_widget.options = self.stop_options

        available_stop_ids = [x[1] for x in self.stop_options]

        if self.default_start_stop_id and self.default_start_stop_id in available_stop_ids:
            self.start_stop_widget.value = self.default_start_stop_id

        if self.default_end_stop_id and self.default_end_stop_id in available_stop_ids:
            self.end_stop_widget.value = self.default_end_stop_id

        self.output.clear_output()
        self.view_output.clear_output()
        with self.view_output:
            display(self.view_path)

    def go_back(self, event):
        self.build_network_button.disabled = False
        with self.view_output:
            self.view_output.clear_output()
            display(self.view_network)

    def format_duration(self, seconds: int) -> str:
        total_minutes = seconds // 60
        hours = int(total_minutes // 60)
        minutes = int(total_minutes % 60)
        return f"{hours}h {minutes}m" if hours else f"{minutes}m"

    def compute_path(self, event):
        self.output.clear_output()
        self.compute_path_button.disabled = True

        self.start_stop_id = self.start_stop_widget.value
        self.end_stop_id = self.end_stop_widget.value
        self.confidence_threshold = self.confidence_threshold_widget.value
        self.path_count = self.path_count_widget.value
        self.rain = self.rain_widget.value
        self.temperature = discretize_temperature_scalar(self.temperature_widget.value)
        
        try:
            self.paths = yens_k_shortest_paths(
                self.graph,
                source_id=self.end_stop_id, # reverse path finding
                target_id=self.start_stop_id,
                K=self.path_count,
                arrival_datetime=self.arrival_datetime,
                maximum_walking_distance=self.maximum_walking_distance,
                walking_speed=WALKING_SPEED,
                default_transfer_time=DEFAULT_TRANSFER_TIME
            )
            
            self.paths = [Path.from_dijkstra_output(
                p,
                arrival_datetime=self.arrival_datetime,
                maximum_walking_distance=self.maximum_walking_distance,
                walking_speed=WALKING_SPEED
            ) for p in self.paths]
    
            self.compressed_paths = [p.compress() for p in self.paths]

            for path in self.compressed_paths:
                confidence = self.predictive_model.get_path_confidence(path, self.rain, self.temperature)
                path.confidence = confidence
    
            self.compressed_paths = list(filter(lambda x: x.confidence >= self.confidence_threshold, self.compressed_paths))
            if len(self.compressed_paths) == 0:
                raise ValueError("No path is satisfying the confidence constraint.")
            self.compressed_paths.sort(key=lambda p: p.cost())
            
            with self.output:
                children = []
                for path in self.compressed_paths:
                    duration = (path.arrival_time - path.departure_time).total_seconds()
                    html_string = f"""
                        <style>
                            h3 {{
                                margin: 0;
                            }}
                            p {{
                                margin: 0;
                            }}
                            td {{
                                padding: 0px 8px;
                                margin: 0px;
                            }}
                            .metadata td:first-child {{
                                font-weight: bold;
                            }}
                            .trip td:last-child {{
                                color: #9ca3af;
                            }}
                        </style>
                        <div style="background-color: #fafafa; padding: 1rem;">
                            <h3>{path.trips[0].source.stop_name} ➡️ {path.trips[-1].target.stop_name}</h3>
                            <div class="metadata" style="display: flex;">
                                <table style="border-collapse: collapse;">
                                    <tr><td>🕓 Departure</td><td>{path.departure_time.time()}</td></tr>
                                    <tr><td>🕓 Arrival</td><td>{path.arrival_time.time()}</td></tr>
                                    <tr><td>⏳ Duration</td><td>{self.format_duration(duration)}</td></tr>
                                </table>
                                <table style="border-collapse: collapse;">
                                    <tr><td>⤵️ Transfers</td><td>{path.transfers_count}</td></tr>
                                    <tr><td>🥾 Walking</td><td>{round(path.walked_distance)}m</td></tr>
                                    <tr><td>📈 Confidence</td><td>{path.confidence:.3f}</td></tr>
                                </table>
                            </div>
                            <div style="position: relative; margin-top: 1rem;">
                                <div style="position: absolute; left: 14px; top: 0; bottom: 0; width: 2px; background-color: #d4d4d8; border-radius: 1px;"></div>
                    """
                    for trip in path.trips:
                        trip_description = trip.trip_description
                        if trip.is_walking():
                            walked_distance = WALKING_SPEED * (trip.target_arrival_time - trip.source_departure_time).total_seconds()
                            trip_description = f"🥾 Walking {int(walked_distance)}m"
                        
                        html_string += f"""
                            <div class="trip" style="position: relative; padding-left: 2rem; margin-bottom: 1rem;">
                                <div style="position: absolute; left: 10px; top: 10px; width: 10px; height: 10px; background-color: #007bff; border-radius: 50%;"></div>
                                <div style="padding: 1rem; background-color: #f4f4f5; border-radius: 2px;">
                                    <p style="font-weight: bold;">{trip_description} ({trip.trip_id})</p>
                                    <table style="border-collapse: collapse;">
                                        <tr><td>🕓 {trip.source_departure_time.time()}</td><td>{trip.source.stop_name}</td><td>{trip.source.stop_id}</td></tr>
                                        <tr><td>🕓 {trip.target_arrival_time.time()}</td><td>{trip.target.stop_name}</td><td>{trip.target.stop_id}</td></tr>
                                    </table>
                                </div>
                            </div>
                        """

                    html_string += """
                        </div>
                    </div>
                    """
                    children.append(widgets.HTML(value=html_string, layout=widgets.Layout(width='750px')))
                tab = widgets.Tab()
                tab.children = children
                tab.titles = [f"Path {i + 1}" for i in range(len(self.paths))]
                display(tab)
        except ValueError as error:
            with self.output:
                display(widgets.HTML(f"""
                    <div style="background-color: #fee2e2; padding: 10px; border-radius: 2px; color: #dc2626; font-weight: bold;">
                        Error - {error}
                    </div>
                """))
        
        self.compute_path_button.disabled = False
