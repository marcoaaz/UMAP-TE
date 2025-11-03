import os
import collections
import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output
import plotly.express as px 
import seaborn as sns

def find_nearest_numpy(array, value):
    """
    Finds the closest value to a target in a NumPy array.
    """
    array = np.asarray(array) # Ensure it's a NumPy array
    idx = (np.abs(array - value)).argmin()
    min_val = array[idx]
    boolean_mask = (array == min_val)

    return boolean_mask

def calculate_xy_limits(table_book, table_coordinates):
    # Determine fixed axis limits
    col_names = table_coordinates.columns
    n_folders = table_book.shape[0]
    xy_list = []
    for i in range(0, n_folders):

        path_temp = table_book.loc[i, "folderpath"] 
        path_temp2 = os.path.basename(path_temp)
        
        coordinate_cols = col_names[col_names.str.contains(path_temp2)]
        coordinate_array = table_coordinates[coordinate_cols]
        
        xy_list.append(coordinate_array)

    xy_array = np.vstack(xy_list) #has NaN

    #old:
    # min0 = np.nanmin(xy_array, axis=0)
    # max0 = np.nanmax(xy_array, axis=0)

    # #new:
    xy_array2 = xy_array[~np.isnan(xy_array).any(axis=1)] #drop nan    
    
    #Interquartile Range (IQR)
    lim_list = []
    for i in range(0, xy_array2.shape[1]):        

        Q1 = np.quantile(xy_array2[:, i], 0.25)
        Q3 = np.quantile(xy_array2[:, i], 0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        lim_list.extend([lower_bound, upper_bound])
    
    min0 = np.array([lim_list[0], lim_list[2]])
    max0 = np.array([lim_list[1], lim_list[3]])
    
    limits_array= [min0, max0]

    return limits_array

def plot_umap_grid_interactive(table_book, table_interactive2, 
                               var1, var2, var3, param1_sel, param2_sel, param3_sel,
                               neighbors_input, min_dist_input, metric_input,                               
                               variable_legend1, list_unique, colour_map, classif_colours, 
                               limits_array, markerSize):
    
    #default
    # markerSize = 5    
    x_min, y_min, x_max, y_max = limits_array[0][0], limits_array[0][1], limits_array[1][0], limits_array[1][1]

    #region Create Dash app
    app = dash.Dash(__name__)
    app.title = "Interactive UMAP in 2D"

    # Layout
    app.layout = html.Div(
    
        style={"display": "flex", "flexDirection": "row", "height": "100vh"},
        
        children=[
            # html.H2("Interactive UMAP in 2D", style={"textAlign": "center"}),

            # --- Left: Plot ---
            html.Div(            
                [
                    dcc.Graph(id='scatter-plot', style={'height': '100%'}), #70vh
                ],
                style={"flex": "2", "padding": "20px"},
            ),
        
            # --- Right: Controls ---
            html.Div(           
                [                                
                    # Legend header
                    html.Label("Legend:"),
                    html.Div(                        
                        id="legend-div",
                        style={
                            "marginBottom": "30px",
                            "textAlign": "center",
                            "fontWeight": "bold",
                            "borderBottom": "2px solid #ccc",
                            "paddingBottom": "10px"
                        }
                    ),

                    # Controls container
                    html.Div(
                        [                        
                            html.Label("Metric:"),
                            dcc.Dropdown(
                                id='category-dropdown',
                                options=[{'label': c, 'value': c} for c in metric_input],
                                value= param3_sel,
                                clearable=False,
                                style={'width': '200px'}
                            ),
                            html.Br(),
                        
                            html.Label("Number of neighbors:"),
                            dcc.Slider(
                                id='param1-slider',            
                                step=None,
                                value=param1_sel,
                                marks={i: str(i) for i in neighbors_input},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Br(),
                        
                            html.Label("Minimum distance:"),
                            dcc.Slider(
                                id='param2-slider',            
                                step=None,
                                value=param2_sel,
                                marks={i: str(i) for i in np.round(min_dist_input, decimals=3)},
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],				
                    ),
                ],
                style={		
                    "flex": "2",
                    "padding": "20px",
                    "borderLeft": "2px solid #ddd",
                    "display": "flex",
                    "flexDirection": "column",
                    "justifyContent": "center", #flex-start                                                        
                    }
            ),
        ],	
    )

    #endregion

    #region Callback for updating the plot

    @app.callback(
        Output('scatter-plot', 'figure'),
        Output("legend-div", "children"),
        Input('param1-slider', 'value'),
        Input('param2-slider', 'value'),
        Input('category-dropdown', 'value'),
    )
    def update_plot(param1, param2, category):
        

        # Filter
        idx1 = (table_book[var1] == param1)    
        idx2 = find_nearest_numpy(table_book[var2], param2)   
        idx3 = (table_book[var3] == category)
        idx = idx1 & idx2 & idx3        
        numeric_indices_np_where = int(np.where(idx)[0][0])    

        path_temp = table_book.loc[numeric_indices_np_where, "folderpath"]    
        path_temp2 = os.path.basename(path_temp)
        
        col_names = table_interactive2.columns.tolist()
        coordinate_cols = list(filter(lambda s: path_temp2 in s, col_names))	
        coordinate_cols.extend(['classif', variable_legend1])
        
        #data
        filtered = table_interactive2[coordinate_cols]   
        filtered.head()
        #Scatter plot
        #default
        box_size = 650 #plot size
        variable_interactive = variable_legend1

        hover_names = list(filtered.columns)	
        dict_names = collections.OrderedDict.fromkeys(hover_names, True)
        dict_names["classif"] = False	

        fig = px.scatter(filtered, x= coordinate_cols[0], y= coordinate_cols[1], color='classif',
                    hover_name= variable_interactive, 
                    hover_data= dict_names,								 
                    width=box_size, height=box_size,
                    title=f"UMAP using: Metric: {category}, Neighbors: {param1}, Min. Distance: {param2:.2f}",
                    )
        
        #Settings
        fig.update_xaxes(range=[x_min, x_max], autorange=False, showgrid=True, zeroline=True, zerolinecolor='black', zerolinewidth=2)
        fig.update_yaxes(range=[y_min, y_max], autorange=False, showgrid=True, zeroline=True, zerolinecolor='black', zerolinewidth=2)	
        fig.update_traces(marker = dict(size= markerSize, color= classif_colours))    
        
        fig.update_layout(		
            
            hoverlabel=dict(
                bgcolor='rgba(255,255,255,0.65)', 
                font=dict(color='black')
                ),
            transition_duration=100, 
            autosize=False, 
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor="white",

            xaxis_gridcolor='darkgray',  # Set x-axis grid color
            yaxis_gridcolor='darkgray',
            xaxis=dict(
                    # tickfont=dict(color="red"),  # Color of tick labels
                    linecolor="blue",            # Color of the axis line
                    # mirror=True,                 # Show axis line on both sides
                    # ticks="outside",             # Show ticks outside the axis line
                    # tickcolor="green"            # Color of the tick marks
                ),
            yaxis=dict(				
                    linecolor="blue", 				
                ),
            )
        
        # Create a mini legend (above controls)	
        legend_html = html.Div(
            [
                html.Div(
                    [
                        html.Span(
                            style={
                                "display": "inline-block",
                                "width": "15px",
                                "height": "15px",
                                "backgroundColor": colour_map[i % len(colour_map)],
                                "marginRight": "8px",
                            }
                        ),
                        html.Span(cls),
                    ],
                    style={"marginBottom": "5px"},
                )
                for i, cls in enumerate(list_unique)
            ]
        )

        return fig, legend_html

    return app
