import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

# Importing datasets
ap_codes = pd.read_csv('airport_codes.csv')
al_codes = pd.read_csv('airline_codes.csv')
ap_locations = pd.read_csv('airport_locations.csv')
network_info = pd.read_csv('network_info.csv')
airlines_data = pd.read_csv('airlines_data.csv')
one_d_data = pd.read_csv('1d-data.csv')

# Function Declaration
def folium_static(fig, width=875, height=800, title='', colors=[], labels=[]): 
    if isinstance(fig, folium.Map): 
        fig = folium.Figure().add_child(fig) 
        add_categorical_legend(m, title, colors, labels)
        return components.html( 
            fig.render(), height=(fig.height or height) + 10, width=width 
            )

def add_categorical_legend(folium_map, title, colors, labels):
    if len(colors) != len(labels):
        raise ValueError("colors and labels must have the same length.")

    color_by_label = dict(zip(labels, colors))
    
    legend_categories = ""     
    for label, color in color_by_label.items():
        legend_categories += f"<li><span style='background:{color}'></span>{label}</li>"
        
    legend_html = f"""
    <div id='maplegend' class='maplegend'>
      <div class='legend-title'>{title}</div>
      <div class='legend-scale'>
        <ul class='legend-labels'>
        {legend_categories}
        </ul>
      </div>
    </div>
    """
    script = f"""
        <script type="text/javascript">
        var oneTimeExecution = (function() {{
                    var executed = false;
                    return function() {{
                        if (!executed) {{
                             var checkExist = setInterval(function() {{
                                       if ((document.getElementsByClassName('leaflet-top leaflet-right').length) || (!executed)) {{
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.display = "flex"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].style.flexDirection = "column"
                                          document.getElementsByClassName('leaflet-top leaflet-right')[0].innerHTML += `{legend_html}`;
                                          clearInterval(checkExist);
                                          executed = true;
                                       }}
                                    }}, 100);
                        }}
                    }};
                }})();
        oneTimeExecution()
        </script>
      """
   

    css = """

    <style type='text/css'>
      .maplegend {
        z-index:9999;
        float:right;
        background-color: rgba(255, 255, 255, 1);
        border-radius: 5px;
        border: 2px solid #bbb;
        padding: 10px;
        font-size:12px;
        positon: relative;
      }
      .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
      .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
      .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
      .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 0px solid #ccc;
        }
      .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
      .maplegend a {
        color: #777;
        }
    </style>
    """

    folium_map.get_root().header.add_child(folium.Element(script + css))

    return folium_map    
# Data cleaning
ap_codes = ap_codes.merge(ap_locations.loc[:, ['IATA', 'AIRPORT', 'CITY', 'STATE', 'LATITUDE', 'LONGITUDE']], left_on='Airport Code', right_on='IATA', how='left')
ap_codes.columns = ap_codes.columns.str.lower()
ap_codes.columns = ap_codes.columns.str.replace(' ', '_')

# Figures
airline_asm = go.Figure()
airline_rpm = go.Figure()

cmap = {'AS': '#8c0dbf',
        'G4': '#ba853b',
        'AA': '#30e8d2',
        'DL': '#04ade5',
        'MQ': '#e8ca45',
        'EV': '#300689',
        'F9': '#c0d142',
        'HA': '#452dfc',
        'B6': '#02f26e',
        'OO': '#0d913d',
        'WN': '#e54475',
        'NK': '#2d4791',
        'UA': '#c6650f'}

asm_rpm_dropdown = [{'label': 'All', 'method': 'update','args':[{'visible': [True, True, True, True, True, True, True, True, True, True, True, True, True]}, {'title': 'All Airlines'}]},
                    {'label': 'Alaska Airlines', 'method': 'update','args':[{'visible': [True, False, False, False, False, False, False, False, False, False, False, False, False]}, {'title': 'Alaska Airlines'}]},  
                    {'label': 'Allegient Air', 'method': 'update','args':[{'visible': [False, True, False, False, False, False, False, False, False, False, False, False, False]}, {'title': 'Allegient Air'}]},
                    {'label': 'American Airlines', 'method': 'update','args':[{'visible': [False, False, True, False, False, False, False, False, False, False, False, False, False]}, {'title': 'American Airlines'}]},
                    {'label': 'Delta Airlines', 'method': 'update','args':[{'visible': [False, False, False, True, False, False, False, False, False, False, False, False, False]}, {'title': 'Delta Airlines'}]},
                    {'label': 'Frontier Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, True, False, False, False, False, False, False, False, False]}, {'title': 'Frontier Airlines'}]},
                    {'label': 'Envoy Air', 'method': 'update','args':[{'visible': [False, False, False, False, False, True, False, False, False, False, False, False, False]}, {'title': 'Envoy Air'}]},
                    {'label': 'ExpressJet Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, True, False, False, False, False, False, False]}, {'title': 'ExpressJet Airlines'}]},
                    {'label': 'Hawaiian Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, False, True, False, False, False, False, False]}, {'title': 'Hawaiian Airlines'}]},
                    {'label': 'JetBlue Airways', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, False, False, True, False, False, False, False]}, {'title': 'JetBlue Airways'}]},
                    {'label': 'SkyWest Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, False, False, False, True, False, False, False]}, {'title': 'SkyWest Airlines'}]},
                    {'label': 'Southwest Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, False, False, False, False, True, False, False]}, {'title': 'Southwest Airlines'}]},
                    {'label': 'Spirit Airlines', 'method': 'update','args':[{'visible': [False, False, False, False, False, False, False, False, False, False, False, True, False]}, {'title': 'Spirit Airlines'}]},  
                    {'label': 'United Airlines', 'method': "update",'args':[{"visible": [False, False, False, False, False, False, False, False, False, False, False, False, True]}, {'title': 'United Airlines'}]}]

for column in airlines_data.columns[airlines_data.columns.str.contains('asm')]:
    airline_asm.add_trace(go.Line(x=airlines_data['Date'], y=airlines_data[column], name=column[:2].upper(), line=dict(color=cmap[column[:2].upper()])))
airline_asm.update_layout(legend={'title': 'Airline'}, title={'text':'Available Seat Miles per airline over time'}, xaxis={'rangeslider':{'visible':True}}, 
                          updatemenus=[{'type': "dropdown",'x': 1.10, 'y': 1.2,'showactive': True,'active': 0,'buttons': asm_rpm_dropdown}])
airline_asm.update_xaxes(title={'text': 'Years'})
airline_asm.update_yaxes(title={'text': 'Available Seat Miles'})

for column in airlines_data.columns[airlines_data.columns.str.contains('rpm')]:
    airline_rpm.add_trace(go.Line(x=airlines_data['Date'], y=airlines_data[column], name=column[:2].upper(), line=dict(color=cmap[column[:2].upper()])))
airline_rpm.update_layout(legend={'title': 'Airline'}, title={'text':'Revenue Passenger Miles per airline over time'}, xaxis={'rangeslider':{'visible':True}},
                          updatemenus=[{'type': "dropdown",'x': 1.10, 'y': 1.2,'showactive': True,'active': 0,'buttons': asm_rpm_dropdown}])
airline_rpm.update_xaxes(title={'text': 'Years'})
airline_rpm.update_yaxes(title={'text': 'Revenue Passenger Miles'})

# Boxplot flights
box = go.Figure()

for column in one_d_data.columns[one_d_data.columns.str.contains('flight')]:
    box.add_trace(go.Box(y= one_d_data[column], name=al_codes[al_codes['Airline Code'] == column[:2].upper()]['Airline Name'].values[0], line=dict(color=cmap[column[:2].upper()])))
box.update_layout(showlegend=False, updatemenus=[{'type': "dropdown",'x': 1., 'y': 1.2,'showactive': True,'active': 0,'buttons': asm_rpm_dropdown}],
                  title={'text': 'Boxplot of monthly flights per airline'})
box.update_xaxes({'title':'Airline'})
box.update_yaxes({'title':'Flights per month'})

# Streamlit initialisation
st.set_page_config(layout='wide', page_title='Domestic air travel in the USA')
st.title('Domestic air travel in the USA')
col1, col2, col3 = st.columns([4, 1, 4])
with col1:
    map1, map2 = st.columns(2)
    with map1:
        ap_select = st.multiselect('Select airport(s)', ['All'] + [*ap_codes['iata']], 'EWR')
        if "All" in ap_select:
          ap_select = ap_codes['iata']
    with map2:
        al_select = st.multiselect('Select airline(s)', ['All'] + [*al_codes['Airline Code']], 'UA')
        if "All" in al_select:
          al_select = al_codes['Airline Code']

    m = folium.Map([38.912226, -97.828435], zoom_start=4)

    data = network_info[(network_info['opcarrier'].isin(al_select)) & (network_info['origin'].isin(ap_select) | (network_info['dest'].isin(ap_select)))]

    for i, row in ap_codes.iterrows():
        folium.Circle([row['latitude'], row['longitude']], radius=20000, popup=row['iata'] + ', ' + row['airport'], fill = True, fill_opacity = 1, color='#000').add_to(m)

    for i, row in data.iterrows():
        folium.PolyLine([[row['origin_lat'], row['origin_lng']], [row['dest_lat'], row['dest_lng']]], color=cmap[row['opcarrier']]).add_to(m)
        
    folium_static(m, 750, 400, 'Airline', cmap.values(), cmap.keys())

with col2:
    st.markdown( '''<p><b>Airline IATA codes</b><br>
                    AS, Alaska Airlines<br>
                    G4, Allegient Air <br>
                    AA, American Airlines<br>
                    DL, Delta Airlines<br>
                    MQ, Frontier Airlines<br>
                    EV, Envoy Air<br>
                    F9, ExpressJet Airlines<br>
                    HA, Hawaiian Airlines<br>
                    B6, JetBlue Airways<br>
                    OO, SkyWest Airlines<br>
                    WN, Southwest Airlines<br>
                    NK, Spirit Airlines<br>
                    UA, United Airlines</p>''' , unsafe_allow_html=True)

with col3:
    asm_rpm = st.radio('Choose ASM or RPM', ['ASM', 'RPM'], 0)
    if asm_rpm == 'ASM':
      st.plotly_chart(airline_asm, True)
    if asm_rpm ==  'RPM':
      st.plotly_chart(airline_rpm, True)
    

col1, col2 = st.columns(2)
with col1:
    # Boxplot of monthly passengers per flight
    st.plotly_chart(box, True)

    # Regression predictions
    predict1, predict2 = st.columns(2)
    with predict1:
      predict_radio = st.radio('Select statistic', ['ASM', 'RPM'])
      if predict_radio == 'ASM':
        suffix = '_asm'
      else:
        suffix = '_rpm'
      
    with predict2:
      predict_select = st.selectbox('Choose airline', al_codes['Airline Name'])
      prefix = al_codes[al_codes['Airline Name'] == predict_select]['Airline Code'].values[0].lower()
    
    predict_df = one_d_data[one_d_data['date'] >= '2010'].copy()
    predict_df['date'] = pd.to_datetime(predict_df['date'])
    input1 = predict_df[prefix + suffix].to_list()
    X1 = predict_df['date']
    y1 = np.log10(input1)

    model = LinearRegression()
    model.fit(X1.values.reshape(-1,1), y1)

    X_predict1 = pd.date_range(start='2017-01-01', end='2030-01-01', freq='MS')
    y_predict1 = model.predict(X_predict1.values.reshape(-1,1).astype("float64"))
    df = pd.DataFrame()
    df['date'] = X_predict1
    df['predictions'] = 10**(y_predict1)

    predict = go.Figure()
    predict.add_trace(go.Scatter(x=predict_df['date'], y= predict_df[prefix + suffix], line={'color':cmap[prefix.upper()]}, name='Historical data'))
    predict.add_trace(go.Scatter(x=df['date'], y=df['predictions'], line={'dash':'dash', 'color':cmap[prefix.upper()]}, name='Predictions'))
    predict.update_layout(title={'text':'Predicted growth of '+ suffix.replace('_', '').upper() + ' per airline'})
    predict.update_xaxes({'title':'Year'})
    predict.update_yaxes({'title':suffix.replace('_', '').upper()})

    st.plotly_chart(predict, True)

with col2:
    eff_slider = st.select_slider('Select time period', one_d_data['date'], [one_d_data['date'].iloc[0], one_d_data['date'].iloc[-1]])
    eff_df = one_d_data[(one_d_data['date']>=eff_slider[0]) & (one_d_data['date'] < eff_slider[1])]
    # Efficiency bar chart
    eff_data = []
    for column in eff_df.columns[eff_df.columns.str.contains('eff')]:
      eff_data.append([column[:2].upper(), al_codes[al_codes['Airline Code'] == column[:2].upper()]['Airline Name'].values[0], eff_df[column].mean()])
    eff_data = pd.DataFrame(eff_data, columns=['Airline Code', 'Airline', 'Seating Efficiency (%)'])


    eff = px.bar(eff_data, 'Airline', 'Seating Efficiency (%)', color='Airline Code', color_discrete_map=cmap, title='Seating efficiency per airline')

    st.plotly_chart(eff, True)
    # Average passengers
    pass_count = go.Figure()

    for column in eff_df.columns[eff_df.columns.str.contains('pass')]:
        pass_count.add_trace(go.Histogram(x=eff_df[column], name=column[:2].upper(), marker=dict(color=cmap[column[:2].upper()])))
    pass_count.update_layout(legend={'title':'Airline Code'}, title={'text':'Average monthly amount of passengers per flight per airline'},
                            xaxis={'rangeslider':{'visible':True}}, updatemenus=[{'type': "dropdown",'x': 1., 'y': 1.2,'showactive': True,'active': 0,'buttons': asm_rpm_dropdown}])
    pass_count.update_xaxes({'title':'Amount of passengers'})
    pass_count.update_yaxes({'title':'Count'})
    st.plotly_chart(pass_count, True)