import folium
from folium.plugins import MarkerCluster

def make_map(df):
    #Create the base Map
    m = folium.Map(location=[48.866667,2.333333], tiles='OpenStreetMap', zoom_start=12)

    markerCluster = MarkerCluster().add_to(m)
    for i, row in df.iterrows():
        lat = df.at[i,'latitude']
        lng = df.at[i,'longitude']

        compteur = df.at[i,'id_compteur']

        sum_day = df.at[i,'mean_day']
        if sum_day > 1236:
            if sum_day > 1962:
                color = 'red'
            else:
                color = 'orange'
        else:
            if sum_day < 879:
                color = 'green'
            else:
                color = 'yellow'

        folium.Marker(location=[lat, lng], popup=sum_day, icon=folium.Icon(color=color)).add_to(markerCluster)

    m.save("bike_counts_by_day.html")