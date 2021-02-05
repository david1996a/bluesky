import MapPlot as mapplt
import time 
import numpy as np

render_plot = mapplt.MapPlot()

time.sleep(2)
n_planes = 5
for i in range(50):
	lat = np.random.randint(33, 65, n_planes)
	lon = np.random.randint(-10, 40, n_planes)
	heading = np.random.randint(0,359, n_planes)
	print(lat, lon)
	render_plot.update(lat, lon, heading, range(n_planes))
