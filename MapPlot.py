import matplotlib.pyplot as plt 
import numpy as np 
import cartopy.crs as crs
import cartopy

class MapPlot:

	def __init__(self):
		self.figure = plt.figure(figsize=(15,13))
		self.central_lon = -10
		self.central_lat =  45
		self.projection = crs.PlateCarree()
		#self.ax = self.figure.add_subplot(1,1,1,projection=crs.EuroPP())
		self.ax = plt.axes(projection = self.projection)
		self.extent = [-40,20,30,60]
		self.ax.set_extent(self.extent)
		self.ax.coastlines(resolution='50m')

	def plot_initial(self, lat, lon, heading):
		self.point = plt.quiver(lat, lon, 0.1, 0.1, angles = heading+90, transform=crs.PlateCarree())

	def update_planes(self, lat, lon, heading, acid):
		self.ax.quiver(lat, lon, 1, 0, angles = heading+90, transform=crs.PlateCarree(), scale=50)
		#plt.text(lat, lon, acid, transform=crs.PlateCarree())
		idx=0
		for ac in acid:
			self.ax.text(lat[idx], lon[idx], ac, transform=crs.PlateCarree())
			idx += 1

	def update_waypoints(self, acidx, lat, lon):
		plt.plot(lat, lon, 'bo', transform=crs.PlateCarree())

	def update(self, lat, lon, heading, acid):
		self.figure.clear()
		self.ax = plt.axes(projection = self.projection)
		self.ax.set_extent(self.extent)
		#self.extent = [-40,20,30,60]
		#self.ax.stock_img()
		self.ax.coastlines(resolution='50m')
		#self.ax.set_extent(self.extent)
		self.update_planes(lat, lon, heading, acid)
		#self.update_waypoints()
		plt.draw()
		plt.pause(.001)

