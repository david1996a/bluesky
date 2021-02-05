import bluesky as bs 
bs.init("server-headless")

for i in range(15):
	print(i)
	bs.sim.step()
	bs.net.step()
