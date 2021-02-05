import bluesky 
bs.init("sim-detached")

for i in range(10):
	bs.sim.step()
	bs.net.step()

print("Ending script...")
